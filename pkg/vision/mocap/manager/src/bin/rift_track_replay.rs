//! rift_track_replay: do the markers track in 3D wherever two cameras see them?
//!
//! Reads a manager log: a recording (<data dir>/recording/<id>.log: the
//! system state, then `blobs` and `points` entries interleaved) or a wanding
//! log (<id>-wanding.log, blobs only), and sets the frames where cameras see
//! markers against the frames with tracked points.
//!
//!   rift_track_replay LOG [--saved CONFIG_PB] [--rematch] [--min-matches N]
//!
//! The calibration is the log's system state config, with --saved's config
//! patch (<data dir>/config.pb) merged over it as the manager merges it when
//! it starts (ManagerConfigContainer::merge_from: per camera intrinsics,
//! extrinsics, enabled and name). When the log has no `points` entries, or
//! with --rematch, BlobMatcher alone runs over the `blobs` entries in order
//! (as inst.rs matching_task does, but without the rigid body and skeleton
//! trackers' back-propagation) and its points are used; with --rematch on a
//! log that has points, the rematched non-ghost points are counted against
//! the logged ones. --min-matches sets matching's min_num_matches for that
//! run (the cameras a new point needs; with 3, two cameras only keep up a
//! point three started, as min_num_rematches allows 2) and implies --rematch.
//! Lines for reading start with "track:".
//!
//! Then:
//!
//! - frames: blobs and points entries, the frame period (median step), the
//!   duration, gaps in the grid, and blobs entries with no points entry of the
//!   same timestamp and the reverse (the recorder reads at most one of each
//!   per wake, so it can drop some);
//! - per camera: frames it is in, frames with a blob, frames by blob count;
//! - loose: frames where two or more cameras each have a blob, against frames
//!   with a non-ghost point;
//! - the wand as the probe: per frame the T finder runs on each camera's
//!   blobs (sightings under wanding's max_wand_frame_error kept). In a frame
//!   where two or more cameras hold the T its four markers are expected to
//!   track; a marker is tracked when a non-ghost point of that frame projects
//!   into a holding camera within matching's max_reprojection_error of the
//!   marker's blob. Counts overall, by the set of holding cameras, frames
//!   with all four / some / none, and markers tracked in frames where only
//!   one camera holds the T;
//! - track continuity: point ids, ghosts per frame, track lengths (frames
//!   with cameras), tracks started against the four markers;
//! - 3D accuracy: the wand's three lengths from the tracked points in frames
//!   with all four markers tracked, against the wand config;
//! - where: the tracked points' bounding box, and expected / tracked markers
//!   on a 0.5 m floor grid (the marker's world position from the holding
//!   camera's own pose fit of the T, carried through its extrinsics), one
//!   line per row of cells, to set against the coverage simulator.

extern crate common;
#[macro_use]
extern crate macros;

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use cluster_client::id::entity_id_to_string;
use common::errors::*;
use file::LocalPath;
use math::matrix::{vec2d, Vector3d};
use mocap_manager::matching::{BlobMatcher, CameraParameters, TrackedPoint};
use mocap_manager::{read_log_file, BlobPatternFinder, BlobPatternMatch, ManagerConfigContainer};
use mocap_proto::mocap::*;
use vision::CameraExtrinsics;

const CELL: f64 = 0.5;

struct Sighting {
    camera_id: u64,
    m: BlobPatternMatch,
}

fn median_u64(v: &mut Vec<u64>) -> u64 {
    v.sort();
    if v.is_empty() { 0 } else { v[v.len() / 2] }
}

fn pct(a: usize, b: usize) -> f64 {
    if b == 0 { 0. } else { 100. * a as f64 / b as f64 }
}

#[executor_main]
async fn main() -> Result<()> {
    let mut log_path: Option<String> = None;
    let mut saved_path: Option<String> = None;
    let mut rematch = false;
    let mut min_matches: Option<u32> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--saved" {
            saved_path = Some(args.next().ok_or_else(|| err_msg("--saved needs a path"))?);
        } else if arg == "--rematch" {
            rematch = true;
        } else if arg == "--min-matches" {
            let v = args.next().ok_or_else(|| err_msg("--min-matches needs a number"))?;
            min_matches = Some(v.parse::<u32>()?);
            rematch = true;
        } else if arg.starts_with("--") {
            return Err(format_err!("unknown option {}", arg));
        } else {
            log_path = Some(arg);
        }
    }
    let log_path = log_path.ok_or_else(|| err_msg("usage: rift_track_replay LOG [--saved CONFIG_PB] [--rematch] [--min-matches N]"))?;

    let entries = read_log_file(LocalPath::new(&log_path)).await?;
    if entries.is_empty() || !entries[0].has_system_state() {
        return Err(err_msg("the log doesn't start with a system state"));
    }
    let state = entries[0].system_state().clone();

    // The config: the log's, with the saved patch merged as MocapManager::create does.
    let mut config = ManagerConfigContainer::create(state.config())?;
    if let Some(path) = &saved_path {
        use protobuf::StaticMessage;
        let saved = MocapManagerConfig::parse(&std::fs::read(path)?)?;
        config.merge_from(&saved)?;
        println!("track: merged {} ({} cameras, {} with extrinsics)", path, saved.per_camera().len(),
            saved.per_camera().iter().filter(|c| c.has_extrinsics()).count());
    }

    let name = |id: u64| -> String {
        let n = config.per_camera().iter().find(|c| c.camera_id() == id)
            .map(|c| c.name().to_string()).unwrap_or_default();
        if n.is_empty() { entity_id_to_string(id).unwrap_or_else(|| format!("{:x}", id)) } else { n }
    };

    let blobs: Vec<&ReadBlobsResponse> = entries[1..].iter().filter(|e| e.has_blobs()).map(|e| e.blobs()).collect();
    let logged: Vec<&ReadTrackedPointsResponse> = entries[1..].iter().filter(|e| e.has_points()).map(|e| e.points()).collect();

    let mc = config.matching();
    println!("track: {}: {} blobs entries, {} points entries", log_path, blobs.len(), logged.len());
    println!("track: matching min_num_matches {}, min_num_rematches {}, max_reprojection_error {} px, min_marker_distance {} m, ghosting {} s, max_point_jump {} m, fast_rematch {}",
        mc.min_num_matches(), mc.min_num_rematches(), mc.max_reprojection_error(), mc.min_marker_distance(),
        mc.ghosting_limit_secs(), mc.max_point_jump(), mc.fast_rematch_old_points());

    // Camera parameters as inst.rs matching_task builds them: every camera
    // with extrinsics and intrinsics in the config (no enabled filter).
    let mut params = vec![];
    for (cam_id, extrinsics) in config.camera_extrinsics() {
        let intrinsics = match config.camera_intrinsics().get(cam_id) {
            Some(v) => v,
            None => continue,
        };
        params.push(CameraParameters { id: *cam_id, extrinsics: extrinsics.clone(), intrinsics: intrinsics.clone() });
    }
    params.sort_by_key(|p| name(p.id));
    for p in &params {
        let pos = p.extrinsics.position();
        println!("track: {} at {:.3} {:.3} {:.3} m, f {:.1}, enabled {}", name(p.id), pos.x(), pos.y(), pos.z(),
            p.intrinsics.focal_length.x(), config.camera_enabled(p.id));
    }
    let param_of: HashMap<u64, &CameraParameters> = params.iter().map(|p| (p.id, p)).collect();

    // The points used, by frame timestamp.
    let mut points: HashMap<u64, Vec<TrackedPoint>> = HashMap::new();
    let run_matcher = logged.is_empty() || rematch;
    if run_matcher {
        let mut matching = config.matching().clone();
        if let Some(n) = min_matches {
            matching.set_min_num_matches(n);
            println!("track: min_num_matches set to {} for the matcher", n);
        }
        let mut matcher = BlobMatcher::new(&matching);
        matcher.set_camera_parameters(&params);
        let mut last = 0u64;
        let mut skipped = 0;
        for b in &blobs {
            if b.frame_timestamp() < last {
                skipped += 1;
                continue;
            }
            last = b.frame_timestamp();
            let pts = matcher.run(b);
            points.insert(b.frame_timestamp(), pts);
        }
        println!("track: points from BlobMatcher alone over the blobs entries ({}; no rigid body or skeleton back-propagation){}",
            if logged.is_empty() { "the log has no points" } else { "--rematch" },
            if skipped > 0 { format!(", {} entries out of time order skipped", skipped) } else { String::new() });
        if !logged.is_empty() {
            let logged_n: usize = logged.iter().map(|p| p.points().iter().filter(|q| !q.camera_ids().is_empty()).count()).sum();
            let re_n: usize = points.values().map(|v| v.iter().filter(|q| !q.camera_ids.is_empty()).count()).sum();
            println!("track: non-ghost points: rematched {}, logged {} ({:+.1}%)", re_n, logged_n, pct(re_n, logged_n) - 100.);
        }
    } else {
        for p in &logged {
            let mut v = vec![];
            for q in p.points() {
                v.push(TrackedPoint::from_proto(q)?);
            }
            points.insert(p.frame_timestamp(), v);
        }
        println!("track: points as logged");
    }

    // 1. Frames.
    let mut ts: Vec<u64> = blobs.iter().map(|b| b.frame_timestamp()).collect();
    ts.sort();
    let dup = ts.len() - { let mut d = ts.clone(); d.dedup(); d.len() };
    ts.dedup();
    let mut steps: Vec<u64> = ts.windows(2).map(|w| w[1] - w[0]).collect();
    let period = median_u64(&mut steps);
    if period > 0 {
        let mut gaps = 0;
        let mut missing = 0u64;
        for w in ts.windows(2) {
            let n = ((w[1] - w[0]) as f64 / period as f64).round() as u64;
            if n >= 2 {
                gaps += 1;
                missing += n - 1;
            }
        }
        println!("track: frame period {:.1} us ({:.3} fps), {:.1} s; {} gaps of 2+ periods, {} frames missing from the grid ({:.2}%){}",
            period as f64 / 1e3, 1e9 / period as f64, (ts[ts.len() - 1] - ts[0]) as f64 / 1e9, gaps, missing,
            pct(missing as usize, ts.len() + missing as usize),
            if dup > 0 { format!("; {} blobs entries repeat a timestamp", dup) } else { String::new() });
    }
    if !logged.is_empty() {
        let bset: HashSet<u64> = ts.iter().cloned().collect();
        let pset: HashSet<u64> = logged.iter().map(|p| p.frame_timestamp()).collect();
        let both = bset.intersection(&pset).count();
        if both == 0 {
            println!("track: no points entry shares a timestamp with any blobs entry: the two streams don't line up");
        } else {
            println!("track: blobs entries with no points entry {}, points entries with no blobs entry {}, both {}",
                bset.difference(&pset).count(), pset.difference(&bset).count(), both);
        }
    }

    // 2. Per camera.
    let mut cams: BTreeMap<String, (usize, usize, [usize; 6])> = BTreeMap::new();
    for b in &blobs {
        for c in b.cameras() {
            let s = cams.entry(name(c.camera_id())).or_default();
            let n = c.results().blobs().len();
            s.0 += 1;
            if n > 0 { s.1 += 1; }
            s.2[n.min(5)] += 1;
        }
    }
    for (n, (inn, with, by)) in &cams {
        println!("track: {} in {} frames, {} with blobs; by blobs 0:{} 1:{} 2:{} 3:{} 4:{} 5+:{}",
            n, inn, with, by[0], by[1], by[2], by[3], by[4], by[5]);
    }

    // 3. Loose measure.
    {
        let (mut two, mut tracked, mut both) = (0, 0, 0);
        for b in &blobs {
            let a = b.cameras().iter().filter(|c| !c.results().blobs().is_empty()).count() >= 2;
            let t = points.get(&b.frame_timestamp())
                .map(|v| v.iter().any(|p| !p.camera_ids.is_empty())).unwrap_or(false);
            if a { two += 1; }
            if t { tracked += 1; }
            if a && t { both += 1; }
        }
        println!("track: loose: {} frames with 2+ cameras seeing blobs, {} with a non-ghost point, {} both",
            two, tracked, both);
    }

    // 4. The wand as the probe.
    let max_frame_error = config.wanding().max_wand_frame_error();
    let limit = mc.max_reprojection_error();
    let w = config.wand();
    println!("track: wand probe: T sightings under {} px, a marker tracked when a point projects within {} px of its blob",
        max_frame_error, limit);

    // The points (index into the frame's points, distance in px) that project
    // within the limit of a marker's blob in a holder.
    let candidates = |pts: &[TrackedPoint], cam: &CameraBlobResults, blob_index: usize, out: &mut Vec<(usize, f64)>| {
        let p = match param_of.get(&cam.camera_id()) { Some(p) => p, None => return };
        let blob = &cam.results().blobs()[blob_index];
        let at = vec2d(blob.x() as f64, blob.y() as f64);
        for (i, pt) in pts.iter().enumerate() {
            if pt.camera_ids.is_empty() {
                continue;
            }
            let local = p.extrinsics.transform(&pt.position);
            if local.z() <= 0. {
                continue;
            }
            let d = (p.intrinsics.project_point(&local) - &at).norm();
            if d <= limit {
                out.push((i, d));
            }
        }
    };

    let world_of = |s: &Sighting, k: usize| -> Option<Vector3d> {
        let p = param_of.get(&s.camera_id)?;
        let world_from = CameraExtrinsics::from_mat4x4(&p.extrinsics.to_mat4x4().inverse()?);
        let wand = CameraExtrinsics { rotation: s.m.rotation.clone(), translation: s.m.translation.clone() };
        Some(world_from.transform(&wand.transform(&s.m.points[k].reference_point)))
    };

    let (mut expected, mut tracked) = (0usize, 0usize);
    let mut by_set: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    let (mut all4, mut some, mut none) = (0, 0, 0);
    let (mut single_frames, mut single_tracked, mut single_others) = (0usize, 0usize, 0usize);
    let mut no_points_entry = 0;
    // Markers whose only nearby points were taken by nearer markers.
    let mut shared = 0usize;
    let mut lengths: [Vec<f64>; 3] = [vec![], vec![], vec![]];
    let mut grid: BTreeMap<(i64, i64), (usize, usize)> = BTreeMap::new();
    let mut by_holders = [0usize; 5];

    for b in &blobs {
        let mut sightings = vec![];
        for c in b.cameras() {
            let p = match param_of.get(&c.camera_id()) { Some(p) => p, None => continue };
            if let Some(m) = BlobPatternFinder::find_t_wand(w, &p.intrinsics, c.results()) {
                if m.error <= max_frame_error {
                    sightings.push(Sighting { camera_id: c.camera_id(), m });
                }
            }
        }
        sightings.sort_by_key(|s| name(s.camera_id));
        by_holders[sightings.len().min(4)] += 1;
        if sightings.is_empty() {
            continue;
        }
        let pts = match points.get(&b.frame_timestamp()) {
            Some(v) => v,
            None => { if sightings.len() >= 2 { no_points_entry += 1; } continue; }
        };
        let cam_of = |id: u64| b.cameras().iter().find(|c| c.camera_id() == id).unwrap();

        // Per marker: the point tracking it. Nearest first over every holder,
        // one point per marker, so a point near two markers' blobs (in
        // different cameras) only tracks the nearer.
        let mut pairs: Vec<(f64, usize, usize)> = vec![];
        for k in 0..4 {
            let mut c = vec![];
            for s in &sightings {
                candidates(pts, cam_of(s.camera_id), s.m.points[k].blob_index, &mut c);
            }
            pairs.extend(c.into_iter().map(|(i, d)| (d, k, i)));
        }
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut marker_point: [Option<usize>; 4] = [None; 4];
        let mut had_candidate = [false; 4];
        for (_, k, i) in pairs {
            had_candidate[k] = true;
            if marker_point[k].is_none() && !marker_point.contains(&Some(i)) {
                marker_point[k] = Some(i);
            }
        }
        if sightings.len() >= 2 {
            shared += (0..4).filter(|k| had_candidate[*k] && marker_point[*k].is_none()).count();
        }
        let n = marker_point.iter().filter(|m| m.is_some()).count();

        if sightings.len() == 1 {
            single_frames += 1;
            single_tracked += n;
            if b.cameras().iter().any(|c| c.camera_id() != sightings[0].camera_id && c.results().blobs().len() >= 3) {
                single_others += 1;
            }
            continue;
        }

        expected += 4;
        tracked += n;
        let set = sightings.iter().map(|s| name(s.camera_id)).collect::<Vec<_>>().join("+");
        let e = by_set.entry(set).or_default();
        e.0 += 4;
        e.1 += n;
        match n { 4 => all4 += 1, 0 => none += 1, _ => some += 1 }

        for k in 0..4 {
            let ws: Vec<Vector3d> = sightings.iter().filter_map(|s| world_of(s, k)).collect();
            if ws.is_empty() { continue; }
            let mut mean = ws[0].clone();
            for v in &ws[1..] { mean = mean + v.clone(); }
            let mean = mean * (1.0 / ws.len() as f64);
            let cell = ((mean.x() / CELL).floor() as i64, (mean.y() / CELL).floor() as i64);
            let g = grid.entry(cell).or_default();
            g.0 += 1;
            if marker_point[k].is_some() { g.1 += 1; }
        }

        if n == 4 {
            let p = |k: usize| &pts[marker_point[k].unwrap()].position;
            lengths[0].push((p(1) - p(0)).norm() - w.left_arm_length());
            lengths[1].push((p(1) - p(2)).norm() - w.right_arm_length());
            lengths[2].push((p(1) - p(3)).norm() - w.bottom_length());
        }
    }

    println!("track: frames by cameras holding the T: 0:{} 1:{} 2:{} 3:{} 4+:{}",
        by_holders[0], by_holders[1], by_holders[2], by_holders[3], by_holders[4]);
    if no_points_entry > 0 {
        println!("track: {} frames with 2+ holders have no points entry and are left out", no_points_entry);
    }
    println!("track: 2+ holders: {} of {} markers tracked ({:.1}%); frames with all four {}, some {}, none {}; {} markers untracked because their only nearby point tracks another marker",
        tracked, expected, pct(tracked, expected), all4, some, none, shared);
    for (set, (e, t)) in &by_set {
        println!("track:   {}: {} of {} ({:.1}%)", set, t, e, pct(*t, *e));
    }
    println!("track: 1 holder: {} frames, {} markers with a tracked point anyway; in {} of the frames another camera has 3+ blobs",
        single_frames, single_tracked, single_others);

    // 5. Track continuity.
    {
        let mut frames_of: HashMap<u64, usize> = HashMap::new();
        let mut ghosts = 0usize;
        let mut nframes = 0usize;
        for v in points.values() {
            nframes += 1;
            for p in v {
                if p.camera_ids.is_empty() {
                    ghosts += 1;
                } else {
                    *frames_of.entry(p.id).or_default() += 1;
                }
            }
        }
        let mut lens: Vec<u64> = frames_of.values().map(|v| *v as u64).collect();
        let med = median_u64(&mut lens);
        let longest = lens.last().cloned().unwrap_or(0);
        let ids: BTreeSet<u64> = points.values().flat_map(|v| v.iter().map(|p| p.id)).collect();
        println!("track: {} point ids ({} ever seen by cameras), {:.2} ghosts per frame; track length median {} frames, longest {}; {} tracks started for 4 markers ({:.1} each)",
            ids.len(), frames_of.len(), ghosts as f64 / nframes.max(1) as f64, med, longest,
            frames_of.len(), frames_of.len() as f64 / 4.);
    }

    // 6. 3D accuracy.
    let names = ["centre-left", "centre-right", "centre-handle"];
    let refs = [w.left_arm_length(), w.right_arm_length(), w.bottom_length()];
    for i in 0..3 {
        let v = &lengths[i];
        let n = v.len().max(1) as f64;
        let mut abs: Vec<f64> = v.iter().map(|e| e.abs()).collect();
        abs.sort_by(|a, b| a.total_cmp(b));
        let at = |p: f64| if abs.is_empty() { 0. } else { abs[((abs.len() - 1) as f64 * p).round() as usize] * 1e3 };
        println!("track: {} ({:.1} mm): {} frames, mean error {:+.2} mm, RMS {:.2} mm; |error| median {:.2}, p95 {:.2}, max {:.2} mm, {} over 10 mm",
            names[i], refs[i] * 1e3, v.len(), v.iter().sum::<f64>() / n * 1e3,
            (v.iter().map(|e| e * e).sum::<f64>() / n).sqrt() * 1e3, at(0.5), at(0.95), at(1.0),
            abs.iter().filter(|e| **e > 0.01).count());
    }

    // 7. Where.
    {
        let mut lo = [f64::MAX; 3];
        let mut hi = [f64::MIN; 3];
        for v in points.values() {
            for p in v.iter().filter(|p| !p.camera_ids.is_empty()) {
                for j in 0..3 {
                    lo[j] = lo[j].min(p.position[j]);
                    hi[j] = hi[j].max(p.position[j]);
                }
            }
        }
        if lo[0] <= hi[0] {
            println!("track: tracked points span x {:.2}..{:.2}, y {:.2}..{:.2}, z {:.2}..{:.2} m",
                lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]);
        }
        println!("track: by {} m floor cell, row by row (y), x of the cell's low edge: tracked/expected", CELL);
        let rows: BTreeSet<i64> = grid.keys().map(|k| k.1).collect();
        for y in rows.iter().rev() {
            let cells: Vec<String> = grid.iter().filter(|(k, _)| k.1 == *y)
                .map(|(k, (e, t))| format!("{:+.1}:{}/{}", k.0 as f64 * CELL, t, e)).collect();
            println!("track:   y {:+.1}..{:+.1}: {}", *y as f64 * CELL, (*y + 1) as f64 * CELL, cells.join("  "));
        }
    }

    Ok(())
}
