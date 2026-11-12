//! rift_wand_replay: what did a wanding run see, and how well does it solve?
//!
//! The manager logs every merged blob frame of a wanding run to
//! <data dir>/recording/<id>-wanding.log, behind the system state it started
//! from. This reads one back and runs what wanding.rs runs: the T finder on
//! each camera's blobs, then WandingCalibrationSolver.
//!
//!   rift_wand_replay LOG [--config PATH] [--shift NAME=FRAMES]... [--saved CONFIG_PB]
//!
//! The config is the one in the log (what the manager ran with) unless
//! --config names a manager.txtpb to try instead; the cameras' saved
//! intrinsics and their status always come from the log. --shift moves one
//! camera's blobs FRAMES frames later (negative: earlier) before anything
//! else, to test whether that camera numbers its frames a slot off: if it
//! does, the shift that undoes it is the one that solves well. Lines for
//! reading start with "replay:". --saved reads the manager's saved config
//! patch (<data dir>/config.pb) and prints where it puts each camera, which
//! way it looks and which way is down in its picture, in the world Set Origin
//! left (z up from the floor), with the distances between cameras against
//! this log's solve: equal if that solve is the one saved.
//!
//! Then:
//!
//! - the wand and wanding config in force, and each camera's intrinsics;
//! - per camera: frames by blob count, how many held a T, the T's pose-fit
//!   error (the finder's reprojection RMS, px) and distance, and how many
//!   show the wand from its back (its +z away from the camera: for a wand
//!   that only shows one side, a reflection);
//! - frames where two or more cameras held a T inside max_wand_frame_error,
//!   by camera pair;
//! - the solve's error and camera positions, the error by camera, by how
//!   fast the wand moved in the image, and the worst sightings;
//! - a cross-check per camera pair that doesn't lean on the bundle
//!   adjustment's object poses: one camera's own pose fit of the wand, carried
//!   through the solved extrinsics into the other camera, against the blobs
//!   that camera saw (RMS px).

#[macro_use]
extern crate common;
#[macro_use]
extern crate macros;

use std::collections::{BTreeMap, HashMap};

use cluster_client::id::entity_id_to_string;
use common::errors::*;
use common::hash::FastHasherBuilder;
use file::LocalPath;
use math::matrix::vec2d;
use mocap_manager::calibration::{wanding_intrinsics, WandingCalibrationSolver};
use mocap_manager::{read_log_file, BlobPatternFinder, BlobPatternMatch, ManagerConfigContainer};
use mocap_proto::mocap::*;
use vision::CameraExtrinsics;

const ERROR_EDGES: [f64; 5] = [0.25, 0.5, 1.0, 2.0, 5.0];

#[derive(Default)]
struct CameraStats {
    /// Frames by blob count: 0, 1, 2, 3, 4, 5 or more.
    by_blobs: [usize; 6],
    found: usize,
    /// T sightings by pose-fit error, under each of ERROR_EDGES.
    by_error: [usize; 5],
    distance_min: f64,
    distance_max: f64,
    /// Sightings whose pose has the wand's marker side (+z) turned away from
    /// the camera, and of those, the ones inside wanding's error limit.
    back: usize,
    back_kept: usize,
}

struct Sighting {
    camera_id: u64,
    m: BlobPatternMatch,
}

#[executor_main]
async fn main() -> Result<()> {
    let mut log_path: Option<String> = None;
    let mut config_path: Option<String> = None;
    let mut shifts: Vec<(String, i64)> = vec![];
    let mut saved_path: Option<String> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--config" {
            config_path = Some(args.next().ok_or_else(|| err_msg("--config needs a path"))?);
        } else if arg == "--saved" {
            saved_path = Some(args.next().ok_or_else(|| err_msg("--saved needs a path"))?);
        } else if arg == "--shift" {
            let v = args.next().ok_or_else(|| err_msg("--shift needs NAME=FRAMES"))?;
            let (n, f) = v.split_once('=').ok_or_else(|| err_msg("--shift needs NAME=FRAMES"))?;
            shifts.push((n.to_string(), f.parse()?));
        } else if arg.starts_with("--") {
            return Err(format_err!("unknown option {}", arg));
        } else {
            log_path = Some(arg);
        }
    }
    let log_path = log_path.ok_or_else(|| err_msg("usage: rift_wand_replay LOG [--config PATH]"))?;

    let mut entries = read_log_file(LocalPath::new(&log_path)).await?;
    if entries.is_empty() || !entries[0].has_system_state() {
        return Err(err_msg("the log doesn't start with a system state"));
    }
    let state = entries[0].system_state().clone();

    // The config: the log's, or the given file with the log's cameras
    // (intrinsics, extrinsics, enabled) merged in.
    let logged = ManagerConfigContainer::create(state.config())?;
    let config = match &config_path {
        None => logged,
        Some(path) => {
            let mut base = MocapManagerConfig::default();
            protobuf::text::parse_text_proto(&std::fs::read_to_string(path)?, &mut base)?;
            base.clear_per_camera();
            let mut c = ManagerConfigContainer::create(&base)?;
            let mut patch = MocapManagerConfig::default();
            for cam in logged.per_camera() {
                patch.add_per_camera((**cam).clone());
            }
            c.merge_from(&patch)?;
            c
        }
    };

    let name = |id: u64| -> String {
        let n = config.per_camera().iter().find(|c| c.camera_id() == id)
            .map(|c| c.name().to_string()).unwrap_or_default();
        if n.is_empty() { entity_id_to_string(id).unwrap_or_else(|| format!("{:x}", id)) } else { n }
    };

    println!("replay: {} ({} frames), config from {}", log_path, entries.len() - 1,
        config_path.as_deref().unwrap_or("the log"));
    let w = config.wand();
    println!("replay: wand {} / {} / {} m, height {}; min distance {} px, bottom {} px, stretch {}, pose fit under {} px; wanding keeps under {} px",
        w.left_arm_length(), w.right_arm_length(), w.bottom_length(), w.height(),
        w.min_blob_center_pixel_distance(), w.min_bottom_line_pixel_distance(),
        w.max_projection_stretch(), w.max_reprojection_error(),
        config.wanding().max_wand_frame_error());

    // As inst.rs starts wanding.
    // --shift: rebuild the frames with the named cameras' blobs moved.
    if !shifts.is_empty() {
        let mut timestamps: Vec<u64> = entries[1..].iter().map(|e| e.blobs().frame_timestamp()).collect();
        timestamps.sort();
        timestamps.dedup();
        let mut steps: Vec<u64> = timestamps.windows(2).map(|w| w[1] - w[0]).collect();
        steps.sort();
        let step = steps[steps.len() / 2] as i64;

        let mut by_time: BTreeMap<u64, Vec<CameraBlobResults>> = BTreeMap::new();
        for entry in &entries[1..] {
            for cam in entry.blobs().cameras() {
                let frames = shifts.iter().find(|(n, _)| *n == name(cam.camera_id())).map(|(_, f)| *f).unwrap_or(0);
                let t = (entry.blobs().frame_timestamp() as i64 + frames * step) as u64;
                by_time.entry(t).or_default().push((**cam).clone());
            }
        }

        entries.truncate(1);
        for (t, cams) in by_time {
            let mut entry = MocapLogEntry::default();
            entry.blobs_mut().set_frame_timestamp(t);
            for cam in cams {
                entry.blobs_mut().add_cameras(cam);
            }
            entries.push(entry);
        }
        println!("replay: shifted {:?} frames of {} ns", shifts, step);
    }

    let intrinsics = wanding_intrinsics(&config, &state);
    println!("replay: intrinsics {}", if config.wanding().freeze_intrinsics() {
        "frozen, from each camera's factory intrinsics" } else { "from the config, solved for" });
    let mut ids: Vec<u64> = state.cameras().iter().map(|c| c.id()).collect();
    ids.sort_by_key(|id| name(*id));
    for id in &ids {
        let cam = state.cameras().iter().find(|c| c.id() == *id).unwrap();
        match intrinsics.get(id) {
            Some(i) => println!("replay: {} f {:.2} {:.2}, centre {:.2} {:.2}, k1 {:.5} k2 {:.5}; enabled {}, accelerometer {}",
                name(*id), i.focal_length.x(), i.focal_length.y(), i.center.x(), i.center.y(), i.k1, i.k2,
                config.camera_enabled(*id), cam.camera_status().accelerometer().has_value()),
            None => println!("replay: {} has no intrinsics in the config", name(*id)),
        }
    }

    // The finder, camera by camera.
    let max_frame_error = config.wanding().max_wand_frame_error();
    let mut stats: BTreeMap<u64, CameraStats> = BTreeMap::new();
    // (index into entries, the sightings wanding would keep)
    let mut frames: Vec<(usize, Vec<Sighting>)> = vec![];
    let mut pairs: BTreeMap<(String, String), usize> = BTreeMap::new();
    let mut by_cameras = [0usize; 8];
    // Where the centre marker was, per camera and frame, for the wand's speed.
    let mut centre_at: HashMap<(u64, u64), (f64, f64)> = HashMap::new();

    for (entry_i, entry) in entries.iter().enumerate().skip(1) {
        let mut sightings = vec![];
        for cam in entry.blobs().cameras() {
            let s = stats.entry(cam.camera_id()).or_insert_with(|| CameraStats {
                distance_min: f64::MAX, ..Default::default()
            });
            s.by_blobs[cam.results().blobs().len().min(5)] += 1;

            let intr = match intrinsics.get(&cam.camera_id()) {
                Some(v) => v,
                None => continue,
            };
            let m = match BlobPatternFinder::find_t_wand(config.wand(), intr, cam.results()) {
                Some(m) => m,
                None => continue,
            };

            s.found += 1;
            {
                let blob = &cam.results().blobs()[m.points[1].blob_index];
                centre_at.insert((cam.camera_id(), entry.blobs().frame_timestamp()), (blob.x() as f64, blob.y() as f64));
            }
            for (i, edge) in ERROR_EDGES.iter().enumerate() {
                if m.error < *edge {
                    s.by_error[i] += 1;
                }
            }
            // The wand's +z in the camera's frame against the line of sight.
            let normal = math::matrix::axis_angle::from_axis_angle(&m.rotation) * math::matrix::vec3d(0., 0., 1.);
            if normal.dot(&m.translation) > 0. {
                s.back += 1;
                if m.error <= max_frame_error {
                    s.back_kept += 1;
                }
            }
            let d = m.translation.norm();
            s.distance_min = s.distance_min.min(d);
            s.distance_max = s.distance_max.max(d);

            if m.error <= max_frame_error {
                sightings.push(Sighting { camera_id: cam.camera_id(), m });
            }
        }

        by_cameras[sightings.len().min(7)] += 1;
        for a in &sightings {
            for b in &sightings {
                if name(a.camera_id) < name(b.camera_id) {
                    *pairs.entry((name(a.camera_id), name(b.camera_id))).or_default() += 1;
                }
            }
        }
        if sightings.len() >= 2 {
            frames.push((entry_i, sightings));
        }
    }

    for id in &ids {
        let s = match stats.get(id) {
            Some(s) => s,
            None => { println!("replay: {} sent no frames", name(*id)); continue; }
        };
        println!("replay: {} frames by blobs 0:{} 1:{} 2:{} 3:{} 4:{} 5+:{}; T in {}; fit under 0.25:{} 0.5:{} 1:{} 2:{} 5:{} px; {:.2}-{:.2} m; seen from the back {} ({} kept)",
            name(*id), s.by_blobs[0], s.by_blobs[1], s.by_blobs[2], s.by_blobs[3], s.by_blobs[4], s.by_blobs[5],
            s.found, s.by_error[0], s.by_error[1], s.by_error[2], s.by_error[3], s.by_error[4],
            if s.found > 0 { s.distance_min } else { 0. }, s.distance_max, s.back, s.back_kept);
    }
    println!("replay: frames by cameras holding a T under {} px: 0:{} 1:{} 2:{} 3:{} 4:{}",
        max_frame_error, by_cameras[0], by_cameras[1], by_cameras[2], by_cameras[3], by_cameras[4]);
    for ((a, b), n) in &pairs {
        println!("replay: {} + {}: {} frames", a, b, n);
    }

    // The solve, as wanding.rs runs it.
    let mut solver = WandingCalibrationSolver::new(config.value().clone(), intrinsics.clone());
    solver.set_initial_status(state.clone());
    for entry in &entries[1..] {
        solver.add_frame(entry)?;
    }
    let st = solver.stats();
    println!("replay: solver frames {} raw, {} with a T, {} after dedup",
        st.raw_num_frames(), st.num_valid_frames(), st.num_deduped_frames());

    let solution = solver.solve()?;
    println!("replay: solve error {:.4} px", solution.error);

    // The error by camera, by image speed, and the worst sightings.
    {
        let mut timestamps: Vec<u64> = entries[1..].iter().map(|e| e.blobs().frame_timestamp()).collect();
        timestamps.sort();
        let mut steps: Vec<u64> = timestamps.windows(2).map(|w| w[1] - w[0]).collect();
        steps.sort();
        let step = steps[steps.len() / 2];

        let speed = |r: &mocap_manager::calibration::WandingSightingResidual| -> Option<f64> {
            let here = centre_at.get(&(r.camera_id, r.frame_timestamp))?;
            let other = centre_at.get(&(r.camera_id, r.frame_timestamp + step))
                .or_else(|| centre_at.get(&(r.camera_id, r.frame_timestamp.wrapping_sub(step))))?;
            Some(((here.0 - other.0).powi(2) + (here.1 - other.1).powi(2)).sqrt())
        };

        let rms = |v: &[f64]| (v.iter().map(|e| e * e).sum::<f64>() / (v.len().max(1) as f64)).sqrt();

        for id in &ids {
            let v: Vec<f64> = solution.residuals.iter().filter(|r| r.camera_id == *id).map(|r| r.error).collect();
            let mut sorted = v.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            if v.is_empty() { continue; }
            println!("replay: {} in the solve: {} sightings, RMS {:.2} px, median {:.2}, worst {:.2}",
                name(*id), v.len(), rms(&v), sorted[sorted.len() / 2], sorted[sorted.len() - 1]);
        }

        let edges = [0.0, 1.0, 3.0, 6.0, 12.0, 1e9];
        for w in edges.windows(2) {
            let v: Vec<f64> = solution.residuals.iter()
                .filter(|r| speed(r).map(|s| s >= w[0] && s < w[1]).unwrap_or(false))
                .map(|r| r.error).collect();
            println!("replay: wand moving {}-{} px a frame: {} sightings, RMS {:.2} px",
                w[0], if w[1] > 1e8 { "up".to_string() } else { w[1].to_string() }, v.len(), rms(&v));
        }

        let mut worst: Vec<_> = solution.residuals.iter().collect();
        worst.sort_by(|a, b| b.error.partial_cmp(&a.error).unwrap());
        for r in worst.iter().take(6) {
            println!("replay: worst: {} at frame {} ({:.1} s): {:.1} px, moving {} px a frame",
                name(r.camera_id), r.frame_timestamp / step, (r.frame_timestamp - timestamps[0]) as f64 / 1e9, r.error,
                speed(r).map(|s| format!("{:.1}", s)).unwrap_or("?".to_string()));
        }
    }

    let mut solved = HashMap::<u64, (vision::CameraIntrinsicsModel, CameraExtrinsics), FastHasherBuilder>::default();
    for p in &solution.params {
        let pos = p.extrinsics.position();
        println!("replay: {} at {:.3} {:.3} {:.3} m; f {:.2} {:.2}, centre {:.2} {:.2}, k1 {:.5} k2 {:.5}",
            name(p.id), pos.x(), pos.y(), pos.z(),
            p.intrinsics.focal_length.x(), p.intrinsics.focal_length.y(),
            p.intrinsics.center.x(), p.intrinsics.center.y(), p.intrinsics.k1, p.intrinsics.k2);
        solved.insert(p.id, (p.intrinsics.clone(), p.extrinsics.clone()));
    }
    for a in &solution.params {
        for b in &solution.params {
            if name(a.id) < name(b.id) {
                println!("replay: {} to {}: {:.3} m", name(a.id), name(b.id),
                    (a.extrinsics.position() - b.extrinsics.position()).norm());
            }
        }
    }

    // The saved calibration, against this solve.
    if let Some(path) = &saved_path {
        use protobuf::{Message, StaticMessage};

        let saved = MocapManagerConfig::parse(&std::fs::read(path)?)?;
        let mut positions = vec![];
        for cam in saved.per_camera() {
            if !cam.has_extrinsics() {
                println!("replay: saved {} has no extrinsics", name(cam.camera_id()));
                continue;
            }

            let e = CameraExtrinsics::from_proto(cam.extrinsics());
            let to_world = math::matrix::axis_angle::from_axis_angle(&e.rotation).transpose();
            let look = to_world.clone() * math::matrix::vec3d(0., 0., 1.);
            let down = to_world * math::matrix::vec3d(0., 1., 0.);
            let pos = e.position();
            let i = vision::CameraIntrinsicsModel::from_proto(cam.intrinsics());
            println!("replay: saved {} at x {:.3} y {:.3} z {:.3} m; looks along {:.2} {:.2} {:.2}; down in its picture is {:.2} {:.2} {:.2}; f {:.2}, centre {:.2} {:.2}, k1 {:.5}",
                name(cam.camera_id()), pos.x(), pos.y(), pos.z(), look.x(), look.y(), look.z(),
                down.x(), down.y(), down.z(), i.focal_length.x(), i.center.x(), i.center.y(), i.k1);
            positions.push((cam.camera_id(), pos));
        }

        let mut worst = 0.0f64;
        for (a, pa) in &positions {
            for (b, pb) in &positions {
                if name(*a) < name(*b) {
                    let saved_d = (pa.clone() - pb.clone()).norm();
                    let solved_d = match (solved.get(a), solved.get(b)) {
                        (Some((_, ea)), Some((_, eb))) => (ea.position() - eb.position()).norm(),
                        _ => continue,
                    };
                    println!("replay: saved {} to {}: {:.3} m (this solve {:.3})", name(*a), name(*b), saved_d, solved_d);
                    worst = worst.max((saved_d - solved_d).abs());
                }
            }
        }
        println!("replay: saved against this solve: distances differ by up to {:.1} mm", worst * 1000.);
    }

    // Cross-check: camera a's own fit of the wand, seen by camera b.
    let mut cross: BTreeMap<(String, String), (f64, usize)> = BTreeMap::new();
    for (entry_i, sightings) in &frames {
        let entry = &entries[*entry_i];
        for a in sightings {
            for b in sightings {
                if a.camera_id == b.camera_id {
                    continue;
                }
                let (ea, (ib, eb)) = match (solved.get(&a.camera_id), solved.get(&b.camera_id)) {
                    (Some((_, ea)), Some(b)) => (ea, b),
                    _ => continue,
                };
                let world_from_a = CameraExtrinsics::from_mat4x4(&ea.to_mat4x4().inverse().unwrap());
                let wand_in_a = CameraExtrinsics {
                    rotation: a.m.rotation.clone(),
                    translation: a.m.translation.clone(),
                };
                let blobs_b = entry.blobs().cameras().iter()
                    .find(|c| c.camera_id() == b.camera_id).unwrap().results().blobs();

                let acc = cross.entry((name(a.camera_id), name(b.camera_id))).or_insert((0., 0));
                for (pa, pb) in a.m.points.iter().zip(b.m.points.iter()) {
                    let world = world_from_a.transform(&wand_in_a.transform(&pa.reference_point));
                    let px = ib.project_point(&eb.transform(&world));
                    let blob = &blobs_b[pb.blob_index];
                    acc.0 += (px - vec2d(blob.x() as f64, blob.y() as f64)).norm_squared();
                    acc.1 += 1;
                }
            }
        }
    }
    for ((a, b), (sum, n)) in &cross {
        println!("replay: {}'s wand seen by {}: {:.2} px RMS over {} points", a, b, (sum / (*n as f64)).sqrt(), n);
    }

    Ok(())
}
