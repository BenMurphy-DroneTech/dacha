//! rift_solve_bench: how long does the manager's per-frame solve take?
//!
//! Runs exactly what inst.rs `matching_task` runs for each merged frame set
//! (BlobMatcher, then the rigid body tracker, then the skeleton tracker, each
//! with its back-propagation) on blobs made up from a moving skeleton, and
//! times every frame. The configs come from config/manager.txtpb.
//!
//! The scene is the project's room plan (riftcv1-mocap
//! docs/coverage_simulator.html): 5.1 × 4.5 m, the origin at the pole, z up,
//! twelve cameras 12 cm in from the walls, 6 at 2.3 m and 6 at 0.8 m, aimed as
//! the simulator aims them. Each camera is the pinhole the Rift camera node
//! gives dacha (f 716 px, centre 640, 480) and sees a marker within 37.4° of
//! its axis both ways and 5.5 m (the simulator's reach, so every marker the
//! window could pan to counts; bodies hiding markers aren't modelled, so the
//! blob count is an upper bound). Blob centres get 0.1 px of noise.
//!
//! The skeleton is dacha's standard_skeleton (the one SkeletonTracker builds).
//! It stands in its default pose at the origin for 3 s, so the tracker's search
//! finds it, then walks a 1 m circle round the pole at about 1.5 m/s, turning,
//! with every joint swinging inside its limits.
//!
//!   rift_solve_bench [--seconds 60] [--false-blobs N] [--speed 1] [--config PATH]
//!                    [--solver-alone | --chain]
//!
//! --speed scales the motion (0 keeps the skeleton still). --false-blobs adds N blobs per camera per frame at random places (stray
//! light, reflections). Prints the solve's median, p90, p99 and max per stage,
//! over the frames after the skeleton went live, then where the fit error
//! comes from: the tracked points against the true markers, the solved
//! skeleton against the points it was given, and whether each marker is
//! labelled with the right point.
//!
//! --solver-alone also runs solve_skeleton_joints_state on its own each frame,
//! from the last frame's true pose onto this frame's true markers; --chain
//! starts each from its own last answer instead (drift shows there), and both
//! print the error per marker.

use std::time::Instant;

use common::errors::*;
use math::matrix::{vec2d, vec3d, Matrix3d, Vector3d};
use math::matrix::axis_angle::to_axis_angle;
use mocap_manager::matching::*;
use mocap_manager::skeleton::*;
use mocap_manager::RigidBodyTracker;
use mocap_proto::mocap::*;
use vision::*;

const FRAME_NS: u64 = 11_111_000;

const ROOM_W: f64 = 5.1;
const ROOM_D: f64 = 4.5;
const INSET: f64 = 0.12;
const HIGH: f64 = 2.3;
const LOW: f64 = 0.8;
const HALF_FOV: f64 = 37.4 * std::f64::consts::PI / 180.0;
const RANGE: f64 = 5.5;
const F_PX: f64 = 716.0;

/// xorshift64*: noise without pulling in a crate.
struct Rng(u64);

impl Rng {
    fn next_f64(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / (1u64 << 53) as f64
    }

    fn gauss(&mut self) -> f64 {
        let u = self.next_f64().max(1e-12);
        let v = self.next_f64();
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

struct Cam {
    params: CameraParameters,
    position: Vector3d,
    forward: Vector3d,
    right: Vector3d,
    up: Vector3d,
}

/// The simulator's cameras: (x, y from the room's bottom-left corner, high, corner).
fn room_cameras() -> Vec<Cam> {
    let plan = [
        (INSET, ROOM_D - INSET, true, true),
        (1.74, ROOM_D - INSET, true, false),
        (3.36, ROOM_D - INSET, false, false),
        (ROOM_W - INSET, ROOM_D - INSET, true, true),
        (ROOM_W - INSET, 2.96, false, false),
        (ROOM_W - INSET, 1.54, false, false),
        (ROOM_W - INSET, INSET, true, true),
        (3.36, INSET, false, false),
        (1.74, INSET, false, false),
        (INSET, INSET, true, true),
        (INSET, 1.54, true, false),
        (INSET, 3.0, false, false),
    ];

    let mut out = vec![];
    for (i, (x, y, high, corner)) in plan.iter().cloned().enumerate() {
        let z = if high { HIGH } else { LOW };
        let target_z = if high { if corner { 1.3 } else { 1.5 } } else { 1.1 };
        let position = vec3d(x - ROOM_W / 2.0, y - ROOM_D / 2.0, z);
        let forward = (vec3d(0.0, 0.0, target_z) - &position).normalized();
        let right = vec3d(forward[1], -forward[0], 0.0).normalized();
        let up = right.cross(&forward);

        // Camera axes as dacha's pinhole wants them: x right, y down, z forward.
        let mut r = Matrix3d::zero();
        for j in 0..3 {
            r[(0, j)] = right[j];
            r[(1, j)] = -up[j];
            r[(2, j)] = forward[j];
        }
        let translation = (&r * &position) * -1.0;
        let extrinsics = CameraExtrinsics { rotation: to_axis_angle(&r), translation };

        let intrinsics = CameraIntrinsicsModel {
            focal_length: vec2d(F_PX, F_PX),
            center: vec2d(640.0, 480.0),
            k1: 0.0,
            k2: 0.0,
        };

        // The aim point has to land on the image centre.
        let aim = intrinsics.project_point(&extrinsics.transform(&vec3d(0.0, 0.0, target_z)));
        assert!((aim[0] - 640.0).abs() < 1e-6 && (aim[1] - 480.0).abs() < 1e-6, "camera {} aim {:?}", i + 1, aim);

        out.push(Cam {
            params: CameraParameters { id: (i + 1) as u64, intrinsics, extrinsics },
            position,
            forward,
            right,
            up,
        });
    }
    out
}

impl Cam {
    fn sees(&self, p: &Vector3d) -> bool {
        let v = p - &self.position;
        if v.norm() > RANGE {
            return false;
        }
        let df = v.dot(&self.forward);
        if df <= 0.0 {
            return false;
        }
        v.dot(&self.right).atan2(df).abs() <= HALF_FOV && v.dot(&self.up).atan2(df).abs() <= HALF_FOV
    }

    fn pixel(&self, p: &Vector3d) -> math::matrix::Vector2d {
        self.params.intrinsics.project_point(&self.params.extrinsics.transform(p))
    }
}

/// The pose at time t (s): still in the default pose for `still` s, then a
/// walk round the pole with every joint swinging.
fn pose(skeleton: &Skeleton, t: f64, still: f64, speed: f64) -> SkeletonJointsState {
    let mut s = SkeletonJointsState::zero(skeleton);
    if t < still {
        return s;
    }
    let t = (t - still) * speed;
    let ramp = (t / 1.0).min(1.0);
    let w = 1.5;   // rad/s round a 1 m circle: 1.5 m/s
    let a = w * t;
    s.translation = vec3d(ramp * (a.cos() - 1.0), ramp * a.sin(), 0.9 + 0.15 * ramp * (3.1 * t).sin());
    s.rotation = vec3d(0.2 * ramp * (1.7 * t).sin(), 0.2 * ramp * (1.3 * t).cos(), ramp * 0.8 * a);

    for (i, bone) in skeleton.bones.iter().enumerate() {
        for j in 0..3 {
            let c = &bone.rotation_constraints[j];
            if c.is_fixed() {
                continue;
            }
            let lo = c.min.max(-1.2);
            let hi = c.max.min(1.2);
            let mid = 0.5 * (lo + hi);
            let amp = 0.45 * (hi - lo);
            let phase = (i * 3 + j) as f64 * 0.7;
            s.bone_rotations[i][j] = ramp * (mid + amp * (2.0 * t + phase).sin());
        }
    }
    s.clamp(skeleton);
    s
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    sorted[((sorted.len() - 1) as f64 * p).round() as usize]
}

fn main() -> Result<()> {
    let mut seconds = 60.0;
    let mut false_blobs = 0usize;
    let mut speed = 1.0;
    let mut solver_alone = false;
    let mut chain = false;
    let mut chained: Option<SkeletonJointsState> = None;
    let mut config_path = "pkg/vision/mocap/config/manager.txtpb".to_string();
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--seconds" => { seconds = args[i + 1].parse()?; i += 1; }
            "--false-blobs" => { false_blobs = args[i + 1].parse()?; i += 1; }
            "--speed" => { speed = args[i + 1].parse()?; i += 1; }
            "--solver-alone" => { solver_alone = true; }
            "--chain" => { solver_alone = true; chain = true; }
            "--config" => { config_path = args[i + 1].clone(); i += 1; }
            a => return Err(format_err!("unknown argument {}", a)),
        }
        i += 1;
    }

    let mut config = MocapManagerConfig::default();
    protobuf::text::parse_text_proto(&std::fs::read_to_string(&config_path)?, &mut config)?;

    let cams = room_cameras();
    let params: Vec<CameraParameters> = cams.iter().map(|c| c.params.clone()).collect();

    let mut matcher = BlobMatcher::new(config.matching());
    matcher.set_camera_parameters(&params);

    let mut rigid_body_tracker = RigidBodyTracker::default();
    rigid_body_tracker.set_config(config.rigid_body_tracker().clone())?;

    let mut skel_config = config.skeleton_tracker().clone();
    skel_config.skeletons_mut().clear();
    let mut proto = standard_skeleton().to_proto();
    proto.set_id(1u32);
    skel_config.add_skeletons(proto);
    let mut skeleton_tracker = SkeletonTracker::default();
    skeleton_tracker.set_config(skel_config)?;
    skeleton_tracker.set_skeleton_searching(1, true);

    let skeleton = standard_skeleton();
    let tree = SkeletonTree::new(&skeleton);
    let mut bone_positions = vec![Vector3d::zero(); skeleton.bones.len()];
    let mut marker_positions = vec![Vector3d::zero(); skeleton.markers.len()];

    let still = 3.0;
    let frames = (seconds * 1e9 / FRAME_NS as f64) as u64;
    let mut rng = Rng(0x9e3779b97f4a7c15);

    let mut live_from = None;
    let mut t_match = vec![];
    let mut t_rigid = vec![];
    let mut t_skel = vec![];
    let mut t_out = vec![];
    let mut t_total = vec![];
    let mut blobs_per_frame = vec![];
    let mut points_per_frame = vec![];
    let mut cams_per_marker = vec![];
    let mut marker_err = vec![];
    let mut alone_ms = vec![];
    let mut alone_err = vec![];
    let mut per_marker = vec![0.0; 64];
    let mut per_marker_n = 0usize;
    let mut point_err = vec![];
    let mut fit_resid = vec![];
    let (mut labelled, mut mislabelled, mut unlabelled) = (0usize, 0usize, 0usize);

    println!("rift_solve_bench: {} cameras, standard_skeleton ({} markers, {} bones), {} s at 90 fps, {} false blobs per camera",
             cams.len(), skeleton.markers.len(), skeleton.bones.len(), seconds, false_blobs);

    for k in 0..frames {
        let ts = 1_000_000_000_000 + k * FRAME_NS;
        let t = k as f64 * FRAME_NS as f64 / 1e9;

        let state = pose(&skeleton, t, still, speed);
        tree.forward_kinematics(&state, &mut bone_positions, &mut marker_positions);

        if solver_alone && t > still + 1.0 {
            let prev = match &chained {
                Some(c) => c.clone(),
                None => pose(&skeleton, t - FRAME_NS as f64 / 1e9, still, speed),
            };
            let targets: Vec<(usize, Vector3d)> = marker_positions.iter().cloned().enumerate().collect();
            let s0 = Instant::now();
            let sol = mocap_manager::skeleton::solver::solve_skeleton_joints_state(&skeleton, &tree, &prev, &targets);
            alone_ms.push(s0.elapsed().as_secs_f64() * 1e3);
            let mut mp = vec![Vector3d::zero(); skeleton.markers.len()];
            tree.forward_kinematics(&sol, &mut [], &mut mp);
            let e = mp.iter().zip(&marker_positions).map(|(a, b)| (a - b).norm()).sum::<f64>() / mp.len() as f64;
            alone_err.push(e * 1e3);
            for (i, (a, b)) in mp.iter().zip(&marker_positions).enumerate() {
                per_marker[i] += (a - b).norm() * 1e3;
            }
            per_marker_n += 1;
            if chain {
                chained = Some(sol);
            }
        }

        let mut res = ReadBlobsResponse::default();
        res.set_frame_timestamp(ts);
        let mut n_blobs = 0;
        let mut seen = vec![0usize; marker_positions.len()];
        for cam in &cams {
            let mut c = CameraBlobResults::default();
            c.set_camera_id(cam.params.id);
            for (m, p) in marker_positions.iter().enumerate() {
                if !cam.sees(p) {
                    continue;
                }
                seen[m] += 1;
                let px = cam.pixel(p);
                let mut b = Blob::default();
                b.set_x((px[0] + 0.1 * rng.gauss()) as f32);
                b.set_y((px[1] + 0.1 * rng.gauss()) as f32);
                b.set_radius_a(2.0);
                b.set_radius_b(2.0);
                c.results_mut().add_blobs(b);
                n_blobs += 1;
            }
            for _ in 0..false_blobs {
                let mut b = Blob::default();
                b.set_x((rng.next_f64() * 1280.0) as f32);
                b.set_y((rng.next_f64() * 960.0) as f32);
                b.set_radius_a(2.0);
                b.set_radius_b(2.0);
                c.results_mut().add_blobs(b);
                n_blobs += 1;
            }
            res.add_cameras(c);
        }

        // inst.rs matching_task, stage by stage.
        let t0 = Instant::now();
        let mut points = matcher.run(&res);
        let t1 = Instant::now();
        rigid_body_tracker.run(&points);
        rigid_body_tracker.backpropagate_predicted_points(&mut matcher);
        points = matcher.points();
        let t2 = Instant::now();
        let solve_points = points.clone();
        skeleton_tracker.run(res.frame_timestamp(), &points);
        skeleton_tracker.backpropagate_predicted_points(&mut matcher);
        points = matcher.points();
        let t3 = Instant::now();

        // The response matching_task sends on.
        let mut out = ReadTrackedPointsResponse::default();
        out.set_frame_timestamp(res.frame_timestamp());
        for pt in &points {
            out.add_points(pt.to_proto());
        }
        for body in rigid_body_tracker.bodies() {
            if body.transform.is_some() {
                out.add_rigid_bodies(body.to_proto());
            }
        }
        let skel = skeleton_tracker.to_state_protos();
        for s in &skel {
            out.add_skeletons(s.clone());
        }
        let t4 = Instant::now();

        if live_from.is_none() && is_live(&skel) {
            live_from = Some(k);
            println!("skeleton live at frame {} ({:.2} s)", k, t);
        }

        if let Some(k0) = live_from {
            if k > k0 + 90 {
                t_match.push((t1 - t0).as_secs_f64() * 1e3);
                t_rigid.push((t2 - t1).as_secs_f64() * 1e3);
                t_skel.push((t3 - t2).as_secs_f64() * 1e3);
                t_out.push((t4 - t3).as_secs_f64() * 1e3);
                t_total.push((t4 - t0).as_secs_f64() * 1e3);
                blobs_per_frame.push(n_blobs as f64);
                points_per_frame.push(points.len() as f64);
                for s in &seen {
                    cams_per_marker.push(*s as f64);
                }
                // How far the solved skeleton's markers sit from the true ones.
                if let Some(err) = solved_marker_error(&skel, &marker_positions) {
                    marker_err.push(err * 1e3);
                }
                // Where the error comes from: the points against the truth, and the
                // fit against the points it was given (and whether they're the right ones).
                for pt in &solve_points {
                    if pt.camera_ids.is_empty() {
                        continue;
                    }
                    let d = marker_positions.iter().map(|m| (&pt.position - m).norm()).fold(f64::MAX, f64::min);
                    point_err.push(d * 1e3);
                }
                if let Some(s) = skel.first() {
                    if s.mode() == SkeletonStateProto_Mode::LIVE {
                        let markers = s.calculated_positions().markers();
                        for (i, id) in s.marker_ids().iter().enumerate() {
                            let pt = match solve_points.iter().find(|p| p.id == *id && !p.camera_ids.is_empty()) {
                                Some(p) => p,
                                None => { unlabelled += 1; continue; }
                            };
                            let v = markers[i].values();
                            fit_resid.push((&vec3d(v[0], v[1], v[2]) - &pt.position).norm() * 1e3);
                            let nearest = (0..marker_positions.len())
                                .min_by(|a, b| (&pt.position - &marker_positions[*a]).norm()
                                    .total_cmp(&(&pt.position - &marker_positions[*b]).norm())).unwrap();
                            if nearest != i {
                                mislabelled += 1;
                            }
                            labelled += 1;
                        }
                    }
                }
            }
        }
    }

    if solver_alone {
        alone_ms.sort_by(|a, b| a.total_cmp(b));
        alone_err.sort_by(|a, b| a.total_cmp(b));
        println!("solver alone, from the last true pose: {} solves, median {:.3} ms, p99 {:.3} ms; marker error median {:.3} mm, p99 {:.3} mm, max {:.3} mm",
                 alone_ms.len(), percentile(&alone_ms, 0.5), percentile(&alone_ms, 0.99),
                 percentile(&alone_err, 0.5), percentile(&alone_err, 0.99), alone_err[alone_err.len() - 1]);
        for (i, mk) in skeleton.markers.iter().enumerate() {
            println!("  {:<32} mean {:7.2} mm", mk.name, per_marker[i] / per_marker_n.max(1) as f64);
        }
    }
    if live_from.is_none() {
        return Err(err_msg("the skeleton never went live"));
    }

    let report = |name: &str, v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.total_cmp(b));
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        println!("{:<28} mean {:8.3}  median {:8.3}  p90 {:8.3}  p99 {:8.3}  max {:8.3}",
                 name, mean, percentile(v, 0.5), percentile(v, 0.9), percentile(v, 0.99), v[v.len() - 1]);
    };
    println!("{} frames timed (from 1 s after the skeleton went live)", t_total.len());
    report("matching (ms)", &mut t_match);
    report("rigid bodies (ms)", &mut t_rigid);
    report("skeleton (ms)", &mut t_skel);
    report("response (ms)", &mut t_out);
    report("solve total (ms)", &mut t_total);
    report("blobs per frame set", &mut blobs_per_frame);
    report("3d points per frame", &mut points_per_frame);
    report("cameras seeing a marker", &mut cams_per_marker);
    if !marker_err.is_empty() {
        report("skeleton marker error (mm)", &mut marker_err);
    }
    report("point error (mm)", &mut point_err);
    if !fit_resid.is_empty() {
        report("fit to its points (mm)", &mut fit_resid);
    }
    println!("labels: {} on a seen point, {} of them on the wrong marker, {} on no seen point",
             labelled, mislabelled, unlabelled);
    Ok(())
}

fn is_live(skel: &[SkeletonStateProto]) -> bool {
    skel.iter().any(|s| s.mode() == SkeletonStateProto_Mode::LIVE)
}

/// Mean distance from the tracked skeleton's markers to the true markers, or
/// None while it isn't live.
fn solved_marker_error(skel: &[SkeletonStateProto], truth: &[Vector3d]) -> Option<f64> {
    let s = skel.first()?;
    if s.mode() != SkeletonStateProto_Mode::LIVE {
        return None;
    }
    let markers = s.calculated_positions().markers();
    if markers.len() != truth.len() {
        return None;
    }
    let mut sum = 0.0;
    for (m, t) in markers.iter().zip(truth) {
        let v = m.values();
        sum += (&vec3d(v[0], v[1], v[2]) - t).norm();
    }
    Some(sum / truth.len() as f64)
}
