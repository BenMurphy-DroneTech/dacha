use common::errors::*;
use mocap_proto::mocap::*;
use math::matrix::{vec3d, Matrix4d};
use vision::CameraExtrinsics;
use math_proto_util::VectorProtoExt;

use crate::matching::TrackedPoint;
use crate::rigid_body::*;
use crate::config::*;


/// Finds a calibration wand in the given point cloud and creates a config
/// patch that will set the origin position/orientaiton based on that wand.
///
/// The wand's whole pose is taken: with it lying flat, markers up, the floor
/// becomes z = 0 and its plane's normal becomes +z, the origin sits under the
/// centre marker, +x runs along the long arm and +y along the handle. For
/// cameras without accelerometers (calibration.rs then leaves the world
/// unlevelled) this is where the floor plane and the up direction come from.
///
/// TODO: Support optionally ignoring any X/Y axis rotations and just
/// taking Z yaw adjustments given the accelerometer based alignment.
pub fn set_origin_with_wand(
    config: &ManagerConfigContainer,
    points: &[TrackedPoint]
) -> Result<MocapManagerConfig> {
    let mut tracker_config = config.rigid_body_tracker().clone();
    tracker_config.clear_bodies();

    let w = config.wand();
    let pts = vec![
        vec3d(0., 0., w.height()),
        vec3d(w.left_arm_length(), 0., w.height()),
        vec3d(-w.right_arm_length(), 0., w.height()),
        vec3d(0., w.bottom_length(), w.height()),
    ];

    let body = tracker_config.new_bodies();
    body.set_id(1u32);

    for pt in pts {
        body.add_points(pt.to_proto());
    }

    let mut tracker = RigidBodyTracker::default();
    tracker.set_config(tracker_config);
    
    tracker.run(points);

    let bodies = tracker.bodies();

    if bodies.len() != 1 || bodies[0].transform.is_none() {
        return Err(err_msg("Wand not found"));
    }

    let mut transform = {
        let (r, t) = bodies[0].transform.as_ref().unwrap();
        
        let mut out = Matrix4d::zero();
        out.block_mut(0, 0).copy_from(&r);
        out.block_mut(0, 3).copy_from(&t);
        out[(3, 3)] = 1.0;
        out
    };

    let mut patch = MocapManagerConfig::default();

    for cam in config.per_camera() {
        let old_extrinsics = match config.camera_extrinsics().get(&cam.camera_id()) {
            Some(v) => v,
            None => continue
        };

        let new_extrinsics = CameraExtrinsics::from_mat4x4(&(
            old_extrinsics.to_mat4x4() * &transform
        ));

        let mut proto = patch.new_per_camera();
        proto.set_camera_id(cam.camera_id());
        proto.set_extrinsics(new_extrinsics.to_proto());
    }

    Ok(patch)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The breadboard wand flat on the floor, markers up, with the world left
    /// in an arbitrary tilted frame (a calibration without accelerometers):
    /// Set Origin must put the floor at z = 0 with z up, the origin under the
    /// centre marker, and leave what the camera sees unchanged.
    #[test]
    fn floor_and_up_from_flat_wand() {
        let mut base = MocapManagerConfig::default();
        protobuf::text::parse_text_proto(
            r#"
            wand {
                left_arm_length: 0.21213
                right_arm_length: 0.10607
                bottom_length: 0.14142
                height: 0.06
            }
            rigid_body_tracker {
                body_config {
                    min_initial_match_points: 4
                    min_rematch_points: 3
                    min_matching_edge_length: 0.05
                    max_matching_edge_length: 1
                    max_error: 0.005
                }
            }
            "#,
            &mut base
        ).unwrap();

        // The room: floor at z = 0, z up. The wand as built, seen from above
        // (x to the right, y up the page, z towards the viewer): from the
        // centre marker, left is down-left, right up-right, handle down-right.
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let centre = vec3d(1.3, -0.4, 0.06);
        let wand_in_room = vec![
            centre.clone(),
            centre.clone() + vec3d(-s, -s, 0.) * 0.21213,
            centre.clone() + vec3d(s, s, 0.) * 0.10607,
            centre.clone() + vec3d(s, -s, 0.) * 0.14142,
        ];

        let camera_in_room = CameraExtrinsics {
            rotation: vec3d(1.9, -0.4, 0.3),
            translation: vec3d(0.2, 0.5, 3.0),
        };

        // Room to the world the calibration left.
        let room_to_world = CameraExtrinsics {
            rotation: vec3d(0.7, -0.3, 1.1),
            translation: vec3d(0.5, 2.0, -1.0),
        };

        let camera_in_world = CameraExtrinsics::from_mat4x4(&(
            camera_in_room.to_mat4x4() * room_to_world.to_mat4x4().inverse().unwrap()
        ));

        {
            let cam = base.new_per_camera();
            cam.set_camera_id(1u64);
            cam.set_enabled(true);
            cam.set_extrinsics(camera_in_world.to_proto());
        }

        let config = ManagerConfigContainer::create(&base).unwrap();

        // Handle, right, centre, left: the order must not matter.
        let mut points = vec![];
        for (n, i) in [3usize, 2, 0, 1].into_iter().enumerate() {
            points.push(TrackedPoint {
                id: (n + 10) as u64,
                position: room_to_world.transform(&wand_in_room[i]),
                camera_ids: vec![1, 2],
            });
        }

        let patch = set_origin_with_wand(&config, &points).unwrap();
        assert_eq!(patch.per_camera().len(), 1);
        let camera_new = CameraExtrinsics::from_proto(patch.per_camera()[0].extrinsics());

        // The camera is as high above the floor as it was in the room, and as
        // far from the centre marker along the floor.
        let was = camera_in_room.position();
        let now = camera_new.position();
        assert!((now.z() - was.z()).abs() < 1e-6, "height {} != {}", now.z(), was.z());

        let floor_dist_was = ((was.x() - centre.x()).powi(2) + (was.y() - centre.y()).powi(2)).sqrt();
        let floor_dist_now = (now.x().powi(2) + now.y().powi(2)).sqrt();
        assert!((floor_dist_now - floor_dist_was).abs() < 1e-6);

        // Each marker is where the wand model says, seen where it was seen.
        let model = vec![
            vec3d(0., 0., 0.06),
            vec3d(0.21213, 0., 0.06),
            vec3d(-0.10607, 0., 0.06),
            vec3d(0., 0.14142, 0.06),
        ];
        for i in 0..4 {
            let seen_was = camera_in_room.transform(&wand_in_room[i]);
            let seen_now = camera_new.transform(&model[i]);
            assert!((seen_now - seen_was).norm() < 1e-6, "marker {}", i);
        }
    }
}
