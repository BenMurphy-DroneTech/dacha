
use std::io::Read;
use std::sync::Arc;
use std::{fs::File, time::Duration};
use std::time::Instant;
use std::collections::HashMap;

use common::errors::*;
use common::io::{Readable, Writeable};
use common::hash::FastHasherBuilder;
use executor::bundle::TaskResultBundle;
use file::LocalPathBuf;
use math::array::Array;
use image::Colorspace;
use file::{project_path, LocalPath};
use mocap_proto::mocap::*;
use sstable::record_log::RecordReader;
use protobuf::{StaticMessage, Message};
use protobuf_json::MessageJsonSerialize;
use math::matrix::axis_angle::*;
use math::matrix::{vec2d, vec3d, Vector2d, Matrix3d, Vector3d, Matrix4d};
use vision::{CameraIntrinsicsModel, CameraExtrinsics, BundleAdjustmentSolver, EightPointCameraExtrinsicsSolver};
use cluster_client::id::{entity_id_from_string, entity_id_to_string};
use math_proto_util::VectorProtoExt;

use crate::wand::*;
use crate::matching::CameraParameters;
use crate::proto_utils::*;
use crate::config::*;


/*
TODO: Eventually do full spatial dedupping instead of just looking at the previous matched frame.

TODO: Need to support calibration using a subset of cameras if we need to disable some (or just all that have data).

*/


/// The intrinsics a wanding run starts from: the config's, except that with
/// wanding.freeze_intrinsics a camera that reports factory intrinsics uses
/// those. The solve then hands them back unchanged, so what an earlier solve
/// left in the config can't carry over.
pub fn wanding_intrinsics(
    config: &ManagerConfigContainer,
    status: &MocapManagerStatus
) -> HashMap<u64, CameraIntrinsicsModel, FastHasherBuilder> {
    let mut out = config.camera_intrinsics().clone();

    if config.wanding().freeze_intrinsics() {
        for cam in status.cameras() {
            let hardware = cam.camera_status().hardware_config();
            if hardware.has_factory_intrinsics() {
                out.insert(cam.id(), CameraIntrinsicsModel::from_proto(hardware.factory_intrinsics()));
            }
        }
    }

    out
}

pub struct WandingCalibrationSolver {
    config: MocapManagerConfig,
    camera_intrinsics: HashMap<u64, CameraIntrinsicsModel, FastHasherBuilder>,
    
    stats: WandingCalibrationStats,

    /// Data after deduplication.
    frames: Vec<FrameData>,

    // TODO: Only need to preserve the extrinsic parameters in this.
    last_match_per_camera: HashMap<u64, BlobPatternMatch, FastHasherBuilder>,

    initial_system_state: Option<MocapManagerStatus>,
}

#[derive(Clone)]
pub struct WandingCalibrationSolution {
    pub error: f64,
    pub params: Vec<CameraParameters>,

    /// How well each sighting the solve used fits the solution.
    pub residuals: Vec<WandingSightingResidual>,
}

/// One camera's view of the wand in one frame, against the solution.
#[derive(Clone, Debug)]
pub struct WandingSightingResidual {
    pub frame_timestamp: u64,
    pub camera_id: u64,

    /// RMS reprojection error of the wand's points, in pixels.
    pub error: f64,
}


impl WandingCalibrationSolver {

    /// Initializes a calibrator with configs and initial intrinsics pulled from the given config.
    pub fn new(
        config: MocapManagerConfig,
        camera_intrinsics: HashMap<u64, CameraIntrinsicsModel, FastHasherBuilder>,
    ) -> Self {
        Self {
            config,
            camera_intrinsics,
            frames: vec![],
            last_match_per_camera: Default::default(),
            initial_system_state: None,
            stats: WandingCalibrationStats::default(),
        }
    }

    pub fn set_initial_status(&mut self, status: MocapManagerStatus) {
        self.initial_system_state = Some(status);
    }

    pub fn stats(&self) -> WandingCalibrationStats {
        self.stats.clone()
    }

    /// TODO: If doing this offline, the wand finding is parallelizable across
    /// all the frames.
    pub fn add_frame(&mut self, entry: &MocapLogEntry) -> Result<()> {
        *self.stats.raw_num_frames_mut() += 1;

        // At least two cameras need to be involved for correlating relative
        // positions between cameras.
        if entry.blobs().cameras().len() < 2 {
            return Ok(());
        }

        let mut cameras_data = HashMap::default();

        let mut found_wand = false;

        for cam in entry.blobs().cameras() {
            let intrinsics = self.camera_intrinsics.get(&cam.camera_id())
                .ok_or_else(|| format_err!(
                    "Missing intrinsics for camera: {}",
                    entity_id_to_string(cam.camera_id()).unwrap()
                ))?;

            let m = match BlobPatternFinder::find_t_wand(self.config.wand(), intrinsics, cam.results()) {
                Some(v) => v,
                None => continue
            };

            if m.error > self.config.wanding().max_wand_frame_error() {
                continue;
            }

            if !found_wand {
                found_wand = true;
                *self.stats.num_valid_frames_mut() += 1;
            }

            // TODO: If there are multiple frames nearby, pick the one with the lowest reprojection error.
            if let Some(last_match) = self.last_match_per_camera.get(&cam.camera_id()) {
                if !pattern_positions_distinct(&m, last_match) {
                    return Ok(());
                }
            }

            let mut points_2d = vec![];
            let mut points_3d = vec![];
            for p in &m.points {
                points_3d.push(p.reference_point.clone());

                let blob = &cam.results().blobs()[p.blob_index];
                points_2d.push(vec2d(blob.x() as f64, blob.y() as f64));
            }

            cameras_data.insert(cam.camera_id(), FrameCameraData {
                points_2d,
                points_3d,
                pattern: m
            });
        }

        if cameras_data.len() < 2 {
            return Ok(());
        }

        // Update last position per camera for future dedupping.
        for (cam_id, data) in &cameras_data {
            self.last_match_per_camera.insert(*cam_id, data.pattern.clone());
        }

        self.frames.push(FrameData {
            frame_timestamp: entry.blobs().frame_timestamp(),
            cameras: cameras_data
        });

        *self.stats.num_deduped_frames_mut() += 1;

        Ok(())
    }

    pub fn solve(&mut self) -> Result<WandingCalibrationSolution> {

        let mut data = vec![];
        core::mem::swap(&mut data, &mut self.frames);

        // println!("Subsetting wands...");
        // let mut data = self.select_frame_subset(&mut data)?;

        // println!("Num cams: {}", num_cameras);

        println!("Remaining entries: {}", data.len());

        if data.len() == 0 {
            return Err(err_msg("No data with good enough matches"));
        }


        // Initialize rough camera extrinsics guesses.
        println!("Extract initial extrinsics...");
        let (anchor_camera_id, camera_initial_extrinsics) = self.extract_initial_extrinsics_pnp(&mut data)?;

        println!("Optimizing...");
        let mut solution = self.optimize(&data, anchor_camera_id, &camera_initial_extrinsics)?;

        // Drop the frames the solution can't fit and solve again from where
        // it got to. The cut comes down by halves from the worst frame, so a
        // solve dragged off by a few wild frames isn't asked to judge the
        // good ones at the final limit straight away.
        let max_error = self.config.wanding().max_solved_frame_error();
        if max_error > 0.0 {
            const MAX_ROUNDS: usize = 8;

            for _ in 0..MAX_ROUNDS {
                let worst = solution.residuals.iter().map(|r| r.error).fold(0.0, f64::max);
                if worst <= max_error {
                    break;
                }

                let cut = max_error.max(worst / 2.0);
                let dropped = solution.residuals.iter()
                    .filter(|r| r.error > cut)
                    .map(|r| r.frame_timestamp)
                    .collect::<std::collections::HashSet<u64>>();

                let num_before = data.len();
                data.retain(|frame| !dropped.contains(&frame.frame_timestamp));
                println!(
                    "Dropped {} of {} frames over {:.2} px (worst {:.2}). Optimizing again...",
                    num_before - data.len(), num_before, cut, worst);

                if data.is_empty() {
                    return Err(err_msg("No frames fit the solution"));
                }

                let mut extrinsics = HashMap::<u64, CameraExtrinsics, FastHasherBuilder>::default();
                for p in &solution.params {
                    extrinsics.insert(p.id, p.extrinsics.clone());
                }

                solution = self.optimize(&data, anchor_camera_id, &extrinsics)?;
            }
        }

        self.align_extrinsics(&mut solution.params)?;

        Ok(solution)
    }

    fn select_frame_subset(&self, frames: &mut Vec<FrameData>) -> Result<Vec<FrameData>> {
        const CROSS_CAMERA_WEIGHT: f64 = 0.1;

        const SINGLE_CAMERA_WEIGHT: f64 = 10.0;

        const PAIR_WEIGHT: f64 = 20.0;

        // Target number of frames to select in total.
        const TARGET_SUBSET_SIZE: usize = 80;

        // Rough number of times that we want each camera to be observed.
        const SINGLE_CAMERA_TARGET_OBSERVATIONS: usize = 10;

        // Target number of observations of each pair of cameras.
        const PAIR_TARGET_COINCIDENCES: usize = 5;

        // Selected frames.
        let mut out = vec![];

        // How many times each camera has appeared so far in the selected data.
        let mut camera_counts = HashMap::<u64, usize, FastHasherBuilder>::default();

        let mut camera_coincidences = HashMap::<(u64, u64), usize, FastHasherBuilder>::default();

        // TODO: Need some randomization for similarly weighted pairs.

        while frames.len() > 0 && out.len() < TARGET_SUBSET_SIZE {

            let mut best_score = 0.0;
            let mut best_i = 0;

            for i in 0..frames.len() {

                let frame = &frames[i];

                let score = {
                    
                    let mut score = 0.0;
                
                    score += CROSS_CAMERA_WEIGHT * (frame.cameras.len() as f64);

                    for cam_id in frame.cameras.keys() {
                        let c = camera_counts.get(cam_id).cloned().unwrap_or(0);
                        score += SINGLE_CAMERA_WEIGHT * ((SINGLE_CAMERA_TARGET_OBSERVATIONS as f64) - (c as f64)).max(0.0);
                    }

                    for cam_id1 in frame.cameras.keys() {
                        for cam_id2 in frame.cameras.keys() {
                            if cam_id1 >= cam_id2 {
                                continue;
                            }

                            let c = camera_coincidences.get(&(*cam_id1, *cam_id2)).cloned().unwrap_or(0);
                            score += PAIR_WEIGHT * ((PAIR_TARGET_COINCIDENCES as f64) - (c as f64)).max(0.0);
                        }
                    }

                    score
                };

                if score >= best_score {
                    best_score = score;
                    best_i = i;
                }
            }

            let frame = frames.swap_remove(best_i);

            {
                for cam_id in frame.cameras.keys() {
                    *camera_counts.entry(*cam_id).or_default() += 1;
                }

                for cam_id1 in frame.cameras.keys() {
                    for cam_id2 in frame.cameras.keys() {
                        if cam_id1 >= cam_id2 {
                            continue;
                        }

                        *camera_coincidences.entry((*cam_id1, *cam_id2)).or_default() += 1;
                    }
                }
            }

            out.push(frame);
        }

        Ok(out)
    }

    fn extract_initial_extrinsics_eight_point(
        &self, data: &[FrameData]
    ) -> Result<(u64, HashMap<u64, CameraExtrinsics, FastHasherBuilder>)> {

        let mut correspondences = HashMap::
            <(u64, u64), EightPointCameraExtrinsicsSolver, FastHasherBuilder>::default();

        for frame in data {

            let mut camera_ids = frame.cameras.keys().cloned().collect::<Vec<u64>>();
            camera_ids.sort();

            for i in 0..camera_ids.len() {
                for j in (i + 1)..camera_ids.len() {
                    let cam_i = frame.cameras.get(&camera_ids[i]).unwrap();
                    let cam_j = frame.cameras.get(&camera_ids[j]).unwrap();

                    let key = (camera_ids[i], camera_ids[j]);
                    if !correspondences.contains_key(&key) {
                        let int_i = self.camera_intrinsics.get(&key.0).unwrap();
                        let int_j = self.camera_intrinsics.get(&key.1).unwrap();
                        correspondences.insert(key, EightPointCameraExtrinsicsSolver::new(int_i.clone(), int_j.clone()));
                    }

                    let out = correspondences.get_mut(&key).unwrap();

                    assert_eq!(&cam_i.points_3d[..], &cam_j.points_3d[..]);
                    out.add_object(&cam_i.points_3d[..], &cam_i.points_2d[..], &cam_j.points_2d[..]);
                }
            }
        }

        // For the solver on all pairs so that we throw out any bad data.
        let mut relative_poses = correspondences
            .into_iter().filter_map(|(key, solver)| {
                solver.solve().map(|e| (key, e))
            })
            .collect::<HashMap<(u64, u64), CameraExtrinsics, FastHasherBuilder>>();

        let mut out = HashMap::<u64, CameraExtrinsics, FastHasherBuilder>::default();

        // Pick the first camera based on which camera connected to the most other cameras.
        //
        // (we are basically doing a greedy MST construction)
        let first_camera_id = {
            let mut camera_scores = HashMap::<u64, usize, FastHasherBuilder>::default();

            for (cam1, cam2) in relative_poses.keys().cloned() {
                *camera_scores.entry(cam1).or_default() += 1;
                *camera_scores.entry(cam2).or_default() += 1;
            }

            let mut camera_scores = camera_scores.into_iter()
                .map(|(id, count)| (count, id)).collect::<Vec<(usize, u64)>>();

            camera_scores.sort();

            camera_scores.pop().ok_or_else(|| err_msg("Unable to get any camera transforms"))?.1
        };

        out.insert(first_camera_id, CameraExtrinsics::default());

        // TODO: Bound by number of cameras.
        for _ in 0..2000 {
            // Find a camera for which we correctly have extrinsics to expand next
            // based on how many connections it has to unsolved cameras.
            let next_cam_id = {
                let mut camera_scores = HashMap::<u64, usize, FastHasherBuilder>::default();

                for (cam1, cam2) in relative_poses.keys().cloned() {
                    if out.contains_key(&cam1) && out.contains_key(&cam2) {
                        continue;
                    }

                    if out.contains_key(&cam1) {
                        *camera_scores.entry(cam1).or_default() += 1;
                    } else if out.contains_key(&cam2) {
                        *camera_scores.entry(cam2).or_default() += 1;
                    }
                }

                let mut camera_scores = camera_scores.into_iter()
                    .map(|(id, count)| (count, id)).collect::<Vec<(usize, u64)>>();

                camera_scores.sort();

                match camera_scores.pop() {
                    Some((_, v)) => v,
                    None => break
                }
            };

            for ((cam_id1, cam_id2), extrinsics) in relative_poses.iter() {
                let mut cam_id1 = *cam_id1;
                let mut cam_id2 = *cam_id2;
                
                if cam_id1 != next_cam_id && cam_id2 != next_cam_id {
                    continue;
                }


                let mut inverted = false;
                if out.contains_key(&cam_id1) && !out.contains_key(&cam_id2) {
                    // Proceed
                } else if !out.contains_key(&cam_id1) && out.contains_key(&cam_id2) {
                    inverted = true;
                    core::mem::swap(&mut cam_id1, &mut cam_id2);
                } else {
                    // Both already present.
                    continue;
                }

                let mut rel_extrinsics = extrinsics.to_mat4x4();

                if inverted {
                    rel_extrinsics = rel_extrinsics.inverse().unwrap();
                }

                let known_extrinsics = out.get(&cam_id1).unwrap().to_mat4x4();

                let ext = CameraExtrinsics::from_mat4x4(&(
                    rel_extrinsics * known_extrinsics
                ));

                out.insert(cam_id2, ext);
            }
        }

        println!("Num initial extrinsics: {}", out.len());

        /*
        let initial_system_state = self.initial_system_state.as_ref()
            .ok_or_else(|| err_msg("Missing initial system status"))?;

        for cam in initial_system_state.cameras() {
            if !out.contains_key(&cam.id()) {
                eprintln!("Missing camera: {} ({:?})", cam.id(), entity_id_to_string(cam.id()));
            }
        }

        let num_cameras = initial_system_state.cameras().len();

        // TODO: Also verify no unknown cameras are present in the data but not in the initial set.
        if out.len() != num_cameras {
            return Err(err_msg("Not all cameras are linked by some frame chain."));
        }
        */

        Ok((first_camera_id, out))
    }

    /// Old way using just wand pose data
    /// (this is very sensitive to noise)
    fn extract_initial_extrinsics_pnp(
        &self, data: &mut Vec<FrameData>
    ) -> Result<(u64, HashMap<u64, CameraExtrinsics, FastHasherBuilder>)> {
        let mut camera_initial_extrinsics = HashMap::<u64, CameraExtrinsics, FastHasherBuilder>::default();

        data.sort_by_key(|v| v.cameras.len());
        data.reverse();

        let mut anchor_camera_id = None;

        // Init with extrinsics from first entry (the one)
        println!("Best frame spans {} cameras", data[0].cameras.len());
        for (id, cam) in &data[0].cameras {
            let extrinsics = CameraExtrinsics {
                rotation: cam.pattern.rotation.clone(),
                translation: cam.pattern.translation.clone(),
            };

            anchor_camera_id = Some(*id);

            camera_initial_extrinsics.insert(*id, extrinsics);
        }

        // TODO: Validate that we have enough high quality linkages between camera subsets and we 
        // enforce at most 2 hops from any camera to another camera.


        // Greedily attempt to get extrinsics for all other cameras relative to existing cameras.
        // TODO: Implement a proper minimum spanning tree.
        let mut changed = true;

        while changed {
            changed = false;

            for entry in &data[1..] {
                //

                let mut known_cam_id = None;

                for (cam_id, _) in &entry.cameras {
                    if camera_initial_extrinsics.contains_key(cam_id) {
                        known_cam_id = Some(*cam_id);
                        break;
                    }
                }

                // TODO: Skip below stuff if there are no unknown cameras in the current entry.

                let known_cam_id = match known_cam_id {
                    Some(v) => v,
                    None => continue
                };

                let known_cam_extrinsics = camera_initial_extrinsics.get(&known_cam_id).unwrap();
                let known_cam_global_mat = known_cam_extrinsics.to_mat4x4();

                let known_cam_data = entry.cameras.get(&known_cam_id).unwrap();
                let known_cam_local_mat = CameraExtrinsics {
                    rotation: known_cam_data.pattern.rotation.clone(),
                    translation: known_cam_data.pattern.translation.clone(),
                }.to_mat4x4();

                for (cam_id, cam_data) in &entry.cameras {
                    if camera_initial_extrinsics.contains_key(cam_id) {
                        continue;
                    }

                    let cam_local_mat = CameraExtrinsics {
                        rotation: cam_data.pattern.rotation.clone(),
                        translation: cam_data.pattern.translation.clone(),
                    }.to_mat4x4();

                    // TODO: Cache the inverse.
                    let rel_mat = cam_local_mat * known_cam_local_mat.inverse().unwrap();

                    let global_mat = rel_mat * &known_cam_global_mat;

                    camera_initial_extrinsics.insert(
                        *cam_id,
                        CameraExtrinsics::from_mat4x4(&global_mat)
                    );
                    changed = true;
                }
            }
        }

        println!("Num initial extrinsics: {}", camera_initial_extrinsics.len());

        /*
        let initial_system_state = self.initial_system_state.as_ref()
            .ok_or_else(|| err_msg("Missing initial system status"))?;

        for cam in initial_system_state.cameras() {
            if !camera_initial_extrinsics.contains_key(&cam.id()) {
                eprintln!("Missing camera: {} ({:?})", cam.id(), entity_id_to_string(cam.id()));
            }
        }

        let num_cameras = initial_system_state.cameras().len();

        // TODO: Also verify no unknown cameras are present in the data but not in the initial set.
        if camera_initial_extrinsics.len() != num_cameras {
            return Err(err_msg("Not all cameras are linked by some frame chain."));
        }
        */

        Ok((anchor_camera_id.unwrap(), camera_initial_extrinsics))
    }


    /// 'anchor_camera_id' is the camera that we will assume has 'zero' extrinsics and
    ///    is locked in space to reduce the number of parameters we need to optimize.
    ///    This means that this camera must be well connected to other cameras in the
    ///    data to avoid massive pivotting around this camera.
    fn optimize(
        &self,
        data: &[FrameData],
        anchor_camera_id: u64,
        camera_initial_extrinsics: &HashMap<u64, CameraExtrinsics, FastHasherBuilder>,
    ) -> Result<WandingCalibrationSolution> {
        // TODO: Use something like a Huber/Cauchy loss to reduce sensitivity to outliers if we
        // messed up want detection.
        let mut solver = BundleAdjustmentSolver::new();

        let mut camera_id_to_index = HashMap::<u64, usize>::default();

        for (id, extrinsics) in camera_initial_extrinsics {
            // TODO: Chose whichever camera has the most coincidences with other cameras.
            let fixed = camera_id_to_index.is_empty();

            let idx = solver.add_camera(
                self.camera_intrinsics.get(id).unwrap(),
                &extrinsics.rotation,
                &extrinsics.translation,
                *id == anchor_camera_id,
            );
            // TODO: If we know that initial intrinsics were calibrated from this camera
            // without modifications, contrain how much the intrinsics are allowed to change.
            if self.config.wanding().freeze_intrinsics() {
                solver.freeze_camera_intrinsics(idx);
            }

            camera_id_to_index.insert(*id, idx);
        }

        for entry in data {

            let mut object_translation = Vector3d::zero();
            let mut object_rotation = Vector3d::zero();
            {
                let mut first = true;

                for (cam_id, cam_data) in entry.cameras.iter() {
                    let (cam_id, cam_data) = entry.cameras.iter().next().unwrap();

                    let cam_idx = camera_id_to_index.get(cam_id).unwrap();

                    // estimated_transform = initial_camera_transform * object_transform
                    // ^ Need to solve for object_transform;

                    let initial_camera_transform = camera_initial_extrinsics.get(cam_id).unwrap()
                        .to_mat4x4();

                    let estimated_transform = CameraExtrinsics {
                        rotation: cam_data.pattern.rotation.clone(),
                        translation: cam_data.pattern.translation.clone(),
                    }.to_mat4x4();

                    // println!("BBB: {:?}", estimated_transform);
                    
                    let object_transform = initial_camera_transform.inverse().unwrap() * estimated_transform;

                    // println!("OBJ TRANSFORM: {:?}", object_transform);

                    let extrinsics = CameraExtrinsics::from_mat4x4(&object_transform);


                    if first {
                        first = false;
                        object_translation = extrinsics.translation;
                        object_rotation = extrinsics.rotation;
                    }
                }

                // object_translation /= (entry.cameras.len() as f64);


                /*
                let mut solver = BundleAdjustmentSolver::new();

                let object_idx = solver.add_object(
                    &object_rotation,
                    &object_translation
                );

                for (cam_id, cam_data) in &entry.cameras {
                    let cam_idx = solver.add_camera(
                        self.camera_intrinsics.get(cam_id).unwrap(),
                        &camera_initial_extrinsics.get(cam_id).unwrap().rotation,
                        &camera_initial_extrinsics.get(cam_id).unwrap().translation,
                        true,
                    );
                    solver.freeze_camera_intrinsics(cam_idx);

                    for i in 0..cam_data.points_2d.len() {
                        let point_2d = &cam_data.points_2d[i];
                        let point_3d = &cam_data.points_3d[i];

                        solver.add_object_point_view(
                            object_idx,
                            cam_idx,
                            point_2d,
                            point_3d
                        );
                    }
                }

                let sol = solver.solve();

                let obj_ext = sol.object_extrinsics(object_idx);

                object_rotation = obj_ext.rotation;
                object_translation = obj_ext.translation;
                */

            }


            let object_idx = {
                solver.add_object(
                    &object_rotation,
                    &object_translation
                )
            };

            for (cam_id, cam_data) in &entry.cameras {
                let cam_idx = *camera_id_to_index.get(cam_id).unwrap();

                for i in 0..cam_data.points_2d.len() {
                    let point_2d = &cam_data.points_2d[i];
                    let point_3d = &cam_data.points_3d[i];

                    solver.add_object_point_view(
                        object_idx,
                        cam_idx,
                        point_2d,
                        point_3d
                    );
                }
            }

        }

        println!("Solving...");

        solver.enable_logging();

        let start = Instant::now();

        let solution = solver.solve();

        let end = Instant::now();

        println!("Solved in {:?}!", end - start);

        let mut out = WandingCalibrationSolution {
            error: solution.error(),
            params: vec![],
            residuals: vec![],
        };

        // Objects were added in the order of 'data'.
        for (object_idx, entry) in data.iter().enumerate() {
            let object = solution.object_extrinsics(object_idx);

            for (cam_id, cam_data) in &entry.cameras {
                let cam_idx = *camera_id_to_index.get(cam_id).unwrap();
                let intrinsics = solution.camera_intrinsics(cam_idx);
                let extrinsics = solution.camera_extrinsics(cam_idx);

                let mut sum = 0.0;
                for (point_2d, point_3d) in cam_data.points_2d.iter().zip(cam_data.points_3d.iter()) {
                    let projected = intrinsics.project_point(
                        &extrinsics.transform(&object.transform(point_3d)));
                    sum += (projected - point_2d).norm_squared();
                }

                out.residuals.push(WandingSightingResidual {
                    frame_timestamp: entry.frame_timestamp,
                    camera_id: *cam_id,
                    error: (sum / (cam_data.points_2d.len() as f64)).sqrt(),
                });
            }
        }

        for (cam_id, cam_idx) in camera_id_to_index {
            out.params.push(CameraParameters {
                id: cam_id,
                intrinsics: solution.camera_intrinsics(cam_idx),
                extrinsics: solution.camera_extrinsics(cam_idx)
            });
        }

        // TODO: Ideally compute RMS over all data and not just the selected subset.
        println!("RMS Error: {}", out.error);

        Ok(out)
    }

    // MAkes 
    fn align_extrinsics(
        &self,
        camera_params: &mut [CameraParameters]
    ) -> Result<()> {
        let mut up_vector = Vector3d::zero();
        let mut num_cams = 0;

        let initial_system_state = self.initial_system_state.as_ref()
            .ok_or_else(|| err_msg("Missing initial system status"))?;

        for cam in initial_system_state.cameras() {
            let cam_id = cam.id();

            let cam_enabled = initial_system_state.config().per_camera()
                .iter().find(|c| c.camera_id() == cam_id).map(|c| c.enabled())
                .unwrap_or(false);

            if !cam_enabled {
                continue;
            }

            // Cameras without an accelerometer (Rift CV1) have no say in
            // which way is up.
            if !cam.camera_status().accelerometer().has_value() {
                continue;
            }

            let extrinsics = &camera_params.iter().find(|p| p.id == cam_id)
                .ok_or_else(|| err_msg("Camera not declared in system_state"))?
                .extrinsics;

            let r = from_axis_angle(&extrinsics.rotation);

            let proto = cam.camera_status().accelerometer().value();
            let accel_vec = Vector3d::from_proto(proto)?;

            let v: Vector3d = r.transpose() * accel_vec.normalized();
            up_vector += v;
            num_cams += 1;
        }

        // No camera knows which way is up, so there is nothing to level
        // against (and the height and yaw steps below assume a level world).
        // The solution stays in the frame of the wand sighting that seeded
        // the solve; Set Origin (origin.rs) then takes the floor plane and
        // the up direction from the wand lying flat.
        if num_cams == 0 {
            println!("No camera reports an accelerometer: not levelled. Use Set Origin with the wand flat on the floor.");
            return Ok(());
        }

        up_vector /= (num_cams as f64);
        up_vector.normalize();

        
        let r_align = {
            let z = vec3d(0., 0., 1.);
            let mut axis = up_vector.cross(&z).normalized();
            let angle = up_vector.dot(&z).acos();
            axis *= angle;
            from_axis_angle(&axis)
        };

        let t_align = {

            let mut cam_center_pos = vec3d(0., 0., 0.);
            let mut num_cams = 0;
            let mut z_max = -1000000.0f64;
            let mut z_min = 1000000.0f64;

            for p in camera_params.iter() {
                let pos = &r_align * p.extrinsics.position();
                z_max = z_max.max(pos.z());
                z_min = z_min.min(pos.z());
                cam_center_pos += pos;
                num_cams += 1;
            }

            cam_center_pos /= (num_cams as f64);

            let t_z = self.config.wanding().camera_height_guess() - z_max;

            vec3d(
                -cam_center_pos.x(),
                -cam_center_pos.y(),
                t_z
            )
        };

        let align_mat = {
            let mut out = Matrix4d::zero();
            out.block_mut(0, 0).copy_from(&r_align.transpose());
            out.block_mut(0, 3).copy_from(&(
                r_align.transpose() * t_align * -1.0   
            ));
            out[(3, 3)] = 1.0;
            out
        };

        for p in camera_params.iter_mut() {
            p.extrinsics = CameraExtrinsics::from_mat4x4(&(
                p.extrinsics.to_mat4x4() * &align_mat
            ));
        }

        // TODO: Only do this if we have an accelerometer lock.

        // Correcting the yaw (rotation around Z) such that the cameras align
        // with orthogonal directions.
        if camera_params.len() > 1 {
            // Order the cameras based on their angle from the center.
            let mut camera_order = vec![];
            for p in camera_params.iter() {
                let pos = p.extrinsics.position();
                camera_order.push((pos.y().atan2(pos.x()), p.id));
            }

            // TODO: Make more resilient to cameras that are stacked vertically.
            camera_order.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let mut dir_sum = vec2d(0., 0.);

            for i in 0..camera_order.len() {
                let j = (i + 1) % camera_order.len();
                
                let i_id = camera_order[i].1;
                let j_id = camera_order[j].1;

                let p_i = camera_params.iter().find(|p| p.id == i_id).unwrap();
                let p_j = camera_params.iter().find(|p| p.id == j_id).unwrap();

                let dir = p_i.extrinsics.position() - p_j.extrinsics.position();
                let angle = dir.y().atan2(dir.x());

                dir_sum[0] += (4. * angle).cos();
                dir_sum[1] += (4. * angle).sin();
            }

            let yaw = dir_sum[1].atan2(dir_sum[0]) / 4.0;

            {
                let align_mat = {
                    let mut out = Matrix4d::identity();
                    // TODO: Use a helper for this.
                    out[(0, 0)] = (-yaw).cos();
                    out[(0, 1)] = -(-yaw).sin();
                    out[(1, 0)] = (-yaw).sin();
                    out[(1, 1)] = (-yaw).cos();
                    out
                };

                for p in camera_params.iter_mut() {
                    p.extrinsics = CameraExtrinsics::from_mat4x4(&(
                        p.extrinsics.to_mat4x4() * &align_mat
                    ));
                }
            }
        }

        Ok(())
    }

}

fn pattern_positions_distinct(a: &BlobPatternMatch, b: &BlobPatternMatch) -> bool {
    let dt = (&a.translation - &b.translation).norm();
    if dt > 0.05 { // 50mm
        return true;
    }

    let a_r = from_axis_angle(&a.rotation);
    let b_r = from_axis_angle(&b.rotation);

    let dr = to_axis_angle(
        &(a_r.inverse().unwrap() * b_r)
    );

    let angle = dr.norm().abs() * 180.0 / std::f64::consts::PI;

    if angle > 20.0 {
        return true;
    }


    false
}

struct FrameData {
    frame_timestamp: u64,
    cameras: HashMap<u64, FrameCameraData, FastHasherBuilder>

}

#[derive(Debug)]
struct FrameCameraData {
    pattern: BlobPatternMatch,
    points_2d: Vec<Vector2d>,
    points_3d: Vec<Vector3d>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn camera(id: u64, rotation: Vector3d, translation: Vector3d) -> CameraParameters {
        CameraParameters {
            id,
            intrinsics: CameraIntrinsicsModel::from_nominal_params(1280, 960, 710., 1.),
            extrinsics: CameraExtrinsics { rotation, translation },
        }
    }

    /// A manager status with one enabled camera per entry, ids from 1.
    fn status(accelerometers: &[Option<Vector3d>]) -> MocapManagerStatus {
        let mut status = MocapManagerStatus::default();
        for (i, accel) in accelerometers.iter().enumerate() {
            let id = (i + 1) as u64;

            {
                let cam = status.new_cameras();
                cam.set_id(id);
                if let Some(v) = accel {
                    cam.camera_status_mut().accelerometer_mut().set_value(v.to_proto());
                }
            }

            let per_cam = status.config_mut().new_per_camera();
            per_cam.set_camera_id(id);
            per_cam.set_enabled(true);
        }
        status
    }

    /// A camera at 'position' looking at 'target', x to its right.
    fn looking_at(id: u64, position: &Vector3d, target: &Vector3d) -> CameraParameters {
        let z = (target.clone() - position.clone()).normalized();
        let x = z.cross(&vec3d(0., 0., 1.)).normalized();
        let y = z.cross(&x);

        let mut axes = Matrix3d::zero();
        axes.col_mut(0).copy_from(&x);
        axes.col_mut(1).copy_from(&y);
        axes.col_mut(2).copy_from(&z);
        let r = axes.transpose();

        let translation = (r.clone() * position.clone()) * -1.0;
        camera(id, to_axis_angle(&r), translation)
    }

    /// The whole wanding solve on made-up sightings, with the shipped config:
    /// three cameras a few metres from a wand that keeps its markers towards
    /// them. Some frames carry what a first wanding on real cameras held and
    /// the solve can't absorb: a view of the wand's back (a reflection), and
    /// sightings that fit their own camera but sit in the wrong place.
    #[test]
    fn solves_through_reflections_and_stray_sightings() {
        let mut config = MocapManagerConfig::default();
        protobuf::text::parse_text_proto(
            include_str!("../../config/manager.txtpb"),
            &mut config
        ).unwrap();
        config.clear_per_camera();
        assert!(config.wanding().freeze_intrinsics());
        assert!(config.wanding().max_solved_frame_error() > 0.0);

        let target = vec3d(0., 0., 1.0);
        let truth = vec![
            looking_at(1, &vec3d(3.0, -1.6, 1.8), &target),
            looking_at(2, &vec3d(3.4, 0.0, 1.2), &target),
            looking_at(3, &vec3d(3.0, 1.6, 2.0), &target),
        ];

        let mut intrinsics = HashMap::<u64, CameraIntrinsicsModel, FastHasherBuilder>::default();
        for cam in &truth {
            intrinsics.insert(cam.id, cam.intrinsics.clone());
        }

        let mut seed = 12345u64;
        let mut random = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / ((1u64 << 31) as f64)
        };

        let w = config.wand().clone();
        let model = [
            vec3d(-w.left_arm_length(), 0., 0.),
            vec3d(0., 0., 0.),
            vec3d(w.right_arm_length(), 0., 0.),
            vec3d(0., -w.bottom_length(), 0.),
        ];

        const STEP: u64 = 11_111_000;
        let reflected = [7u64, 23];
        let stray = [11u64, 31, 40];

        let mut solver = WandingCalibrationSolver::new(config.clone(), intrinsics);
        solver.set_initial_status(status(&[None, None, None]));

        for frame in 0..60u64 {
            // Markers towards the cameras (+x), turned and tipped a little.
            let position = vec3d(random() - 0.5, random() - 0.5, 0.6 + 0.8 * random());
            let pose = {
                let towards = (vec3d(3.2, 0., 1.6) - position.clone()).normalized();
                let side = towards.cross(&vec3d(0., 0., 1.)).normalized();
                let mut r = Matrix3d::zero();
                r.col_mut(0).copy_from(&side);
                r.col_mut(1).copy_from(&towards.cross(&side));
                r.col_mut(2).copy_from(&towards);

                let roll = from_axis_angle(&vec3d(0., 0., 6.28 * random()));
                let tip = from_axis_angle(&vec3d(0.3 * (random() - 0.5), 0.3 * (random() - 0.5), 0.));
                CameraExtrinsics { rotation: to_axis_angle(&(r * tip * roll)), translation: position }
            };

            let mut entry = MocapLogEntry::default();
            entry.blobs_mut().set_frame_timestamp(frame * STEP);

            for cam in &truth {
                let results = entry.blobs_mut().new_cameras();
                results.set_camera_id(cam.id);

                for point in &model {
                    let mut world = pose.transform(point);
                    if reflected.contains(&frame) && cam.id == 2 {
                        // In a mirror on the wall at y = -2.
                        world[1] = -4.0 - world[1];
                    }

                    let mut px = cam.intrinsics.project_point(&cam.extrinsics.transform(&world));
                    if stray.contains(&frame) && cam.id == 3 {
                        px += vec2d(25., -15.);
                    }

                    let blob = results.results_mut().new_blobs();
                    blob.set_x(px.x() as f32);
                    blob.set_y(px.y() as f32);
                    blob.set_radius_a(2.0);
                    blob.set_radius_b(2.0);
                }
            }

            solver.add_frame(&entry).unwrap();
        }

        let solution = solver.solve().unwrap();

        // The blobs are f32, good to about 1e-4 px.
        assert!(solution.error < 0.01, "error {}", solution.error);

        // The stray sightings fit their own camera, so they got as far as the
        // solve, and were dropped from it.
        for r in &solution.residuals {
            assert!(!stray.contains(&(r.frame_timestamp / STEP)) || r.camera_id != 3,
                "stray frame {} kept", r.frame_timestamp / STEP);
            assert!(!reflected.contains(&(r.frame_timestamp / STEP)) || r.camera_id != 2,
                "reflected frame {} kept", r.frame_timestamp / STEP);
        }
        assert!(solution.residuals.len() > 100);

        // The cameras stand as they were set, each with its own lens model.
        for a in &truth {
            let solved_a = solution.params.iter().find(|p| p.id == a.id).unwrap();
            assert_eq!(solved_a.intrinsics.focal_length, a.intrinsics.focal_length);
            assert_eq!(solved_a.intrinsics.center, a.intrinsics.center);

            for b in &truth {
                let solved_b = solution.params.iter().find(|p| p.id == b.id).unwrap();
                let set = (a.extrinsics.position() - b.extrinsics.position()).norm();
                let solved = (solved_a.extrinsics.position() - solved_b.extrinsics.position()).norm();
                assert!((set - solved).abs() < 0.001, "{} to {}: {} m, set {}", a.id, b.id, solved, set);
            }
        }
    }

    /// Rift cameras report no accelerometer: the solve must still finish,
    /// with the extrinsics left as solved for Set Origin to level.
    #[test]
    fn align_without_accelerometers() {
        let mut solver = WandingCalibrationSolver::new(
            MocapManagerConfig::default(), HashMap::default());
        solver.set_initial_status(status(&[None, None]));

        let mut params = vec![
            camera(1, vec3d(0.1, 0.2, 0.3), vec3d(0.4, 0.5, 0.6)),
            camera(2, vec3d(-0.3, 0.9, 0.1), vec3d(-1.0, 0.2, 2.5)),
        ];
        let before = params.clone();

        solver.align_extrinsics(&mut params).unwrap();

        for (a, b) in params.iter().zip(before.iter()) {
            assert_eq!(a.extrinsics.rotation, b.extrinsics.rotation);
            assert_eq!(a.extrinsics.translation, b.extrinsics.translation);
        }
    }

    /// Cameras with accelerometers are still levelled, and one without is
    /// left out of the up vector rather than failing the calibration.
    #[test]
    fn align_with_some_accelerometers() {
        let mut config = MocapManagerConfig::default();
        config.wanding_mut().set_camera_height_guess(2.1);

        let mut solver = WandingCalibrationSolver::new(config, HashMap::default());

        // Camera 1 has no rotation and reads its own +y as up, so the
        // world's +y is up before alignment.
        solver.set_initial_status(status(&[Some(vec3d(0., 1., 0.)), None]));

        let mut params = vec![
            camera(1, vec3d(0., 0., 0.), vec3d(0., 0., 0.)),
            camera(2, vec3d(0., 0., 0.), vec3d(-2., 0., 0.)),
        ];

        solver.align_extrinsics(&mut params).unwrap();

        // The new world's +z is what camera 1 reads as up.
        let up_in_camera = from_axis_angle(&params[0].extrinsics.rotation) * vec3d(0., 0., 1.);
        assert!((up_in_camera - vec3d(0., 1., 0.)).norm() < 1e-6);

        // The highest camera sits at the height guess.
        let z_max = params.iter().map(|p| p.extrinsics.position().z()).fold(f64::MIN, f64::max);
        assert!((z_max - 2.1).abs() < 1e-6);
    }
}


