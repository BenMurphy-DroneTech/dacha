//! Blob detection for one Rift camera: Dennis's FrameProcessor, called the
//! way his camera node calls it, between two Rift-specific steps:
//!
//! 1. before: the camera's hot pixels are repaired in the frame (hotpix.rs);
//! 2. Dennis's `FrameProcessor::process` with the threshold and blob filter
//!    the manager last configured (his node does reset + process_lines +
//!    finish over chunks of the frame; uvcvideo hands over whole frames, so
//!    this is the one call that does the same);
//! 3. after: each blob centre moves from the 960x576 window to full-sensor
//!    coordinates and is undistorted with the camera's factory fisheye
//!    calibration (lens.rs), so dacha gets a pinhole camera. Radii and angle
//!    stay as measured in the image.
//!
//! The settings are copied out of the shared config only when the manager
//! changes them (one atomic load per frame otherwise), so a frame does no
//! locking or allocation beyond what FrameProcessor itself does.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use mocap_camera_core::FrameProcessor;
use mocap_proto::mocap::{BlobFilterConfig, BlobResults};
use rift_camera::lens::FisheyeCalibration;

use crate::config::{FRAME_HEIGHT, FRAME_WIDTH};
use crate::hotpix::{HotPixels, COLUMN_START};
use crate::slots::monotonic_ns;

/// Frames kept for the p99 in a report (10 s at 90 fps is 900).
const PUBLISH_SAMPLES: usize = 2048;

/// The blob filter for Rift markers down to 2.6 px (a 20 mm sphere at 5.5 m),
/// from rift_blob_sim running synthetic markers through Dennis's
/// FrameProcessor: only elliptical_area_error moves from dacha's 0.25 (which
/// rejects every blob under about 5 px; small blobs always cover more pixels
/// than pi * a * b) and the skew limit tightens to 1 px. The simulation
/// passed 100% of 2.6-60 px blobs at 1.5 (worst -1.48), but a real diffused
/// LED's saturated core and wide faint halo reach -1.52 to -1.68, so 3.0:
/// at 1 or more the check only rejects blobs covering over (1 + limit) times
/// their ellipse, and aspect, radius ratio and skew still catch merged or odd
/// shapes. Noise, hot pixels and pixel pairs don't pass (min_area 6 also keeps
/// out the diagonal pairs whose radius_b comes out NaN). The manager's config
/// (config/manager.txtpb) matches.
pub const RIFT_THRESHOLD: u32 = 30;
pub const RIFT_BLOB_FILTER: &str = r#"
    min_area: 6
    max_area: 10000
    reject_edge: true
    max_bbox_aspect_ratio: 1.5
    elliptical_area_error: 3.0
    max_radius_ratio: 1.5
    max_centroid_bbox_skew { abs: 1  rel: 0.05 }
"#;

/// The blob settings the manager configures, shared between the Camera
/// service (writer) and the capture loop (reader).
pub struct BlobSettings {
    version: AtomicU64,
    value: Mutex<(u8, BlobFilterConfig)>,
}

impl BlobSettings {
    pub fn new(threshold: u32, filter: BlobFilterConfig) -> Self {
        Self { version: AtomicU64::new(1), value: Mutex::new((clamp_threshold(threshold), filter)) }
    }

    pub fn set(&self, threshold: u32, filter: BlobFilterConfig) {
        *self.value.lock().unwrap() = (clamp_threshold(threshold), filter);
        self.version.fetch_add(1, Ordering::Release);
    }
}

/// As Dennis's node validates pixel_threshold: 1 to 255.
fn clamp_threshold(t: u32) -> u8 {
    t.clamp(1, 255) as u8
}

pub struct Detector {
    processor: FrameProcessor,
    hot: HotPixels,
    lens: Option<FisheyeCalibration>,
    row_start: f32,

    version: u64,
    threshold: u8,
    filter: BlobFilterConfig,

    // For the report; cleared each time it is printed.
    frames: u64,
    blobs: u64,
    rejected: u64,
    process_ns_sum: i64,
    process_ns_max: i64,
    publish_ns_sum: i64,
    publish_ns_max: i64,
    /// Each frame's dequeue to publish, for the p99 (preallocated, never grown).
    publish_ns: Vec<i64>,

    /// Save the next frame (before and after the hot-pixel repair) for a look.
    dump: Option<String>,
}

impl Detector {
    pub fn new(hot: HotPixels, row_start: u32) -> Self {
        let mut hot = hot;
        hot.set_window(row_start);
        Self {
            processor: FrameProcessor::new(FRAME_WIDTH as usize, FRAME_HEIGHT as usize),
            hot,
            lens: None,
            row_start: row_start as f32,
            version: 0,
            threshold: 255,
            filter: BlobFilterConfig::default(),
            frames: 0, blobs: 0, rejected: 0,
            process_ns_sum: 0, process_ns_max: 0, publish_ns_sum: 0, publish_ns_max: 0,
            publish_ns: Vec::with_capacity(PUBLISH_SAMPLES),
            dump: None,
        }
    }

    pub fn set_lens(&mut self, lens: Option<FisheyeCalibration>) {
        self.lens = lens;
    }

    pub fn hot_pixels_in_window(&self) -> usize {
        self.hot.in_window()
    }

    /// Detects the blobs in one full 960x576 frame, in place.
    #[inline]
    pub fn detect(&mut self, frame: &mut [u8], settings: &BlobSettings) -> BlobResults {
        let t0 = monotonic_ns();

        let v = settings.version.load(Ordering::Acquire);
        if v != self.version {
            let value = settings.value.lock().unwrap();
            self.threshold = value.0;
            self.filter = value.1.clone();
            self.version = v;
        }

        if let Some(path) = self.dump.take() {
            let _ = std::fs::write(format!("{}.raw", path), &frame[..]);
            let mut repaired = frame.to_vec();
            self.hot.repair(&mut repaired);
            let _ = std::fs::write(format!("{}.repaired", path), &repaired);
            println!("dumped a frame to {}.raw / .repaired (960x576 8-bit)", path);
        }
        self.hot.repair(frame);
        let mut results = self.processor.process(frame, self.threshold, &self.filter);

        for blob in results.blobs_mut().iter_mut() {
            let x = blob.x() + COLUMN_START as f32;
            let y = blob.y() + self.row_start;
            // Dennis's centres put pixel i's centre at i + 0.5; the factory
            // calibration (OpenCV's convention) at i. Undistort in the
            // calibration's convention and come back; the pinhole model in
            // Status carries the same half-pixel shift (service.rs).
            let (x, y) = match &self.lens {
                Some(lens) => {
                    let (ux, uy) = lens.undistort(x as f64 - 0.5, y as f64 - 0.5);
                    (ux as f32 + 0.5, uy as f32 + 0.5)
                }
                None => (x, y),
            };
            blob.set_x(x);
            blob.set_y(y);
        }

        let dt = monotonic_ns() - t0;
        self.frames += 1;
        self.blobs += results.blobs().len() as u64;
        self.rejected += results.num_rejected_blobs() as u64;
        self.process_ns_sum += dt;
        self.process_ns_max = self.process_ns_max.max(dt);
        results
    }

    /// Dequeue to publish for a frame, for the report.
    #[inline]
    pub fn published(&mut self, dq_to_publish_ns: i64) {
        self.publish_ns_sum += dq_to_publish_ns;
        self.publish_ns_max = self.publish_ns_max.max(dq_to_publish_ns);
        if self.publish_ns.len() < PUBLISH_SAMPLES {
            self.publish_ns.push(dq_to_publish_ns);
        }
    }

    /// Arms a dump of the next detected frame if /dev/shm/rift_dump_<serial>
    /// exists (removing it). Checked with the report, not per frame.
    pub fn check_dump_request(&mut self, serial: &str) {
        let trigger = format!("/dev/shm/rift_dump_{}", serial);
        if std::fs::remove_file(&trigger).is_ok() {
            self.dump = Some(format!("/dev/shm/rift_frame_{}", serial));
        }
    }

    fn publish_p99(&mut self) -> i64 {
        if self.publish_ns.is_empty() {
            return 0;
        }
        self.publish_ns.sort_unstable();
        self.publish_ns[(self.publish_ns.len() - 1) * 99 / 100]
    }

    pub fn report(&mut self) -> String {
        let n = self.frames.max(1) as f64;
        let line = if self.frames == 0 {
            "blobs: idle (no subscribers)".to_string()
        } else {
            format!(
                "blobs: {:.2} per frame, {:.1} rejected; detect {:.0} us mean, {:.0} max; dequeue to publish {:.0} us mean, {:.0} max, {:.0} p99",
                self.blobs as f64 / n, self.rejected as f64 / n,
                self.process_ns_sum as f64 / n / 1e3, self.process_ns_max as f64 / 1e3,
                self.publish_ns_sum as f64 / n / 1e3, self.publish_ns_max as f64 / 1e3, self.publish_p99() as f64 / 1e3)
        };
        self.publish_ns.clear();
        self.frames = 0;
        self.blobs = 0;
        self.rejected = 0;
        self.process_ns_sum = 0;
        self.process_ns_max = 0;
        self.publish_ns_sum = 0;
        self.publish_ns_max = 0;
        line
    }
}
