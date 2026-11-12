//! MJPEG previews for ReadFrames: Dennis's MJPEGEncoder, unchanged, on a
//! thread of its own, as his node's mjpeg_processing_thread.
//!
//! The capture loop offers every frame after its blobs are published and
//! before its buffer goes back to the driver. Only while ReadFrames has a
//! subscriber, and no more often than the manager's mjpeg.max_fps, is the
//! frame copied into one of two preallocated buffers and handed to the
//! encoder thread; the buffer comes back when the JPEG is out. Between
//! previews an offer is one compare. A preview the encoder is still busy
//! with is skipped, never waited for, and the encoder runs at SCHED_IDLE.
//!
//! The copy is taken after the blobs, so it shows the frame with its hot
//! pixels repaired whenever blob detection ran on it.
//!
//! The preview is drawn in the frame the blob centres are reported in, so the
//! manager's camera view puts each blob on its marker: the full 1280x960
//! sensor, undistorted to the pinhole model in Status (detect.rs), the 960x576
//! window's pixels remapped into it and the rest black. Every centre the
//! window can see lands inside (x about 46-1206, y 128-834 at row_start 192).
//! The remap is a table built on the encoder thread whenever the lens or the
//! window changes (set_view, from the capture loop after each sync setup),
//! about 1 ms a preview.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};

use common::bytes::Bytes;
use common::errors::*;
use executor_multitask::BroadcastChannel;
use image::ImageRef;
use mocap_camera_core::MJPEGEncoder;
use mocap_proto::mocap::MJPEGEncoderConfig;

use rift_camera::lens::{FisheyeCalibration, SENSOR_HEIGHT, SENSOR_WIDTH};

use crate::config::{FRAME_HEIGHT, FRAME_WIDTH};
use crate::hotpix::COLUMN_START;

/// Frame copies in flight: one being encoded, one being filled.
const NUM_COPIES: usize = 2;

/// The preview settings the manager configures (mjpeg and pixel_threshold),
/// shared between the Camera service (writer), the capture loop (the
/// interval, one atomic load an offer) and the encoder thread.
pub struct PreviewSettings {
    /// 1 / max_fps; 0 while max_fps is 0 (no previews).
    interval_ns: AtomicU64,
    value: Mutex<(u8, MJPEGEncoderConfig)>,
    /// The blobs' frame: (version, lens, row_start); version 0 until the
    /// capture loop sets it, and no previews until then.
    view: Mutex<(u64, Option<FisheyeCalibration>, u32)>,
}

impl PreviewSettings {
    pub fn new(threshold: u32, config: MJPEGEncoderConfig) -> Arc<Self> {
        let s = Arc::new(Self { interval_ns: AtomicU64::new(0), value: Mutex::new((0, MJPEGEncoderConfig::default())),
                                view: Mutex::new((0, None, 0)) });
        s.set(threshold, config);
        s
    }

    pub fn set(&self, threshold: u32, config: MJPEGEncoderConfig) {
        let fps = config.max_fps();
        self.interval_ns.store(if fps == 0 { 0 } else { 1_000_000_000 / fps as u64 }, Ordering::Relaxed);
        *self.value.lock().unwrap() = (threshold.clamp(1, 255) as u8, config);
    }

    /// The frame the blob centres are in: undistorted with `lens` (None: left
    /// distorted), the window starting at sensor row `row_start`.
    pub fn set_view(&self, lens: Option<FisheyeCalibration>, row_start: u32) {
        let mut view = self.view.lock().unwrap();
        if view.0 == 0 || view.1 != lens || view.2 != row_start {
            *view = (view.0 + 1, lens, row_start);
        }
    }
}

/// The window pixel each pixel of the preview shows: for preview pixel (u, v),
/// the index into a window copy, or FRAME_WIDTH * FRAME_HEIGHT (the copy's
/// guard byte, always 0) where it falls outside the window.
fn build_remap(lens: Option<FisheyeCalibration>, row_start: u32) -> Vec<u32> {
    let outside = FRAME_WIDTH * FRAME_HEIGHT;
    let mut remap = Vec::with_capacity((SENSOR_WIDTH * SENSOR_HEIGHT) as usize);
    for v in 0..SENSOR_HEIGHT {
        for u in 0..SENSOR_WIDTH {
            // Pixel (u, v) covers blob centres u to u + 1 (Dennis's pixel
            // centres at + 0.5): its centre is (u, v) in the calibration's
            // convention, where sensor pixel i is centred on i.
            let (x, y) = match &lens {
                Some(lens) => lens.distort(u as f64, v as f64),
                None => (u as f64, v as f64),
            };
            let c = x.round() - COLUMN_START as f64;
            let r = y.round() - row_start as f64;
            remap.push(if c >= 0.0 && c < FRAME_WIDTH as f64 && r >= 0.0 && r < FRAME_HEIGHT as f64 {
                r as u32 * FRAME_WIDTH + c as u32
            } else {
                outside
            });
        }
    }
    remap
}

/// The capture loop's end: offers frames to the encoder thread.
pub struct Previews {
    frames: Arc<BroadcastChannel<Bytes>>,
    settings: Arc<PreviewSettings>,
    to_encoder: mpsc::SyncSender<Vec<u8>>,
    free: mpsc::Receiver<Vec<u8>>,
    next_ns: i64,
}

impl Previews {
    /// Starts the encoder thread, which sends each JPEG to `frames`.
    pub fn start(frames: Arc<BroadcastChannel<Bytes>>, settings: Arc<PreviewSettings>) -> Result<Self> {
        let (to_encoder, from_capture) = mpsc::sync_channel::<Vec<u8>>(NUM_COPIES);
        let (to_capture, free) = mpsc::sync_channel::<Vec<u8>>(NUM_COPIES);
        for _ in 0..NUM_COPIES {
            // (one byte over: the remap's guard, never written)
            to_capture.send(vec![0u8; (FRAME_WIDTH * FRAME_HEIGHT) as usize + 1]).unwrap();
        }

        let (out, s) = (frames.clone(), settings.clone());
        std::thread::Builder::new().name("rift_mjpeg".into()).spawn(move || {
            // SCHED_IDLE: the encoder only gets a core nothing else wants,
            // so it never preempts a capture loop mid-detection.
            let param = libc::sched_param { sched_priority: 0 };
            if unsafe { libc::sched_setscheduler(0, libc::SCHED_IDLE, &param) } != 0 {
                eprintln!("MJPEG preview: SCHED_IDLE refused; encoding at normal priority");
            }
            let mut encoder = MJPEGEncoder::new();
            let mut sensor = vec![0u8; (SENSOR_WIDTH * SENSOR_HEIGHT) as usize];
            let (mut remap, mut remap_version) = (Vec::new(), 0);
            while let Ok(copy) = from_capture.recv() {
                let (version, lens, row_start) = *s.view.lock().unwrap();
                if version != remap_version {
                    remap = build_remap(lens, row_start);
                    remap_version = version;
                }
                if remap_version != 0 {
                    for (px, &i) in sensor.iter_mut().zip(remap.iter()) {
                        *px = copy[i as usize];
                    }
                    let (threshold, config) = s.value.lock().unwrap().clone();
                    let image = ImageRef { width: SENSOR_WIDTH as usize, height: SENSOR_HEIGHT as usize, channels: 1, data: &sensor };
                    match encoder.encode(image, threshold, &config) {
                        Ok(data) => { out.send(data.into()); }
                        Err(e) => eprintln!("MJPEG preview: {}", e),
                    }
                }
                if to_capture.send(copy).is_err() {
                    break;
                }
            }
        })?;

        Ok(Self { frames, settings, to_encoder, free, next_ns: 0 })
    }

    /// The blobs' frame, after each sync setup (PreviewSettings::set_view).
    pub fn set_view(&self, lens: Option<FisheyeCalibration>, row_start: u32) {
        self.settings.set_view(lens, row_start);
    }

    /// One frame, in place in its capture buffer.
    #[inline]
    pub fn offer(&mut self, now_ns: i64, frame: &[u8]) {
        if now_ns < self.next_ns {
            return;
        }
        let interval = self.settings.interval_ns.load(Ordering::Relaxed);
        if interval == 0 {
            // Previews off: look again in a second.
            self.next_ns = now_ns + 1_000_000_000;
            return;
        }
        self.next_ns = now_ns + interval as i64;

        if !self.frames.active() {
            return;
        }
        if let Ok(mut copy) = self.free.try_recv() {
            copy[..frame.len()].copy_from_slice(frame);
            let _ = self.to_encoder.try_send(copy);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A blob centre as detect.rs reports it (window pixel to full sensor,
    /// undistorted, Dennis's + 0.5 convention) lands on a preview pixel that
    /// shows that same window pixel, across the window.
    #[test]
    fn blobs_land_on_their_pixels() {
        let lens = FisheyeCalibration { f: 716.206, cx: 663.816, cy: 478.217,
            k: [0.072958, -0.025257, 0.005886, -0.000611] };
        let row_start = 192;
        for lens in [Some(lens), None] {
            let remap = build_remap(lens, row_start);
            for (c, r) in [(0u32, 0u32), (480, 288), (959, 575), (5, 570), (950, 3), (300, 100)] {
                // The centre of window pixel (c, r), as a blob centre
                let (x, y) = ((c + COLUMN_START) as f64 + 0.5, (r + row_start) as f64 + 0.5);
                let (ux, uy) = match &lens {
                    Some(l) => {
                        let (ux, uy) = l.undistort(x - 0.5, y - 0.5);
                        (ux + 0.5, uy + 0.5)
                    }
                    None => (x, y),
                };
                let i = remap[uy.floor() as usize * SENSOR_WIDTH as usize + ux.floor() as usize];
                assert!(i < FRAME_WIDTH * FRAME_HEIGHT, "{:?}: ({}, {}) outside", lens, c, r);
                let (gc, gr) = ((i % FRAME_WIDTH) as i64, (i / FRAME_WIDTH) as i64);
                assert!((gc - c as i64).abs() <= 1 && (gr - r as i64).abs() <= 1,
                        "{:?}: ({}, {}) shows ({}, {})", lens, c, r, gc, gr);
            }
        }
    }
}
