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

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};

use common::bytes::Bytes;
use common::errors::*;
use executor_multitask::BroadcastChannel;
use image::ImageRef;
use mocap_camera_core::MJPEGEncoder;
use mocap_proto::mocap::MJPEGEncoderConfig;

use crate::config::{FRAME_HEIGHT, FRAME_WIDTH};

/// Frame copies in flight: one being encoded, one being filled.
const NUM_COPIES: usize = 2;

/// The preview settings the manager configures (mjpeg and pixel_threshold),
/// shared between the Camera service (writer), the capture loop (the
/// interval, one atomic load an offer) and the encoder thread.
pub struct PreviewSettings {
    /// 1 / max_fps; 0 while max_fps is 0 (no previews).
    interval_ns: AtomicU64,
    value: Mutex<(u8, MJPEGEncoderConfig)>,
}

impl PreviewSettings {
    pub fn new(threshold: u32, config: MJPEGEncoderConfig) -> Arc<Self> {
        let s = Arc::new(Self { interval_ns: AtomicU64::new(0), value: Mutex::new((0, MJPEGEncoderConfig::default())) });
        s.set(threshold, config);
        s
    }

    pub fn set(&self, threshold: u32, config: MJPEGEncoderConfig) {
        let fps = config.max_fps();
        self.interval_ns.store(if fps == 0 { 0 } else { 1_000_000_000 / fps as u64 }, Ordering::Relaxed);
        *self.value.lock().unwrap() = (threshold.clamp(1, 255) as u8, config);
    }
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
            to_capture.send(vec![0u8; (FRAME_WIDTH * FRAME_HEIGHT) as usize]).unwrap();
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
            while let Ok(copy) = from_capture.recv() {
                let (threshold, config) = s.value.lock().unwrap().clone();
                let image = ImageRef { width: FRAME_WIDTH as usize, height: FRAME_HEIGHT as usize, channels: 1, data: &copy };
                match encoder.encode(image, threshold, &config) {
                    Ok(data) => { out.send(data.into()); }
                    Err(e) => eprintln!("MJPEG preview: {}", e),
                }
                if to_capture.send(copy).is_err() {
                    break;
                }
            }
        })?;

        Ok(Self { frames, settings, to_encoder, free, next_ns: 0 })
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
            copy.copy_from_slice(frame);
            let _ = self.to_encoder.try_send(copy);
        }
    }
}
