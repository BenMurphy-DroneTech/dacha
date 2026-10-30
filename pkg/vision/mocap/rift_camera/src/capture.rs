//! The capture loop: one Rift sensor through uvcvideo, synced to the beacon.
//!
//! Open the camera by serial, set 960x576 GREY (the bridge's 90 fps crop),
//! STREAMON, run the sync setup, then dequeue frames for as long as the camera
//! stays plugged in; when it goes away, wait for it and start again.
//!
//! The frame loop is the hot path, so it never copies, allocates or prints:
//! each frame is read in place from its mmap'd buffer and the buffer goes back
//! to the driver straight after. Counts are plain integers, reported every
//! REPORT_INTERVAL.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use common::errors::*;
use executor_multitask::BroadcastChannel;
use mocap_proto::mocap::ReadBlobsResponse;

use crate::bridge::Bridge;
use crate::config::*;
use crate::slots::{monotonic_ns, Slot, Slotter};

/// Buffers queued in the driver: 4 frames (44 ms at 90 fps) of slack.
const NUM_BUFFERS: usize = 4;
/// A frame later than this means the stream has stalled.
const FRAME_TIMEOUT: Duration = Duration::from_secs(2);
/// Frames ignored after the sync setup while the sensor moves onto the beacon grid.
const SETTLE_FRAMES: u32 = 10;
const REPORT_INTERVAL: Duration = Duration::from_secs(10);

#[derive(Clone, Debug)]
pub struct CaptureOptions {
    pub period_us: u32,
    pub exposure_us: f64,
    pub gain: u32,
}

/// Totals since the process started, readable from other tasks.
#[derive(Default)]
pub struct CaptureCounters {
    pub frames: AtomicU64,
    pub missed: AtomicU64,
    pub short_frames: AtomicU64,
    pub error_frames: AtomicU64,
    pub restarts: AtomicU64,
}

/// Counts for one stretch of streaming. Per-frame work is a few compares.
struct FrameStats {
    period_ns: u64,
    frames: u64,
    missed: u64,
    seq_gaps: u64,
    short_frames: u64,
    error_frames: u64,
    no_timestamp: u64,
    max_interval_ns: u64,
    last_ts_ns: Option<u64>,
    last_seq: Option<u32>,
}

impl FrameStats {
    fn new(period_ns: u64) -> Self {
        Self {
            period_ns, frames: 0, missed: 0, seq_gaps: 0, short_frames: 0, error_frames: 0, no_timestamp: 0,
            max_interval_ns: 0, last_ts_ns: None, last_seq: None,
        }
    }

    /// Returns the frames this one shows were missed before it.
    #[inline]
    fn record(&mut self, seq: u32, ts_ns: Option<u64>, full: bool) -> u64 {
        self.frames += 1;
        if !full {
            self.short_frames += 1;
        }
        if let Some(last) = self.last_seq {
            if seq != last.wrapping_add(1) {
                self.seq_gaps += 1;
            }
        }
        self.last_seq = Some(seq);

        let mut missed = 0;
        match ts_ns {
            Some(ts) => {
                if let Some(last) = self.last_ts_ns {
                    let dt = ts.saturating_sub(last);
                    self.max_interval_ns = self.max_interval_ns.max(dt);
                    // Gaps over 1.5 periods, as rift_lib.dropped(); count the frames they hide.
                    if dt * 2 > self.period_ns * 3 {
                        missed = (dt + self.period_ns / 2) / self.period_ns - 1;
                        self.missed += missed;
                    }
                }
                self.last_ts_ns = Some(ts);
            }
            None => self.no_timestamp += 1,
        }
        missed
    }

    fn report(&self, serial: &str, elapsed: Duration) -> String {
        format!(
            "{}: {} frames in {:.1} s ({:.2} fps), missed {}, sequence gaps {}, short {}, errors {}, longest interval {:.2} ms{}",
            serial, self.frames, elapsed.as_secs_f64(),
            self.frames as f64 / elapsed.as_secs_f64().max(1e-9),
            self.missed, self.seq_gaps, self.short_frames, self.error_frames,
            self.max_interval_ns as f64 / 1e6,
            if self.no_timestamp > 0 { format!(", {} without timestamps", self.no_timestamp) } else { String::new() })
    }
}

/// Runs a blocking closure on its own thread, so the executor (and the RPC
/// server on it) keeps running.
async fn run_blocking<T: Send + 'static, F: FnOnce() -> T + Send + 'static>(f: F) -> Result<T> {
    let (sender, receiver) = executor::channel::oneshot::channel();
    std::thread::spawn(move || {
        let _ = sender.send(f());
    });
    receiver.recv().await.map_err(|_| err_msg("Blocking task ended without a result"))
}

fn hw_timestamps_on() -> bool {
    std::fs::read_to_string("/sys/module/uvcvideo/parameters/hwtimestamps")
        .map(|s| s.trim() == "1")
        .unwrap_or(false)
}

pub async fn run(config: RiftCameraConfig, options: CaptureOptions, counters: Arc<CaptureCounters>,
                 blobs: Arc<BroadcastChannel<ReadBlobsResponse>>) -> Result<()> {
    if !hw_timestamps_on() {
        eprintln!("warning: uvcvideo hwtimestamps is off; frame intervals will step by about 4 ms \
                   (echo 1 | sudo tee /sys/module/uvcvideo/parameters/hwtimestamps)");
    }

    let path = config.device_path();
    let mut slotter = Slotter::new(&config.serial);
    loop {
        if !std::path::Path::new(&path).exists() {
            println!("{}: waiting for {}", config.serial, path);
            while !std::path::Path::new(&path).exists() {
                executor::sleep(Duration::from_millis(500)).await?;
            }
            // Let uvcvideo finish binding the camera.
            executor::sleep(Duration::from_secs(1)).await?;
        }

        slotter.restart();
        match stream(&config, &options, &counters, &blobs, &mut slotter).await {
            Ok(()) => {}
            Err(e) => eprintln!("{}: {}", config.serial, e),
        }
        counters.restarts.fetch_add(1, Ordering::Relaxed);
        executor::sleep(Duration::from_secs(1)).await?;
    }
}

/// One stretch of streaming, until the camera goes away or stalls.
async fn stream(config: &RiftCameraConfig, options: &CaptureOptions, counters: &CaptureCounters,
                blobs: &BroadcastChannel<ReadBlobsResponse>, slotter: &mut Slotter) -> Result<()> {
    let path = config.device_path();
    let frame_bytes = (FRAME_WIDTH * FRAME_HEIGHT) as usize;

    let device = v4l2::Device::open(&path).await?;
    let mut unconfigured = device.new_capture_stream()?;
    let mut format = unconfigured.get_format().await?;
    format.set_width(FRAME_WIDTH);
    format.set_height(FRAME_HEIGHT);
    format.set_pixelformat(v4l2::V4L2_PIX_FMT_GREY);
    unconfigured.set_format(format).await?;
    let format = unconfigured.get_format().await?;
    if format.width() != FRAME_WIDTH || format.height() != FRAME_HEIGHT || format.pixelformat() != v4l2::V4L2_PIX_FMT_GREY {
        return Err(format_err!("The camera gave {}x{} {:?}, not {}x{} GREY", format.width(), format.height(),
                               v4l2::PixelFormat(format.pixelformat()), FRAME_WIDTH, FRAME_HEIGHT));
    }

    let (stream, buffers) = unconfigured.configure_mmap(NUM_BUFFERS).await?;
    for buffer in buffers {
        stream.enqueue_buffer(buffer).await?;
    }
    stream.turn_on().await?;

    // The sync setup talks to the bridge for about half a second; frames that
    // arrive meanwhile wait in the queued buffers or are dropped by uvcvideo.
    let setup = {
        let path = path.clone();
        let (exposure, gain, row_start) = (options.exposure_us, options.gain, config.row_start as u16);
        run_blocking(move || -> Result<_> {
            let mut bridge = Bridge::open(&path)?;
            bridge.sync_setup(exposure, gain, row_start, FRAME_HEIGHT as u16)
        }).await??
    };
    println!("{}: synced: exposure {} lines = {:.1} us at {:.2} MHz, radio exposure {} us, rows {}-{}, {} warnings",
             config.serial, setup.exposure_lines, setup.exposure_us, setup.pixel_clock_mhz,
             setup.radio_exposure_us, config.row_start, config.row_start + FRAME_HEIGHT - 1, setup.warnings);

    let period_ns = options.period_us as u64 * 1000;
    let mut settle = SETTLE_FRAMES;
    let mut stats = FrameStats::new(period_ns);
    let mut total = FrameStats::new(period_ns);
    let started = Instant::now();
    let mut window_start = started;
    let mut next_report = started + REPORT_INTERVAL;

    loop {
        let buffer = match executor::timeout(FRAME_TIMEOUT, stream.dequeue_buffer_with_errors()).await {
            Ok(Ok(b)) => b,
            Ok(Err(e)) => {
                if total.frames > 0 {
                    println!("stream total: {}", total.report(&config.serial, started.elapsed()));
                }
                return Err(format_err!("Stream ended: {}", e));
            }
            Err(_) => {
                if total.frames > 0 {
                    println!("stream total: {}", total.report(&config.serial, started.elapsed()));
                }
                // Not triggered, or lost on the way? Switch the sensor to
                // free-running and see whether frames come through.
                let free = {
                    let path = path.clone();
                    run_blocking(move || -> Result<_> {
                        let mut bridge = Bridge::open(&path)?;
                        bridge.open_i2c()?;
                        bridge.free_run()
                    }).await?
                };
                match free {
                    Ok(reset) => {
                        let mut n = 0;
                        let deadline = Instant::now() + Duration::from_secs(1);
                        while let Some(left) = deadline.checked_duration_since(Instant::now()) {
                            match executor::timeout(left, stream.dequeue_buffer_with_errors()).await {
                                Ok(Ok(b)) => { n += 1; stream.enqueue_buffer(b).await?; }
                                _ => break,
                            }
                        }
                        println!("{}: stalled: reset_register was 0x{:04x}; free-running gave {} frames in 1 s",
                                 config.serial, reset, n);
                    }
                    Err(e) => println!("{}: stalled: free-running check failed: {}", config.serial, e),
                }
                return Err(format_err!("No frame for {:?}", FRAME_TIMEOUT));
            }
        };

        // A frame uvcvideo flagged as corrupt goes straight back; its slot shows
        // up as a missed frame in the next interval.
        if buffer.flags() & v4l2::V4L2_BUF_FLAG_ERROR != 0 {
            stream.enqueue_buffer(buffer).await?;
            if settle == 0 {
                stats.error_frames += 1;
                total.error_frames += 1;
                counters.error_frames.fetch_add(1, Ordering::Relaxed);
            }
            continue;
        }

        let dq_ns = monotonic_ns();
        let seq = buffer.sequence();
        let ts_ns = buffer.monotonic_timestamp().map(|t| t.as_nanos() as u64);
        let full = buffer.used_memory().len() == frame_bytes;

        // Blob detection reads buffer.used_memory() in place here (step 5),
        // before the buffer goes back to the driver.

        stream.enqueue_buffer(buffer).await?;

        if settle > 0 {
            settle -= 1;
            if settle == 0 {
                window_start = Instant::now();
                next_report = window_start + REPORT_INTERVAL;
            }
            continue;
        }

        // The frame's number on the beacon grid, then straight out to the
        // manager (blobs land here in step 5).
        if let (Some(ts), true) = (ts_ns, full) {
            if let Slot::Publish(frame_timestamp) = slotter.place(ts as i64, dq_ns) {
                if blobs.active() {
                    let mut res = ReadBlobsResponse::default();
                    res.set_frame_timestamp(frame_timestamp);
                    res.new_cameras();
                    blobs.send(res);
                }
            }
        }

        let missed = stats.record(seq, ts_ns, full);
        total.record(seq, ts_ns, full);
        counters.frames.fetch_add(1, Ordering::Relaxed);
        if missed > 0 {
            counters.missed.fetch_add(missed, Ordering::Relaxed);
        }
        if !full {
            counters.short_frames.fetch_add(1, Ordering::Relaxed);
        }

        let now = Instant::now();
        if now >= next_report {
            println!("{}; {}", stats.report(&config.serial, now - window_start), slotter.report());
            stats = FrameStats { last_ts_ns: stats.last_ts_ns, last_seq: stats.last_seq, ..FrameStats::new(period_ns) };
            window_start = now;
            next_report = now + REPORT_INTERVAL;
        }
    }
}
