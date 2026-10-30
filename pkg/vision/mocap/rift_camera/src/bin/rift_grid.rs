//! rift_grid: the beacon grid daemon, one per Pi.
//!
//! The receiver XIAO pulses its D10 pin 20 us wide, centred on every beacon
//! frame boundary (`led on`, `width 20`, `off 0` on its serial port), and D10
//! is wired to a GPIO on the Pi. rift_grid stamps the rising edges on
//! CLOCK_MONOTONIC through the GPIO character device, fits a line through the
//! last ~10 s of boundaries (period and phase on the Pi's clock) and
//! publishes it to /dev/shm/rift_grid (rift_camera::grid) for the camera
//! nodes, which number their frames on it.
//!
//!   rift_grid [--chip /dev/gpiochip0] [--line 17] [--serial /dev/serial/by-id/...]
//!
//! The serial port defaults to the one whose name has `SlimeNRF_Receiver`;
//! rift_grid only opens it to (re)send the flash commands when edges stop,
//! so other tools can use it the rest of the time.
//!
//! A grid generation lasts while the boundaries stay on one line. A gap of
//! more than a second, or a run of edges off the line that agree with each
//! other (the beacon's clock wrap moves the grid ~10.2 ms every 71.6 min; a
//! beacon restart moves it anywhere), starts a new one, published after a
//! second of clean edges. frame_timestamp only ever increases: within a
//! generation each re-anchoring carries the previous mapping on, and a new
//! generation (or a restart of rift_grid that finds a grid in /dev/shm)
//! starts strictly after anything published before.

#[macro_use]
extern crate common;

use std::collections::VecDeque;
use std::ffi::CString;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::time::Duration;

use common::errors::*;
use rift_camera::grid::{Grid, GridMap, FRAME_TIMESTAMP_STEP_NS};

/// The beacon's nominal frame period (ns).
const NOMINAL_NS: f64 = 11_111_000.0;
/// The flash rises this long before the boundary (half its 20 us width).
const FLASH_LEAD_NS: i64 = 10_000;
const FLASH_COMMANDS: [&str; 3] = ["led on", "width 20", "off 0"];

/// Boundaries kept for the fit.
const WINDOW_NS: i64 = 10_000_000_000;
/// An edge further than this off the fitted line is an outlier.
const OUTLIER_NS: f64 = 50_000.0;
/// While warming up, an interval must be this close to whole periods.
const WHOLE_TOL_NS: f64 = 100_000.0;
/// This many consecutive outliers that agree with each other mean the grid
/// moved.
const SHIFT_RUN: usize = 5;
/// No edge for this long ends the generation.
const GAP_NS: i64 = 1_000_000_000;
/// Clean edges needed before a generation is published.
const WARMUP_NS: i64 = 1_000_000_000;
const WARMUP_EDGES: usize = 60;

const PUBLISH_NS: i64 = 100_000_000;
const LOG_NS: i64 = 10_000_000_000;
/// Without edges for this long, send the flash commands...
const QUIET_NS: i64 = 1_000_000_000;
/// ...and again this often while they stay missing.
const FLASH_RETRY_NS: i64 = 5_000_000_000;

// ---------- GPIO character device, uAPI v2 (<linux/gpio.h>) ----------

#[repr(C)]
#[derive(Clone, Copy)]
struct GpioV2LineAttribute {
    id: u32,
    padding: u32,
    value: u64, // union of flags / values / debounce_period_us
}

#[repr(C)]
#[derive(Clone, Copy)]
struct GpioV2LineConfigAttribute {
    attr: GpioV2LineAttribute,
    mask: u64,
}

#[repr(C)]
struct GpioV2LineConfig {
    flags: u64,
    num_attrs: u32,
    padding: [u32; 5],
    attrs: [GpioV2LineConfigAttribute; 10],
}

#[repr(C)]
struct GpioV2LineRequest {
    offsets: [u32; 64],
    consumer: [u8; 32],
    config: GpioV2LineConfig,
    num_lines: u32,
    event_buffer_size: u32,
    padding: [u32; 5],
    fd: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct GpioV2LineEvent {
    timestamp_ns: u64,
    id: u32,
    offset: u32,
    seqno: u32,
    line_seqno: u32,
    padding: [u32; 6],
}

const _: () = assert!(std::mem::size_of::<GpioV2LineRequest>() == 592);
const _: () = assert!(std::mem::size_of::<GpioV2LineEvent>() == 48);

const GPIO_V2_LINE_FLAG_INPUT: u64 = 1 << 2;
const GPIO_V2_LINE_FLAG_EDGE_RISING: u64 = 1 << 4;
const GPIO_V2_LINE_EVENT_RISING_EDGE: u32 = 1;
/// _IOWR(0xB4, 0x07, struct gpio_v2_line_request)
const GPIO_V2_GET_LINE_IOCTL: u64 =
    (3 << 30) | ((std::mem::size_of::<GpioV2LineRequest>() as u64) << 16) | (0xB4 << 8) | 0x07;

/// Requests one line as an input with rising-edge events, stamped on
/// CLOCK_MONOTONIC (the default event clock).
fn request_line(chip: &str, line: u32) -> Result<OwnedFd> {
    let path = CString::new(chip)?;
    let chip_fd = unsafe { libc::open(path.as_ptr(), libc::O_RDWR | libc::O_CLOEXEC) };
    if chip_fd < 0 {
        return Err(format_err!("{}: {}", chip, std::io::Error::last_os_error()));
    }
    let chip_fd = unsafe { OwnedFd::from_raw_fd(chip_fd) };

    let mut req: GpioV2LineRequest = unsafe { std::mem::zeroed() };
    req.offsets[0] = line;
    let consumer = b"rift_grid";
    req.consumer[..consumer.len()].copy_from_slice(consumer);
    req.config.flags = GPIO_V2_LINE_FLAG_INPUT | GPIO_V2_LINE_FLAG_EDGE_RISING;
    req.num_lines = 1;
    req.event_buffer_size = 256;
    let r = unsafe { libc::ioctl(chip_fd.as_raw_fd(), GPIO_V2_GET_LINE_IOCTL as _, &mut req) };
    if r < 0 {
        return Err(format_err!("{} line {}: {}", chip, line, std::io::Error::last_os_error()));
    }
    Ok(unsafe { OwnedFd::from_raw_fd(req.fd) })
}

/// Waits up to timeout_ms for edges; appends their kernel timestamps (ns).
fn read_edges(fd: &OwnedFd, timeout_ms: i32, out: &mut Vec<i64>) -> Result<()> {
    let mut pfd = libc::pollfd { fd: fd.as_raw_fd(), events: libc::POLLIN, revents: 0 };
    let r = unsafe { libc::poll(&mut pfd, 1, timeout_ms) };
    if r < 0 {
        let e = std::io::Error::last_os_error();
        if e.kind() == std::io::ErrorKind::Interrupted {
            return Ok(());
        }
        return Err(format_err!("poll: {}", e));
    }
    if r == 0 {
        return Ok(());
    }
    let mut buf = [GpioV2LineEvent { timestamp_ns: 0, id: 0, offset: 0, seqno: 0, line_seqno: 0, padding: [0; 6] }; 32];
    let n = unsafe {
        libc::read(fd.as_raw_fd(), buf.as_mut_ptr() as *mut libc::c_void, std::mem::size_of_val(&buf))
    };
    if n < 0 {
        let e = std::io::Error::last_os_error();
        if e.kind() == std::io::ErrorKind::Interrupted {
            return Ok(());
        }
        return Err(format_err!("read line events: {}", e));
    }
    let count = n as usize / std::mem::size_of::<GpioV2LineEvent>();
    for ev in &buf[..count] {
        if ev.id == GPIO_V2_LINE_EVENT_RISING_EDGE {
            out.push(ev.timestamp_ns as i64);
        }
    }
    Ok(())
}

// ---------- clocks ----------

fn clock_ns(clock: libc::clockid_t) -> i64 {
    let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    unsafe { libc::clock_gettime(clock, &mut ts) };
    ts.tv_sec as i64 * 1_000_000_000 + ts.tv_nsec as i64
}

fn mono_ns() -> i64 {
    clock_ns(libc::CLOCK_MONOTONIC)
}

// ---------- the receiver's serial port ----------

fn find_serial() -> Option<String> {
    let dir = std::fs::read_dir("/dev/serial/by-id").ok()?;
    let mut names: Vec<String> = dir
        .filter_map(|e| e.ok())
        .map(|e| e.path().to_string_lossy().into_owned())
        .filter(|p| p.contains("SlimeNRF_Receiver"))
        .collect();
    names.sort();
    names.into_iter().next()
}

/// Opens the port raw, sends the flash commands and closes it again. Returns
/// how many status lines (`PERIOD ...`) came back.
fn send_flash(path: &str) -> Result<usize> {
    let cpath = CString::new(path)?;
    let fd = unsafe {
        libc::open(cpath.as_ptr(), libc::O_RDWR | libc::O_NOCTTY | libc::O_NONBLOCK | libc::O_CLOEXEC)
    };
    if fd < 0 {
        return Err(format_err!("{}: {}", path, std::io::Error::last_os_error()));
    }
    let fd = unsafe { OwnedFd::from_raw_fd(fd) };
    let raw = fd.as_raw_fd();
    unsafe {
        let mut t: libc::termios = std::mem::zeroed();
        if libc::tcgetattr(raw, &mut t) == 0 {
            libc::cfmakeraw(&mut t);
            libc::cfsetispeed(&mut t, libc::B115200);
            libc::cfsetospeed(&mut t, libc::B115200);
            libc::tcsetattr(raw, libc::TCSANOW, &t);
        }
        libc::tcflush(raw, libc::TCIFLUSH);
    }
    for cmd in FLASH_COMMANDS {
        let line = format!("{}\n", cmd);
        let n = unsafe { libc::write(raw, line.as_ptr() as *const libc::c_void, line.len()) };
        if n != line.len() as isize {
            return Err(format_err!("write {}: {}", path, std::io::Error::last_os_error()));
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    // Drain what comes back for a moment and count the status lines.
    let mut text = Vec::new();
    let end = mono_ns() + 400_000_000;
    while mono_ns() < end {
        let mut buf = [0u8; 1024];
        let n = unsafe { libc::read(raw, buf.as_mut_ptr() as *mut libc::c_void, buf.len()) };
        if n > 0 {
            text.extend_from_slice(&buf[..n as usize]);
        } else {
            std::thread::sleep(Duration::from_millis(20));
        }
    }
    let text = String::from_utf8_lossy(&text);
    Ok(text.lines().filter(|l| l.starts_with("PERIOD ")).count())
}

// ---------- the fit ----------

/// boundary(i) = t0 + offset + (i - i0) * period, on CLOCK_MONOTONIC.
#[derive(Clone, Copy)]
struct Fit {
    t0: i64,
    i0: i64,
    offset: f64,
    period: f64,
    rms: f64,
}

impl Fit {
    fn time(&self, i: i64) -> f64 {
        self.t0 as f64 + self.offset + (i - self.i0) as f64 * self.period
    }

    /// The boundary index nearest t and t's residual from it.
    fn nearest(&self, t: i64) -> (i64, f64) {
        let x = ((t - self.t0) as f64 - self.offset) / self.period;
        let i = self.i0 + x.round() as i64;
        (i, (t - self.t0) as f64 - self.offset - (i - self.i0) as f64 * self.period)
    }
}

/// Least squares of boundary time against index, relative to the newest point.
fn fit_line(points: &VecDeque<(i64, i64)>) -> Option<Fit> {
    if points.len() < 2 {
        return None;
    }
    let &(i0, t0) = points.back()?;
    let n = points.len() as f64;
    let (mut sx, mut sy) = (0.0, 0.0);
    for &(i, t) in points {
        sx += (i - i0) as f64;
        sy += (t - t0) as f64;
    }
    let (mx, my) = (sx / n, sy / n);
    let (mut sxx, mut sxy) = (0.0, 0.0);
    for &(i, t) in points {
        let dx = (i - i0) as f64 - mx;
        sxx += dx * dx;
        sxy += dx * ((t - t0) as f64 - my);
    }
    if sxx <= 0.0 {
        return None;
    }
    let period = sxy / sxx;
    let offset = my - period * mx;
    let mut fit = Fit { t0, i0, offset, period, rms: 0.0 };
    let mut ss = 0.0;
    for &(i, t) in points {
        let r = t as f64 - fit.time(i);
        ss += r * r;
    }
    fit.rms = (ss / n).sqrt();
    Some(fit)
}

// ---------- the tracker ----------

struct Tracker {
    map: GridMap,
    /// The last grid written (or the one a previous rift_grid left).
    published: Option<Grid>,
    generation: u64,
    gen_published: bool,
    window: VecDeque<(i64, i64)>, // (index, boundary ns)
    last: Option<(i64, i64)>,     // newest accepted boundary
    fit: Option<Fit>,
    /// Consecutive outliers while fitted, (boundary, residual).
    suspects: Vec<(i64, f64)>,
    last_raw: i64,
    lost_logged: bool,
    last_published_index: Option<i64>,
    // Counters since the last log line.
    edges: u64,
    missed: u64,
    outliers: u64,
    new_gens: u64,
}

impl Tracker {
    fn new(map: GridMap) -> Self {
        let published = map.last();
        let generation = published.map(|g| g.generation + 1).unwrap_or(1);
        if let Some(g) = &published {
            println!("Previous grid: generation {}, anchor ts {} ns; carrying frame_timestamp on as generation {}",
                     g.generation, g.anchor_ts_ns, generation);
        } else {
            println!("No previous grid: generation 1 starts from the real-time clock");
        }
        Self {
            map, published, generation, gen_published: false, window: VecDeque::new(), last: None,
            fit: None, suspects: Vec::new(), last_raw: mono_ns(), lost_logged: false,
            last_published_index: None, edges: 0, missed: 0, outliers: 0, new_gens: 0,
        }
    }

    /// Ends the generation (or restarts the warm-up if it was never published).
    fn new_generation(&mut self, reason: &str) {
        if self.gen_published {
            self.generation += 1;
            self.gen_published = false;
            self.new_gens += 1;
            println!("Generation {}: {}; waiting for {:.0} s of clean edges", self.generation, reason,
                     WARMUP_NS as f64 / 1e9);
        } else {
            println!("Generation {} warm-up restarted: {}", self.generation, reason);
        }
        self.window.clear();
        self.last = None;
        self.fit = None;
        self.suspects.clear();
        self.last_published_index = None;
    }

    fn accept(&mut self, i: i64, b: i64) {
        if let Some((li, _)) = self.last {
            self.missed += (i - li - 1).max(0) as u64;
        }
        self.window.push_back((i, b));
        self.last = Some((i, b));
    }

    /// A boundary while warming up: indexed by whole periods from the last.
    fn warm(&mut self, b: i64) {
        let Some((li, lt)) = self.last else {
            self.accept(0, b);
            return;
        };
        let dt = (b - lt) as f64;
        let n = (dt / NOMINAL_NS).round();
        if n < 1.0 {
            self.outliers += 1; // a second edge within half a period
            return;
        }
        let off = dt - n * NOMINAL_NS;
        if off.abs() > WHOLE_TOL_NS {
            self.new_generation(&format!("interval {:.3} ms is {:+.1} us off whole periods", dt / 1e6, off / 1e3));
            self.accept(0, b);
            return;
        }
        self.accept(li + n as i64, b);
    }

    fn edge(&mut self, b: i64) {
        self.edges += 1;
        self.last_raw = b;
        self.lost_logged = false;
        if let Some((_, lt)) = self.last {
            if b - lt > GAP_NS {
                self.new_generation(&format!("edges stopped for {:.2} s", (b - lt) as f64 / 1e9));
                self.accept(0, b);
                return;
            }
        }
        let Some(fit) = self.fit else {
            self.warm(b);
            return;
        };
        let (i, r) = fit.nearest(b);
        let li = self.last.map(|l| l.0).unwrap_or(i64::MIN);
        if r.abs() <= OUTLIER_NS && i > li {
            self.accept(i, b);
            // Slots skipped for outliers had an edge: they weren't missed.
            self.missed = self.missed.saturating_sub(self.suspects.len() as u64);
            self.suspects.clear();
            return;
        }
        self.outliers += 1;
        self.suspects.push((b, r));
        if self.suspects.len() >= SHIFT_RUN {
            // Do the outliers sit on a grid of their own?
            let agree = self.suspects.windows(2).all(|w| {
                let dt = (w[1].0 - w[0].0) as f64;
                let n = (dt / NOMINAL_NS).round();
                n >= 1.0 && (dt - n * NOMINAL_NS).abs() <= WHOLE_TOL_NS
            });
            if agree {
                let run = std::mem::take(&mut self.suspects);
                self.outliers -= run.len() as u64;
                let shift = run.iter().map(|s| s.1).sum::<f64>() / run.len() as f64;
                self.new_generation(&format!("grid moved {:+.1} us", shift / 1e3));
                for (t, _) in run {
                    self.warm(t);
                }
            } else {
                self.suspects.remove(0);
            }
        }
    }

    /// Fits the window, rejecting points off the line.
    fn refit(&mut self) {
        let newest = match self.window.back() {
            Some(&(_, t)) => t,
            None => return,
        };
        while self.window.front().map_or(false, |&(_, t)| newest - t > WINDOW_NS) {
            self.window.pop_front();
        }
        if self.fit.is_none() {
            let span = newest - self.window.front().map_or(newest, |p| p.1);
            if span < WARMUP_NS || self.window.len() < WARMUP_EDGES {
                return;
            }
        }
        for _ in 0..3 {
            let Some(fit) = fit_line(&self.window) else { return };
            let before = self.window.len();
            self.window.retain(|&(i, t)| (t as f64 - fit.time(i)).abs() <= OUTLIER_NS);
            let dropped = before - self.window.len();
            self.outliers += dropped as u64;
            if dropped == 0 || self.window.len() < WARMUP_EDGES {
                self.fit = Some(fit);
                return;
            }
        }
        self.fit = fit_line(&self.window);
    }

    fn publish(&mut self) {
        self.refit();
        let (Some(fit), Some((i_new, b_new))) = (self.fit, self.last) else { return };
        if self.last_published_index == Some(i_new) {
            return; // nothing new
        }
        let anchor_mono = fit.time(i_new).round() as i64;
        let step = FRAME_TIMESTAMP_STEP_NS as i64;
        let anchor_ts = match (&self.published, self.gen_published) {
            // Re-anchoring: the previous mapping's number for this boundary.
            (Some(p), true) => {
                let k = ((anchor_mono - p.anchor_mono_ns) as f64 / p.period_ns).round() as i64;
                (p.anchor_ts_ns as i64 + k * step) as u64
            }
            // A new generation: strictly after anything published before.
            (Some(p), false) => {
                let d = (anchor_mono - p.anchor_mono_ns).max(0);
                let c = (d + step - 1) / step;
                (p.anchor_ts_ns as i64 + c * step + step) as u64
            }
            (None, _) => {
                let now = clock_ns(libc::CLOCK_REALTIME) as u64;
                now / FRAME_TIMESTAMP_STEP_NS * FRAME_TIMESTAMP_STEP_NS
            }
        };
        let grid = Grid {
            generation: self.generation,
            anchor_mono_ns: anchor_mono,
            anchor_ts_ns: anchor_ts,
            period_ns: fit.period,
            last_edge_mono_ns: b_new,
            fit_rms_ns: fit.rms,
        };
        self.map.write(&grid);
        if !self.gen_published {
            println!("Generation {} published: period {:.1} ns ({:+.2} ppm), rms {:.2} us, anchor ts {} ns, \
                      {} boundaries", self.generation, fit.period, ppm(fit.period), fit.rms / 1e3, anchor_ts,
                     self.window.len());
        }
        self.published = Some(grid);
        self.gen_published = true;
        self.last_published_index = Some(i_new);
    }

    fn log(&mut self, seconds: f64) {
        let fit = match self.fit {
            Some(f) => format!("period {:.1} ns ({:+.2} ppm), rms {:.2} us", f.period, ppm(f.period), f.rms / 1e3),
            None => "not fitted".to_string(),
        };
        println!("edges {} ({:.1}/s), missed {}, outliers {}, {}, generation {}{}{}",
                 self.edges, self.edges as f64 / seconds, self.missed, self.outliers, fit, self.generation,
                 if self.gen_published { "" } else { " (warming up)" },
                 if self.new_gens > 0 { format!(", {} new generations", self.new_gens) } else { String::new() });
        self.edges = 0;
        self.missed = 0;
        self.outliers = 0;
        self.new_gens = 0;
    }
}

fn ppm(period: f64) -> f64 {
    (period - NOMINAL_NS) / NOMINAL_NS * 1e6
}

// ---------- main ----------

struct Args {
    chip: String,
    line: u32,
    serial: Option<String>,
}

fn parse_args() -> Result<Args> {
    let mut args = Args { chip: "/dev/gpiochip0".into(), line: 17, serial: None };
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        let (key, value) = match a.split_once('=') {
            Some((k, v)) => (k.to_string(), v.to_string()),
            None => {
                let v = it.next().ok_or_else(|| format_err!("{} needs a value", a))?;
                (a.clone(), v)
            }
        };
        match key.as_str() {
            "--chip" => args.chip = value,
            "--line" => args.line = value.parse().map_err(|_| format_err!("--line {}: not a number", value))?,
            "--serial" => args.serial = Some(value),
            _ => return Err(format_err!("unknown option {} (--chip, --line, --serial)", key)),
        }
    }
    Ok(args)
}

fn run() -> Result<()> {
    let args = parse_args()?;
    let line_fd = request_line(&args.chip, args.line)?;
    let map = GridMap::create()?;
    println!("rift_grid: {} line {} (boundary = rising edge + {} us), publishing to {}",
             args.chip, args.line, FLASH_LEAD_NS / 1000, rift_camera::grid::GRID_PATH);

    let mut tracker = Tracker::new(map);
    let start = mono_ns();
    let (mut last_publish, mut last_log, mut last_flash) = (start, start, i64::MIN / 2);
    let mut edges = Vec::new();
    loop {
        edges.clear();
        read_edges(&line_fd, 50, &mut edges)?;
        for &t in &edges {
            tracker.edge(t + FLASH_LEAD_NS);
        }
        let now = mono_ns();

        let quiet = now - tracker.last_raw;
        if quiet > GAP_NS && tracker.last.is_some() && !tracker.lost_logged {
            println!("No edges for {:.1} s", quiet as f64 / 1e9);
            tracker.lost_logged = true;
        }
        if quiet > QUIET_NS && now - last_flash > FLASH_RETRY_NS {
            last_flash = now;
            match args.serial.clone().or_else(find_serial) {
                None => println!("No edges, and no SlimeNRF_Receiver serial port to send the flash commands to"),
                Some(port) => match send_flash(&port) {
                    Ok(n) => println!("No edges for {:.1} s: sent '{}' to {} ({} status lines back)",
                                      quiet as f64 / 1e9, FLASH_COMMANDS.join("', '"), port, n),
                    Err(e) => println!("No edges for {:.1} s: couldn't send the flash commands: {}",
                                       quiet as f64 / 1e9, e),
                },
            }
        }

        if now - last_publish >= PUBLISH_NS {
            last_publish = now;
            tracker.publish();
        }
        if now - last_log >= LOG_NS {
            tracker.log((now - last_log) as f64 / 1e9);
            last_log = now;
        }
    }
}

fn main() {
    if let Err(e) = run() {
        eprintln!("rift_grid: {}", e);
        std::process::exit(1);
    }
}
