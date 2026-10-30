//! Frame numbers on the beacon grid (first milestone step 4).
//!
//! A frame's uvcvideo hardware timestamp sits a fixed, per-camera offset from
//! its beacon frame boundary, and that offset can be several milliseconds,
//! so it can't be found from the timestamps alone (taking ts mod period would
//! put cameras whose offsets straddle the cut a whole frame apart). Each
//! stream learns it first: a frame is dequeued between DQ_LEAD_MIN and
//! DQ_LEAD_MIN + one period after its boundary (uvcvideo hands frames over at
//! a median +13.3 ms, never seen past +18.9 ms), which names the boundary
//! without ambiguity, and the median of hwts - boundary over LEARN_FRAMES
//! frames is the offset. From then on each frame goes in the slot nearest
//! hwts - offset on rift_grid's grid, and a frame too far from it is dropped.
//!
//! Per frame this is a seqlock read of the grid and a few float operations.

use rift_camera::grid::{Grid, GridMap};

/// A frame is dequeued at least this long after its boundary.
const DQ_LEAD_MIN_NS: i64 = 9_470_000;
const LEARN_FRAMES: usize = 200;
/// The learned offsets must agree to within this (10th to 90th percentile).
const LEARN_SPREAD_MAX_NS: i64 = 1_000_000;
/// A frame further than this from its slot's boundary is dropped. The
/// timestamps scatter about 45 us on average with a tail past 450 us; slots
/// are +-5.6 ms wide, so 2 ms keeps them unambiguous.
const SLOT_TOLERANCE_NS: f64 = 2_000_000.0;
/// This many dropped frames in a row means the offset no longer holds: relearn.
const RELEARN_AFTER: u32 = 90;
/// No boundary for this long and the grid is stale.
const GRID_STALE_NS: i64 = 500_000_000;

pub enum Slot {
    /// The frame's frame_timestamp.
    Publish(u64),
    Learning,
    NoGrid,
    Rejected,
}

pub struct Slotter {
    serial: String,
    map: Option<GridMap>,
    open_retry: u32,
    offset_ns: Option<i64>,
    samples: Vec<i64>,
    leads: Vec<i64>,
    last_ts: u64,
    rejects_in_row: u32,
    last_generation: u64,

    // For the report; cleared each time it is printed.
    published: u64,
    learning: u64,
    no_grid: u64,
    rejected: u64,
    resid_sum_ns: f64,
    resid_abs_sum_ns: f64,
    resid_abs_max_ns: f64,
}

#[inline]
pub fn monotonic_ns() -> i64 {
    let mut t = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut t) };
    t.tv_sec as i64 * 1_000_000_000 + t.tv_nsec as i64
}

fn percentile(sorted: &[i64], p: f64) -> i64 {
    sorted[((sorted.len() - 1) as f64 * p).round() as usize]
}

impl Slotter {
    pub fn new(serial: &str) -> Self {
        Self {
            serial: serial.to_string(),
            map: None,
            open_retry: 0,
            offset_ns: None,
            samples: Vec::with_capacity(LEARN_FRAMES),
            leads: Vec::with_capacity(LEARN_FRAMES),
            last_ts: 0,
            rejects_in_row: 0,
            last_generation: 0,
            published: 0, learning: 0, no_grid: 0, rejected: 0,
            resid_sum_ns: 0.0, resid_abs_sum_ns: 0.0, resid_abs_max_ns: 0.0,
        }
    }

    /// A new stream: the camera's offset may have changed, so learn it again.
    pub fn restart(&mut self) {
        self.offset_ns = None;
        self.samples.clear();
        self.leads.clear();
        self.rejects_in_row = 0;
    }

    fn grid(&mut self, dq_ns: i64) -> Option<Grid> {
        if self.map.is_none() {
            // Opening is a few system calls: try about once a second.
            if self.open_retry > 0 {
                self.open_retry -= 1;
                return None;
            }
            match GridMap::open() {
                Ok(m) => self.map = Some(m),
                Err(_) => {
                    self.open_retry = 90;
                    return None;
                }
            }
        }
        let grid = self.map.as_ref().unwrap().read()?;
        if dq_ns - grid.last_edge_mono_ns > GRID_STALE_NS {
            return None;
        }
        Some(grid)
    }

    /// Places one frame: hwts_ns its uvcvideo hardware timestamp, dq_ns when it
    /// was dequeued (both CLOCK_MONOTONIC).
    #[inline]
    pub fn place(&mut self, hwts_ns: i64, dq_ns: i64) -> Slot {
        let grid = match self.grid(dq_ns) {
            Some(g) => g,
            None => {
                self.no_grid += 1;
                return Slot::NoGrid;
            }
        };
        if grid.generation != self.last_generation {
            if self.last_generation != 0 {
                println!("{}: beacon grid generation {}", self.serial, grid.generation);
            }
            self.last_generation = grid.generation;
        }

        let offset = match self.offset_ns {
            Some(o) => o,
            None => {
                self.learn(&grid, hwts_ns, dq_ns);
                self.learning += 1;
                return Slot::Learning;
            }
        };

        let (k, resid) = grid.slot(hwts_ns - offset);
        if resid.abs() > SLOT_TOLERANCE_NS {
            self.rejected += 1;
            self.rejects_in_row += 1;
            if self.rejects_in_row >= RELEARN_AFTER {
                println!("{}: {} frames in a row off their slots (last {:.0} us); learning the offset again",
                         self.serial, self.rejects_in_row, resid / 1000.0);
                self.restart();
            }
            return Slot::Rejected;
        }
        self.rejects_in_row = 0;

        let ts = grid.frame_timestamp(k);
        if ts <= self.last_ts {
            self.rejected += 1;
            return Slot::Rejected;
        }
        self.last_ts = ts;

        self.published += 1;
        let r = resid.abs();
        self.resid_sum_ns += resid;
        self.resid_abs_sum_ns += r;
        if r > self.resid_abs_max_ns {
            self.resid_abs_max_ns = r;
        }
        Slot::Publish(ts)
    }

    fn learn(&mut self, grid: &Grid, hwts_ns: i64, dq_ns: i64) {
        // The boundary in (dq - DQ_LEAD_MIN - period, dq - DQ_LEAD_MIN].
        let x = (dq_ns - DQ_LEAD_MIN_NS - grid.anchor_mono_ns) as f64 / grid.period_ns;
        let boundary = grid.boundary(x.floor() as i64);
        self.samples.push(hwts_ns - boundary);
        self.leads.push(dq_ns - boundary);
        if self.samples.len() < LEARN_FRAMES {
            return;
        }

        let mut s = std::mem::take(&mut self.samples);
        let mut l = std::mem::take(&mut self.leads);
        s.sort_unstable();
        l.sort_unstable();
        let (p10, p50, p90) = (percentile(&s, 0.1), percentile(&s, 0.5), percentile(&s, 0.9));
        println!(
            "{}: hwts - boundary over {} frames: median {:.1} us, 10-90% {:.1} to {:.1} us, min {:.1}, max {:.1}; \
             dequeued +{:.2} ms median, +{:.2} to +{:.2} ms",
            self.serial, s.len(), p50 as f64 / 1e3, p10 as f64 / 1e3, p90 as f64 / 1e3,
            s[0] as f64 / 1e3, s[s.len() - 1] as f64 / 1e3,
            percentile(&l, 0.5) as f64 / 1e6, l[0] as f64 / 1e6, l[l.len() - 1] as f64 / 1e6);
        if p90 - p10 > LEARN_SPREAD_MAX_NS {
            println!("{}: offsets spread over {:.0} us; learning again", self.serial, (p90 - p10) as f64 / 1e3);
        } else {
            self.offset_ns = Some(p50);
        }
        s.clear();
        l.clear();
        self.samples = s;
        self.leads = l;
    }

    /// One line for the 10 s report, then the counts start again.
    pub fn report(&mut self) -> String {
        let line = format!(
            "slots: published {}, learning {}, no grid {}, rejected {}, offset {}, error mean {:+.1} us, |error| mean {:.1} us, max {:.1} us",
            self.published, self.learning, self.no_grid, self.rejected,
            self.offset_ns.map(|o| format!("{:.1} us", o as f64 / 1e3)).unwrap_or("-".into()),
            if self.published > 0 { self.resid_sum_ns / self.published as f64 / 1e3 } else { 0.0 },
            if self.published > 0 { self.resid_abs_sum_ns / self.published as f64 / 1e3 } else { 0.0 },
            self.resid_abs_max_ns / 1e3);
        self.published = 0;
        self.learning = 0;
        self.no_grid = 0;
        self.rejected = 0;
        self.resid_sum_ns = 0.0;
        self.resid_abs_sum_ns = 0.0;
        self.resid_abs_max_ns = 0.0;
        line
    }
}
