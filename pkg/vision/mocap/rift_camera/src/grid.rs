//! The beacon grid, shared between rift_grid (the one writer on a Pi) and the
//! rift_camera nodes (readers) through a small file in /dev/shm.
//!
//! rift_grid times the beacon's frame boundaries on CLOCK_MONOTONIC (from the
//! receiver's D10 flash on a GPIO) and publishes a line through them: an
//! anchor boundary, its frame_timestamp and the period on the Pi's clock. A
//! node maps a frame's corrected time t to
//!
//!   k = round((t - anchor_mono_ns) / period_ns)
//!   frame_timestamp = anchor_ts_ns + k * FRAME_TIMESTAMP_STEP_NS
//!
//! so every camera on the Pi gives the same beacon frame the same number.
//! frame_timestamp only ever increases, across re-anchoring, grid
//! generations (a beacon restart or clock wrap moves the grid) and rift_grid
//! restarts: dacha's manager drops anything not newer than what it has seen.
//!
//! Reading is a seqlock: no system call, lock or allocation, so it costs a
//! few loads per frame.

use std::fs::OpenOptions;
use std::os::fd::AsRawFd;
use std::sync::atomic::{fence, AtomicU64, Ordering};

use common::errors::*;

pub const GRID_PATH: &str = "/dev/shm/rift_grid";
const MAGIC: u64 = 0x5249_4654_4752_4431; // "RIFTGRD1"

/// frame_timestamp advances by the nominal beacon period per frame (ns).
pub const FRAME_TIMESTAMP_STEP_NS: u64 = 11_111_000;

/// One published grid. All times are CLOCK_MONOTONIC ns unless named _ts.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct Grid {
    /// Bumped whenever the grid jumps (beacon restart, clock wrap, lost edges).
    pub generation: u64,
    /// A frame boundary (the middle of the beacon's exposure window).
    pub anchor_mono_ns: i64,
    /// frame_timestamp of that boundary.
    pub anchor_ts_ns: u64,
    /// The beacon period measured on the Pi's clock.
    pub period_ns: f64,
    /// The latest boundary seen; a reader treats the grid as stale past
    /// about a second without one.
    pub last_edge_mono_ns: i64,
    /// RMS of the boundaries about the fitted line.
    pub fit_rms_ns: f64,
}

impl Grid {
    /// The slot k of a corrected time t (k may be negative for times before
    /// the anchor) and its residual t - boundary_k, in ns.
    #[inline]
    pub fn slot(&self, t_mono_ns: i64) -> (i64, f64) {
        let x = (t_mono_ns - self.anchor_mono_ns) as f64 / self.period_ns;
        let k = x.round();
        (k as i64, (x - k) * self.period_ns)
    }

    #[inline]
    pub fn frame_timestamp(&self, k: i64) -> u64 {
        (self.anchor_ts_ns as i64 + k * FRAME_TIMESTAMP_STEP_NS as i64) as u64
    }

    /// The boundary k periods from the anchor, on CLOCK_MONOTONIC.
    #[inline]
    pub fn boundary(&self, k: i64) -> i64 {
        self.anchor_mono_ns + (k as f64 * self.period_ns).round() as i64
    }
}

#[repr(C)]
struct Shared {
    magic: u64,
    /// Odd while the writer is mid-update.
    seq: AtomicU64,
    grid: Grid,
}

/// The mapped file, for either side.
pub struct GridMap {
    ptr: *mut Shared,
}

unsafe impl Send for GridMap {}
unsafe impl Sync for GridMap {}

impl Drop for GridMap {
    fn drop(&mut self) {
        unsafe { libc::munmap(self.ptr as *mut libc::c_void, std::mem::size_of::<Shared>()) };
    }
}

impl GridMap {
    fn map(write: bool) -> Result<Self> {
        let file = OpenOptions::new().read(true).write(true).create(write).open(GRID_PATH)
            .map_err(|e| format_err!("{}: {}", GRID_PATH, e))?;
        let size = std::mem::size_of::<Shared>();
        if write {
            file.set_len(size as u64)?;
        } else if (file.metadata()?.len() as usize) < size {
            return Err(format_err!("{} is not a grid yet", GRID_PATH));
        }
        let ptr = unsafe {
            libc::mmap(std::ptr::null_mut(), size, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_SHARED,
                       file.as_raw_fd(), 0)
        };
        if ptr == libc::MAP_FAILED {
            return Err(format_err!("mmap {}: {}", GRID_PATH, std::io::Error::last_os_error()));
        }
        Ok(Self { ptr: ptr as *mut Shared })
    }

    /// The writer's side (rift_grid). Keeps whatever grid a previous rift_grid
    /// left, so a restart can carry frame_timestamp on from it.
    pub fn create() -> Result<Self> {
        let map = Self::map(true)?;
        let shared = unsafe { &mut *map.ptr };
        if shared.magic != MAGIC {
            shared.seq.store(0, Ordering::Relaxed);
            shared.grid = Grid::default();
            fence(Ordering::Release);
            shared.magic = MAGIC;
        }
        Ok(map)
    }

    /// A reader's side (rift_camera).
    pub fn open() -> Result<Self> {
        let map = Self::map(false)?;
        if unsafe { (*map.ptr).magic } != MAGIC {
            return Err(format_err!("{} has no grid", GRID_PATH));
        }
        Ok(map)
    }

    /// The current grid, or None if none has been published yet.
    #[inline]
    pub fn read(&self) -> Option<Grid> {
        let seq = unsafe { &(*self.ptr).seq };
        loop {
            let s1 = seq.load(Ordering::Acquire);
            if s1 & 1 == 1 {
                std::hint::spin_loop();
                continue;
            }
            let grid = unsafe { std::ptr::read_volatile(std::ptr::addr_of!((*self.ptr).grid)) };
            fence(Ordering::Acquire);
            if seq.load(Ordering::Relaxed) == s1 {
                return if s1 == 0 { None } else { Some(grid) };
            }
        }
    }

    /// Publishes a grid (writer only; there is one writer per Pi).
    pub fn write(&self, grid: &Grid) {
        let seq = unsafe { &(*self.ptr).seq };
        let s = seq.load(Ordering::Relaxed);
        seq.store(s + 1, Ordering::Relaxed);
        fence(Ordering::Release);
        unsafe { std::ptr::write_volatile(std::ptr::addr_of_mut!((*self.ptr).grid), *grid) };
        seq.store(s + 2, Ordering::Release);
    }

    /// The grid as last written, even mid-generation (writer's restart).
    pub fn last(&self) -> Option<Grid> {
        self.read()
    }
}
