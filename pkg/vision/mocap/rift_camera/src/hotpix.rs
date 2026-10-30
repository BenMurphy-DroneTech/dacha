//! Hot pixels: each camera's map from riftcv1-mocap
//! (`data/captures/hot_pixels_<serial>.txt`, made by `rift_test.py hotmap` at
//! 2x and 188 us: pixels at least 12 counts over the dark level). Before
//! Dennis's blob detection sees a frame, every mapped pixel in the window is
//! replaced by the median of its eight neighbours, as rift_lib's
//! HotPixels.repair does, so a hot pixel can't become a blob and one beside a
//! marker doesn't pull its centre. The frame is the driver's mmap'd buffer;
//! uvcvideo overwrites it on the next fill.
//!
//! Per frame: up to about 1,400 pixels, each nine loads and a store.

use common::errors::*;

use crate::config::{FRAME_HEIGHT, FRAME_WIDTH};

/// The window's first sensor column (the crop's fixed columns 160-1119).
pub const COLUMN_START: u32 = 160;

pub struct HotPixels {
    /// Sensor (column, row) of every mapped pixel.
    sensor: Vec<(u32, u32)>,
    /// Their offsets in the 960x576 frame for the current row window, not on
    /// the frame's border (the 8 neighbours must exist).
    window: Vec<u32>,
}

impl HotPixels {
    pub fn none() -> Self {
        Self { sensor: vec![], window: vec![] }
    }

    /// Reads a map: `#` header lines, then `column row level` per pixel in
    /// sensor coordinates.
    pub fn load(path: &str) -> Result<Self> {
        let text = std::fs::read_to_string(path).map_err(|e| format_err!("{}: {}", path, e))?;
        if !text.lines().take_while(|l| l.starts_with('#')).any(|l| l.contains("sensor coordinates")) {
            return Err(format_err!("{}: not a map in sensor coordinates", path));
        }
        let mut sensor = vec![];
        for line in text.lines() {
            if line.starts_with('#') || line.trim().is_empty() {
                continue;
            }
            let mut f = line.split_whitespace();
            let (Some(c), Some(r)) = (f.next(), f.next()) else { continue };
            sensor.push((c.parse()?, r.parse()?));
        }
        Ok(Self { sensor, window: vec![] })
    }

    pub fn len(&self) -> usize {
        self.sensor.len()
    }

    /// Places the map on the row window starting at sensor row row_start.
    pub fn set_window(&mut self, row_start: u32) {
        self.window.clear();
        for &(c, r) in &self.sensor {
            if c < COLUMN_START + 1 || r < row_start + 1 {
                continue;
            }
            let (x, y) = (c - COLUMN_START, r - row_start);
            if x + 1 >= FRAME_WIDTH || y + 1 >= FRAME_HEIGHT {
                continue;
            }
            self.window.push(y * FRAME_WIDTH + x);
        }
        // In memory order, so the repairs walk the frame forwards.
        self.window.sort_unstable();
    }

    pub fn in_window(&self) -> usize {
        self.window.len()
    }

    /// Repairs the frame in place.
    #[inline]
    pub fn repair(&self, frame: &mut [u8]) {
        let w = FRAME_WIDTH as usize;
        for &i in &self.window {
            let i = i as usize;
            let mut n = [
                frame[i - w - 1], frame[i - w], frame[i - w + 1],
                frame[i - 1], frame[i + 1],
                frame[i + w - 1], frame[i + w], frame[i + w + 1],
            ];
            n.sort_unstable();
            // The median of eight: the mean of the middle two.
            frame[i] = ((n[3] as u16 + n[4] as u16 + 1) / 2) as u8;
        }
    }
}
