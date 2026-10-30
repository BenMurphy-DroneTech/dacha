//! rift_lens: reads each Rift sensor's factory lens calibration from its
//! bridge's flash and prints it with a few undistorted points (centre, edges,
//! corners) and the undistortion's speed. Safe on a streaming camera (a second
//! handle, XU queries only), but not while its node runs a sync setup.
//!
//!   rift_lens [--hex] [/dev/v4l/by-id/usb-Oculus_VR_Rift_Sensor_<serial>-video-index0 ...]
//!
//! With no paths, every Rift sensor in /dev/v4l/by-id.

#[macro_use]
extern crate common;

#[allow(dead_code)]
#[path = "../bridge.rs"]
mod bridge;

use std::time::Instant;

use common::errors::*;
use rift_camera::lens::*;

fn serial(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
        .trim_start_matches("usb-Oculus_VR_Rift_Sensor_")
        .trim_end_matches("-video-index0")
}

fn one(path: &str, hex: bool) -> Result<()> {
    let mut b = bridge::Bridge::open(path)?;
    let t0 = Instant::now();
    let block = b.read_calibration()?;
    let read_ms = t0.elapsed().as_secs_f64() * 1e3;
    println!("{}  (flash read {:.1} ms)", serial(path), read_ms);
    if hex {
        for (i, row) in block.chunks(16).enumerate() {
            println!("  {:04x}: {:02x?}", 0x1d000 + i * 16, row);
        }
    }
    let c = FisheyeCalibration::parse(&block)?;
    println!("  f {:.3}  cx {:.3}  cy {:.3}  k {:.6} {:.6} {:.6} {:.6}",
        c.f, c.cx, c.cy, c.k[0], c.k[1], c.k[2], c.k[3]);

    let (w, h) = (SENSOR_WIDTH as f64, SENSOR_HEIGHT as f64);
    let points = [
        ("centre", c.cx, c.cy),
        ("left edge", 0.0, c.cy), ("right edge", w, c.cy),
        ("top edge", c.cx, 0.0), ("bottom edge", c.cx, h),
        ("corner TL", 0.0, 0.0), ("corner TR", w, 0.0),
        ("corner BL", 0.0, h), ("corner BR", w, h),
    ];
    for (name, x, y) in points {
        let (ux, uy) = c.undistort(x, y);
        let r = ((ux - c.cx).powi(2) + (uy - c.cy).powi(2)).sqrt();
        let theta = (r / c.f).atan().to_degrees();
        let (bx, by) = c.distort(ux, uy);
        println!("  {:<11} ({:7.1}, {:6.1}) -> ({:8.2}, {:8.2})  r {:7.1} px  {:4.1} deg  back {:.1e} px",
            name, x, y, ux, uy, r, theta, ((bx - x).powi(2) + (by - y).powi(2)).sqrt());
    }

    // Speed: a million points spread over the sensor.
    let n = 1_000_000;
    let t0 = Instant::now();
    let mut sum = 0.0;
    for i in 0..n {
        let x = (i % 1280) as f64 + 0.25;
        let y = ((i / 1280) % 960) as f64 + 0.5;
        let (ux, uy) = c.undistort(std::hint::black_box(x), std::hint::black_box(y));
        sum += ux + uy;
    }
    let ns = t0.elapsed().as_secs_f64() * 1e9 / n as f64;
    std::hint::black_box(sum);
    println!("  undistort {:.0} ns per point", ns);
    let p = c.pinhole_intrinsics();
    println!("  pinhole: focal_length {:?} center {:?} k1 {} k2 {}", p.focal_length(), p.center(), p.k1(), p.k2());
    Ok(())
}

fn main() -> Result<()> {
    let mut hex = false;
    let mut paths = vec![];
    for a in std::env::args().skip(1) {
        if a == "--hex" {
            hex = true;
        } else {
            paths.push(a);
        }
    }
    if paths.is_empty() {
        for e in std::fs::read_dir("/dev/v4l/by-id")? {
            let p = e?.path().to_string_lossy().to_string();
            if p.contains("Rift_Sensor") && p.ends_with("-video-index0") {
                paths.push(p);
            }
        }
        paths.sort();
    }
    let mut failed = 0;
    for p in &paths {
        if let Err(e) = one(p, hex) {
            eprintln!("{}: {}", serial(p), e);
            failed += 1;
        }
    }
    if failed > 0 {
        return Err(format_err!("{} of {} cameras failed", failed, paths.len()));
    }
    Ok(())
}
