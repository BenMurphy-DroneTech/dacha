//! rift_blob_sim: which BlobFilterConfig tracks the smallest Rift CV1 marker
//! blobs (2.6 px across: a 20 mm sphere at 5.5 m)?
//!
//! Renders synthetic 960x576 frames like the Rift camera's (dark level 18,
//! Gaussian read noise, self-lit diffuse spheres as discs blurred by a small
//! Gaussian PSF, integrated over each pixel, clipped at 255) and runs them
//! through dacha's own mocap_camera_core::FrameProcessor, unmodified. Beside
//! the processor it computes each marker's component exactly as dacha does
//! (area, bbox, intensity-weighted moments with weights v - threshold), so
//! every filter check can be evaluated for any config and each rejection
//! explained; the two are compared for agreement.
//!
//!   rift_blob_sim [--trials 4000] [--seed 1] [--sigma 0.5] [--noise 1.5]
//!                 [--shot 0] [--prop "min_area,max_area,aspect,ellip,ratio,abs,rel"]
//!                 [--only accept|false|bad|time]

use std::time::Instant;

use mocap_camera_core::FrameProcessor;
use mocap_proto::mocap::*;

const W: usize = 960;
const H: usize = 576;
const DARK: f64 = 18.0;
const FULL: f64 = 255.0;

const THRESHOLDS: [u8; 2] = [30, 40];
const DIAMS: [f64; 6] = [2.6, 3.0, 4.8, 8.0, 20.0, 60.0];
const BRIGHT: [f64; 3] = [1.5, 3.0, 5.0];

const CHECK_NAMES: [&str; 7] = ["min_area", "max_area", "edge", "aspect", "radius_ratio", "ellip_area", "skew"];

// ---------------------------------------------------------------- random

struct Rng {
    s: u64,
    spare: Option<f64>,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { s: seed.wrapping_mul(0x9E3779B97F4A7C15) | 1, spare: None }
    }
    fn next(&mut self) -> u64 {
        let mut x = self.s;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.s = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn unif(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn range(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.unif()
    }
    fn int(&mut self, a: i64, b_incl: i64) -> i64 {
        a + (self.next() % ((b_incl - a + 1) as u64)) as i64
    }
    fn gauss(&mut self) -> f64 {
        if let Some(v) = self.spare.take() {
            return v;
        }
        loop {
            let u = 2.0 * self.unif() - 1.0;
            let v = 2.0 * self.unif() - 1.0;
            let s = u * u + v * v;
            if s > 0.0 && s < 1.0 {
                let m = (-2.0 * s.ln() / s).sqrt();
                self.spare = Some(v * m);
                return u * m;
            }
        }
    }
}

// ---------------------------------------------------------------- optics

fn erfc(x: f64) -> f64 {
    // Numerical Recipes erfcc, fractional error < 1.2e-7.
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let r = t * (-z * z - 1.26551223
        + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806
        + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587
        + t * (-0.82215223 + t * 0.17087277))))))))).exp();
    if x >= 0.0 { r } else { 2.0 - r }
}

fn erf(x: f64) -> f64 {
    1.0 - erfc(x)
}

/// Radial profile of a uniform disc (radius `r_disc`, unit radiance)
/// convolved with an isotropic Gaussian of `sigma`, tabulated against
/// distance from the centre.
struct Profile {
    step: f64,
    table: Vec<f32>,
}

impl Profile {
    fn new(r_disc: f64, sigma: f64) -> Self {
        let step = 0.002;
        let r_max = r_disc + 6.0 * sigma + 1.0;
        let n = (r_max / step) as usize + 2;
        let mut table = Vec::with_capacity(n);
        let s2 = sigma * std::f64::consts::SQRT_2;
        let norm = 1.0 / (sigma * (2.0 * std::f64::consts::PI).sqrt());
        for i in 0..n {
            let r = i as f64 * step;
            // f(r) = integral over x of g(x - r) * P(|y| < sqrt(R^2 - x^2)).
            let a = (-r_disc).max(r - 6.0 * sigma);
            let b = r_disc.min(r + 6.0 * sigma);
            let mut sum = 0.0;
            if b > a {
                let m = 600;
                let h = (b - a) / m as f64;
                for k in 0..m {
                    let x = a + (k as f64 + 0.5) * h;
                    let hh = (r_disc * r_disc - x * x).max(0.0).sqrt();
                    let dx = x - r;
                    sum += norm * (-dx * dx / (2.0 * sigma * sigma)).exp() * erf(hh / s2) * h;
                }
            }
            table.push(sum as f32);
        }
        Self { step, table }
    }

    #[inline]
    fn at(&self, r: f64) -> f64 {
        let p = r / self.step;
        let i = p as usize;
        if i + 1 >= self.table.len() {
            return 0.0;
        }
        let t = (p - i as f64) as f32;
        (self.table[i] * (1.0 - t) + self.table[i + 1] * t) as f64
    }
}

#[derive(Clone, Copy)]
enum Kind {
    Disc,
    /// A second, identical marker offset by (dx, dy): the light adds.
    Pair { dx: f64, dy: f64 },
    /// A sharp occluder: light kept only where nx*(x-cx) + ny*(y-cy) < c.
    Occluded { nx: f64, ny: f64, c: f64 },
}

#[derive(Clone, Copy)]
struct Marker {
    cx: f64,
    cy: f64,
    amp: f64,
    kind: Kind,
}

/// Adds a marker's signal (counts over dark) into `sig`, integrating over
/// each pixel with `ss` x `ss` subsamples. Pixel i covers [i, i+1).
fn render(sig: &mut [f32], prof: &Profile, r_disc: f64, sigma: f64, m: &Marker, ss: usize) {
    let reach = r_disc + 5.0 * sigma + 1.0;
    let (ox, oy) = match m.kind {
        Kind::Pair { dx, dy } => (dx, dy),
        _ => (0.0, 0.0),
    };
    let x0 = ((m.cx + ox.min(0.0) - reach).floor().max(0.0)) as usize;
    let x1 = ((m.cx + ox.max(0.0) + reach).ceil().min((W - 1) as f64)) as usize;
    let y0 = ((m.cy + oy.min(0.0) - reach).floor().max(0.0)) as usize;
    let y1 = ((m.cy + oy.max(0.0) + reach).ceil().min((H - 1) as f64)) as usize;
    let inv = 1.0 / (ss * ss) as f64;
    for py in y0..=y1 {
        for px in x0..=x1 {
            let mut acc = 0.0;
            for sy in 0..ss {
                let y = py as f64 + (sy as f64 + 0.5) / ss as f64;
                for sx in 0..ss {
                    let x = px as f64 + (sx as f64 + 0.5) / ss as f64;
                    let dx = x - m.cx;
                    let dy = y - m.cy;
                    let mut v = prof.at((dx * dx + dy * dy).sqrt());
                    match m.kind {
                        Kind::Disc => {}
                        Kind::Pair { dx: qx, dy: qy } => {
                            let ex = dx - qx;
                            let ey = dy - qy;
                            v += prof.at((ex * ex + ey * ey).sqrt());
                        }
                        Kind::Occluded { nx, ny, c } => {
                            if nx * dx + ny * dy >= c {
                                v = 0.0;
                            }
                        }
                    }
                    acc += v;
                }
            }
            sig[py * W + px] += (m.amp * acc * inv) as f32;
        }
    }
}

/// Dark level plus read noise (and optional shot noise, variance shot * signal)
/// plus signal, rounded and clipped to 8 bits.
fn expose(sig: &[f32], frame: &mut [u8], rng: &mut Rng, noise: f64, shot: f64) {
    for i in 0..W * H {
        let s = sig[i] as f64;
        let sd = if shot > 0.0 && s > 0.0 { (noise * noise + shot * s).sqrt() } else { noise };
        let v = DARK + s + sd * rng.gauss();
        frame[i] = v.round().clamp(0.0, 255.0) as u8;
    }
}

// ---------------------------------------------------------------- filter

#[derive(Clone)]
struct Filt {
    name: String,
    min_area: u32,
    max_area: u32,
    edge: bool,
    aspect: f32,
    ellip: f32,
    ratio: f32,
    abs: f32,
    rel: f32,
}

impl Filt {
    fn proto(&self) -> BlobFilterConfig {
        let mut c = BlobFilterConfig::default();
        c.set_min_area(self.min_area);
        c.set_max_area(self.max_area);
        c.set_reject_edge(self.edge);
        c.set_max_bbox_aspect_ratio(self.aspect);
        c.set_elliptical_area_error(self.ellip);
        c.set_max_radius_ratio(self.ratio);
        c.max_centroid_bbox_skew_mut().set_abs(self.abs);
        c.max_centroid_bbox_skew_mut().set_rel(self.rel);
        c
    }

    fn with(&self, name: &str, f: impl Fn(&mut Filt)) -> Filt {
        let mut c = self.clone();
        c.name = name.to_string();
        f(&mut c);
        c
    }

    fn text(&self) -> String {
        format!(
            "min_area: {} max_area: {} reject_edge: {} max_bbox_aspect_ratio: {} elliptical_area_error: {} max_radius_ratio: {} max_centroid_bbox_skew {{ abs: {} rel: {} }}",
            self.min_area, self.max_area, self.edge, self.aspect, self.ellip, self.ratio, self.abs, self.rel
        )
    }

    /// Bitmask of failing checks, in dacha's order (bit i = CHECK_NAMES[i]).
    fn fails(&self, s: &St) -> u8 {
        let mut m = 0u8;
        if s.area < self.min_area { m |= 1 << 0; }
        if s.area > self.max_area { m |= 1 << 1; }
        if self.edge && s.edge { m |= 1 << 2; }
        if s.aspect > self.aspect { m |= 1 << 3; }
        if s.ratio > self.ratio { m |= 1 << 4; }
        if s.aerr.abs() > self.ellip { m |= 1 << 5; }
        let e = self.abs.max(self.rel * s.rmax);
        if s.skew2 > e * e { m |= 1 << 6; }
        m
    }
}

/// Dacha's manager.txtpb blob_filter (what the cameras run).
fn dacha_filter() -> Filt {
    Filt { name: "dacha".into(), min_area: 6, max_area: 10000, edge: true, aspect: 1.5, ellip: 0.25, ratio: 1.5, abs: 2.0, rel: 0.05 }
}

fn parse_prop(s: &str) -> Filt {
    let v: Vec<f32> = s.split(',').map(|x| x.trim().parse().expect("bad --prop")).collect();
    assert_eq!(v.len(), 7, "--prop wants min_area,max_area,aspect,ellip,ratio,abs,rel");
    Filt { name: "proposed".into(), min_area: v[0] as u32, max_area: v[1] as u32, edge: true, aspect: v[2], ellip: v[3], ratio: v[4], abs: v[5], rel: v[6] }
}

// ---------------------------------------------------------------- component stats (as dacha)

#[derive(Clone, Copy, Default)]
struct St {
    area: u32,
    edge: bool,
    aspect: f32,
    ratio: f32,
    /// Signed (expected - area) / expected.
    aerr: f32,
    skew2: f32,
    rmax: f32,
    cx: f32,
    cy: f32,
    ra: f32,
    rb: f32,
}

#[derive(Default, Clone, Copy)]
struct Acc {
    min_x: u16, max_x: u16, min_y: u16, max_y: u16,
    area: u32, mass: u64, mx: u64, my: u64, mxx: u64, mxy: u64, myy: u64,
}

fn stats_of(a: &Acc) -> St {
    let mass = a.mass as f64;
    let inv = 1.0 / mass;
    let mean_x = a.mx as f64 * inv;
    let mean_y = a.my as f64 * inv;
    let vx = (a.mxx as f64 - mean_x * a.mx as f64) * inv;
    let vy = (a.myy as f64 - mean_y * a.my as f64) * inv;
    let cv = (a.mxy as f64 - mean_x * a.my as f64) * inv;
    let em = (vx + vy) / 2.0;
    let ed = (vx - vy) / 2.0;
    let r = (ed * ed + cv * cv).sqrt();
    let ra = (2.0 * (em + r).sqrt()) as f32;
    let rb = (2.0 * (em - r).sqrt()) as f32;
    let cx = (mean_x + 0.5) as f32;
    let cy = (mean_y + 0.5) as f32;

    let bw = (a.max_x - a.min_x) as usize + 1;
    let bh = (a.max_y - a.min_y) as usize + 1;
    let mut aspect = bw as f32 / bh as f32;
    if aspect < 1.0 { aspect = 1.0 / aspect; }
    let expected = std::f32::consts::PI * ra * rb;
    let aerr = (expected - a.area as f32) / expected;
    let bx = a.min_x as f32 + bw as f32 / 2.0;
    let by = a.min_y as f32 + bh as f32 / 2.0;
    let skew2 = (cx - bx) * (cx - bx) + (cy - by) * (cy - by);
    let edge = a.min_x == 0 || a.min_y == 0 || a.max_x == (W - 1) as u16 || a.max_y == (H - 1) as u16;
    St { area: a.area, edge, aspect, ratio: ra / rb, aerr, skew2, rmax: ra.max(rb), cx, cy, ra, rb }
}

/// 8-connected components of pixels > thr inside the window; returns the
/// component with the most mass and the number of components.
fn analyse(frame: &[u8], thr: u8, x0: usize, y0: usize, x1: usize, y1: usize,
           label: &mut Vec<u32>, stack: &mut Vec<(usize, usize)>) -> (Option<St>, u32) {
    let ww = x1 - x0 + 1;
    let wh = y1 - y0 + 1;
    label.clear();
    label.resize(ww * wh, 0);
    let mut best: Option<(u64, Acc)> = None;
    let mut n = 0u32;
    for y in y0..=y1 {
        for x in x0..=x1 {
            let li = (y - y0) * ww + (x - x0);
            if label[li] != 0 || frame[y * W + x] <= thr { continue; }
            n += 1;
            let mut a = Acc { min_x: u16::MAX, min_y: u16::MAX, ..Default::default() };
            label[li] = n;
            stack.clear();
            stack.push((x, y));
            while let Some((px, py)) = stack.pop() {
                let v = (frame[py * W + px] - thr) as u64;
                a.min_x = a.min_x.min(px as u16); a.max_x = a.max_x.max(px as u16);
                a.min_y = a.min_y.min(py as u16); a.max_y = a.max_y.max(py as u16);
                let (xx, yy) = (px as u64, py as u64);
                a.area += 1; a.mass += v;
                a.mx += xx * v; a.mxx += xx * xx * v;
                a.my += yy * v; a.myy += yy * yy * v; a.mxy += xx * yy * v;
                for dy in -1i64..=1 {
                    for dx in -1i64..=1 {
                        let qx = px as i64 + dx;
                        let qy = py as i64 + dy;
                        if qx < x0 as i64 || qx > x1 as i64 || qy < y0 as i64 || qy > y1 as i64 { continue; }
                        let (qx, qy) = (qx as usize, qy as usize);
                        let qi = (qy - y0) * ww + (qx - x0);
                        if label[qi] == 0 && frame[qy * W + qx] > thr {
                            label[qi] = n;
                            stack.push((qx, qy));
                        }
                    }
                }
            }
            if best.map_or(true, |(m, _)| a.mass > m) {
                best = Some((a.mass, a));
            }
        }
    }
    (best.map(|(_, a)| stats_of(&a)), n)
}

// ---------------------------------------------------------------- helpers

fn quantile(v: &mut Vec<f64>, q: f64) -> f64 {
    if v.is_empty() { return f64::NAN; }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let i = ((v.len() - 1) as f64 * q).round() as usize;
    v[i]
}

fn pct(a: usize, n: usize) -> f64 {
    if n == 0 { f64::NAN } else { 100.0 * a as f64 / n as f64 }
}

struct Layout {
    spacing: f64,
    nx: usize,
    ny: usize,
    ox: f64,
    oy: f64,
}

fn layout(spacing: f64) -> Layout {
    let margin = 8.0;
    let nx = ((W as f64 - 2.0 * margin) / spacing) as usize;
    let ny = ((H as f64 - 2.0 * margin) / spacing) as usize;
    let ox = (W as f64 - nx as f64 * spacing) / 2.0;
    let oy = (H as f64 - ny as f64 * spacing) / 2.0;
    Layout { spacing, nx, ny, ox, oy }
}

fn spacing_for(d: f64) -> f64 {
    (d * 1.6 + 16.0).max(24.0).ceil()
}

/// FrameProcessor blobs assigned to grid cells: per cell, the blobs' (x, y).
fn assign(res: &BlobResults, l: &Layout) -> (Vec<Vec<(f32, f32)>>, usize) {
    let mut cells = vec![vec![]; l.nx * l.ny];
    let mut stray = 0;
    for b in res.blobs() {
        let gx = ((b.x() as f64 - l.ox) / l.spacing).floor();
        let gy = ((b.y() as f64 - l.oy) / l.spacing).floor();
        if gx < 0.0 || gy < 0.0 || gx as usize >= l.nx || gy as usize >= l.ny {
            stray += 1;
            continue;
        }
        cells[gy as usize * l.nx + gx as usize].push((b.x(), b.y()));
    }
    (cells, stray)
}

// ---------------------------------------------------------------- acceptance test

struct Rec {
    ti: usize,
    di: usize,
    bi: usize,
    st: Option<St>,
    ncomp: u32,
    tx: f64,
    ty: f64,
    /// FrameProcessor blobs in the cell: dacha filter, proposed filter.
    fp_n: [u8; 2],
    fp_err: [f32; 2],
}

struct Opts {
    trials: usize,
    seed: u64,
    sigma: f64,
    noise: f64,
    shot: f64,
    prop: Filt,
    only: Option<String>,
}

fn run_accept(o: &Opts) {
    let filts = [dacha_filter(), o.prop.clone()];
    let protos: Vec<BlobFilterConfig> = filts.iter().map(|f| f.proto()).collect();
    let mut fp = FrameProcessor::new(W, H);
    let mut rng = Rng::new(o.seed);
    let mut sig = vec![0f32; W * H];
    let mut frame = vec![0u8; W * H];
    let mut recs: Vec<Rec> = vec![];
    let mut label = vec![];
    let mut stack = vec![];
    let mut stray = [[0usize; 2]; 2];
    let mut frames_total = 0;

    for (di, &d) in DIAMS.iter().enumerate() {
        let r = d / 2.0;
        let prof = Profile::new(r, o.sigma);
        let ss = if d < 10.0 { 8 } else { 4 };
        let l = layout(spacing_for(d));
        let per = l.nx * l.ny;
        for (bi, &b) in BRIGHT.iter().enumerate() {
            let amp = b * FULL;
            let mut done = 0;
            while done < o.trials {
                sig.iter_mut().for_each(|v| *v = 0.0);
                let mut mk = Vec::with_capacity(per);
                for gy in 0..l.ny {
                    for gx in 0..l.nx {
                        let m = Marker {
                            cx: l.ox + (gx as f64 + 0.5) * l.spacing + rng.range(-0.5, 0.5),
                            cy: l.oy + (gy as f64 + 0.5) * l.spacing + rng.range(-0.5, 0.5),
                            amp,
                            kind: Kind::Disc,
                        };
                        render(&mut sig, &prof, r, o.sigma, &m, ss);
                        mk.push(m);
                    }
                }
                expose(&sig, &mut frame, &mut rng, o.noise, o.shot);
                frames_total += 1;
                for (ti, &thr) in THRESHOLDS.iter().enumerate() {
                    let mut cells = vec![];
                    for k in 0..2 {
                        let res = fp.process(&frame, thr, &protos[k]);
                        let (c, s) = assign(&res, &l);
                        stray[ti][k] += s;
                        cells.push(c);
                    }
                    for (i, m) in mk.iter().enumerate() {
                        let reach = r + 4.0;
                        let x0 = (m.cx - reach).floor().max(0.0) as usize;
                        let x1 = (m.cx + reach).ceil().min((W - 1) as f64) as usize;
                        let y0 = (m.cy - reach).floor().max(0.0) as usize;
                        let y1 = (m.cy + reach).ceil().min((H - 1) as f64) as usize;
                        let (st, ncomp) = analyse(&frame, thr, x0, y0, x1, y1, &mut label, &mut stack);
                        let mut fp_n = [0u8; 2];
                        let mut fp_err = [f32::NAN; 2];
                        for k in 0..2 {
                            let c = &cells[k][i];
                            fp_n[k] = c.len().min(255) as u8;
                            if let Some(&(bx, by)) = c.first() {
                                let ex = bx as f64 - m.cx;
                                let ey = by as f64 - m.cy;
                                fp_err[k] = (ex * ex + ey * ey).sqrt() as f32;
                            }
                        }
                        recs.push(Rec { ti, di, bi, st, ncomp, tx: m.cx, ty: m.cy, fp_n, fp_err });
                    }
                }
                done += per;
            }
        }
    }

    println!("== acceptance: {} trials per case, {} frames, PSF sigma {} px, read noise {} counts, shot {}",
             o.trials, frames_total, o.sigma, o.noise, o.shot);
    println!("dacha    = {}", filts[0].text());
    println!("proposed = {}", filts[1].text());

    // Component statistics per case.
    println!("\n-- component stats (own, as dacha computes them): area min/p1/med | aspect p99/max | radius ratio p99.9/max | ellip err p0.1/p1/med/p99.9 (signed, (pi*a*b - area)/(pi*a*b)) | skew max | comps>1 | centroid err rms/max px");
    for ti in 0..2 {
        for di in 0..DIAMS.len() {
            for bi in 0..BRIGHT.len() {
                let rs: Vec<&Rec> = recs.iter().filter(|r| r.ti == ti && r.di == di && r.bi == bi).collect();
                let sts: Vec<St> = rs.iter().filter_map(|r| r.st).collect();
                let missing = rs.len() - sts.len();
                let mut area: Vec<f64> = sts.iter().map(|s| s.area as f64).collect();
                let mut asp: Vec<f64> = sts.iter().map(|s| s.aspect as f64).collect();
                let mut rat: Vec<f64> = sts.iter().map(|s| if s.ratio.is_finite() { s.ratio as f64 } else { 1e9 }).collect();
                let mut ae: Vec<f64> = sts.iter().map(|s| s.aerr as f64).collect();
                let mut sk: Vec<f64> = sts.iter().map(|s| (s.skew2 as f64).sqrt()).collect();
                let multi = rs.iter().filter(|r| r.ncomp > 1).count();
                let errs: Vec<f64> = rs.iter().filter_map(|r| r.st.map(|s| {
                    let ex = s.cx as f64 - r.tx; let ey = s.cy as f64 - r.ty; (ex * ex + ey * ey).sqrt()
                })).collect();
                let rms = (errs.iter().map(|e| e * e).sum::<f64>() / errs.len().max(1) as f64).sqrt();
                let emax = errs.iter().cloned().fold(0.0, f64::max);
                println!(
                    "thr {} D {:>4} {}x: area {:>4}/{:>4}/{:>5} | asp {:.2}/{:.2} | rr {:.2}/{:.2} | ae {:+.2}/{:+.2}/{:+.2}/{:+.2} | sk {:.2} | multi {} none {} | err {:.3}/{:.3}",
                    THRESHOLDS[ti], DIAMS[di], BRIGHT[bi],
                    quantile(&mut area, 0.0), quantile(&mut area, 0.01), quantile(&mut area, 0.5),
                    quantile(&mut asp, 0.99), quantile(&mut asp, 1.0),
                    quantile(&mut rat, 0.999), quantile(&mut rat, 1.0),
                    quantile(&mut ae, 0.001), quantile(&mut ae, 0.01), quantile(&mut ae, 0.5), quantile(&mut ae, 0.999),
                    quantile(&mut sk, 1.0), multi, missing, rms, emax
                );
            }
        }
    }

    // Acceptance from own stats for a family of configs.
    let p = &o.prop;
    let mut variants = vec![filts[0].clone(), p.clone()];
    for m in [4u32, 6, 8, 10, 12] { variants.push(p.with(&format!("prop min_area {}", m), |f| f.min_area = m)); }
    for e in [0.25f32, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0] { variants.push(p.with(&format!("prop ellip {}", e), |f| f.ellip = e)); }
    for a in [1.25f32, 1.5, 1.75] { variants.push(p.with(&format!("prop aspect {}", a), |f| f.aspect = a)); }
    for r in [1.2f32, 1.3] { variants.push(p.with(&format!("prop ratio {}", r), |f| f.ratio = r)); }
    for a in [0.5f32, 0.75, 1.0] { variants.push(p.with(&format!("prop skew abs {}", a), |f| f.abs = a)); }
    println!("\n-- acceptance % from own stats (columns: D; each cell 1.5x/3x/5x)");
    for f in &variants {
        for ti in 0..2 {
            let mut line = format!("{:<20} thr {}:", f.name, THRESHOLDS[ti]);
            for di in 0..DIAMS.len() {
                let mut cell = vec![];
                for bi in 0..BRIGHT.len() {
                    let rs: Vec<&Rec> = recs.iter().filter(|r| r.ti == ti && r.di == di && r.bi == bi).collect();
                    let ok = rs.iter().filter(|r| r.st.map_or(false, |s| f.fails(&s) == 0)).count();
                    cell.push(format!("{:.1}", pct(ok, rs.len())));
                }
                line += &format!(" D{} {}", DIAMS[di], cell.join("/"));
            }
            println!("{}", line);
        }
    }

    // Rejection reasons (first failing check in dacha's order, and any failing).
    println!("\n-- rejections: first failing check (any failing) as % of trials, nonzero only");
    for (k, f) in filts.iter().enumerate() {
        let _ = k;
        for ti in 0..2 {
            for di in 0..DIAMS.len() {
                for bi in 0..BRIGHT.len() {
                    let rs: Vec<&Rec> = recs.iter().filter(|r| r.ti == ti && r.di == di && r.bi == bi).collect();
                    let mut first = [0usize; 7];
                    let mut any = [0usize; 7];
                    let mut none = 0;
                    for r in &rs {
                        match r.st {
                            None => none += 1,
                            Some(s) => {
                                let m = f.fails(&s);
                                if m != 0 { first[m.trailing_zeros() as usize] += 1; }
                                for c in 0..7 { if m & (1 << c) != 0 { any[c] += 1; } }
                            }
                        }
                    }
                    let mut parts = vec![];
                    if none > 0 { parts.push(format!("nothing above thr {:.1}", pct(none, rs.len()))); }
                    for c in 0..7 {
                        if any[c] > 0 {
                            parts.push(format!("{} {:.1} ({:.1})", CHECK_NAMES[c], pct(first[c], rs.len()), pct(any[c], rs.len())));
                        }
                    }
                    if !parts.is_empty() {
                        println!("{:<8} thr {} D {:>4} {}x: {}", f.name, THRESHOLDS[ti], DIAMS[di], BRIGHT[bi], parts.join(", "));
                    }
                }
            }
        }
    }

    // FrameProcessor's own results, agreement, centroid error.
    println!("\n-- FrameProcessor (unmodified) acceptance %, agreement with own evaluation, centroid error of accepted blobs rms/max px");
    for (k, f) in filts.iter().enumerate() {
        let mut agree = 0;
        let mut total = 0;
        let mut extra = 0;
        for ti in 0..2 {
            let mut line = format!("{:<8} thr {}:", f.name, THRESHOLDS[ti]);
            for di in 0..DIAMS.len() {
                let mut cell = vec![];
                let mut errs = vec![];
                for bi in 0..BRIGHT.len() {
                    let rs: Vec<&Rec> = recs.iter().filter(|r| r.ti == ti && r.di == di && r.bi == bi).collect();
                    let ok = rs.iter().filter(|r| r.fp_n[k] >= 1).count();
                    for r in &rs {
                        let own = r.st.map_or(false, |s| f.fails(&s) == 0);
                        total += 1;
                        if own == (r.fp_n[k] >= 1) { agree += 1; }
                        if r.fp_n[k] > 1 { extra += 1; }
                        if r.fp_n[k] >= 1 { errs.push(r.fp_err[k] as f64); }
                    }
                    cell.push(format!("{:.1}", pct(ok, rs.len())));
                }
                let rms = (errs.iter().map(|e| e * e).sum::<f64>() / errs.len().max(1) as f64).sqrt();
                let emax = errs.iter().cloned().fold(0.0, f64::max);
                line += &format!(" D{} {} [{:.3}/{:.3}]", DIAMS[di], cell.join("/"), rms, emax);
            }
            println!("{}", line);
        }
        println!("{:<8} agreement {}/{} ({:.4}%), cells with >1 blob {}, stray blobs thr30 {} thr40 {}",
                 f.name, agree, total, pct(agree, total), extra, stray[0][k], stray[1][k]);
    }
}

// ---------------------------------------------------------------- false blobs

fn run_false(o: &Opts) {
    let p = &o.prop;
    let loose = Filt { name: String::new(), min_area: 1, max_area: 100000, edge: true, aspect: 1e9, ellip: 1e9, ratio: 1e9, abs: 1e9, rel: 0.0 };
    let filts = vec![
        dacha_filter(),
        p.clone(),
        p.with("prop min_area 1", |f| f.min_area = 1),
        p.with("prop min_area 2", |f| f.min_area = 2),
        p.with("prop min_area 3", |f| f.min_area = 3),
        p.with("prop min_area 5", |f| f.min_area = 5),
        loose.with("min_area 1 only", |f| f.min_area = 1),
        loose.with("min_area 2 only", |f| f.min_area = 2),
        loose.with("min_area 3 only", |f| f.min_area = 3),
    ];
    let protos: Vec<BlobFilterConfig> = filts.iter().map(|f| f.proto()).collect();
    let mut fp = FrameProcessor::new(W, H);
    let mut rng = Rng::new(o.seed ^ 0x5555);
    let sig = vec![0f32; W * H];
    let mut frame = vec![0u8; W * H];

    println!("\n== false blobs (FrameProcessor, unmodified); hot pixels replace the pixel with a value uniform in 30..60");
    // kinds: 0 noise only, 1 single, 2 pair H, 3 pair V, 4 pair diagonal
    let kinds = ["noise only", "single hot", "pair horiz", "pair vert", "pair diag", "L of 3", "2x2 block"];
    let frames_per_kind = (o.trials / 400).max(10);
    for (kind, kname) in kinds.iter().enumerate() {
        // [filter][thr] accepted blobs; components above threshold
        let mut acc = vec![[0usize; 2]; filts.len()];
        let mut comps = [0usize; 2];
        let mut placed = 0usize;
        let mut above = [0usize; 2];
        for _ in 0..frames_per_kind {
            expose(&sig, &mut frame, &mut rng, o.noise, 0.0);
            let mut hot = vec![];
            if kind > 0 {
                let step = 16;
                for gy in (step..H - step).step_by(step) {
                    for gx in (step..W - step).step_by(step) {
                        let x = gx + rng.int(0, 7) as usize;
                        let y = gy + rng.int(0, 7) as usize;
                        let offs: &[(usize, usize)] = match kind {
                            1 => &[(0, 0)],
                            2 => &[(0, 0), (1, 0)],
                            3 => &[(0, 0), (0, 1)],
                            4 => &[(0, 0), (1, 1)],
                            5 => &[(0, 0), (1, 0), (0, 1)],
                            _ => &[(0, 0), (1, 0), (0, 1), (1, 1)],
                        };
                        let mut mx = 0u8;
                        for &(dx, dy) in offs {
                            let v = rng.int(30, 60) as u8;
                            frame[(y + dy) * W + x + dx] = v;
                            mx = mx.max(v);
                        }
                        hot.push(mx);
                        placed += 1;
                    }
                }
            }
            for (ti, &thr) in THRESHOLDS.iter().enumerate() {
                above[ti] += hot.iter().filter(|&&v| v > thr).count();
                for (k, pr) in protos.iter().enumerate() {
                    let res = fp.process(&frame, thr, pr);
                    acc[k][ti] += res.blobs().len();
                    if k == 0 { comps[ti] += res.blobs().len() + res.num_rejected_blobs() as usize; }
                }
            }
        }
        println!("{:<11}: {} frames, {} placed; components found thr30 {} thr40 {} (hot sites with a pixel above thr: {} / {})",
                 kname, frames_per_kind, placed, comps[0], comps[1], above[0], above[1]);
        for (k, f) in filts.iter().enumerate() {
            println!("    {:<16} accepted thr30 {:>6} thr40 {:>6}", f.name, acc[k][0], acc[k][1]);
        }
    }
}

// ---------------------------------------------------------------- merged and occluded markers

fn run_bad(o: &Opts) {
    let filts = [dacha_filter(), o.prop.clone(), o.prop.with("prop ellip 0.25", |f| f.ellip = 0.25)];
    let protos: Vec<BlobFilterConfig> = filts.iter().map(|f| f.proto()).collect();
    let mut fp = FrameProcessor::new(W, H);
    let mut rng = Rng::new(o.seed ^ 0xBAD);
    let mut sig = vec![0f32; W * H];
    let mut frame = vec![0u8; W * H];
    let thr = 30u8;
    println!("\n== merged pairs (second marker at 0.5-1.0 D, light adds) and occluded markers (sharp edge leaving 40-85% of the disc), 3x, thr 30");
    println!("   accepted % and, for accepted, error to the nearest true centre mean/max px (for occluded: to the full disc's centre)");
    for &d in &[4.8, 8.0, 20.0, 60.0] {
        let r = d / 2.0;
        let prof = Profile::new(r, o.sigma);
        let ss = if d < 10.0 { 8 } else { 4 };
        let l = layout(spacing_for(2.0 * d));
        let per = l.nx * l.ny;
        for kind in 0..2 {
            let mut acc = vec![0usize; filts.len()];
            let mut errs: Vec<Vec<f64>> = vec![vec![]; filts.len()];
            let mut n = 0;
            let target = (o.trials / 4).max(500);
            while n < target {
                sig.iter_mut().for_each(|v| *v = 0.0);
                let mut mk = vec![];
                for gy in 0..l.ny {
                    for gx in 0..l.nx {
                        let cx = l.ox + (gx as f64 + 0.5) * l.spacing + rng.range(-0.5, 0.5);
                        let cy = l.oy + (gy as f64 + 0.5) * l.spacing + rng.range(-0.5, 0.5);
                        let ang = rng.range(0.0, std::f64::consts::TAU);
                        let k = if kind == 0 {
                            let s = rng.range(0.5, 1.0) * d;
                            Kind::Pair { dx: s * ang.cos(), dy: s * ang.sin() }
                        } else {
                            // Chord at signed distance c from the centre keeps fraction
                            // (theta - sin theta)/(2 pi) with theta = 2 acos(c/r) on the
                            // occluded side; pick c for a kept fraction 0.4-0.85.
                            let keep = rng.range(0.4, 0.85);
                            let mut lo = -r; let mut hi = r;
                            for _ in 0..50 {
                                let c = (lo + hi) / 2.0;
                                let th = 2.0 * (c / r).clamp(-1.0, 1.0).acos();
                                let cut = (th - th.sin()) / (2.0 * std::f64::consts::PI);
                                if 1.0 - cut < keep { lo = c; } else { hi = c; }
                            }
                            Kind::Occluded { nx: ang.cos(), ny: ang.sin(), c: (lo + hi) / 2.0 }
                        };
                        // For pairs the first marker sits half a separation back so the pair is centred.
                        let (mx, my) = match k { Kind::Pair { dx, dy } => (cx - dx * 0.5, cy - dy * 0.5), _ => (cx, cy) };
                        let m = Marker { cx: mx, cy: my, amp: 3.0 * FULL, kind: k };
                        render(&mut sig, &prof, r, o.sigma, &m, ss);
                        mk.push(m);
                    }
                }
                expose(&sig, &mut frame, &mut rng, o.noise, o.shot);
                for (fk, pr) in protos.iter().enumerate() {
                    let res = fp.process(&frame, thr, pr);
                    let (cells, _) = assign(&res, &l);
                    for (i, m) in mk.iter().enumerate() {
                        if let Some(&(bx, by)) = cells[i].first() {
                            acc[fk] += 1;
                            let mut e = ((bx as f64 - m.cx).powi(2) + (by as f64 - m.cy).powi(2)).sqrt();
                            if let Kind::Pair { dx, dy } = m.kind {
                                e = e.min(((bx as f64 - m.cx - dx).powi(2) + (by as f64 - m.cy - dy).powi(2)).sqrt());
                            }
                            errs[fk].push(e);
                        }
                    }
                }
                n += per;
            }
            let name = if kind == 0 { "merged pair" } else { "occluded" };
            let mut line = format!("D {:>4} {:<11}:", d, name);
            for (fk, f) in filts.iter().enumerate() {
                let mean = errs[fk].iter().sum::<f64>() / errs[fk].len().max(1) as f64;
                let mx = errs[fk].iter().cloned().fold(0.0, f64::max);
                line += &format!("  {} {:.1}% [{:.2}/{:.2}]", f.name, pct(acc[fk], n), mean, mx);
            }
            println!("{}", line);
        }
    }
}

// ---------------------------------------------------------------- timing

fn run_time(o: &Opts) {
    let pr = o.prop.proto();
    let mut fp = FrameProcessor::new(W, H);
    let mut rng = Rng::new(o.seed ^ 0x7777);
    let mut sig = vec![0f32; W * H];
    let mut frame = vec![0u8; W * H];
    println!("\n== FrameProcessor time per 960x576 frame (proposed filter, 2000 runs each)");
    let cases: Vec<(&str, Vec<f64>)> = vec![
        ("empty (dark + noise)", vec![]),
        ("10 x 2.6 px", vec![2.6; 10]),
        ("10 x 4.8 px", vec![4.8; 10]),
        ("10 mixed 2.6-20 px", vec![2.6, 3.0, 4.8, 4.8, 8.0, 8.0, 8.0, 20.0, 20.0, 20.0]),
        ("10 x 60 px", vec![60.0; 10]),
    ];
    for (name, ds) in cases {
        sig.iter_mut().for_each(|v| *v = 0.0);
        for (i, &d) in ds.iter().enumerate() {
            let prof = Profile::new(d / 2.0, o.sigma);
            let m = Marker { cx: 60.0 + (i % 5) as f64 * 180.0 + rng.unif(), cy: 150.0 + (i / 5) as f64 * 250.0 + rng.unif(), amp: 3.0 * FULL, kind: Kind::Disc };
            render(&mut sig, &prof, d / 2.0, o.sigma, &m, 4);
        }
        expose(&sig, &mut frame, &mut rng, o.noise, o.shot);
        for &thr in &THRESHOLDS {
            let mut t = vec![];
            let mut nb = 0;
            for _ in 0..2000 {
                let s = Instant::now();
                let res = fp.process(&frame, thr, &pr);
                t.push(s.elapsed().as_secs_f64() * 1e6);
                nb = res.blobs().len();
            }
            let mean = t.iter().sum::<f64>() / t.len() as f64;
            println!("{:<22} thr {}: {} blobs, mean {:.0} us, p50 {:.0} us, p99 {:.0} us, max {:.0} us",
                     name, thr, nb, mean, quantile(&mut t, 0.5), quantile(&mut t, 0.99), quantile(&mut t, 1.0));
        }
    }
}

// ---------------------------------------------------------------- main

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut o = Opts {
        trials: 4000,
        seed: 1,
        sigma: 0.5,
        noise: 1.5,
        shot: 0.0,
        prop: parse_prop("3,10000,2.0,0.6,1.5,1.0,0.05"),
        only: None,
    };
    let mut i = 1;
    while i < args.len() {
        let v = args.get(i + 1).cloned().unwrap_or_default();
        match args[i].as_str() {
            "--trials" => o.trials = v.parse().unwrap(),
            "--seed" => o.seed = v.parse().unwrap(),
            "--sigma" => o.sigma = v.parse().unwrap(),
            "--noise" => o.noise = v.parse().unwrap(),
            "--shot" => o.shot = v.parse().unwrap(),
            "--prop" => o.prop = parse_prop(&v),
            "--only" => o.only = Some(v),
            a => panic!("unknown argument {}", a),
        }
        i += 2;
    }
    let want = |s: &str| o.only.as_deref().map_or(true, |x| x == s);
    if want("accept") { run_accept(&o); }
    if want("false") { run_false(&o); }
    if want("bad") { run_bad(&o); }
    if want("time") { run_time(&o); }
}
