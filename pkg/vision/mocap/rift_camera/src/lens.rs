//! The Rift CV1 sensor's lens: the factory fisheye calibration every sensor
//! stores in its bridge's flash, and the undistortion that turns a blob centre
//! into a pinhole pixel for dacha's CameraIntrinsicsModel (a pinhole with k1
//! and k2 only, which fits these lenses poorly; see riftcv1-mocap's camera
//! system reference, "Lens calibration").
//!
//! The model is OpenCV fisheye (Kannala–Brandt): a ray at angle θ from the
//! optical axis lands at radius r = f·θd from (cx, cy), with
//! θd = θ (1 + k1θ² + k2θ⁴ + k3θ⁶ + k4θ⁸). The pinhole it maps to keeps f, cx
//! and cy, with r = f·tan θ. Every coordinate is in full-sensor 1280×960
//! pixels: add the crop's column and row offsets to a cropped frame's blobs
//! first.
//!
//! Pure: the flash read is Bridge::read_calibration.

use common::errors::*;
use vision_proto::vision::CameraIntrinsicsModelProto;

/// The full sensor the calibration's pixels refer to.
pub const SENSOR_WIDTH: u32 = 1280;
pub const SENSOR_HEIGHT: u32 = 960;

/// Where the calibration lives in the bridge's flash, and its size.
pub const CALIBRATION_ADDRESS: u32 = 0x1d000;
pub const CALIBRATION_LEN: usize = 128;

/// Offsets of the little-endian f32 fields in the block.
const OFF_F: usize = 0x30;
const OFF_CX: usize = 0x34;
const OFF_CY: usize = 0x38;
const OFF_K: [usize; 4] = [0x48, 0x4c, 0x50, 0x54];

/// The largest ray angle undistortion returns (89°): tan θ grows without
/// bound at 90°, and nothing on the sensor comes near (the corners are at 62°).
const THETA_MAX: f64 = 89.0 * std::f64::consts::PI / 180.0;

/// Newton steps from the fixed-point start: 3 reach 1e-12 rad at the corners,
/// the 4th is margin.
const NEWTON_STEPS: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FisheyeCalibration {
    /// Focal length in pixels (the same in x and y).
    pub f: f64,
    /// Optical centre in full-sensor pixels.
    pub cx: f64,
    pub cy: f64,
    /// k1–k4 of θd = θ (1 + k1θ² + k2θ⁴ + k3θ⁶ + k4θ⁸).
    pub k: [f64; 4],
}

fn f32_at(block: &[u8], off: usize) -> f64 {
    f32::from_le_bytes([block[off], block[off + 1], block[off + 2], block[off + 3]]) as f64
}

impl FisheyeCalibration {
    /// Parses the 128-byte flash block (at least the first 0x58 bytes) and
    /// checks it looks like a Rift CV1 lens.
    pub fn parse(block: &[u8]) -> Result<Self> {
        if block.len() < OFF_K[3] + 4 {
            return Err(format_err!("Calibration block is {} bytes, too short", block.len()));
        }
        let c = Self {
            f: f32_at(block, OFF_F),
            cx: f32_at(block, OFF_CX),
            cy: f32_at(block, OFF_CY),
            k: OFF_K.map(|o| f32_at(block, o)),
        };
        c.check()?;
        Ok(c)
    }

    /// Fails unless the values are in the range every CV1 sensor has shown:
    /// f about 716 px, the centre within some 20 px of the middle, small k.
    pub fn check(&self) -> Result<()> {
        let ok = (600.0..=800.0).contains(&self.f)
            && (500.0..=800.0).contains(&self.cx)
            && (350.0..=600.0).contains(&self.cy)
            && self.k.iter().all(|k| k.is_finite() && k.abs() < 1.0);
        if !ok {
            return Err(format_err!("Implausible lens calibration: {:?}", self));
        }
        Ok(())
    }

    /// θd for a ray angle θ.
    #[inline]
    pub fn theta_d(&self, theta: f64) -> f64 {
        let t2 = theta * theta;
        let [k1, k2, k3, k4] = self.k;
        theta * (1.0 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
    }

    /// The ray angle θ for a distorted angle θd: Newton's method on
    /// θd(θ) − θd, from θd / (1 + k1θd² + ...), which is within 1e-3 rad at the
    /// corners for these lenses.
    #[inline]
    pub fn theta(&self, theta_d: f64) -> f64 {
        let [k1, k2, k3, k4] = self.k;
        let td2 = theta_d * theta_d;
        let mut t = theta_d / (1.0 + td2 * (k1 + td2 * (k2 + td2 * (k3 + td2 * k4))));
        for _ in 0..NEWTON_STEPS {
            let t2 = t * t;
            let p = 1.0 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4)));
            let dp = 1.0 + t2 * (3.0 * k1 + t2 * (5.0 * k2 + t2 * (7.0 * k3 + t2 * 9.0 * k4)));
            t -= (t * p - theta_d) / dp;
        }
        t.clamp(0.0, THETA_MAX)
    }

    /// A distorted full-sensor pixel (a blob centre) to the pinhole pixel with
    /// the same f, cx and cy.
    #[inline]
    pub fn undistort(&self, x: f64, y: f64) -> (f64, f64) {
        let dx = x - self.cx;
        let dy = y - self.cy;
        let rd = (dx * dx + dy * dy).sqrt();
        if rd < 1e-9 {
            return (x, y);
        }
        let theta = self.theta(rd / self.f);
        let s = self.f * theta.tan() / rd;
        (self.cx + dx * s, self.cy + dy * s)
    }

    /// A pinhole pixel to the distorted full-sensor pixel (the inverse of
    /// undistort; for tests and for drawing pinhole points on frames).
    #[inline]
    pub fn distort(&self, x: f64, y: f64) -> (f64, f64) {
        let dx = x - self.cx;
        let dy = y - self.cy;
        let ru = (dx * dx + dy * dy).sqrt();
        if ru < 1e-9 {
            return (x, y);
        }
        let theta = (ru / self.f).atan();
        let s = self.f * self.theta_d(theta) / ru;
        (self.cx + dx * s, self.cy + dy * s)
    }

    /// dacha's model for the undistorted blobs: a pinhole with this f and
    /// centre and no distortion. The pinhole image is 1280×960 pixels' worth of
    /// rays but reaches about 1,340 px from the centre at the corners
    /// (−420 to 1,613 in x for a typical sensor), so its pixels run outside the
    /// 1280×960 frame. The proto has no image size field.
    pub fn pinhole_intrinsics(&self) -> CameraIntrinsicsModelProto {
        let mut out = CameraIntrinsicsModelProto::default();
        out.focal_length_mut().extend_from_slice(&[self.f, self.f]);
        out.center_mut().extend_from_slice(&[self.cx, self.cy]);
        out.set_k1(0.0);
        out.set_k2(0.0);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WMTD3062200XGN, from the camera system reference.
    fn xgn_block() -> [u8; CALIBRATION_LEN] {
        let mut b = [0u8; CALIBRATION_LEN];
        let mut put = |off: usize, v: f32| b[off..off + 4].copy_from_slice(&v.to_le_bytes());
        put(OFF_F, 716.206);
        put(OFF_CX, 663.816);
        put(OFF_CY, 478.217);
        put(OFF_K[0], 0.072958);
        put(OFF_K[1], -0.025257);
        put(OFF_K[2], 0.005886);
        put(OFF_K[3], -0.000611);
        b
    }

    fn xgn() -> FisheyeCalibration {
        FisheyeCalibration::parse(&xgn_block()).unwrap()
    }

    #[test]
    fn parses_xgn() {
        let c = xgn();
        let close = |a: f64, b: f64| (a - b).abs() <= b.abs() * 1e-6 + 1e-9;
        assert!(close(c.f, 716.206) && close(c.cx, 663.816) && close(c.cy, 478.217), "{:?}", c);
        for (a, b) in c.k.iter().zip([0.072958, -0.025257, 0.005886, -0.000611]) {
            assert!(close(*a, b), "{:?}", c);
        }
    }

    #[test]
    fn rejects_garbage() {
        assert!(FisheyeCalibration::parse(&[0u8; CALIBRATION_LEN]).is_err());
        assert!(FisheyeCalibration::parse(&[0xffu8; CALIBRATION_LEN]).is_err());
        assert!(FisheyeCalibration::parse(&[0u8; 40]).is_err());
    }

    /// Against riftcv1-mocap host/rift_calib_convert.py's fisheye_theta (50
    /// Newton steps in Python), with the f32-rounded values the block holds.
    #[test]
    fn undistorts_like_the_python() {
        // The exact decimal values, so the reference needs no f32 rounding.
        let c = FisheyeCalibration { f: 716.206, cx: 663.816, cy: 478.217,
            k: [0.072958, -0.025257, 0.005886, -0.000611] };
        let cases = [
            ((0.0, 0.0), (-420.491446600, -302.924542522)),
            ((1280.0, 0.0), (1610.103311211, -256.191357090)),
            ((0.0, 960.0), (-424.416744549, 1268.032304794)),
            ((1280.0, 960.0), (1613.265577650, 1220.574259956)),
            ((0.0, 478.217), (-215.764298982, 478.217)),
            ((663.816, 0.0), (663.816, -66.590784143)),
        ];
        for ((x, y), (ux, uy)) in cases {
            let (gx, gy) = c.undistort(x, y);
            assert!((gx - ux).abs() < 1e-6 && (gy - uy).abs() < 1e-6,
                "({}, {}) -> ({}, {}), expected ({}, {})", x, y, gx, gy, ux, uy);
        }
    }

    #[test]
    fn round_trips_across_the_field() {
        let c = xgn();
        // Distorted pixels over the whole sensor and a margin past it.
        let mut worst: f64 = 0.0;
        for j in -2..=50 {
            for i in -2..=66 {
                let (x, y) = (i as f64 * 20.0, j as f64 * 20.0);
                let (ux, uy) = c.undistort(x, y);
                let (dx, dy) = c.distort(ux, uy);
                worst = worst.max((dx - x).abs()).max((dy - y).abs());
            }
        }
        assert!(worst < 1e-6, "undistort->distort worst {} px", worst);

        // Pinhole points out to 70° from the axis in every direction.
        let mut worst: f64 = 0.0;
        for a in 0..72 {
            let phi = a as f64 * 5.0f64.to_radians();
            for t in 0..=70 {
                let r = c.f * (t as f64).to_radians().tan();
                let (x, y) = (c.cx + r * phi.cos(), c.cy + r * phi.sin());
                let (dx, dy) = c.distort(x, y);
                let (ux, uy) = c.undistort(dx, dy);
                worst = worst.max((ux - x).abs()).max((uy - y).abs());
            }
        }
        assert!(worst < 1e-6, "distort->undistort worst {} px", worst);
    }

    #[test]
    fn centre_is_fixed_and_pinhole_matches() {
        let c = xgn();
        assert_eq!(c.undistort(c.cx, c.cy), (c.cx, c.cy));
        let p = c.pinhole_intrinsics();
        assert_eq!(p.focal_length(), &[c.f, c.f]);
        assert_eq!(p.center(), &[c.cx, c.cy]);
        assert_eq!((p.k1(), p.k2()), (0.0, 0.0));
    }
}
