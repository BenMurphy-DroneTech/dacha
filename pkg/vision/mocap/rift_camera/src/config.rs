use std::net::Ipv4Addr;

use common::errors::*;
use crypto::hasher::Hasher;

/// Rows the 90 fps crop reads out, and the sensor's full height.
pub const FRAME_WIDTH: u32 = 960;
pub const FRAME_HEIGHT: u32 = 576;
pub const SENSOR_HEIGHT: u32 = 960;

/// One camera's line in the per-Pi config file.
#[derive(Clone, Debug)]
pub struct RiftCameraConfig {
    pub serial: String,
    pub ip: Ipv4Addr,
    pub row_start: u32,
}

impl RiftCameraConfig {
    /// The line for `serial` in a config file of `serial ip row_start` lines
    /// ('#' starts a comment).
    pub fn load(text: &str, serial: &str) -> Result<Self> {
        for (i, line) in text.lines().enumerate() {
            let line = line.split('#').next().unwrap().trim();
            if line.is_empty() {
                continue;
            }

            let fields = line.split_whitespace().collect::<Vec<_>>();
            if fields.len() != 3 {
                return Err(format_err!("Config line {}: expected 'serial ip row_start'", i + 1));
            }
            if fields[0] != serial {
                continue;
            }

            let ip = fields[1].parse::<Ipv4Addr>()
                .map_err(|_| format_err!("Config line {}: bad IP {}", i + 1, fields[1]))?;
            let row_start = fields[2].parse::<u32>()
                .map_err(|_| format_err!("Config line {}: bad row_start {}", i + 1, fields[2]))?;
            if row_start + FRAME_HEIGHT > SENSOR_HEIGHT {
                return Err(format_err!("Config line {}: row_start must be 0 to {}", i + 1, SENSOR_HEIGHT - FRAME_HEIGHT));
            }

            return Ok(Self { serial: serial.to_string(), ip, row_start });
        }

        Err(format_err!("No line for camera {} in the config", serial))
    }

    /// dacha's camera id: the first 8 bytes (big-endian) of the SHA-256 of the
    /// serial, as host/rift_mdns.py in riftcv1-mocap names them.
    pub fn camera_id(&self) -> u64 {
        let mut hasher = crypto::sha256::SHA256Hasher::default();
        hasher.update(self.serial.as_bytes());
        let hash = hasher.finish();
        u64::from_be_bytes(hash[0..8].try_into().unwrap())
    }

    /// The id as it appears in mDNS names: CL64 base32, 13 characters.
    pub fn camera_id_string(&self) -> String {
        base_radix::base32_encode_cl64(self.camera_id())
    }

    /// The /dev/v4l/by-id capture node for this camera.
    pub fn device_path(&self) -> String {
        format!("/dev/v4l/by-id/usb-Oculus_VR_Rift_Sensor_{}-video-index0", self.serial)
    }
}
