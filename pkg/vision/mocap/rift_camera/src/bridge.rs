//! The Rift sensor's eSP770U bridge, through its UVC extension unit: AR0134
//! registers, the bridge's own registers and the SPI link to the sensor's
//! nRF51 radio. A port of rift_sync_setup (riftcv1-mocap host/rift_lib.py,
//! itself from OpenHMD's esp770u.c and ar0134.c), in its --trigger-only form:
//! it keeps the bridge's 960x576 crop timing and only sets the exposure, gain,
//! row window and external trigger, then gives the radio its address.
//!
//! Every call is a USB control transfer through uvcvideo (UVCIOC_CTRL_QUERY on
//! a second handle to the capture node), so it is slow (about 1 ms) and never
//! runs inside the frame loop except for window moves.

use std::fs::File;
use std::os::fd::AsRawFd;
use std::time::Duration;

use common::errors::*;

const XU_UNIT: u8 = 4;
const SEL_I2C: u8 = 2;      // AR0134 registers
const SEL_REG: u8 = 3;      // bridge registers
const SEL_CONTROL: u8 = 11; // SPI control
const SEL_DATA: u8 = 12;    // SPI data
const UVC_SET_CUR: u8 = 0x01;
const UVC_GET_CUR: u8 = 0x81;
const UVC_GET_LEN: u8 = 0x85;
const UVCIOC_CTRL_QUERY: libc::c_ulong = 0xC010_7521; // _IOWR('u', 0x21, struct uvc_xu_control_query)

/// The beacon's radio address, the same for every system.
pub const RADIO_ADDRESS: [u8; 5] = [0xb0, 0x53, 0xc2, 0x78, 0xc8];
const EXTCLK_MHZ: f64 = 25.0;
const FINE_CLOCKS: u16 = 646;

#[repr(C)]
struct XuQuery {
    unit: u8,
    selector: u8,
    query: u8,
    size: u16,
    data: *mut u8,
}

/// What the setup did, for the log.
#[derive(Debug, Default)]
pub struct SetupReport {
    pub exposure_lines: u16,
    pub exposure_us: f64,
    pub pixel_clock_mhz: f64,
    pub radio_exposure_us: u32,
    pub warnings: u32,
}

pub struct Bridge {
    file: File,
    len: [u16; 16],
    pub warnings: u32,
}

impl Bridge {
    pub fn open(path: &str) -> Result<Self> {
        let file = std::fs::OpenOptions::new().read(true).write(true).open(path)
            .map_err(|e| format_err!("{}: {}", path, e))?;
        Ok(Self { file, len: [0; 16], warnings: 0 })
    }

    /// One UVCIOC_CTRL_QUERY on buf. The bridge answers EBUSY while it finishes
    /// an SPI transfer to the radio, so retry for about 100 ms.
    fn query(&self, sel: u8, query: u8, buf: &mut [u8]) -> Result<()> {
        let mut q = XuQuery { unit: XU_UNIT, selector: sel, query, size: buf.len() as u16, data: buf.as_mut_ptr() };
        for _ in 0..50 {
            let r = unsafe { libc::ioctl(self.file.as_raw_fd(), UVCIOC_CTRL_QUERY, &mut q as *mut XuQuery) };
            if r == 0 {
                return Ok(());
            }
            let e = std::io::Error::last_os_error();
            match e.raw_os_error() {
                Some(libc::EBUSY) => std::thread::sleep(Duration::from_millis(2)),
                Some(libc::ENOBUFS) => return Err(format_err!(
                    "uvcvideo cached a different length for selector {}: replug the sensor (or reload uvcvideo)", sel)),
                _ => return Err(format_err!("UVC selector {} query 0x{:02x}: {}", sel, query, e)),
            }
        }
        Err(format_err!("The bridge stayed busy (selector {})", sel))
    }

    fn set(&self, sel: u8, data: &[u8]) -> Result<()> {
        let mut buf = vec![0u8; self.len[sel as usize] as usize];
        let n = data.len().min(buf.len());
        buf[..n].copy_from_slice(&data[..n]);
        self.query(sel, UVC_SET_CUR, &mut buf)
    }

    fn get(&self, sel: u8) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; self.len[sel as usize] as usize];
        self.query(sel, UVC_GET_CUR, &mut buf)?;
        Ok(buf)
    }

    fn get_len(&self, sel: u8) -> Result<u16> {
        let mut buf = [0u8; 2];
        self.query(sel, UVC_GET_LEN, &mut buf)?;
        Ok(u16::from_le_bytes(buf))
    }

    /// Reads every selector's length as the kernel will hold it. The bridge
    /// reports the SPI data selector's length from the last SPI control command
    /// (1 byte after power-up), and the kernel caches the first length it sees,
    /// so set it to 127 before touching selector 12, then finish that SPI
    /// transaction with an empty poll as every radio command ends.
    pub fn open_links(&mut self) -> Result<()> {
        for sel in [SEL_I2C, SEL_REG, SEL_CONTROL, SEL_DATA] {
            if sel == SEL_DATA {
                self.spi_control(0x81, 127)?;
            }
            let n = self.get_len(sel)
                .map_err(|e| format_err!("Extension unit selector {} not available: {}", sel, e))?;
            if n > 1024 {
                return Err(format_err!("Selector {} has an unexpectedly large length {}", sel, n));
            }
            self.len[sel as usize] = n;
        }
        if self.len[SEL_DATA as usize] != 127 {
            return Err(format_err!(
                "The radio data selector has length {} instead of 127: replug the sensor (or reload uvcvideo)",
                self.len[SEL_DATA as usize]));
        }
        self.set(SEL_DATA, &[0u8; 127])?;
        self.spi_control(0x41, 127)?;
        self.get(SEL_DATA)?;
        Ok(())
    }

    // ---- eSP770U registers ----

    fn esp_read(&self, reg: u8) -> Result<u8> {
        self.set(SEL_REG, &[0x82, 0xf0, reg, 0])?;
        Ok(self.get(SEL_REG)?[1])
    }

    fn esp_write(&mut self, reg: u8, val: u8) -> Result<()> {
        self.set(SEL_REG, &[0x02, 0xf0, reg, val])?;
        let b = self.get(SEL_REG)?;
        if b[0] != 0x02 || b[2] != reg || b[3] != val {
            eprintln!("warning: bridge write 0x{:02x}=0x{:02x} echoed {:02x?}", reg, val, &b[..4]);
            self.warnings += 1;
        }
        Ok(())
    }

    fn esp_init(&mut self) -> Result<()> {
        self.set(SEL_REG, &[0xa0, 0x03, 0x00, 0x00])?;
        let b = self.get(SEL_REG)?;
        if b[0] != 0xa0 || b[1] != 0xb2 {
            eprintln!("warning: bridge a0 check returned {:02x?}", &b[..3]);
            self.warnings += 1;
        }
        self.esp_read(0x5a)?;
        self.esp_write(0x5a, 0x01)?;
        self.esp_read(0x18)?;
        self.esp_write(0x18, 0x0f)?;
        self.esp_read(0x17)?;
        self.esp_write(0x17, 0xed)?;
        self.esp_write(0x17, 0xec)?;
        self.esp_write(0x18, 0x0e)?;
        self.esp_read(0x14)?;
        Ok(())
    }

    // ---- AR0134 registers ----

    pub fn ar_read(&self, reg: u16) -> Result<u16> {
        self.set(SEL_I2C, &[0x86, 0x20, (reg >> 8) as u8, reg as u8, 0, 0])?;
        let b = self.get(SEL_I2C)?;
        Ok((b[2] as u16) << 8 | b[1] as u16)
    }

    pub fn ar_write(&mut self, reg: u16, val: u16) -> Result<()> {
        self.set(SEL_I2C, &[0x06, 0x20, (reg >> 8) as u8, reg as u8, (val >> 8) as u8, val as u8])?;
        let b = self.get(SEL_I2C)?;
        if !(b[0] == 0x06 && b[2] == (reg >> 8) as u8 && b[3] == reg as u8 && b[4] == (val >> 8) as u8 && b[5] == val as u8) {
            eprintln!("warning: AR0134 write 0x{:04x}=0x{:04x} echoed {:02x?}", reg, val, &b[..6]);
            self.warnings += 1;
        }
        Ok(())
    }

    /// The pixel clock from the PLL registers (77.08 MHz in the crop).
    fn pixel_clock_mhz(&self) -> Result<f64> {
        let m = self.ar_read(0x3030)? as f64;  // pll_multiplier
        let n = self.ar_read(0x302e)? as f64;  // pre_pll_clk_div
        let p_pix = self.ar_read(0x302a)? as f64; // vt_pix_clk_div
        let p_sys = self.ar_read(0x302c)? as f64; // vt_sys_clk_div
        if m == 0.0 || n == 0.0 || p_pix == 0.0 || p_sys == 0.0 {
            return Ok(74.25);
        }
        Ok(EXTCLK_MHZ * m / (n * p_pix * p_sys))
    }

    /// Moves the 576-row window to start at `start`, all on one frame (grouped
    /// hold, 0x3022 high byte). The write order never shrinks the window if a
    /// frame starts between the writes: moving down, the end goes first.
    pub fn move_rows(&mut self, start: u16, rows: u16, current: u16) -> Result<()> {
        self.ar_write(0x3022, 0x0100)?;            // grouped_parameter_hold
        if start > current {
            self.ar_write(0x3006, start + rows - 1)?; // y_addr_end
            self.ar_write(0x3002, start)?;            // y_addr_start
        } else {
            self.ar_write(0x3002, start)?;
            self.ar_write(0x3006, start + rows - 1)?;
        }
        self.ar_write(0x3022, 0x0000)
    }

    // ---- the nRF51 radio, over the bridge's SPI link ----

    fn spi_control(&self, a: u8, length: u8) -> Result<()> {
        let mut c = [0u8; 16];
        c[1] = a;
        c[2] = 0x80;
        c[3] = 0x01;
        c[9] = length;
        self.set(SEL_CONTROL, &c)
    }

    fn radio_write(&mut self, cmd: &[u8]) -> Result<()> {
        let mut data = [0u8; 127];
        for (i, v) in cmd.iter().enumerate() {
            data[i] = *v;
            data[126] = data[126].wrapping_sub(*v); // checksum
        }
        self.spi_control(0x81, 127)?;
        self.set(SEL_DATA, &data)?;
        self.spi_control(0x41, 127)?;
        self.get(SEL_DATA)?;
        self.spi_control(0x81, 127)?;
        self.set(SEL_DATA, &[0u8; 127])?;
        self.spi_control(0x41, 127)?;
        let r = self.get(SEL_DATA)?;
        if r[0] != cmd[0] || r[1] != cmd[1] {
            eprintln!("warning: radio command {:02x} {:02x} got reply {:02x} {:02x}", cmd[0], cmd[1], r[0], r[1]);
            self.warnings += 1;
        }
        Ok(())
    }

    fn radio_setup(&mut self, address: &[u8; 5], radio_exposure_us: u32) -> Result<()> {
        let e = radio_exposure_us.to_le_bytes();
        let mut addr = vec![0x40, 0x10];
        addr.extend_from_slice(address);
        self.radio_write(&addr)?;
        self.radio_write(&[0x50, 0x11, e[0], e[1], e[2], e[3], 0x67, 0xff, 0xff, 0xff])?;
        self.radio_write(&[0x61, 0x12])?;
        self.radio_write(&[0x71, 0x85])?;
        self.radio_write(&[0x81, 0x86])?;
        Ok(())
    }

    /// The sync setup for a streaming sensor: bridge init, AR0134 external
    /// trigger with the given exposure and analog gain (1, 2, 4 or 8), auto
    /// exposure off, the row window at row_start, then the radio address with
    /// the real exposure as its radio value so every exposure stays centred on
    /// the frame boundary. Call after STREAMON.
    pub fn sync_setup(&mut self, exposure_us: f64, gain: u32, row_start: u16, rows: u16) -> Result<SetupReport> {
        self.open_links()?;
        self.esp_init()?;
        std::thread::sleep(Duration::from_millis(100));

        let mut chip = 0;
        for _ in 0..10 {
            chip = self.ar_read(0x3000)?;  // chip_version_reg
            if chip == 0x2406 {
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        if chip != 0x2406 {
            return Err(format_err!("AR0134 chip id 0x{:04x}, expected 0x2406", chip));
        }

        // Exposure in whole lines plus 646 fine clocks, at the pixel clock the PLL gives.
        let ll = self.ar_read(0x300c)?;  // line_length_pck
        let fl = self.ar_read(0x300a)?;  // frame_length_lines
        let pc = self.pixel_clock_mhz()?;
        let coarse = (((exposure_us * pc - FINE_CLOCKS as f64) / ll as f64) + 0.5).floor()
            .max(1.0).min((fl - 4) as f64) as u16;
        self.ar_write(0x3012, coarse)?;        // coarse_integration_time
        self.ar_write(0x3014, FINE_CLOCKS)?;   // fine_integration_time

        let g: u16 = match gain { 8.. => 3, 4..=7 => 2, 2..=3 => 1, _ => 0 };
        let dt = self.ar_read(0x30b0)?;        // digital_test: column gain in bits 5:4
        self.ar_write(0x30b0, (dt & !0x0030) | (g << 4))?;

        let ae = self.ar_read(0x3100)?;        // ae_ctrl_reg
        self.ar_write(0x3100, ae & !1)?;

        let current = self.ar_read(0x3002)?;
        if current != row_start {
            self.move_rows(row_start, rows, current)?;
        }

        let coarse = self.ar_read(0x3012)?;
        let fine = self.ar_read(0x3014)?;
        let us = (coarse as f64 * ll as f64 + fine as f64) / pc;
        let radio_exposure_us = (us + 0.5) as u32;

        // reset_register: keep the PLL (bit 11), trigger input on (bit 8), free-running off (bit 2).
        let r = self.ar_read(0x301a)?;
        self.ar_write(0x301a, (r | (1 << 11) | (1 << 8)) & !(1 << 2))?;

        self.radio_setup(&RADIO_ADDRESS, radio_exposure_us)?;

        Ok(SetupReport {
            exposure_lines: coarse,
            exposure_us: us,
            pixel_clock_mhz: pc,
            radio_exposure_us,
            warnings: self.warnings,
        })
    }
}
