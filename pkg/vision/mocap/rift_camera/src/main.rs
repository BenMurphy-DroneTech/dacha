//! rift_camera: dacha's camera node for one Oculus Rift CV1 sensor.
//!
//! One process per camera. Each binds its own eth0 alias from the per-Pi
//! config, serves dacha's Camera and TimeSync services there on the RPC port
//! (82), answers the manager's mDNS query from that alias, and streams its
//! sensor synced to the beacon.
//!
//!   rift_camera --config=pkg/vision/mocap/rift_camera/config/pi1.cfg --serial=WMTD305350189Q
//!
//! TimeSync is dacha's dummy node: four processes on one Pi can't all steer
//! the Pi's PTP clock, and frames are numbered on the beacon grid instead
//! (first milestone step 4).

#[macro_use]
extern crate common;
#[macro_use]
extern crate macros;

mod bridge;
mod capture;
mod config;
mod mdns;
mod service;

use std::sync::Arc;

use common::errors::*;
use executor_multitask::RootResource;
use mocap_proto::mocap::CameraIntoService;
use ptp_proto::ptp::*;

use crate::capture::{CaptureCounters, CaptureOptions};
use crate::config::RiftCameraConfig;

#[derive(Args)]
struct Args {
    /// The per-Pi config file: `serial ip row_start` per camera.
    config: String,

    /// This camera's serial, as in /dev/v4l/by-id.
    serial: String,

    #[arg(default = 82)]
    rpc_port: u16,

    /// The beacon's frame period.
    #[arg(default = 11111)]
    period_us: u32,

    /// Exposure, set in whole lines: 188 us is 10 lines in the crop.
    #[arg(default = 188)]
    exposure_us: u32,

    /// Analog gain: 1, 2, 4 or 8.
    #[arg(default = 2)]
    gain: u32,
}

#[executor_main]
async fn main() -> Result<()> {
    let args = common::args::parse_args::<Args>()?;

    let text = std::fs::read_to_string(&args.config)
        .map_err(|e| format_err!("{}: {}", args.config, e))?;
    let config = RiftCameraConfig::load(&text, &args.serial)?;
    let camera_id = config.camera_id_string();
    println!("Rift camera {} at {}:{}, id {}", config.serial, config.ip, args.rpc_port, camera_id);

    {
        let (ip, id) = (config.ip, camera_id.clone());
        std::thread::spawn(move || {
            if let Err(e) = mdns::serve(ip, &id) {
                eprintln!("mDNS: {}", e);
                std::process::exit(1);
            }
        });
    }

    let service = RootResource::new();

    let mut server = rpc::Http2Server::new(Some(args.rpc_port));
    server.http_options_mut().ip = net::ip::IPAddress::V4(config.ip.octets());

    let camera = Arc::new(service::RiftCameraService::new(&config, args.period_us, args.exposure_us as f64));
    server.add_service(camera.clone().into_service())?;

    let time_sync = Arc::new(ptp_core::DummyTimeSyncNode::create());
    server.add_service(time_sync.clone().into_service())?;

    let counters = Arc::new(CaptureCounters::default());
    let options = CaptureOptions {
        period_us: args.period_us,
        exposure_us: args.exposure_us as f64,
        gain: args.gain,
    };
    let _capture = executor::spawn(capture::run(config.clone(), options, counters.clone()));

    service.register_dependency(server.start()).await;

    println!("Running...");
    service.wait().await
}
