//! rift_camera: dacha's camera node for one Oculus Rift CV1 sensor.
//!
//! One process per camera. Each binds its own eth0 alias from the per-Pi
//! config, serves dacha's Camera and TimeSync services there on the RPC port
//! (82), answers the manager's mDNS query from that alias, and streams its
//! sensor synced to the beacon.
//!
//!   rift_camera --config=pkg/vision/mocap/rift_camera/config/pi1.cfg --serial=WMTD305350189Q
//!
//! TimeSync is dacha's dummy node, reporting the beacon grid (timesync.rs):
//! four processes on one Pi can't all steer the Pi's PTP clock. Frames are
//! numbered on the beacon grid instead, which rift_grid publishes in /dev/shm
//! (see grid.rs and slots.rs).

#[macro_use]
extern crate common;
#[macro_use]
extern crate macros;

mod bridge;
mod capture;
mod config;
mod detect;
mod hotpix;
mod mdns;
mod preview;
mod service;
mod slots;
mod timesync;

use std::sync::Arc;

use common::errors::*;
use executor_multitask::{BroadcastChannel, RootResource};
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

    /// Where the hot-pixel maps are (hot_pixels_<serial>.txt), riftcv1-mocap's data/captures.
    #[arg(default = "/home/ben/riftcv1-mocap/data/captures")]
    hot_pixels_dir: String,
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
    // The manager's streams never end on their own, so a graceful shutdown
    // only ever waits out its timeout (5 s by default) before a stop or restart.
    let stop_wait = std::time::Duration::from_millis(200);
    server.http_options_mut().graceful_shutdown_timeout = stop_wait;
    server.http_options_mut().connection_options_v2.graceful_shutdown_timeout = stop_wait;

    // Until the manager configures them (with the same values).
    let threshold = detect::RIFT_THRESHOLD;
    let mut filter = mocap_proto::mocap::BlobFilterConfig::default();
    protobuf::text::parse_text_proto(detect::RIFT_BLOB_FILTER, &mut filter)?;
    let settings = Arc::new(detect::BlobSettings::new(threshold, filter.clone()));

    // Dennis's preview defaults (5 fps, quality 100, full size) until the manager configures them.
    let mjpeg = mocap_camera_core::MJPEGEncoder::default_config()?;
    let preview_settings = preview::PreviewSettings::new(threshold, mjpeg.clone());
    let frames = Arc::new(BroadcastChannel::default());
    let previews = preview::Previews::start(frames.clone(), preview_settings.clone())?;

    let blobs = Arc::new(BroadcastChannel::default());
    let camera = Arc::new(service::RiftCameraService::new(&config, args.period_us, args.exposure_us as f64,
                                                          blobs.clone(), settings.clone(), threshold, filter,
                                                          frames, preview_settings, mjpeg));
    server.add_service(camera.clone().into_service())?;

    let time_sync = Arc::new(timesync::RiftTimeSync::create());
    server.add_service(time_sync.clone().into_service())?;

    let counters = Arc::new(CaptureCounters::default());
    let options = CaptureOptions {
        period_us: args.period_us,
        exposure_us: args.exposure_us as f64,
        gain: args.gain,
        hot_pixels: format!("{}/hot_pixels_{}.txt", args.hot_pixels_dir, config.serial),
    };
    let _capture = executor::spawn(capture::run(config.clone(), options, counters.clone(), blobs, settings,
                                                camera.clone(), previews));

    service.register_dependency(server.start()).await;

    println!("Running...");
    service.wait().await
}
