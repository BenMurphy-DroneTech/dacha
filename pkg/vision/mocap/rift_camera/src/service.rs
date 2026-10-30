//! dacha's Camera service for one Rift sensor. Status and Configure keep the
//! manager's config (it re-sends it whenever the epoch differs); exposure and
//! gain stay fixed at the crop's 188 us and 2x. ReadBlobs streams one
//! response per frame placed on the beacon grid; ReadFrames streams the MJPEG
//! previews (preview.rs).

use std::sync::{Arc, Mutex};

use common::bytes::Bytes;
use common::errors::*;
use executor_multitask::BroadcastChannel;
use mocap_proto::mocap::*;

use crate::config::*;
use crate::detect::BlobSettings;
use crate::preview::PreviewSettings;
use vision_proto::vision::CameraIntrinsicsModelProto;

pub struct RiftCameraService {
    status: Mutex<MocapCameraStatus>,
    blobs: Arc<BroadcastChannel<ReadBlobsResponse>>,
    settings: Arc<BlobSettings>,
    frames: Arc<BroadcastChannel<Bytes>>,
    preview: Arc<PreviewSettings>,
}

impl RiftCameraService {
    pub fn new(config: &RiftCameraConfig, period_us: u32, exposure_us: f64,
               blobs: Arc<BroadcastChannel<ReadBlobsResponse>>, settings: Arc<BlobSettings>,
               threshold: u32, filter: BlobFilterConfig,
               frames: Arc<BroadcastChannel<Bytes>>, preview: Arc<PreviewSettings>, mjpeg: MJPEGEncoderConfig) -> Self {
        let mut status = MocapCameraStatus::default();

        let sensor = status.sensor_mut();
        sensor.set_model_name("ar0134 (Rift CV1)");
        sensor.set_max_frame_rate((1_000_000 / period_us) as u32);
        sensor.set_frame_width(FRAME_WIDTH);
        sensor.set_frame_height(FRAME_HEIGHT);
        sensor.set_max_exposure_micros(exposure_us as u32);
        sensor.set_min_exposure_micros(exposure_us as u32);

        status.hardware_config_mut().set_camera_id(config.camera_id());

        let c = status.config_mut();
        c.set_frame_rate((1_000_000 / period_us) as u32);
        c.set_exposure_micros(exposure_us as u32);
        c.set_pixel_threshold(threshold);
        c.set_blob_filter(filter);
        c.set_mjpeg(mjpeg);

        Self { status: Mutex::new(status), blobs, settings, frames, preview }
    }
}

impl RiftCameraService {
    /// The camera's lens as dacha should model it (a pinhole for undistorted
    /// blob centres); the manager copies it into its config when it has none.
    pub fn set_factory_intrinsics(&self, intrinsics: CameraIntrinsicsModelProto) {
        self.status.lock().unwrap().hardware_config_mut().set_factory_intrinsics(intrinsics);
    }
}

#[async_trait]
impl CameraService for RiftCameraService {
    async fn Status(
        &self,
        _request: rpc::ServerRequest<StatusRequest>,
        response: &mut rpc::ServerResponse<MocapCameraStatus>,
    ) -> Result<()> {
        response.value = self.status.lock().unwrap().clone();
        Ok(())
    }

    async fn Configure(
        &self,
        request: rpc::ServerRequest<MocapCameraConfigureRequest>,
        _response: &mut rpc::ServerResponse<ConfigureResponse>,
    ) -> Result<()> {
        let mut status = self.status.lock().unwrap();
        let mut config = request.value.clone();
        // The sensor's settings are fixed for now: report them, not the request.
        config.set_frame_rate(status.config().frame_rate());
        config.set_exposure_micros(status.config().exposure_micros());
        // Dennis's blob detection takes the threshold and filter from here.
        self.settings.set(config.pixel_threshold(), config.blob_filter().clone());
        self.preview.set(config.pixel_threshold(), config.mjpeg().clone());
        status.set_config(config);
        Ok(())
    }

    async fn ReadBlobs(
        &self,
        _request: rpc::ServerRequest<ReadBlobsRequest>,
        response: &mut rpc::ServerStreamResponse<ReadBlobsResponse>,
    ) -> Result<()> {
        // As dacha's camera node: subscribe before the head, then pass every
        // frame on (max_rate is only used by UI previews).
        let mut subscriber = self.blobs.subscribe(1024);
        response.send_head().await?;
        loop {
            let res = subscriber.recv().await?;
            response.send(res).await?;
        }
    }

    async fn ReadFrames(
        &self,
        _request: rpc::ServerRequest<ReadFramesRequest>,
        response: &mut rpc::ServerStreamResponse<ReadFramesResponse>,
    ) -> Result<()> {
        // As dacha's camera node.
        let mut subscriber = self.frames.subscribe(8);
        loop {
            let frame = subscriber.recv().await?;
            let mut res = ReadFramesResponse::default();
            res.set_mjpeg(&frame[..]);
            response.send(res).await?;
        }
    }

    async fn FlashMCU(
        &self,
        _request: rpc::ServerRequest<FlashMCURequest>,
        _response: &mut rpc::ServerResponse<FlashMCUResponse>,
    ) -> Result<()> {
        Ok(())
    }
}
