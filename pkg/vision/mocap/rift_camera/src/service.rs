//! dacha's Camera service for one Rift sensor. Status and Configure keep the
//! manager's config (it re-sends it whenever the epoch differs); exposure and
//! gain stay fixed at the crop's 188 us and 2x. ReadBlobs streams nothing
//! until blob detection lands (first milestone step 5).

use std::sync::Mutex;
use std::time::Duration;

use common::errors::*;
use mocap_proto::mocap::*;

use crate::config::*;

pub struct RiftCameraService {
    status: Mutex<MocapCameraStatus>,
}

impl RiftCameraService {
    pub fn new(config: &RiftCameraConfig, period_us: u32, exposure_us: f64) -> Self {
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
        c.set_pixel_threshold(35u32);

        Self { status: Mutex::new(status) }
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
        status.set_config(config);
        Ok(())
    }

    async fn ReadBlobs(
        &self,
        _request: rpc::ServerRequest<ReadBlobsRequest>,
        response: &mut rpc::ServerStreamResponse<ReadBlobsResponse>,
    ) -> Result<()> {
        response.send_head().await?;
        loop {
            executor::sleep(Duration::from_secs(3600)).await?;
        }
    }

    async fn ReadFrames(
        &self,
        _request: rpc::ServerRequest<ReadFramesRequest>,
        _response: &mut rpc::ServerStreamResponse<ReadFramesResponse>,
    ) -> Result<()> {
        Ok(())
    }

    async fn FlashMCU(
        &self,
        _request: rpc::ServerRequest<FlashMCURequest>,
        _response: &mut rpc::ServerResponse<FlashMCUResponse>,
    ) -> Result<()> {
        Ok(())
    }
}
