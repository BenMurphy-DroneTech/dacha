//! TimeSync for a Rift node: dacha's DummyTimeSyncNode, reporting the beacon
//! grid instead of a fixed follower status.
//!
//! Four processes on one Pi can't all steer the Pi's PTP clock, so nothing
//! here syncs: frames are numbered on the beacon grid (grid.rs, slots.rs).
//! The manager still configures one camera as PTP leader and the rest as
//! followers, and its Cameras page shows the leader's basic_client and each
//! follower's follower status. Both report the grid: synced while rift_grid
//! has seen a boundary in the last second, the error is the boundaries' RMS
//! about the fitted line, and the age is the time since the last boundary.

use std::sync::Mutex;

use common::errors::*;
use ptp_proto::ptp::*;

use rift_camera::grid::GridMap;
use crate::slots::monotonic_ns;

/// No boundary for this long and the grid counts as lost.
const STALE_NS: i64 = 1_000_000_000;

pub struct RiftTimeSync {
    config: Mutex<TimeSyncConfig>,
    /// Opened on the first Status that finds rift_grid's file.
    grid: Mutex<Option<GridMap>>,
}

impl RiftTimeSync {
    pub fn create() -> Self {
        Self { config: Mutex::new(TimeSyncConfig::default()), grid: Mutex::new(None) }
    }

    fn grid_status(&self) -> FollowerStatus {
        let mut status = FollowerStatus::default();
        let mut map = self.grid.lock().unwrap();
        if map.is_none() {
            *map = GridMap::open().ok();
        }
        if let Some(grid) = map.as_ref().and_then(|m| m.read()) {
            let age_ns = monotonic_ns() - grid.last_edge_mono_ns;
            status.set_got_sync(age_ns < STALE_NS);
            status.set_last_leader_error(grid.fit_rms_ns * 1e-9);
            status.set_last_sync_age(age_ns as f64 * 1e-9);
        }
        status
    }
}

#[async_trait]
impl TimeSyncService for RiftTimeSync {
    async fn Status(
        &self,
        _request: rpc::ServerRequest<StatusRequest>,
        response: &mut rpc::ServerResponse<StatusResponse>
    ) -> Result<()> {
        let config = self.config.lock().unwrap().clone();
        let mut out = StatusResponse::default();
        if config.has_follower() {
            out.set_follower(self.grid_status());
        }
        if config.has_basic_client() {
            out.set_basic_client(self.grid_status());
        }
        out.set_config(config);
        response.value = out;
        Ok(())
    }

    async fn Configure(
        &self,
        request: rpc::ServerRequest<ConfigureRequest>,
        _response: &mut rpc::ServerResponse<ConfigureResponse>
    ) -> Result<()> {
        *self.config.lock().unwrap() = request.config().clone();
        Ok(())
    }

    async fn Sync(
        &self,
        _request: rpc::ServerRequest<TimeSyncPoint>,
        _response: &mut rpc::ServerResponse<SyncResponse>
    ) -> Result<()> {
        Ok(())
    }
}
