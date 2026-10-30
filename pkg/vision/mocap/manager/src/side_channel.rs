use std::collections::HashMap;
use std::sync::Arc;

use common::errors::*;
use common::bytes::Bytes;
use executor::channel;
use executor::channel::oneshot;
use executor::sync::SyncMutex;

/// Carries camera frames to the app's WebSocket connections, each of which
/// registers its own receiver. A side_channel_id names the connection in its
/// top 8 bits (the app sets them, see `side_channel_id`) and the page's own
/// stream in the low 24, so each page gets its own frames however many are
/// open.
pub struct DataSideChannel {
    state: SyncMutex<State>,
}

struct State {
    sockets: HashMap<u32, channel::Sender<Packet>>,
    last_socket: u32,
}

struct Packet {
    stream_id: u32,
    data: Bytes,
    returner: oneshot::Sender<()>,
}

/// One connection's end of the side channel; unregisters when dropped.
pub struct DataSideReceiver {
    owner: Arc<DataSideChannel>,
    socket: u32,
    receiver: channel::Receiver<Packet>,
}

const STREAM_BITS: u32 = 24;
const STREAM_MASK: u32 = (1 << STREAM_BITS) - 1;

/// The side_channel_id the manager sees for a page's stream on a connection.
pub fn side_channel_id(socket: u32, stream_id: u32) -> u32 {
    (socket << STREAM_BITS) | (stream_id & STREAM_MASK)
}

impl DataSideChannel {
    pub fn create() -> Self {
        Self {
            state: SyncMutex::new(State { sockets: HashMap::new(), last_socket: 0 }),
        }
    }

    /// Registers a connection: the receiver its frames arrive on, which knows
    /// its socket number (for `side_channel_id`).
    pub fn register(self: &Arc<Self>) -> Result<DataSideReceiver> {
        let (sender, receiver) = channel::unbounded();
        let socket = self.state.apply(|s| {
            // 1 to 255, skipping any still open (0 is no side channel).
            for _ in 0..255 {
                s.last_socket = s.last_socket % 255 + 1;
                if !s.sockets.contains_key(&s.last_socket) {
                    s.sockets.insert(s.last_socket, sender);
                    return Ok(s.last_socket);
                }
            }
            Err(err_msg("Too many side channel connections"))
        })??;

        Ok(DataSideReceiver { owner: self.clone(), socket, receiver })
    }

    /// NOTE: We intentionally block for the other end to receive the data before returning
    /// so that camera data is processed in a round robin manner across cameras with reasonable
    /// flow control.
    pub async fn push(&self, side_channel_id: u32, data: Bytes) -> Result<()> {
        let socket = side_channel_id >> STREAM_BITS;
        let sender = self.state.apply(|s| s.sockets.get(&socket).cloned())?
            .ok_or_else(|| err_msg("Side channel connection closed"))?;

        let (returner, receiver) = oneshot::channel();

        sender.send(Packet {
            stream_id: side_channel_id & STREAM_MASK,
            data,
            returner
        }).await?;

        let _ = receiver.recv().await;

        Ok(())
    }
}

impl DataSideReceiver {
    pub fn socket(&self) -> u32 {
        self.socket
    }

    /// The next frame: the page's stream id and the data.
    pub async fn recv(&self) -> Result<(u32, Bytes)> {
        let packet = self.receiver.recv().await?;
        packet.returner.send(());
        Ok((packet.stream_id, packet.data))
    }
}

impl Drop for DataSideReceiver {
    fn drop(&mut self) {
        let _ = self.owner.state.apply(|s| s.sockets.remove(&self.socket));
    }
}
