use std::sync::{Arc, Weak};
use std::collections::HashMap;

use common::errors::*;
use common::hash::FastHasherBuilder;
use executor::sync::SyncMutex;
use executor::child_task::ChildTask;
use websocket::*;
use reflection::ParseFrom;
use rpc::Channel;


use crate::protocol::*;


/// NOTE: There is a single instance of this per WebSocket connection.
pub struct AppRpcServer {
    shared: Arc<Shared>,
}

struct Shared {
    channel: rpc::LocalChannel,
    socket: Arc<WebSocket>,
    pending_rpcs: SyncMutex<HashMap<usize, ChildTask<()>, FastHasherBuilder>>,
    /// This connection's number in the frames side channel.
    side_channel_socket: u32,
}

impl AppRpcServer {

    pub fn new(service: Arc<dyn rpc::Service>, socket: Arc<WebSocket>, side_channel_socket: u32) -> Self {
        let channel = rpc::LocalChannel::new_json(service);

        Self {
            shared: Arc::new(Shared {
                channel,
                socket,
                pending_rpcs: Default::default(),
                side_channel_socket,
            })
        }
    }

    /// A request naming a side channel stream gets this connection's number
    /// in it, so the frames come back on this connection only.
    fn tag_side_channel(request: String, socket: u32) -> Result<String> {
        if !request.contains("side_channel_id") {
            return Ok(request);
        }

        let mut value = json::parse(&request)?;
        let id = match value.get_field("side_channel_id").and_then(|v| v.get_number()) {
            Some(v) if v > 0.0 => v as u32,
            _ => return Ok(request),
        };
        if let Some(field) = value.get_field_mut("side_channel_id") {
            *field = json::Value::Number(mocap_manager::side_channel::side_channel_id(socket, id) as f64);
        }

        json::stringify(&value)
    }
 
    pub fn handle_message(&self, message_data: &str) -> Result<()> {
        let message_json = json::parse(message_data)?;

        let mut message = WebMessage::parse_from(json::ValueParser::new(&message_json))?;

        if let Some(m) = message.start_rpc.take() {
            let id = m.request_id;
            let task = ChildTask::spawn(Self::rpc_thread(Arc::downgrade(&self.shared), m));

            // TODO: Disallow duplicates.
            self.shared.pending_rpcs.apply(|v| {
                v.insert(id, task);
            })?;
        } else if let Some(m) = message.cancel_rpc.take() {
            self.shared.pending_rpcs.apply(|v| {
                v.remove(&m.request_id);
            })?;
        } else {
            println!("Unknown message type: {:?}", message);
        }

        Ok(())
    }

    async fn rpc_thread(shared: Weak<Shared>, msg: StartRpcMessage) {
        
        let request_id = msg.request_id;

        if let Err(e) = Self::rpc_thread_inner(shared.clone(), msg).await {
            println!("RPC failed: {}", e);
        }

        if let Some(shared) = shared.upgrade() {
            let _ = shared.pending_rpcs.apply(|s| {
                s.remove(&request_id);
            });
        }
    }

    async fn rpc_thread_inner(shared: Weak<Shared>, msg: StartRpcMessage) -> Result<()> {
        let shared_strong = match shared.upgrade() {
            Some(v) => v,
            None => return Ok(())
        };

        let (mut req_stream, mut res_stream) = shared_strong.channel.call_raw(
            &msg.service_name,
            &msg.method_name,
            &rpc::ClientRequestContext::default()
        ).await;

        let request = Self::tag_side_channel(msg.request, shared_strong.side_channel_socket)?;

        // So that there is no cyclic loop preventing this future from getting dropped.
        drop(shared_strong);

        let _ = req_stream.send_bytes(request.into()).await;
        req_stream.close().await;

        // TODO: In most unary rpc cases, we should be able to merge the post_message
        // for data and status into one.

        while let Some(res) = res_stream.recv_bytes().await {            
            if let Some(shared) = shared.upgrade() {
                shared.socket.write_text(
                    WebMessage::build_data_response(
                        msg.request_id,
                        std::str::from_utf8(&res)?
                    ).as_bytes()
                ).await?;
            }
        }

        let res = res_stream.finish().await;

        // TODO: Also do this on error.
        if let Some(shared) = shared.upgrade() {
            shared.socket.write_text(WebMessage::build_result_response(msg.request_id, res).as_bytes()).await?;
        }

        Ok(())
    }

}