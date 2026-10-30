//! rift_check: does every frame of the Rift camera nodes merge?
//!
//! Subscribes to each node's ReadBlobs stream exactly as dacha's mocap
//! manager does (manager/src/networking.rs `connect`, inst.rs
//! `camera_read_blobs_thread`), records every response's frame_timestamp and
//! receive time, and reports per node (responses, out-of-order or duplicate
//! timestamps, steps other than one frame) and across nodes (frames seen by
//! all, 3, 2 or 1 of them; the spread of receive times for complete frames,
//! which is the manager's merge wait).
//!
//!   rift_check [--seconds 60] [169.254.10.1 169.254.10.2 ...]
//!
//! Exits non-zero if any frame in the middle of the window (the first and
//! last 0.5 s ignored) lacks a node.

#[macro_use]
extern crate common;
#[macro_use]
extern crate macros;

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use common::errors::*;
use mocap_proto::mocap::*;
use rift_camera::grid::FRAME_TIMESTAMP_STEP_NS;

const DEFAULT_NODES: [&str; 4] = ["169.254.10.1", "169.254.10.2", "169.254.10.3", "169.254.10.4"];
const EDGE: Duration = Duration::from_millis(500);
/// The manager publishes a frame this long after its first response.
const MERGE_TIMEOUT: Duration = Duration::from_millis(20);
const REPORT_EVERY: Duration = Duration::from_secs(10);

#[derive(Default)]
struct NodeLog {
    /// (frame_timestamp, receive time) per response, in arrival order.
    responses: Vec<(u64, Instant)>,
    opened: Option<Instant>,
    error: Option<String>,
}

struct Shared {
    nodes: Vec<Mutex<NodeLog>>,
}

async fn connect(endpoint: &str) -> Result<Arc<CameraStub>> {
    let channel_options: rpc::Http2ChannelOptions = format!("http://{}:82", endpoint)
        .as_str()
        .try_into_result()?;
    let channel = Arc::new(rpc::Http2Channel::create(channel_options).await?);
    Ok(Arc::new(CameraStub::new(channel)))
}

async fn read_blobs(shared: Arc<Shared>, index: usize, endpoint: String) {
    let result = read_blobs_inner(&shared, index, &endpoint).await;
    let msg = match result {
        Ok(()) => "ReadBlobs terminated from the camera".to_string(),
        Err(e) => format!("{}", e),
    };
    eprintln!("node {} ({}): {}", index + 1, endpoint, msg);
    shared.nodes[index].lock().unwrap().error = Some(msg);
}

async fn read_blobs_inner(shared: &Shared, index: usize, endpoint: &str) -> Result<()> {
    let stub = connect(endpoint).await?;

    let mut ctx = rpc::ClientRequestContext::default();
    ctx.http.wait_for_ready = true;

    let req = ReadBlobsRequest::default();
    let mut res_stream = stub.ReadBlobs(&ctx, &req).await;
    shared.nodes[index].lock().unwrap().opened = Some(Instant::now());

    while let Some(res) = res_stream.recv().await {
        let rx_time = Instant::now();
        if res.cameras().len() != 1 {
            eprintln!("node {}: bad blobs format", index + 1);
            continue;
        }
        shared.nodes[index].lock().unwrap().responses.push((res.frame_timestamp(), rx_time));
    }

    res_stream.finish().await?;
    Ok(())
}

struct NodeStats {
    responses: usize,
    out_of_order: usize,
    duplicates: usize,
    /// Steps of a whole number (> 1) of frames, and the frames they skip.
    gaps: usize,
    skipped: u64,
    /// Steps that aren't a whole number of frames.
    odd_steps: Vec<i128>,
}

fn node_stats(responses: &[(u64, Instant)]) -> NodeStats {
    let mut s = NodeStats { responses: responses.len(), out_of_order: 0, duplicates: 0, gaps: 0, skipped: 0,
                            odd_steps: vec![] };
    for w in responses.windows(2) {
        let step = w[1].0 as i128 - w[0].0 as i128;
        if step == 0 {
            s.duplicates += 1;
        } else if step < 0 {
            s.out_of_order += 1;
        } else if step != FRAME_TIMESTAMP_STEP_NS as i128 {
            if step % FRAME_TIMESTAMP_STEP_NS as i128 == 0 {
                s.gaps += 1;
                s.skipped += (step / FRAME_TIMESTAMP_STEP_NS as i128 - 1) as u64;
            } else {
                s.odd_steps.push(step);
            }
        }
    }
    s
}

struct Merge {
    frames: usize,
    /// by_count[k] = frames seen by k nodes.
    by_count: Vec<usize>,
    missing_by_node: Vec<usize>,
    /// Grid frames between the first and last seen that no node reported.
    seen_by_none: u64,
    /// Receive-time spread (last minus first) of complete frames, ms.
    spreads_ms: Vec<f64>,
    last_by_node: Vec<usize>,
    /// Responses arriving more than MERGE_TIMEOUT after their frame's first:
    /// the manager has already published that frame without them.
    late: usize,
}

/// Merges on exactly equal frame_timestamp, over frames whose first receive
/// time lies in [from, to].
fn merge(logs: &[Vec<(u64, Instant)>], from: Instant, to: Instant) -> Merge {
    let n = logs.len();
    let mut frames: BTreeMap<u64, Vec<Option<Instant>>> = BTreeMap::new();
    for (i, log) in logs.iter().enumerate() {
        for &(ts, rx) in log {
            let e = frames.entry(ts).or_insert_with(|| vec![None; n]);
            if e[i].is_none() {
                e[i] = Some(rx);
            }
        }
    }

    let mut m = Merge { frames: 0, by_count: vec![0; n + 1], missing_by_node: vec![0; n], seen_by_none: 0,
                        spreads_ms: vec![], last_by_node: vec![0; n], late: 0 };
    let mut first_ts: Option<u64> = None;
    let mut last_ts = 0;
    for (&ts, rx) in &frames {
        let first = rx.iter().flatten().min().copied().unwrap();
        if first < from || first > to {
            continue;
        }
        first_ts.get_or_insert(ts);
        last_ts = ts;
        m.frames += 1;
        let count = rx.iter().flatten().count();
        m.by_count[count] += 1;
        for (i, r) in rx.iter().enumerate() {
            match r {
                None => m.missing_by_node[i] += 1,
                Some(t) if *t - first > MERGE_TIMEOUT => m.late += 1,
                _ => {}
            }
        }
        if count == n {
            let (last_i, last) = rx.iter().enumerate().map(|(i, r)| (i, r.unwrap())).max_by_key(|x| x.1).unwrap();
            m.last_by_node[last_i] += 1;
            m.spreads_ms.push((last - first).as_secs_f64() * 1e3);
        }
    }
    if let Some(first_ts) = first_ts {
        let span = (last_ts - first_ts) / FRAME_TIMESTAMP_STEP_NS + 1;
        m.seen_by_none = span.saturating_sub(m.frames as u64);
    }
    m.spreads_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    m
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    sorted[((sorted.len() - 1) as f64 * p).round() as usize]
}

fn snapshot(shared: &Shared) -> Vec<Vec<(u64, Instant)>> {
    shared.nodes.iter().map(|n| n.lock().unwrap().responses.clone()).collect()
}

fn first_response(logs: &[Vec<(u64, Instant)>]) -> Option<Instant> {
    logs.iter().filter_map(|l| l.first().map(|x| x.1)).min()
}

#[executor_main]
async fn main() -> Result<()> {
    let mut seconds: u64 = 60;
    let mut endpoints: Vec<String> = vec![];
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if let Some(v) = arg.strip_prefix("--seconds=") {
            seconds = v.parse()?;
        } else if arg == "--seconds" {
            seconds = args.next().ok_or_else(|| err_msg("--seconds needs a value"))?.parse()?;
        } else if arg.starts_with("--") {
            return Err(format_err!("unknown option {}", arg));
        } else {
            endpoints.push(arg);
        }
    }
    if endpoints.is_empty() {
        endpoints = DEFAULT_NODES.iter().map(|s| s.to_string()).collect();
    }
    let n = endpoints.len();

    let shared = Arc::new(Shared { nodes: (0..n).map(|_| Mutex::new(NodeLog::default())).collect() });

    println!("rift_check: {} nodes ({}), {} s", n, endpoints.join(", "), seconds);
    let start = Instant::now();
    let _tasks: Vec<_> = endpoints.iter().enumerate()
        .map(|(i, e)| executor::spawn(read_blobs(shared.clone(), i, e.clone())))
        .collect();

    let end = start + Duration::from_secs(seconds);
    let mut next_report = start + REPORT_EVERY;
    loop {
        let now = Instant::now();
        if now >= end {
            break;
        }
        let wake = next_report.min(end);
        executor::sleep(wake - now).await?;
        if Instant::now() >= next_report && next_report < end {
            let logs = snapshot(&shared);
            let from = next_report - REPORT_EVERY;
            let m = merge(&logs, from.max(first_response(&logs).unwrap_or(from) + EDGE), next_report - EDGE);
            let counts: Vec<String> = logs.iter().map(|l| l.len().to_string()).collect();
            println!("{:>4} s: responses {}; last 10 s: {} frames, {} complete, missing by node {:?}, merge wait median {:.2} ms max {:.2} ms",
                     (next_report - start).as_secs(), counts.join("/"), m.frames, m.by_count[n], m.missing_by_node,
                     percentile(&m.spreads_ms, 0.5), percentile(&m.spreads_ms, 1.0));
            next_report += REPORT_EVERY;
        }
    }

    let stop = Instant::now();
    let logs = snapshot(&shared);

    println!();
    println!("Per node:");
    for (i, log) in logs.iter().enumerate() {
        let s = node_stats(log);
        let node = shared.nodes[i].lock().unwrap();
        let opened = node.opened.map(|t| format!("{:.0} ms", (t - start).as_secs_f64() * 1e3))
            .unwrap_or_else(|| "never".into());
        let first = log.first().map(|x| format!("{:.0} ms", (x.1 - start).as_secs_f64() * 1e3))
            .unwrap_or_else(|| "none".into());
        let mut odd: Vec<String> = s.odd_steps.iter().take(5).map(|x| x.to_string()).collect();
        if s.odd_steps.len() > 5 {
            odd.push("...".into());
        }
        println!("  node {} {}: {} responses ({:.2} fps), opened {}, first {}; {} out of order, {} duplicate, {} gaps skipping {} frames, {} odd steps{}{}",
                 i + 1, endpoints[i], s.responses,
                 if log.len() > 1 { (log.len() - 1) as f64 / (log[log.len() - 1].1 - log[0].1).as_secs_f64() } else { 0.0 },
                 opened, first, s.out_of_order, s.duplicates, s.gaps, s.skipped, s.odd_steps.len(),
                 if odd.is_empty() { String::new() } else { format!(" [{}] ns", odd.join(", ")) },
                 node.error.as_ref().map(|e| format!("; stream ended: {}", e)).unwrap_or_default());
    }

    let Some(first) = first_response(&logs) else {
        println!("No responses from any node.");
        std::process::exit(2);
    };
    let from = first + EDGE;
    let to = stop - EDGE;
    let m = merge(&logs, from, to);

    println!();
    println!("Across nodes ({:.1} s, first and last 0.5 s ignored): {} frame_timestamps seen",
             (to - from).as_secs_f64(), m.frames);
    for k in (1..=n).rev() {
        println!("  seen by {} of {}: {} ({:.3}%)", k, n, m.by_count[k],
                 100.0 * m.by_count[k] as f64 / m.frames.max(1) as f64);
    }
    println!("  grid frames seen by none (between first and last seen): {}", m.seen_by_none);
    let missing: Vec<String> = m.missing_by_node.iter().enumerate()
        .map(|(i, c)| format!("node {} {}", i + 1, c)).collect();
    println!("  missing by node: {}", missing.join(", "));
    println!("  responses > {} ms after their frame's first (the manager would drop them): {}",
             MERGE_TIMEOUT.as_millis(), m.late);
    let last: Vec<String> = m.last_by_node.iter().enumerate()
        .map(|(i, c)| format!("node {} {}", i + 1, c)).collect();
    println!("  merge wait (last minus first receive, complete frames): median {:.3} ms, p99 {:.3} ms, max {:.3} ms",
             percentile(&m.spreads_ms, 0.5), percentile(&m.spreads_ms, 0.99), percentile(&m.spreads_ms, 1.0));
    println!("  last to arrive: {}", last.join(", "));

    let ok = m.frames > 0 && m.by_count[n] == m.frames;
    println!();
    println!("{}", if ok { "PASS: every frame merged" } else { "FAIL: frames lack a node" });
    std::process::exit(if ok { 0 } else { 1 });
}
