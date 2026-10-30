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
//! It also reports each node's blobs per frame (mean, min, max, a histogram
//! of 0..5+) and their mean radius.
//!
//!   rift_check [--seconds 60] [--frames] [--blink[=CMD]] [--blink-every 3]
//!              [169.254.10.1 169.254.10.2 ...]
//!
//! --frames also subscribes to each node's ReadFrames (the MJPEG previews),
//! reports their rate and size, and keeps node N's latest JPEG in
//! /dev/shm/rift_preview_N.jpg.
//!
//! --blink sends CMD (default `blink 0`: every band leaves channel 0, D10,
//! dark in one frame 30 frames ahead) to the band receiver's serial port
//! (/dev/serial/by-id/*SlimeNRF_Receiver*) 3 s into the run and every
//! --blink-every seconds after, up to 2 s before the end; `--blink=rollcall`
//! sends a roll call instead (frames N to N+4, each LED dark where its marker
//! ID has a bit set). A node's usual blob count is the mode of its counts, the
//! nodes whose usual count is above 0 and holds in half their frames or more
//! see the band, and a frame is dark on a node when its count drops below
//! usual. Every dark frame must be dark on all
//! nodes that see the band (the same frame_timestamp).
//!
//! Exits non-zero if any frame in the middle of the window (the first and
//! last 0.5 s ignored) lacks a node, or, with --blink, if a dark frame isn't
//! dark on every node that sees the band or no frame went dark at all.

#[macro_use]
extern crate common;
#[macro_use]
extern crate macros;

use std::collections::BTreeMap;
use std::ffi::CString;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
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

/// One frame of the beacon grid (11,111 us).
const FRAME_PERIOD: Duration = Duration::from_nanos(FRAME_TIMESTAMP_STEP_NS);
const BLINK_FIRST: Duration = Duration::from_secs(3);
/// The last command goes out at least this long before the end.
const BLINK_TAIL: Duration = Duration::from_secs(2);
/// How long the receiver's answer is waited for.
const REPLY_WAIT: Duration = Duration::from_millis(600);
/// The receiver's default frames ahead (riftcv1-receiver CMD_LEAD_FRAMES).
const DEFAULT_LEAD: u32 = 30;

#[derive(Clone, Copy, Default)]
struct FrameBlobs {
    count: u32,
    /// Sum over the blobs with finite radii of (radius_a + radius_b) / 2,
    /// and how many blobs had a radius that isn't finite.
    radius_sum: f64,
    radius_bad: u32,
    rejected: u32,
}

#[derive(Default)]
struct NodeLog {
    /// (frame_timestamp, receive time) per response, in arrival order.
    responses: Vec<(u64, Instant)>,
    /// Each response's blobs, in the same order.
    blobs: Vec<FrameBlobs>,
    opened: Option<Instant>,
    error: Option<String>,
}

/// One band command sent on the receiver's serial port.
struct BlinkSent {
    cmd: String,
    sent: Instant,
    /// Frames ahead, and how many frames from there the command can darken.
    lead: u32,
    span: u32,
    /// The receiver's `CMD ...` answer and when it came, or why there was none.
    reply: std::result::Result<(Instant, String), String>,
    /// Its other lines (`# ...`), e.g. a usage message.
    notes: Vec<String>,
}

struct Shared {
    nodes: Vec<Mutex<NodeLog>>,
    blinks: Mutex<Vec<BlinkSent>>,
    previews: Vec<Mutex<PreviewLog>>,
}

#[derive(Default)]
struct PreviewLog {
    count: u64,
    bytes: u64,
    first: Option<Instant>,
    last: Option<Instant>,
    error: Option<String>,
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
        let results = res.cameras()[0].results();
        let mut fb = FrameBlobs { count: results.blobs().len() as u32, rejected: results.num_rejected_blobs(),
                                  ..Default::default() };
        for b in results.blobs() {
            let r = (b.radius_a() + b.radius_b()) as f64 / 2.0;
            if r.is_finite() {
                fb.radius_sum += r;
            } else {
                fb.radius_bad += 1;
            }
        }
        let mut node = shared.nodes[index].lock().unwrap();
        node.responses.push((res.frame_timestamp(), rx_time));
        node.blobs.push(fb);
    }

    res_stream.finish().await?;
    Ok(())
}

async fn read_frames(shared: Arc<Shared>, index: usize, endpoint: String) {
    if let Err(e) = read_frames_inner(&shared, index, &endpoint).await {
        shared.previews[index].lock().unwrap().error = Some(format!("{}", e));
    }
}

async fn read_frames_inner(shared: &Shared, index: usize, endpoint: &str) -> Result<()> {
    let stub = connect(endpoint).await?;
    let mut ctx = rpc::ClientRequestContext::default();
    ctx.http.wait_for_ready = true;
    let mut res_stream = stub.ReadFrames(&ctx, &ReadFramesRequest::default()).await;
    while let Some(res) = res_stream.recv().await {
        let now = Instant::now();
        let _ = std::fs::write(format!("/dev/shm/rift_preview_{}.jpg", index + 1), res.mjpeg());
        let mut p = shared.previews[index].lock().unwrap();
        p.count += 1;
        p.bytes += res.mjpeg().len() as u64;
        p.first.get_or_insert(now);
        p.last = Some(now);
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

fn first_response(logs: &[Vec<(u64, Instant)>]) -> Option<Instant> {
    logs.iter().filter_map(|l| l.first().map(|x| x.1)).min()
}

/// Each node's responses and blobs, taken together so they line up.
fn snapshot_both(shared: &Shared) -> (Vec<Vec<(u64, Instant)>>, Vec<Vec<FrameBlobs>>) {
    shared.nodes.iter().map(|n| {
        let n = n.lock().unwrap();
        (n.responses.clone(), n.blobs.clone())
    }).unzip()
}

// ---------- blobs ----------

/// The most common blob count (the larger on a tie), or None with no frames.
fn usual_count(blobs: &[FrameBlobs]) -> Option<u32> {
    let mut freq: BTreeMap<u32, usize> = BTreeMap::new();
    for b in blobs {
        *freq.entry(b.count).or_default() += 1;
    }
    freq.iter().max_by_key(|(c, f)| (**f, **c)).map(|(c, _)| *c)
}

fn blob_line(blobs: &[FrameBlobs]) -> String {
    if blobs.is_empty() {
        return "no frames".into();
    }
    let mut hist = [0usize; 6];
    let (mut total, mut rsum, mut bad, mut rejected) = (0u64, 0.0, 0u64, 0u64);
    for b in blobs {
        hist[(b.count as usize).min(5)] += 1;
        total += b.count as u64;
        rsum += b.radius_sum;
        bad += b.radius_bad as u64;
        rejected += b.rejected as u64;
    }
    let good = total - bad;
    let min = blobs.iter().map(|b| b.count).min().unwrap();
    let max = blobs.iter().map(|b| b.count).max().unwrap();
    let hist: Vec<String> = hist.iter().enumerate()
        .map(|(k, c)| format!("{}{}:{}", k, if k == 5 { "+" } else { "" }, c)).collect();
    format!("blobs/frame mean {:.3} min {} max {} usual {}; frames by count {}; mean radius {}; rejected/frame {:.3}",
            total as f64 / blobs.len() as f64, min, max, usual_count(blobs).unwrap(), hist.join(" "),
            match (good, bad) {
                (0, 0) => "-".to_string(),
                (0, b) => format!("- ({} blobs with a radius that isn't finite)", b),
                (g, 0) => format!("{:.2} px", rsum / g as f64),
                (g, b) => format!("{:.2} px ({} blobs with a radius that isn't finite)", rsum / g as f64, b),
            },
            rejected as f64 / blobs.len() as f64)
}

// ---------- the band command (--blink) ----------

fn find_serial() -> Option<String> {
    let dir = std::fs::read_dir("/dev/serial/by-id").ok()?;
    let mut names: Vec<String> = dir
        .filter_map(|e| e.ok())
        .map(|e| e.path().to_string_lossy().into_owned())
        .filter(|p| p.contains("SlimeNRF_Receiver"))
        .collect();
    names.sort();
    names.into_iter().next()
}

/// Frames ahead and frames darkened for a receiver command: `blink <channel>
/// [lead]` or `rollcall [lead]`. Anything else (sleepall above all) is refused.
fn parse_band_command(cmd: &str) -> Result<(u32, u32)> {
    let t: Vec<&str> = cmd.split_whitespace().collect();
    let (lead, span) = match t.first().copied() {
        Some("blink") if t.len() == 2 || t.len() == 3 => {
            if !matches!(t[1], "0" | "1") {
                return Err(format_err!("blink takes channel 0 or 1: {:?}", cmd));
            }
            (t.get(2), 1)
        }
        Some("rollcall") if t.len() <= 2 => (t.get(1), 5),
        _ => return Err(format_err!("--blink takes `blink <0|1> [lead]` or `rollcall [lead]`, not {:?}", cmd)),
    };
    let lead = match lead {
        Some(v) => v.parse::<u32>()?,
        None => DEFAULT_LEAD,
    };
    if !(5..=9000).contains(&lead) {
        return Err(format_err!("frames ahead: 5 to 9000"));
    }
    Ok((lead, span))
}

/// Opens the receiver's port raw, writes the command line, reads lines until
/// its `CMD ...` answer (or REPLY_WAIT) and closes it again.
fn send_band_command(path: &str, cmd: &str, lead: u32, span: u32) -> BlinkSent {
    let mut out = BlinkSent { cmd: cmd.to_string(), sent: Instant::now(), lead, span,
                              reply: Err("no answer".into()), notes: vec![] };
    let cpath = match CString::new(path) {
        Ok(p) => p,
        Err(e) => {
            out.reply = Err(e.to_string());
            return out;
        }
    };
    let fd = unsafe {
        libc::open(cpath.as_ptr(), libc::O_RDWR | libc::O_NOCTTY | libc::O_NONBLOCK | libc::O_CLOEXEC)
    };
    if fd < 0 {
        out.reply = Err(format!("{}: {}", path, std::io::Error::last_os_error()));
        return out;
    }
    let fd = unsafe { OwnedFd::from_raw_fd(fd) };
    let raw = fd.as_raw_fd();
    unsafe {
        let mut t: libc::termios = std::mem::zeroed();
        if libc::tcgetattr(raw, &mut t) == 0 {
            libc::cfmakeraw(&mut t);
            libc::tcsetattr(raw, libc::TCSANOW, &t);
        }
        libc::tcflush(raw, libc::TCIFLUSH);
    }
    let line = format!("{}\n", cmd);
    out.sent = Instant::now();
    let n = unsafe { libc::write(raw, line.as_ptr() as *const libc::c_void, line.len()) };
    if n != line.len() as isize {
        out.reply = Err(format!("write {}: {}", path, std::io::Error::last_os_error()));
        return out;
    }
    let mut buf = Vec::new();
    while out.sent.elapsed() < REPLY_WAIT {
        let mut chunk = [0u8; 1024];
        let n = unsafe { libc::read(raw, chunk.as_mut_ptr() as *mut libc::c_void, chunk.len()) };
        if n <= 0 {
            std::thread::sleep(Duration::from_millis(1));
            continue;
        }
        let now = Instant::now();
        buf.extend_from_slice(&chunk[..n as usize]);
        while let Some(p) = buf.iter().position(|&c| c == b'\n') {
            let l = String::from_utf8_lossy(&buf[..p]).trim().to_string();
            buf.drain(..=p);
            if l.starts_with("CMD ") {
                out.reply = Ok((now, l));
                return out;
            } else if l.starts_with("# ") {
                out.notes.push(l);
            }
        }
    }
    out
}

/// The --blink check: over frames that every band-seeing node reported in
/// [from, to], is each frame that is dark on one of them dark on all of them?
fn blink_check(logs: &[Vec<(u64, Instant)>], blobs: &[Vec<FrameBlobs>], from: Instant, to: Instant,
               start: Instant, blinks: &[BlinkSent]) -> bool {
    let n = logs.len();
    println!();
    println!("Blink check:");
    for (k, b) in blinks.iter().enumerate() {
        let reply = match &b.reply {
            Ok((t, l)) => format!("{} ({:.0} ms after sending)", l, (*t - b.sent).as_secs_f64() * 1e3),
            Err(e) => format!("no CMD answer: {}", e),
        };
        let notes = if b.notes.is_empty() { String::new() } else { format!("; {}", b.notes.join("; ")) };
        println!("  command {} at {:.1} s, {:?}: {}{}", k + 1, (b.sent - start).as_secs_f64(), b.cmd, reply, notes);
    }

    let usual: Vec<Option<u32>> = blobs.iter().map(|b| usual_count(b)).collect();
    // The share of a node's frames with its usual count: below STEADY, the
    // count is noise (stray blobs) and a drop says nothing about the band.
    const STEADY: f64 = 0.5;
    let steady: Vec<f64> = blobs.iter().zip(&usual).map(|(b, u)| match u {
        Some(u) => b.iter().filter(|x| x.count == *u).count() as f64 / b.len() as f64,
        None => 0.0,
    }).collect();
    let see: Vec<usize> = (0..n).filter(|&i| usual[i].unwrap_or(0) > 0 && steady[i] >= STEADY).collect();
    let usual_text: Vec<String> = usual.iter().enumerate()
        .map(|(i, u)| format!("node {} {}", i + 1, match u {
            Some(u) => format!("{} ({:.0}% of frames)", u, 100.0 * steady[i]),
            None => "-".into(),
        }))
        .collect();
    println!("  usual blob count: {}; nodes that see the band (usual above 0 in {:.0}% of frames or more): {:?}",
             usual_text.join(", "), 100.0 * STEADY, see.iter().map(|i| i + 1).collect::<Vec<_>>());
    if see.is_empty() {
        println!("  FAIL: no node sees the band with a steady blob count");
        return false;
    }

    // frame_timestamp -> (first receive time, count per node)
    let mut frames: BTreeMap<u64, (Instant, Vec<Option<u32>>)> = BTreeMap::new();
    for i in 0..n {
        for (&(ts, rx), b) in logs[i].iter().zip(&blobs[i]) {
            let e = frames.entry(ts).or_insert_with(|| (rx, vec![None; n]));
            e.0 = e.0.min(rx);
            if e.1[i].is_none() {
                e.1[i] = Some(b.count);
            }
        }
    }
    let base = frames.keys().next().copied().unwrap_or(0);

    // The command each frame could belong to: its answer's time plus lead
    // frames, give or take two for the latency of the answer and the blobs.
    let near_command = |rx: Instant| -> Option<(usize, f64)> {
        blinks.iter().enumerate().find_map(|(k, b)| {
            let (t, _) = b.reply.as_ref().ok()?;
            let lo = *t + FRAME_PERIOD * (b.lead - 2);
            let hi = *t + FRAME_PERIOD * (b.lead + b.span + 2);
            (rx >= lo && rx <= hi).then(|| (k, (rx - *t).as_secs_f64() / FRAME_PERIOD.as_secs_f64()))
        })
    };

    let (mut judged, mut unjudged, mut dark, mut agreed, mut near) = (0usize, 0usize, 0usize, 0usize, 0usize);
    let mut disagree: Vec<String> = vec![];
    let mut agreed_lines: Vec<String> = vec![];
    for (&ts, (rx, counts)) in &frames {
        if *rx < from || *rx > to {
            continue;
        }
        if see.iter().any(|&i| counts[i].is_none()) {
            unjudged += 1;
            continue;
        }
        judged += 1;
        let dark_on: Vec<usize> = see.iter().copied().filter(|&i| counts[i].unwrap() < usual[i].unwrap()).collect();
        if dark_on.is_empty() {
            continue;
        }
        dark += 1;
        let cmd = near_command(*rx);
        near += cmd.is_some() as usize;
        let counts_text: Vec<String> = counts.iter()
            .map(|c| c.map(|c| c.to_string()).unwrap_or_else(|| "-".into()))
            .collect();
        let line = format!("frame {} (ts {}, {:.2} s): counts {} (node 1..{}), dark on {:?}, {}",
                           (ts - base) / FRAME_TIMESTAMP_STEP_NS, ts, (*rx - start).as_secs_f64(),
                           counts_text.join("/"), n, dark_on.iter().map(|i| i + 1).collect::<Vec<_>>(),
                           match cmd {
                               Some((k, f)) => format!("{:.1} frames after command {}'s answer", f, k + 1),
                               None => "near no command".into(),
                           });
        if dark_on.len() == see.len() {
            agreed += 1;
            agreed_lines.push(line);
        } else {
            disagree.push(line);
        }
    }

    println!("  frames judged {} (not judged, a band-seeing node lacks them: {}); dark {}: on all {} band-seeing nodes {}, on some only {}; near a command {}, elsewhere {}",
             judged, unjudged, dark, see.len(), agreed, disagree.len(), near, dark - near);
    for (title, lines) in [("Dark on every band-seeing node:", &agreed_lines),
                           ("Dark on some nodes but not others:", &disagree)] {
        if lines.is_empty() {
            continue;
        }
        println!("  {}", title);
        for l in lines.iter().take(20) {
            println!("    {}", l);
        }
        if lines.len() > 20 {
            println!("    ... {} more", lines.len() - 20);
        }
    }
    let ok = dark > 0 && disagree.is_empty();
    println!("  {}", if ok {
        "PASS: every dark frame is dark on all nodes that see the band"
    } else if dark == 0 {
        "FAIL: no frame went dark"
    } else {
        "FAIL: nodes disagree on dark frames"
    });
    ok
}

#[executor_main]
async fn main() -> Result<()> {
    let mut seconds: u64 = 60;
    let mut blink: Option<String> = None;
    let mut blink_every: f64 = 3.0;
    let mut frames = false;
    let mut endpoints: Vec<String> = vec![];
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if let Some(v) = arg.strip_prefix("--seconds=") {
            seconds = v.parse()?;
        } else if arg == "--seconds" {
            seconds = args.next().ok_or_else(|| err_msg("--seconds needs a value"))?.parse()?;
        } else if arg == "--frames" {
            frames = true;
        } else if arg == "--blink" {
            blink = Some("blink 0".into());
        } else if let Some(v) = arg.strip_prefix("--blink=") {
            blink = Some(v.trim().to_string());
        } else if let Some(v) = arg.strip_prefix("--blink-every=") {
            blink_every = v.parse()?;
        } else if arg == "--blink-every" {
            blink_every = args.next().ok_or_else(|| err_msg("--blink-every needs a value"))?.parse()?;
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

    let mut blink_setup: Option<(String, u32, u32, String)> = None;
    if let Some(cmd) = &blink {
        let (lead, span) = parse_band_command(cmd)?;
        let port = find_serial().ok_or_else(|| err_msg("--blink: no /dev/serial/by-id/*SlimeNRF_Receiver* port"))?;
        if Duration::from_secs(seconds) < BLINK_FIRST + BLINK_TAIL {
            return Err(err_msg("--blink needs --seconds 5 or more"));
        }
        if !(blink_every >= 0.5) {
            return Err(err_msg("--blink-every: 0.5 s or more"));
        }
        blink_setup = Some((cmd.clone(), lead, span, port));
    }

    let shared = Arc::new(Shared { nodes: (0..n).map(|_| Mutex::new(NodeLog::default())).collect(),
                                   blinks: Mutex::new(vec![]),
                                   previews: (0..n).map(|_| Mutex::new(PreviewLog::default())).collect() });

    println!("rift_check: {} nodes ({}), {} s{}", n, endpoints.join(", "), seconds,
             blink_setup.as_ref().map(|(c, lead, _, port)| format!(", --blink {:?} ({} frames ahead) every {} s on {}",
                                                                 c, lead, blink_every, port))
                 .unwrap_or_default());
    let start = Instant::now();
    let _tasks: Vec<_> = endpoints.iter().enumerate()
        .map(|(i, e)| executor::spawn(read_blobs(shared.clone(), i, e.clone())))
        .collect();
    let _frame_tasks: Vec<_> = endpoints.iter().enumerate().filter(|_| frames)
        .map(|(i, e)| executor::spawn(read_frames(shared.clone(), i, e.clone())))
        .collect();

    let end = start + Duration::from_secs(seconds);
    let mut next_report = start + REPORT_EVERY;
    let mut next_blink = blink_setup.as_ref().map(|_| start + BLINK_FIRST);
    let mut blink_threads = vec![];
    loop {
        let now = Instant::now();
        if now >= end {
            break;
        }
        let wake = next_report.min(end).min(next_blink.unwrap_or(end));
        if wake > now {
            executor::sleep(wake - now).await?;
        }
        if let (Some(t), Some((cmd, lead, span, port))) = (next_blink, &blink_setup) {
            if Instant::now() >= t {
                let (cmd, lead, span, port, shared) = (cmd.clone(), *lead, *span, port.clone(), shared.clone());
                blink_threads.push(std::thread::spawn(move || {
                    let b = send_band_command(&port, &cmd, lead, span);
                    match &b.reply {
                        Ok((_, l)) => println!("  receiver: {}", l),
                        Err(e) => println!("  receiver: no CMD answer to {:?}: {}", cmd, e),
                    }
                    for l in &b.notes {
                        println!("  receiver: {}", l);
                    }
                    shared.blinks.lock().unwrap().push(b);
                }));
                let next = t + Duration::from_secs_f64(blink_every);
                next_blink = (next + BLINK_TAIL <= end).then_some(next);
            }
        }
        if Instant::now() >= next_report && next_report < end {
            let (logs, blobs) = snapshot_both(&shared);
            let from = next_report - REPORT_EVERY;
            let m = merge(&logs, from.max(first_response(&logs).unwrap_or(from) + EDGE), next_report - EDGE);
            let counts: Vec<String> = logs.iter().map(|l| l.len().to_string()).collect();
            let means: Vec<String> = logs.iter().zip(&blobs).map(|(l, b)| {
                let recent: Vec<u32> = l.iter().zip(b).filter(|((_, rx), _)| *rx >= from).map(|(_, b)| b.count).collect();
                format!("{:.2}", recent.iter().sum::<u32>() as f64 / recent.len().max(1) as f64)
            }).collect();
            println!("{:>4} s: responses {}; last 10 s: {} frames, {} complete, missing by node {:?}, merge wait median {:.2} ms max {:.2} ms, blobs/frame {}",
                     (next_report - start).as_secs(), counts.join("/"), m.frames, m.by_count[n], m.missing_by_node,
                     percentile(&m.spreads_ms, 0.5), percentile(&m.spreads_ms, 1.0), means.join("/"));
            next_report += REPORT_EVERY;
        }
    }

    let stop = Instant::now();
    for t in blink_threads {
        let _ = t.join();
    }
    let (logs, blobs) = snapshot_both(&shared);

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

    println!();
    println!("Blobs by node:");
    for (i, b) in blobs.iter().enumerate() {
        println!("  node {} {}: {}", i + 1, endpoints[i], blob_line(b));
    }

    if frames {
        println!();
        println!("Previews by node (ReadFrames):");
        for (i, p) in shared.previews.iter().enumerate() {
            let p = p.lock().unwrap();
            let span = match (p.first, p.last) { (Some(a), Some(b)) => (b - a).as_secs_f64(), _ => 0.0 };
            println!("  node {} {}: {} JPEGs ({:.2} fps), mean {:.0} KB{}", i + 1, endpoints[i], p.count,
                     if p.count > 1 { (p.count - 1) as f64 / span } else { 0.0 },
                     p.bytes as f64 / p.count.max(1) as f64 / 1e3,
                     p.error.as_ref().map(|e| format!("; stream ended: {}", e)).unwrap_or_default());
        }
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

    let blink_ok = if blink_setup.is_some() {
        let mut blinks = std::mem::take(&mut *shared.blinks.lock().unwrap());
        blinks.sort_by_key(|b| b.sent);
        blink_check(&logs, &blobs, from, to, start, &blinks)
    } else {
        true
    };

    let ok = m.frames > 0 && m.by_count[n] == m.frames;
    println!();
    println!("{}", if ok { "PASS: every frame merged" } else { "FAIL: frames lack a node" });
    if blink_setup.is_some() {
        println!("{}", if blink_ok { "PASS: blink check" } else { "FAIL: blink check" });
    }
    std::process::exit(if ok && blink_ok { 0 } else { 1 });
}
