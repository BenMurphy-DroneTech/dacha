//! The camera's mDNS answer to the manager's `_mocap._tcp.local.` PTR query.
//!
//! The manager takes a camera's address from the source IP of its answer, and
//! dacha's own responder (camera/supervisor) answers from 0.0.0.0, so with
//! four cameras on one Pi every answer would carry the same IP. Each
//! rift_camera process answers from its own eth0 alias instead: one socket
//! hears the group (every process on the Pi gets each query), another bound
//! to the alias sends the answer and hears queries sent to that alias.

use std::net::{Ipv4Addr, SocketAddrV4, UdpSocket};
use std::os::fd::{AsRawFd, FromRawFd};

use common::errors::*;

const GROUP: Ipv4Addr = Ipv4Addr::new(224, 0, 0, 251);
const PORT: u16 = 5353;
const SERVICE: &str = "_mocap._tcp.local";
const TYPE_PTR: u16 = 12;
const TYPE_ANY: u16 = 255;
const CLASS_IN: u16 = 1;

fn setsockopt<T>(fd: i32, level: i32, name: i32, value: T) -> Result<()> {
    let r = unsafe {
        libc::setsockopt(fd, level, name, &value as *const T as *const libc::c_void,
                         std::mem::size_of::<T>() as libc::socklen_t)
    };
    if r != 0 {
        return Err(format_err!("setsockopt {}/{}: {}", level, name, std::io::Error::last_os_error()));
    }
    Ok(())
}

/// A UDP socket on addr:5353 that shares the port with Avahi and the other
/// camera processes.
fn shared_socket(addr: Ipv4Addr) -> Result<UdpSocket> {
    let fd = unsafe { libc::socket(libc::AF_INET, libc::SOCK_DGRAM | libc::SOCK_CLOEXEC, 0) };
    if fd < 0 {
        return Err(format_err!("socket: {}", std::io::Error::last_os_error()));
    }
    let sock = unsafe { UdpSocket::from_raw_fd(fd) };
    setsockopt(fd, libc::SOL_SOCKET, libc::SO_REUSEADDR, 1i32)?;
    setsockopt(fd, libc::SOL_SOCKET, libc::SO_REUSEPORT, 1i32)?;

    let sa = libc::sockaddr_in {
        sin_family: libc::AF_INET as libc::sa_family_t,
        sin_port: PORT.to_be(),
        sin_addr: libc::in_addr { s_addr: u32::from_ne_bytes(addr.octets()) },
        sin_zero: [0; 8],
    };
    let r = unsafe {
        libc::bind(fd, &sa as *const _ as *const libc::sockaddr, std::mem::size_of::<libc::sockaddr_in>() as u32)
    };
    if r != 0 {
        return Err(format_err!("bind {}:{}: {}", addr, PORT, std::io::Error::last_os_error()));
    }
    Ok(sock)
}

fn encode_name(name: &str, out: &mut Vec<u8>) {
    for label in name.trim_end_matches('.').split('.') {
        out.push(label.len() as u8);
        out.extend_from_slice(label.as_bytes());
    }
    out.push(0);
}

/// Reads a (possibly compressed) name at pos; returns it and the position after it.
fn read_name(data: &[u8], mut pos: usize) -> Option<(String, usize)> {
    let mut labels = vec![];
    let mut end = None;
    for _ in 0..64 {
        let n = *data.get(pos)? as usize;
        if n & 0xC0 == 0xC0 {
            end.get_or_insert(pos + 2);
            pos = ((n & 0x3F) << 8) | *data.get(pos + 1)? as usize;
        } else if n == 0 {
            return Some((labels.join("."), end.unwrap_or(pos + 1)));
        } else {
            labels.push(String::from_utf8_lossy(data.get(pos + 1..pos + 1 + n)?).into_owned());
            pos += 1 + n;
        }
    }
    None
}

fn u16_at(data: &[u8], pos: usize) -> Option<u16> {
    Some(u16::from_be_bytes([*data.get(pos)?, *data.get(pos + 1)?]))
}

/// If data is a query for our service: (query id, whether a unicast answer was asked for).
fn parse_query(data: &[u8]) -> Option<(u16, bool)> {
    let id = u16_at(data, 0)?;
    let flags = u16_at(data, 2)?;
    if flags & 0x8000 != 0 {
        return None; // a response, not a query
    }
    let mut pos = 12;
    for _ in 0..u16_at(data, 4)? {
        let (name, next) = read_name(data, pos)?;
        let typ = u16_at(data, next)?;
        let class = u16_at(data, next + 2)?;
        pos = next + 4;
        if name.eq_ignore_ascii_case(SERVICE) && (typ == TYPE_PTR || typ == TYPE_ANY) && class & 0x7FFF == CLASS_IN {
            return Some((id, class & 0x8000 != 0));
        }
    }
    None
}

fn ptr_answer(id: u16, target: &str) -> Vec<u8> {
    let mut out = vec![];
    for v in [id, 0x8400, 1, 1, 0, 0] {
        out.extend_from_slice(&v.to_be_bytes());
    }
    encode_name(SERVICE, &mut out);
    out.extend_from_slice(&TYPE_PTR.to_be_bytes());
    out.extend_from_slice(&CLASS_IN.to_be_bytes());

    let mut rdata = vec![];
    encode_name(target, &mut rdata);
    encode_name(SERVICE, &mut out);
    out.extend_from_slice(&TYPE_PTR.to_be_bytes());
    out.extend_from_slice(&CLASS_IN.to_be_bytes());
    out.extend_from_slice(&120u32.to_be_bytes());
    out.extend_from_slice(&(rdata.len() as u16).to_be_bytes());
    out.extend_from_slice(&rdata);
    out
}

/// Answers `_mocap._tcp.local.` with `camera-<camera_id>._mocap._tcp.local.`
/// from `ip`, forever. Runs on its own thread.
pub fn serve(ip: Ipv4Addr, camera_id: &str) -> Result<()> {
    let target = format!("camera-{}.{}.", camera_id, SERVICE);

    let group = shared_socket(Ipv4Addr::UNSPECIFIED)?;
    // Only the groups this socket joins, not every group joined on the host.
    setsockopt(group.as_raw_fd(), libc::IPPROTO_IP, libc::IP_MULTICAST_ALL, 0i32)?;
    group.join_multicast_v4(&GROUP, &ip)
        .map_err(|e| format_err!("Joining {} on {}: {}", GROUP, ip, e))?;

    let own = shared_socket(ip)?;
    setsockopt(own.as_raw_fd(), libc::IPPROTO_IP, libc::IP_MULTICAST_IF,
               libc::in_addr { s_addr: u32::from_ne_bytes(ip.octets()) })?;

    println!("mDNS: answering {}. as {} from {}", SERVICE, target, ip);

    let mut fds = [
        libc::pollfd { fd: group.as_raw_fd(), events: libc::POLLIN, revents: 0 },
        libc::pollfd { fd: own.as_raw_fd(), events: libc::POLLIN, revents: 0 },
    ];
    let mut buf = [0u8; 9000];

    loop {
        let r = unsafe { libc::poll(fds.as_mut_ptr(), fds.len() as libc::nfds_t, -1) };
        if r < 0 {
            let e = std::io::Error::last_os_error();
            if e.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            return Err(format_err!("poll: {}", e));
        }

        for (i, sock) in [&group, &own].into_iter().enumerate() {
            if fds[i].revents & libc::POLLIN == 0 {
                continue;
            }
            let (n, src) = match sock.recv_from(&mut buf) {
                Ok(v) => v,
                Err(_) => continue,
            };
            let src = match src {
                std::net::SocketAddr::V4(v) => v,
                _ => continue,
            };
            // Only the manager's link-local queries; never our own or another camera's packets.
            if !src.ip().is_link_local() || *src.ip() == ip {
                continue;
            }
            let (id, unicast_wanted) = match parse_query(&buf[..n]) {
                Some(v) => v,
                None => continue,
            };

            // A legacy (one-shot) query from a port other than 5353 gets a unicast
            // answer echoing its id, as the manager's client expects; a full mDNS
            // querier gets the answer on the group unless it asked for unicast.
            let legacy = src.port() != PORT;
            let answer = ptr_answer(if legacy { id } else { 0 }, &target);
            let dest = if legacy || unicast_wanted { src } else { SocketAddrV4::new(GROUP, PORT) };
            let _ = own.send_to(&answer, dest);
        }
    }
}
