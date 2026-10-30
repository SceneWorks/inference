//! The **host TCP collective transport** of the Iris distributed trainers (story sc-25689; moved
//! out of `gen_core_iris::train::dist` in sc-27316 so the contract crate holds no network
//! transport or socket FFI).
//!
//! The backend-neutral half — strategy and mesh resolution, the [`World`] description and its
//! single-world proof, the per-tensor shard layout, the typed [`DistError`] failures and the
//! [`Collective`] trait itself — stays in [`gen_core_iris::train::dist`]; this crate provides
//! [`TcpCollective`], the Candle trainers' transport (CPU / CUDA, including Windows where NCCL
//! does not exist) and the MLX trainer's host side channel: a host-staged transport over
//! `std::net` with bounded socket timeouts — the minimal all-reduce / all-gather / reduce-scatter
//! / broadcast primitive. Deterministic: the coordinator reduces in rank order and returns the
//! same bytes to every member.

use std::cell::Cell;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::time::{Duration, Instant};

use serde_json::{json, Value as JsonValue};

use gen_core::{CancelFlag, Error, Result};
use gen_core_iris::train::dist::{
    assert_single_world, dist_error, reduce_in_rank_order, Collective, DistError, Partition,
    Precision, ReduceOp, World, DIST_TOKEN_ENV,
};

// ---- the host TCP transport ------------------------------------------------------------------

const MAGIC: u64 = 0x4952_4953_4453_5431; // "IRISDST1"

/// Message kinds of the TCP protocol.
const OP_REDUCE: u8 = 1;
const OP_GATHER: u8 = 2;
const OP_BCAST: u8 = 3;
const OP_BARRIER: u8 = 4;
const OP_REDUCE64: u8 = 5;

/// The largest hello (or rendezvous answer) a connection may send before it is admitted.
const HELLO_MAX: usize = 4096;
/// The longest the coordinator waits for one accepted connection's hello before dropping it (a
/// silent connection must not stall the rendezvous of the real ranks).
const HELLO_WAIT: Duration = Duration::from_secs(10);
/// The rendezvous answer that admits a rank.
const ADMITTED: &[u8] = b"ok";
/// The prefix of the answer that refuses an authenticated rank whose world disagrees.
const REFUSED: &[u8] = b"refused: ";

/// A host-staged collective over TCP: rank 0 coordinates (every rank holds one connection to it);
/// each collective is "every rank sends its contribution, the coordinator validates and reduces per
/// group in rank order and answers every rank". Every socket wait is bounded by the world's
/// timeout; an EOF, an error or an expired wait is a [`DistError::RankLost`] naming the peer and
/// the operation; a frame of the wrong kind, sequence, size, op or precision is a
/// [`DistError::Desync`].
///
/// **Frame sizes.** A frame's length prefix is never trusted for allocation: a frame is refused
/// ([`DistError::Desync`]) when it exceeds the bound the receiver knows locally (a hello: 4 KiB; an
/// all-reduce or all-gather contribution: at most the coordinator's own — every caller reduces or
/// gathers whole tensors or `torch.chunk` row shards, and shard 0, which the coordinator holds, is
/// the largest; an all-reduce answer: exactly the caller's own length; a barrier: the tag), every
/// all-reduce contribution must match its group's length, and every payload is read incrementally,
/// so memory grows only with the bytes that actually arrive (a lying prefix on an answer whose size
/// the receiver cannot know — a gather result, a broadcast — ends in the bounded wait).
///
/// **Rendezvous.** The coordinator keeps accepting until every rank joined or the deadline passed;
/// a connection that sends no hello in time, a malformed hello, the wrong protocol, the wrong
/// [`World::token`], an out-of-range rank or a rank that already joined is dropped and the
/// rendezvous continues. An authenticated rank whose world description disagrees is refused (it
/// fails fast with [`DistError::SplitWorld`]) and, when the world never completes, named in the
/// coordinator's [`DistError::SplitWorld`].
#[derive(Debug)]
pub struct TcpCollective {
    world: World,
    /// Coordinator: one stream per peer rank (index = rank, `None` at 0). Peer: the stream to 0.
    peers: Vec<Option<TcpStream>>,
    seq: u64,
    /// Every collective wait: bounded by [`World::timeout`], interrupted by the run's cancel flag
    /// after [`World::cancel_grace`].
    pace: Pace,
}

fn io_lost(rank: usize, op: &str, e: impl std::fmt::Display) -> Error {
    DistError::RankLost {
        rank,
        op: op.to_string(),
        detail: e.to_string(),
    }
    .into_error()
}

/// The shortest admission wait a joined rank is given (the rest of the rendezvous bound, at least
/// this).
const MIN_ADMISSION_WAIT: Duration = Duration::from_millis(200);

/// TCP keepalive (sc-25690): a peer that vanishes without a reset (power loss, a partitioned
/// network) stops answering the kernel's probes, which fails the connection after about
/// `idle + interval x probes` (~90 s) instead of the whole collective timeout. A live peer that is
/// merely busy computing answers the probes from its kernel and is never affected.
const KEEPALIVE_IDLE_SECS: u32 = 30;
const KEEPALIVE_INTERVAL_SECS: u32 = 10;
const KEEPALIVE_PROBES: u32 = 6;

/// How a transport wait is bounded and interrupted.
#[derive(Clone)]
struct Pace {
    /// The longest a wait may go without progress (bytes moving) before the peer is lost.
    limit: Duration,
    /// The run's cancel flag.
    cancel: CancelFlag,
    /// How long waits may continue once a cancel is pending: zero during the rendezvous (no
    /// training state exists yet); during training, long enough for the peers (cancelled too) to
    /// reach the collective cancel agreement at their next step boundary, so a cooperative cancel
    /// still ends in the coordinated durable save ([`World::cancel_grace`]).
    grace: Duration,
    /// Whether a pending cancel may end a wait at all ([`Collective::set_cancel_armed`]).
    armed: Cell<bool>,
    /// When a pending cancel was first observed by any wait of this transport: the grace is one
    /// budget for the whole transport, never renewed per frame or per collective.
    cancel_seen: Cell<Option<Instant>>,
}

impl Pace {
    fn new(limit: Duration, cancel: &CancelFlag, grace: Duration) -> Self {
        Self {
            limit,
            cancel: cancel.clone(),
            grace,
            armed: Cell::new(true),
            cancel_seen: Cell::new(None),
        }
    }

    /// A pace with another bound, sharing nothing else's state.
    fn with_limit(&self, limit: Duration) -> Self {
        Self::new(limit, &self.cancel, self.grace)
    }
}

impl std::fmt::Debug for Pace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pace")
            .field("limit", &self.limit)
            .field("cancelled", &self.cancel.is_cancelled())
            .field("grace", &self.grace)
            .field("armed", &self.armed.get())
            .finish()
    }
}

/// One wait's bookkeeping: an inactivity deadline (reset by progress); the cancel bookkeeping is
/// the transport's ([`Pace::cancel_seen`]).
struct Waiter<'a> {
    pace: &'a Pace,
    deadline: Instant,
    spins: u32,
}

impl<'a> Waiter<'a> {
    fn new(pace: &'a Pace) -> Self {
        Self {
            pace,
            deadline: Instant::now() + pace.limit,
            spins: 0,
        }
    }

    fn progress(&mut self) {
        self.deadline = Instant::now() + self.pace.limit;
        self.spins = 0;
    }

    /// Pause before the next attempt of a call that would block: a few yields (an answer that is
    /// about to arrive costs no sleep), then 1 ms sleeps.
    fn pause(&mut self) {
        if self.spins < 64 {
            self.spins += 1;
            std::thread::yield_now();
        } else {
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    /// A slice passed without progress: keep waiting, or end the wait typed: [`Error::Canceled`]
    /// once a pending cancel outlived the grace, [`DistError::RankLost`] once the bound passed.
    fn idle(&mut self, peer: usize, op: &str) -> Result<()> {
        if self.pace.armed.get() && self.pace.cancel.is_cancelled() {
            let seen = match self.pace.cancel_seen.get() {
                Some(t) => t,
                None => {
                    let now = Instant::now();
                    self.pace.cancel_seen.set(Some(now));
                    now
                }
            };
            if seen.elapsed() >= self.pace.grace {
                return Err(Error::Canceled);
            }
        }
        if Instant::now() >= self.deadline {
            return Err(io_lost(
                peer,
                op,
                format!(
                    "no answer within the {}s transport timeout",
                    self.pace.limit.as_secs_f64()
                ),
            ));
        }
        Ok(())
    }
}

/// The transport sockets are non-blocking: a call that would block returns at once, so a wait
/// observes its bound and the cancel flag between attempts, and no byte is ever lost or duplicated
/// to an expired socket timeout (a timed-out blocking `send` / `recv` leaves a Windows socket in
/// an indeterminate state). Between attempts a waiter yields, then sleeps 1 ms.
///
/// Only `WouldBlock` retries: on a non-blocking socket `TimedOut` means TCP keepalive declared the
/// peer dead, a lost rank (sc-25690).
fn would_block(e: &std::io::Error) -> bool {
    e.kind() == std::io::ErrorKind::WouldBlock
}

fn write_all(s: &mut TcpStream, buf: &[u8], w: &mut Waiter, peer: usize, op: &str) -> Result<()> {
    let mut sent = 0;
    while sent < buf.len() {
        match s.write(&buf[sent..]) {
            Ok(0) => return Err(io_lost(peer, op, "the connection accepts no more bytes")),
            Ok(n) => {
                sent += n;
                w.progress();
            }
            Err(e) if would_block(&e) => {
                w.idle(peer, op)?;
                w.pause();
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(io_lost(peer, op, e)),
        }
    }
    Ok(())
}

/// Fill `buf` completely, keeping every byte that arrived before a call would have blocked
/// (`read_exact` would drop them).
fn read_exact(
    s: &mut TcpStream,
    buf: &mut [u8],
    w: &mut Waiter,
    peer: usize,
    op: &str,
) -> Result<()> {
    let mut filled = 0;
    while filled < buf.len() {
        match s.read(&mut buf[filled..]) {
            Ok(0) => {
                return Err(io_lost(
                    peer,
                    op,
                    format!("connection closed after {filled} of {} bytes", buf.len()),
                ))
            }
            Ok(n) => {
                filled += n;
                w.progress();
            }
            Err(e) if would_block(&e) => {
                w.idle(peer, op)?;
                w.pause();
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(io_lost(peer, op, e)),
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn send_frame(
    s: &mut TcpStream,
    pace: &Pace,
    kind: u8,
    seq: u64,
    payload: &[u8],
    peer: usize,
    op: &str,
) -> Result<()> {
    let mut w = Waiter::new(pace);
    let mut head = Vec::with_capacity(17);
    head.push(kind);
    head.extend_from_slice(&seq.to_le_bytes());
    head.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    write_all(s, &head, &mut w, peer, op)?;
    write_all(s, payload, &mut w, peer, op)?;
    s.flush().map_err(|e| io_lost(peer, op, e))
}

/// Receive one frame of `kind` / `seq` holding at most `max_len` bytes. The length prefix is
/// checked against `max_len` before anything is read, and the payload is read incrementally (no
/// allocation sized by the untrusted prefix).
#[allow(clippy::too_many_arguments)]
fn recv_frame(
    s: &mut TcpStream,
    pace: &Pace,
    kind: u8,
    seq: u64,
    peer: usize,
    op: &str,
    max_len: usize,
) -> Result<Vec<u8>> {
    let mut w = Waiter::new(pace);
    let mut head = [0u8; 17];
    read_exact(s, &mut head, &mut w, peer, op)?;
    let got_kind = head[0];
    let got_seq = u64::from_le_bytes(head[1..9].try_into().expect("8"));
    let len = u64::from_le_bytes(head[9..17].try_into().expect("8"));
    if got_kind != kind || got_seq != seq {
        return Err(DistError::Desync {
            rank: peer,
            expected: format!("{op} #{seq}"),
            got: format!("kind {got_kind} #{got_seq}"),
        }
        .into_error());
    }
    if len > max_len as u64 {
        return Err(DistError::Desync {
            rank: peer,
            expected: format!("{op} #{seq} of at most {max_len} bytes"),
            got: format!("a {len}-byte frame"),
        }
        .into_error());
    }
    let mut payload = Vec::new();
    let mut chunk = vec![0u8; (len as usize).min(1 << 16)];
    while (payload.len() as u64) < len {
        let want = ((len - payload.len() as u64) as usize).min(chunk.len());
        if let Err(e) = read_exact(s, &mut chunk[..want], &mut w, peer, op) {
            let closed = matches!(
                dist_error(&e),
                Some(DistError::RankLost { detail, .. }) if detail.starts_with("connection closed")
            );
            return Err(if closed {
                io_lost(
                    peer,
                    op,
                    format!("connection closed after {} of {len} bytes", payload.len()),
                )
            } else {
                e
            });
        }
        payload.extend_from_slice(&chunk[..want]);
    }
    Ok(payload)
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn bytes_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().expect("4")))
        .collect()
}

fn f64_bytes(v: &[f64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn bytes_f64(b: &[u8]) -> Vec<f64> {
    b.chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().expect("8")))
        .collect()
}

fn op_code(op: ReduceOp) -> u8 {
    match op {
        ReduceOp::Sum => 0,
        ReduceOp::Min => 1,
        ReduceOp::Max => 2,
    }
}

fn op_of(code: u8) -> ReduceOp {
    match code {
        1 => ReduceOp::Min,
        2 => ReduceOp::Max,
        _ => ReduceOp::Sum,
    }
}

fn prec_code(p: Precision) -> u8 {
    match p {
        Precision::Fp32 => 0,
        Precision::Bf16 => 1,
        Precision::Fp16 => 2,
    }
}

fn prec_of(code: u8) -> Precision {
    match code {
        1 => Precision::Bf16,
        2 => Precision::Fp16,
        _ => Precision::Fp32,
    }
}

/// Byte-wise equality whose time does not depend on where the inputs differ.
fn same_secret(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// Why the coordinator did not admit one accepted connection.
enum Rejection {
    /// Not (provably) a rank of this job: silent, malformed, another protocol, the wrong token, an
    /// out-of-range or already-joined rank. Dropped without an answer.
    Stranger(String),
    /// An authenticated rank of this job whose world description disagrees: refused (it fails fast)
    /// and reported if the world never completes.
    Split(String),
    /// The run was cancelled while the hello was awaited.
    Canceled,
}

/// The coordinator's check of one accepted connection's hello.
fn admit(
    s: &mut TcpStream,
    pace: &Pace,
    world: &World,
    peers: &[Option<TcpStream>],
) -> std::result::Result<usize, Rejection> {
    let stranger = |m: String| Rejection::Stranger(m);
    let payload = match recv_frame(s, pace, 0, 0, usize::MAX, "rendezvous", HELLO_MAX) {
        Ok(p) => p,
        Err(Error::Canceled) => return Err(Rejection::Canceled),
        Err(e) => return Err(stranger(format!("no valid hello: {e}"))),
    };
    let v: JsonValue =
        serde_json::from_slice(&payload).map_err(|e| stranger(format!("bad hello: {e}")))?;
    if v["magic"].as_u64() != Some(MAGIC) {
        return Err(stranger("it speaks another protocol".into()));
    }
    let theirs = v["token"].as_str().unwrap_or("");
    let mine = world.token.as_ref().map(|t| t.0.as_str()).unwrap_or("");
    if !same_secret(theirs.as_bytes(), mine.as_bytes()) {
        return Err(stranger(format!(
            "its rendezvous token ({DIST_TOKEN_ENV}) does not match"
        )));
    }
    let r = match v["rank"].as_u64() {
        Some(r) if r >= 1 && (r as usize) < world.world_size => r as usize,
        other => return Err(stranger(format!("out-of-range rank {other:?}"))),
    };
    if peers[r].is_some() {
        return Err(stranger(format!("rank {r} already joined")));
    }
    for (k, mine) in [
        ("world_size", world.world_size),
        ("local_world_size", world.local_world_size),
        ("num_nodes", world.num_nodes),
    ] {
        if v[k].as_u64() != Some(mine as u64) {
            return Err(Rejection::Split(format!(
                "rank {r} declares {k}={} but rank 0 declares {mine}",
                v[k]
            )));
        }
    }
    Ok(r)
}

impl TcpCollective {
    /// Join the job described by `world`: rank 0 listens on `master_addr:master_port` and accepts
    /// until `world_size - 1` ranks joined (dropping every connection that is not a rank of this
    /// job; see the type docs); every other rank connects (retrying) and presents its world
    /// description and [`World::token`].
    ///
    /// The whole rendezvous (accepting, connect retries, the admission answer) is bounded by
    /// [`World::rendezvous_bound`] and polls `cancel` between its attempts: a cancelled
    /// rendezvous is [`Error::Canceled`] at once (there is no training state to save yet). The
    /// returned collective keeps `cancel` for every later wait ([`World::cancel_grace`]).
    pub fn connect(world: &World, cancel: &CancelFlag) -> Result<Self> {
        assert_single_world(world).map_err(DistError::into_error)?;
        let bound = world.rendezvous_bound();
        let deadline = Instant::now() + bound;
        let rendezvous = Pace::new(bound, cancel, Duration::ZERO);
        let pace = Pace::new(world.timeout, cancel, world.cancel_grace);
        let mut peers: Vec<Option<TcpStream>> = (0..world.world_size).map(|_| None).collect();
        let hello = json!({
            "magic": MAGIC,
            "rank": world.rank,
            "world_size": world.world_size,
            "local_world_size": world.local_world_size,
            "num_nodes": world.num_nodes,
            "node_rank": world.node_rank,
            "token": world.token.as_ref().map(|t| t.0.as_str()).unwrap_or(""),
        })
        .to_string();
        if world.world_size == 1 {
            return Ok(Self {
                world: world.clone(),
                peers,
                seq: 0,
                pace,
            });
        }
        if world.rank == 0 {
            let bind = format!("{}:{}", bind_host(&world.master_addr), world.master_port);
            let listener = TcpListener::bind(&bind)
                .map_err(|e| io_lost(0, "rendezvous bind", format!("{bind}: {e}")))?;
            listener
                .set_nonblocking(true)
                .map_err(|e| io_lost(0, "rendezvous", e))?;
            let hello_pace = rendezvous.with_limit(bound.min(HELLO_WAIT));
            let mut joined = 1usize;
            let mut dropped: Vec<String> = Vec::new();
            let mut split: Vec<String> = Vec::new();
            while joined < world.world_size {
                if cancel.is_cancelled() {
                    return Err(Error::Canceled);
                }
                match listener.accept() {
                    Ok((mut s, from)) => {
                        let ready = configure(&s).is_ok();
                        let verdict = if ready {
                            admit(&mut s, &hello_pace, world, &peers)
                        } else {
                            Err(Rejection::Stranger("socket setup failed".into()))
                        };
                        match verdict {
                            Ok(r) => {
                                peers[r] = Some(s);
                                joined += 1;
                            }
                            Err(Rejection::Canceled) => return Err(Error::Canceled),
                            Err(Rejection::Stranger(why)) => {
                                eprintln!(
                                    "iris distributed: rendezvous dropped a connection from \
                                     {from}: {why}"
                                );
                                dropped.push(format!("{from}: {why}"));
                            }
                            Err(Rejection::Split(why)) => {
                                let mut answer = REFUSED.to_vec();
                                answer.extend_from_slice(why.as_bytes());
                                let _ = send_frame(
                                    &mut s,
                                    &hello_pace,
                                    0,
                                    0,
                                    &answer,
                                    usize::MAX,
                                    "rendezvous",
                                );
                                eprintln!("iris distributed: rendezvous refused {from}: {why}");
                                split.push(why);
                            }
                        }
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        if Instant::now() > deadline {
                            if !split.is_empty() {
                                return Err(DistError::SplitWorld(split.join("; ")).into_error());
                            }
                            let missing: Vec<usize> = (1..world.world_size)
                                .filter(|&r| peers[r].is_none())
                                .collect();
                            let mut detail =
                                format!("ranks {missing:?} never joined within {bound:?}");
                            if !dropped.is_empty() {
                                detail.push_str(&format!(
                                    "; dropped {} connection(s): {}",
                                    dropped.len(),
                                    dropped.join("; ")
                                ));
                            }
                            return Err(DistError::RankLost {
                                rank: missing[0],
                                op: "rendezvous".into(),
                                detail,
                            }
                            .into_error());
                        }
                        std::thread::sleep(Duration::from_millis(20));
                    }
                    Err(e) => return Err(io_lost(0, "rendezvous accept", e)),
                }
            }
            // Release every peer only once the whole world joined.
            for (r, peer) in peers.iter_mut().enumerate().skip(1) {
                let s = peer.as_mut().expect("joined");
                send_frame(s, &rendezvous, 0, 0, ADMITTED, r, "rendezvous")?;
            }
        } else {
            let addr = resolve(&world.master_addr, world.master_port)?;
            let mut stream = loop {
                if cancel.is_cancelled() {
                    return Err(Error::Canceled);
                }
                match TcpStream::connect_timeout(&addr, Duration::from_millis(500)) {
                    Ok(s) => break s,
                    Err(e) => {
                        if Instant::now() > deadline {
                            return Err(io_lost(0, "rendezvous connect", e));
                        }
                        std::thread::sleep(Duration::from_millis(50));
                    }
                }
            };
            configure(&stream)?;
            send_frame(
                &mut stream,
                &rendezvous,
                0,
                0,
                hello.as_bytes(),
                0,
                "rendezvous",
            )?;
            // The admission answer comes once the whole world joined: bounded by what is left of
            // the rendezvous.
            let left = rendezvous.with_limit(
                deadline
                    .saturating_duration_since(Instant::now())
                    .max(MIN_ADMISSION_WAIT),
            );
            let answer = recv_frame(&mut stream, &left, 0, 0, 0, "rendezvous", HELLO_MAX)?;
            if answer != ADMITTED {
                let why = answer
                    .strip_prefix(REFUSED)
                    .map(|w| String::from_utf8_lossy(w).into_owned())
                    .unwrap_or_else(|| "rendezvous refused".into());
                return Err(DistError::SplitWorld(why).into_error());
            }
            peers[0] = Some(stream);
        }
        Ok(Self {
            world: world.clone(),
            peers,
            seq: 0,
            pace,
        })
    }

    /// When this transport first observed a pending cancel ([`World::cancel_grace`] is one budget
    /// from then on); a collective layered on this one (MLX's) shares it.
    #[doc(hidden)]
    pub fn cancel_first_seen(&self) -> &Cell<Option<Instant>> {
        &self.pace.cancel_seen
    }

    fn next_seq(&mut self) -> u64 {
        self.seq += 1;
        self.seq
    }

    /// One exchange. A peer sends `mine` and receives its answer (at most `answer_max` bytes). The
    /// coordinator receives every member's payload (at most `member_max(rank)` bytes), validates
    /// each with `check(rank, payloads)` (an `Err` names what the member sent → a
    /// [`DistError::Desync`]), then answers every rank with `answer(rank, payloads)`.
    #[allow(clippy::too_many_arguments)]
    fn coordinate(
        &mut self,
        kind: u8,
        mine: Vec<u8>,
        op: &str,
        member_max: impl Fn(usize) -> usize,
        answer_max: usize,
        check: impl Fn(usize, &[Vec<u8>]) -> std::result::Result<(), String>,
        answer: impl Fn(usize, &[Vec<u8>]) -> Vec<u8>,
    ) -> Result<Vec<u8>> {
        let seq = self.next_seq();
        if self.world.world_size == 1 {
            return Ok(answer(0, &[mine]));
        }
        let pace = &self.pace;
        if self.world.rank != 0 {
            let s = self.peers[0].as_mut().expect("connected");
            send_frame(s, pace, kind, seq, &mine, 0, op)?;
            return recv_frame(s, pace, kind, seq, 0, op, answer_max);
        }
        let mut payloads = vec![mine];
        for r in 1..self.world.world_size {
            let s = self.peers[r].as_mut().expect("connected");
            payloads.push(recv_frame(s, pace, kind, seq, r, op, member_max(r))?);
        }
        for r in 1..self.world.world_size {
            check(r, &payloads).map_err(|got| {
                DistError::Desync {
                    rank: r,
                    expected: format!("{op} #{seq} matching its group"),
                    got,
                }
                .into_error()
            })?;
        }
        for r in 1..self.world.world_size {
            let out = answer(r, &payloads);
            let s = self.peers[r].as_mut().expect("connected");
            send_frame(s, &self.pace, kind, seq, &out, r, op)?;
        }
        Ok(answer(0, &payloads))
    }
}

/// The coordinator's check of an all-reduce contribution: its `header` op / precision bytes must
/// equal rank 0's, and its length the first member's of its own reduction group (sub-group
/// reductions — hybrid sharding's replicate groups — reduce row shards of different lengths per
/// group; `groups = None` is the whole world).
fn same_reduce(
    header: usize,
    groups: Option<Partition>,
) -> impl Fn(usize, &[Vec<u8>]) -> std::result::Result<(), String> {
    move |r, payloads| {
        let theirs = &payloads[r];
        let lead = groups
            .as_ref()
            .and_then(|g| g.members(r).first().copied())
            .unwrap_or(0);
        let want = &payloads[lead];
        if theirs.len() != want.len() || theirs.len() < header {
            return Err(format!(
                "{} bytes of values (rank {lead} of its group has {})",
                theirs.len().saturating_sub(header),
                want.len().saturating_sub(header)
            ));
        }
        if theirs[..header] != payloads[0][..header] {
            return Err(format!(
                "op/precision bytes {:?} (rank 0 has {:?})",
                &theirs[..header],
                &payloads[0][..header]
            ));
        }
        Ok(())
    }
}

fn bind_host(addr: &str) -> String {
    if addr.contains(':') && !addr.starts_with('[') {
        format!("[{addr}]")
    } else {
        addr.to_string()
    }
}

fn resolve(host: &str, port: u16) -> Result<SocketAddr> {
    (bind_host(host).as_str(), port)
        .to_socket_addrs()
        .map_err(|e| io_lost(0, "rendezvous resolve", e))?
        .next()
        .ok_or_else(|| io_lost(0, "rendezvous resolve", format!("{host} has no address")))
}

/// Every transport socket: no Nagle delay, non-blocking (the waits poll their bound and the cancel
/// flag between attempts, [`would_block`]), and TCP keepalive.
fn configure(s: &TcpStream) -> Result<()> {
    s.set_nodelay(true)
        .map_err(|e| io_lost(0, "configure", e))?;
    s.set_nonblocking(true)
        .map_err(|e| io_lost(0, "configure", e))?;
    if let Err(e) = keepalive::enable(s) {
        // Never fatal: a socket without keepalive still works; only a vanished peer is then
        // detected at the transport timeout instead of by the probes.
        eprintln!("iris distributed: TCP keepalive unavailable on this socket ({e})");
    }
    Ok(())
}

/// `SO_KEEPALIVE` with the probe timing of [`KEEPALIVE_IDLE_SECS`] / [`KEEPALIVE_INTERVAL_SECS`] /
/// [`KEEPALIVE_PROBES`] (std exposes none of it). Only `SO_KEEPALIVE` itself is required; the
/// timing is best effort (sc-25690): a host without the per-socket TCP options (Windows before
/// 10 1709 / Server 2019) gets the timing through `SIO_KEEPALIVE_VALS` (all but the probe count,
/// which is the system's), else keeps the system's timing.
mod keepalive {
    use super::{KEEPALIVE_IDLE_SECS, KEEPALIVE_INTERVAL_SECS, KEEPALIVE_PROBES};
    use std::net::TcpStream;

    /// One socket option: `(level, name, value)`.
    pub(super) type Set<'a> = dyn Fn(i32, i32, u32) -> std::io::Result<()> + 'a;

    /// The option numbers of one platform.
    pub(super) struct Opts {
        pub sol_socket: i32,
        pub so_keepalive: i32,
        pub ipproto_tcp: i32,
        pub idle: i32,
        pub interval: i32,
        pub probes: i32,
    }

    /// Enable keepalive through `set`; `Ok(false)` when the timing could not be applied (the
    /// caller may fall back), `Err` only when `SO_KEEPALIVE` itself failed.
    pub(super) fn apply(set: &Set, o: &Opts) -> std::io::Result<bool> {
        set(o.sol_socket, o.so_keepalive, 1)?;
        let timed = [
            (o.idle, KEEPALIVE_IDLE_SECS),
            (o.interval, KEEPALIVE_INTERVAL_SECS),
            (o.probes, KEEPALIVE_PROBES),
        ]
        .iter()
        .all(|&(name, v)| set(o.ipproto_tcp, name, v).is_ok());
        Ok(timed)
    }

    #[cfg(unix)]
    pub fn enable(s: &TcpStream) -> std::io::Result<()> {
        use std::os::fd::AsRawFd;
        let fd = s.as_raw_fd();
        let set = |level: i32, name: i32, v: u32| -> std::io::Result<()> {
            let v = v as libc::c_int;
            // SAFETY: a valid socket descriptor and a pointer/length to one live c_int.
            let r = unsafe {
                libc::setsockopt(
                    fd,
                    level,
                    name,
                    (&v as *const libc::c_int).cast(),
                    std::mem::size_of::<libc::c_int>() as libc::socklen_t,
                )
            };
            if r == 0 {
                Ok(())
            } else {
                Err(std::io::Error::last_os_error())
            }
        };
        let opts = Opts {
            sol_socket: libc::SOL_SOCKET,
            so_keepalive: libc::SO_KEEPALIVE,
            ipproto_tcp: libc::IPPROTO_TCP,
            #[cfg(target_vendor = "apple")]
            idle: libc::TCP_KEEPALIVE,
            #[cfg(not(target_vendor = "apple"))]
            idle: libc::TCP_KEEPIDLE,
            interval: libc::TCP_KEEPINTVL,
            probes: libc::TCP_KEEPCNT,
        };
        if !apply(&set, &opts)? {
            eprintln!(
                "iris distributed: TCP keepalive timing unavailable on this host; the system's \
                 keepalive timing applies"
            );
        }
        Ok(())
    }

    #[cfg(windows)]
    pub fn enable(s: &TcpStream) -> std::io::Result<()> {
        use std::os::windows::io::AsRawSocket;
        use windows_sys::Win32::Networking::WinSock::{
            setsockopt, tcp_keepalive, WSAIoctl, IPPROTO_TCP, SIO_KEEPALIVE_VALS, SOCKET,
            SOL_SOCKET, SO_KEEPALIVE, TCP_KEEPCNT, TCP_KEEPIDLE, TCP_KEEPINTVL,
        };
        let sock = s.as_raw_socket() as SOCKET;
        let set = |level: i32, name: i32, v: u32| -> std::io::Result<()> {
            let v = v as i32;
            // SAFETY: a valid socket handle and a pointer/length to one live i32 (Winsock's DWORD
            // / BOOL options).
            let r = unsafe {
                setsockopt(
                    sock,
                    level,
                    name,
                    (&v as *const i32).cast(),
                    std::mem::size_of::<i32>() as i32,
                )
            };
            if r == 0 {
                Ok(())
            } else {
                Err(std::io::Error::last_os_error())
            }
        };
        let opts = Opts {
            sol_socket: SOL_SOCKET,
            so_keepalive: SO_KEEPALIVE,
            ipproto_tcp: IPPROTO_TCP,
            idle: TCP_KEEPIDLE,
            interval: TCP_KEEPINTVL,
            probes: TCP_KEEPCNT,
        };
        if apply(&set, &opts)? {
            return Ok(());
        }
        // Pre-1709 Windows: the per-socket timing through the keepalive ioctl (milliseconds; the
        // probe count stays the system's).
        let vals = tcp_keepalive {
            onoff: 1,
            keepalivetime: KEEPALIVE_IDLE_SECS * 1000,
            keepaliveinterval: KEEPALIVE_INTERVAL_SECS * 1000,
        };
        let mut returned = 0u32;
        // SAFETY: a valid socket, an input buffer of one live `tcp_keepalive`, no output buffer,
        // no overlapped I/O (a blocking-style ioctl that completes at once).
        let r = unsafe {
            WSAIoctl(
                sock,
                SIO_KEEPALIVE_VALS,
                (&vals as *const tcp_keepalive).cast(),
                std::mem::size_of::<tcp_keepalive>() as u32,
                std::ptr::null_mut(),
                0,
                &mut returned,
                std::ptr::null_mut(),
                None,
            )
        };
        if r != 0 {
            eprintln!(
                "iris distributed: TCP keepalive timing unavailable on this host ({}); the \
                 system's keepalive timing applies",
                std::io::Error::last_os_error()
            );
        }
        Ok(())
    }

    #[cfg(not(any(unix, windows)))]
    pub fn enable(_: &TcpStream) -> std::io::Result<()> {
        Ok(())
    }
}

impl Collective for TcpCollective {
    fn world(&self) -> &World {
        &self.world
    }

    fn all_reduce(
        &mut self,
        data: &mut [f32],
        op: ReduceOp,
        precision: Precision,
        groups: &Partition,
    ) -> Result<()> {
        let mut msg = vec![op_code(op), prec_code(precision)];
        msg.extend(f32_bytes(data));
        let len = msg.len();
        let parts = groups.clone();
        let out = self.coordinate(
            OP_REDUCE,
            msg,
            "all_reduce",
            |_| len,
            4 * data.len(),
            same_reduce(2, Some(groups.clone())),
            move |rank, payloads| {
                let members = parts.members(rank);
                let vals: Vec<Vec<f32>> = members
                    .iter()
                    .map(|&m| bytes_f32(&payloads[m][2..]))
                    .collect();
                let refs: Vec<&[f32]> = vals.iter().map(Vec::as_slice).collect();
                let op = op_of(payloads[0][0]);
                let prec = prec_of(payloads[0][1]);
                f32_bytes(&reduce_in_rank_order(&refs, op, prec))
            },
        )?;
        if out.len() != 4 * data.len() {
            return Err(DistError::Desync {
                rank: self.world.rank,
                expected: format!("{} elements", data.len()),
                got: format!("{} bytes", out.len()),
            }
            .into_error());
        }
        data.copy_from_slice(&bytes_f32(&out));
        Ok(())
    }

    /// Contributions are `torch.chunk` row shards of one tensor (every caller's use): none may be
    /// larger than the coordinator's own (shard 0, the largest) — a larger one is a
    /// [`DistError::Desync`].
    fn all_gather(&mut self, data: &[f32], groups: &Partition) -> Result<Vec<Vec<f32>>> {
        let parts = groups.clone();
        let mine = f32_bytes(data);
        let bound = mine.len();
        let out = self.coordinate(
            OP_GATHER,
            mine,
            "all_gather",
            |_| bound,
            usize::MAX,
            |r, payloads: &[Vec<u8>]| {
                let theirs = &payloads[r];
                if theirs.len().is_multiple_of(4) {
                    Ok(())
                } else {
                    Err(format!("{} bytes (not whole f32 values)", theirs.len()))
                }
            },
            move |rank, payloads| {
                let mut buf = Vec::new();
                for m in parts.members(rank) {
                    buf.extend((payloads[m].len() as u64 / 4).to_le_bytes());
                    buf.extend_from_slice(&payloads[m]);
                }
                buf
            },
        )?;
        let malformed = || {
            DistError::Desync {
                rank: 0,
                expected: "an all_gather answer of length-prefixed f32 runs".into(),
                got: format!("{} malformed bytes", out.len()),
            }
            .into_error()
        };
        let mut res = Vec::new();
        let mut i = 0;
        while i < out.len() {
            let head = out.get(i..i + 8).ok_or_else(malformed)?;
            let n = u64::from_le_bytes(head.try_into().expect("8")) as usize;
            i += 8;
            let end = n
                .checked_mul(4)
                .and_then(|b| b.checked_add(i))
                .filter(|&e| e <= out.len())
                .ok_or_else(malformed)?;
            res.push(bytes_f32(&out[i..end]));
            i = end;
        }
        Ok(res)
    }

    fn broadcast(&mut self, data: &mut Vec<u8>, root: usize) -> Result<()> {
        let mut mine = (root as u64).to_le_bytes().to_vec();
        if self.world.rank == root {
            mine.extend_from_slice(data);
        }
        let out = self.coordinate(
            OP_BCAST,
            mine,
            "broadcast",
            |r| if r == root { usize::MAX } else { 8 },
            usize::MAX,
            |r, payloads: &[Vec<u8>]| {
                let theirs = &payloads[r];
                if theirs.get(..8) == Some(&payloads[0][..8]) {
                    Ok(())
                } else {
                    Err(format!(
                        "a broadcast from another root ({:?})",
                        theirs.get(..8)
                    ))
                }
            },
            move |_, payloads| payloads[root][8..].to_vec(),
        )?;
        *data = out;
        Ok(())
    }

    fn barrier(&mut self, tag: &str) -> Result<()> {
        let op = format!("barrier {tag}");
        let len = tag.len();
        self.coordinate(
            OP_BARRIER,
            tag.as_bytes().to_vec(),
            &op,
            |_| len,
            0,
            |r, payloads: &[Vec<u8>]| {
                let theirs = &payloads[r];
                if *theirs == payloads[0] {
                    Ok(())
                } else {
                    Err(format!("barrier {:?}", String::from_utf8_lossy(theirs)))
                }
            },
            |_, _| Vec::new(),
        )
        .map(|_| ())
    }

    fn all_reduce_f64(&mut self, data: &mut [f64], op: ReduceOp) -> Result<()> {
        let mut msg = vec![op_code(op)];
        msg.extend(f64_bytes(data));
        let len = msg.len();
        let out = self.coordinate(
            OP_REDUCE64,
            msg,
            "all_reduce_f64",
            |_| len,
            8 * data.len(),
            same_reduce(1, None),
            |_, payloads| {
                let vals: Vec<Vec<f64>> = payloads.iter().map(|p| bytes_f64(&p[1..])).collect();
                let mut acc = vals[0].clone();
                for v in &vals[1..] {
                    for (a, b) in acc.iter_mut().zip(v) {
                        *a = match op_of(payloads[0][0]) {
                            ReduceOp::Min => a.min(*b),
                            ReduceOp::Max => a.max(*b),
                            ReduceOp::Sum => *a + b,
                        };
                    }
                }
                f64_bytes(&acc)
            },
        )?;
        if out.len() != 8 * data.len() {
            return Err(DistError::Desync {
                rank: self.world.rank,
                expected: format!("{} elements", data.len()),
                got: format!("{} bytes", out.len()),
            }
            .into_error());
        }
        data.copy_from_slice(&bytes_f64(&out));
        Ok(())
    }

    fn set_cancel_armed(&mut self, armed: bool) -> bool {
        self.pace.armed.replace(armed)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use gen_core_iris::train::dist::{parse_dist, DistToken};

    fn world(rank: usize, n: usize, port: u16) -> World {
        World {
            rank,
            local_rank: rank,
            world_size: n,
            local_world_size: n,
            node_rank: 0,
            num_nodes: 1,
            master_addr: "127.0.0.1".into(),
            master_port: port,
            expected_nodes: None,
            transport: "tcp".into(),
            hostfile: None,
            timeout: Duration::from_secs(10),
            rendezvous_timeout: Duration::from_secs(10),
            cancel_grace: Duration::from_secs(2),
            token: None,
            share_device: false,
        }
    }

    /// A raw-frame test's wait: 5 s, never cancelled.
    fn test_pace() -> Pace {
        Pace::new(Duration::from_secs(5), &CancelFlag::new(), Duration::ZERO)
    }

    /// A 5 s wait whose run is already cancelled (no grace).
    fn cancelled_pace() -> Pace {
        Pace::new(Duration::from_secs(5), &cancelled(), Duration::ZERO)
    }

    fn free_port() -> u16 {
        TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port()
    }

    #[test]
    fn tcp_collectives_reduce_gather_and_broadcast_across_processes_threads() {
        let port = free_port();
        let handles: Vec<_> = (0..3)
            .map(|r| {
                std::thread::spawn(move || {
                    let mut c =
                        TcpCollective::connect(&world(r, 3, port), &CancelFlag::new()).unwrap();
                    let mut v = vec![r as f32 + 1.0, 10.0 * r as f32];
                    c.all_reduce(&mut v, ReduceOp::Sum, Precision::Fp32, &Partition::whole(3))
                        .unwrap();
                    // groups {0, 2} and {1}
                    let groups = Partition(vec![0, 1, 0]);
                    let mut g = vec![r as f32];
                    c.all_reduce(&mut g, ReduceOp::Max, Precision::Fp32, &groups)
                        .unwrap();
                    let gathered = c.all_gather(&vec![r as f32; 3 - r], &groups).unwrap();
                    let mut b = if r == 2 {
                        b"hello".to_vec()
                    } else {
                        Vec::new()
                    };
                    c.broadcast(&mut b, 2).unwrap();
                    let mut f = [r as f64];
                    c.all_reduce_f64(&mut f, ReduceOp::Min).unwrap();
                    c.barrier("end").unwrap();
                    (v, g, gathered, b, f[0])
                })
            })
            .collect();
        let out: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        for (r, (v, g, gathered, b, f)) in out.into_iter().enumerate() {
            assert_eq!(v, vec![6.0, 30.0]);
            assert_eq!(g, vec![if r == 1 { 1.0 } else { 2.0 }]);
            if r == 1 {
                assert_eq!(gathered, vec![vec![1.0, 1.0]]);
            } else {
                assert_eq!(gathered, vec![vec![0.0, 0.0, 0.0], vec![2.0]]);
            }
            assert_eq!(b, b"hello");
            assert_eq!(f, 0.0);
        }
    }

    #[test]
    fn a_lost_rank_is_a_typed_bounded_failure() {
        let port = free_port();
        let mut w0 = world(0, 2, port);
        w0.timeout = Duration::from_secs(2);
        let mut w1 = w0.clone();
        w1.rank = 1;
        w1.local_rank = 1;
        let peer = std::thread::spawn(move || {
            let c = TcpCollective::connect(&w1, &CancelFlag::new()).unwrap();
            drop(c); // rank 1 dies right after the rendezvous
        });
        let mut c = TcpCollective::connect(&w0, &CancelFlag::new()).unwrap();
        peer.join().unwrap();
        let mut v = vec![1.0f32];
        let e = c
            .all_reduce(&mut v, ReduceOp::Sum, Precision::Fp32, &Partition::whole(2))
            .unwrap_err();
        assert!(
            matches!(dist_error(&e), Some(DistError::RankLost { rank: 1, .. })),
            "{e}"
        );
        // a rank that never joins is bounded too
        let port = free_port();
        let mut w = world(0, 2, port);
        w.timeout = Duration::from_millis(300);
        let e = TcpCollective::connect(&w, &CancelFlag::new()).unwrap_err();
        assert!(
            matches!(dist_error(&e), Some(DistError::RankLost { op, .. }) if op == "rendezvous")
        );
    }

    /// A raw connection to a listening coordinator (retrying until it listens).
    fn dial(port: u16) -> TcpStream {
        for _ in 0..200 {
            if let Ok(s) = TcpStream::connect(("127.0.0.1", port)) {
                s.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
                return s;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        panic!("coordinator never listened on {port}");
    }

    fn hello_of(w: &World, rank: usize, token: &str) -> Vec<u8> {
        json!({"magic": MAGIC, "rank": rank, "world_size": w.world_size,
               "local_world_size": w.local_world_size, "num_nodes": w.num_nodes,
               "node_rank": 0, "token": token})
        .to_string()
        .into_bytes()
    }

    #[test]
    fn a_huge_length_prefix_is_a_typed_refusal_not_an_allocation() {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = l.local_addr().unwrap().port();
        let sender = std::thread::spawn(move || {
            let mut s = dial(port);
            let mut head = vec![0u8];
            head.extend(0u64.to_le_bytes());
            head.extend(u64::MAX.to_le_bytes()); // 17 bytes claiming 16 EiB
            s.write_all(&head).unwrap();
            std::thread::sleep(Duration::from_millis(200));
        });
        let (mut s, _) = l.accept().unwrap();
        configure(&s).unwrap();
        let e = recv_frame(&mut s, &test_pace(), 0, 0, 1, "rendezvous", HELLO_MAX).unwrap_err();
        assert!(
            matches!(dist_error(&e), Some(DistError::Desync { got, .. }) if got.contains("18446744073709551615-byte")),
            "{e}"
        );
        sender.join().unwrap();
        // A prefix within the bound but never delivered ends in the bounded wait, having allocated
        // only what arrived.
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = l.local_addr().unwrap().port();
        let sender = std::thread::spawn(move || {
            let mut s = dial(port);
            let mut head = vec![0u8];
            head.extend(0u64.to_le_bytes());
            head.extend(1_000u64.to_le_bytes());
            s.write_all(&head).unwrap();
            s.write_all(b"partial").unwrap();
        });
        let (mut s, _) = l.accept().unwrap();
        configure(&s).unwrap();
        sender.join().unwrap();
        let e = recv_frame(&mut s, &test_pace(), 0, 0, 1, "x", HELLO_MAX).unwrap_err();
        assert!(
            matches!(dist_error(&e), Some(DistError::RankLost { .. })),
            "{e}"
        );
    }

    /// Rank `r` of `w0`'s world joining after `after` ms, then all-reducing its rank number.
    fn delayed_rank(w0: &World, r: usize, after: u64) -> std::thread::JoinHandle<Vec<f32>> {
        let mut w = w0.clone();
        w.rank = r;
        w.local_rank = r;
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(after));
            let mut c = TcpCollective::connect(&w, &CancelFlag::new()).unwrap();
            let mut v = vec![r as f32];
            c.all_reduce(&mut v, ReduceOp::Sum, Precision::Fp32, &Partition::whole(3))
                .unwrap();
            v
        })
    }

    #[test]
    fn rendezvous_drops_strangers_and_duplicates_and_keeps_accepting() {
        let port = free_port();
        let mut w0 = world(0, 3, port);
        w0.token = Some(DistToken("s3cret".into()));
        let w_bad = w0.clone();
        let strangers = std::thread::spawn(move || {
            // garbage bytes, a connection that closes without a hello, and a well-formed hello
            // with the wrong token claiming rank 1 — all before any real rank joins
            let mut g = dial(port);
            g.write_all(b"GET / HTTP/1.1\r\n\r\n").unwrap();
            drop(dial(port));
            let mut t = dial(port);
            send_frame(
                &mut t,
                &test_pace(),
                0,
                0,
                &hello_of(&w_bad, 1, "guess"),
                0,
                "x",
            )
            .unwrap();
            let wrong_token_dropped =
                recv_frame(&mut t, &test_pace(), 0, 0, 0, "x", HELLO_MAX).is_err();
            // then, after the real rank 1 joined, a duplicate rank 1 with the right token
            std::thread::sleep(Duration::from_millis(600));
            let mut dup = dial(port);
            send_frame(
                &mut dup,
                &test_pace(),
                0,
                0,
                &hello_of(&w_bad, 1, "s3cret"),
                0,
                "x",
            )
            .unwrap();
            let duplicate_dropped =
                recv_frame(&mut dup, &test_pace(), 0, 0, 0, "x", HELLO_MAX).is_err();
            (g, wrong_token_dropped, duplicate_dropped)
        });
        let one = delayed_rank(&w0, 1, 300);
        let two = delayed_rank(&w0, 2, 1200);
        let mut c = TcpCollective::connect(&w0, &CancelFlag::new()).unwrap();
        let mut v = vec![0.0f32];
        c.all_reduce(&mut v, ReduceOp::Sum, Precision::Fp32, &Partition::whole(3))
            .unwrap();
        assert_eq!(v, vec![3.0]);
        assert_eq!(one.join().unwrap(), vec![3.0]);
        assert_eq!(two.join().unwrap(), vec![3.0]);
        let (_g, wrong_token_dropped, duplicate_dropped) = strangers.join().unwrap();
        assert!(
            wrong_token_dropped,
            "a wrong-token hello must not be admitted"
        );
        assert!(duplicate_dropped, "a duplicate rank must not be admitted");
    }

    #[test]
    fn a_wrong_token_never_takes_a_rank() {
        let port = free_port();
        let mut w0 = world(0, 2, port);
        w0.token = Some(DistToken("right".into()));
        w0.timeout = Duration::from_secs(3);
        let mut impostor = w0.clone();
        impostor.rank = 1;
        impostor.local_rank = 1;
        impostor.token = Some(DistToken("wrong".into()));
        let mut real = impostor.clone();
        real.token = Some(DistToken("right".into()));
        let first =
            std::thread::spawn(move || TcpCollective::connect(&impostor, &CancelFlag::new()));
        std::thread::sleep(Duration::from_millis(300));
        let second = std::thread::spawn(move || TcpCollective::connect(&real, &CancelFlag::new()));
        let c0 = TcpCollective::connect(&w0, &CancelFlag::new());
        assert!(
            c0.is_ok(),
            "the real rank 1 joins after the impostor was dropped"
        );
        assert!(second.join().unwrap().is_ok());
        let e = first.join().unwrap().unwrap_err();
        assert!(
            matches!(dist_error(&e), Some(DistError::RankLost { rank: 0, .. })),
            "{e}"
        );
        // With only an impostor, the world never completes: a bounded RankLost naming the drop.
        let port = free_port();
        let mut w0 = world(0, 2, port);
        w0.token = Some(DistToken("right".into()));
        w0.timeout = Duration::from_millis(800);
        let mut impostor = w0.clone();
        impostor.rank = 1;
        impostor.local_rank = 1;
        impostor.token = None;
        let t = std::thread::spawn(move || TcpCollective::connect(&impostor, &CancelFlag::new()));
        let e = TcpCollective::connect(&w0, &CancelFlag::new()).unwrap_err();
        assert!(
            matches!(dist_error(&e), Some(DistError::RankLost { detail, .. }) if detail.contains("token")),
            "{e}"
        );
        let _ = t.join();
    }

    #[test]
    fn mismatched_member_payloads_are_a_desync_on_the_coordinator() {
        type Case = fn(&mut TcpCollective, usize) -> Result<()>;
        let cases: [(&str, Case); 5] = [
            ("longer", |c, r| {
                let mut v = vec![0.0f32; 1 + r];
                c.all_reduce(&mut v, ReduceOp::Sum, Precision::Fp32, &Partition::whole(2))
            }),
            ("shorter", |c, r| {
                let mut v = vec![0.0f32; 2 - r];
                c.all_reduce(&mut v, ReduceOp::Sum, Precision::Fp32, &Partition::whole(2))
            }),
            ("op", |c, r| {
                let op = if r == 0 { ReduceOp::Sum } else { ReduceOp::Max };
                c.all_reduce(&mut [0.0], op, Precision::Fp32, &Partition::whole(2))
            }),
            ("precision", |c, r| {
                let p = if r == 0 {
                    Precision::Fp32
                } else {
                    Precision::Bf16
                };
                c.all_reduce(&mut [0.0], ReduceOp::Sum, p, &Partition::whole(2))
            }),
            ("f64 shorter", |c, r| {
                let mut v = vec![0.0f64; 2 - r];
                c.all_reduce_f64(&mut v, ReduceOp::Sum)
            }),
        ];
        for (name, case) in cases {
            let port = free_port();
            let mut w0 = world(0, 2, port);
            w0.timeout = Duration::from_secs(2);
            let mut w1 = w0.clone();
            w1.rank = 1;
            w1.local_rank = 1;
            let peer = std::thread::spawn(move || {
                let mut c = TcpCollective::connect(&w1, &CancelFlag::new()).unwrap();
                case(&mut c, 1)
            });
            let mut c = TcpCollective::connect(&w0, &CancelFlag::new()).unwrap();
            let e = case(&mut c, 0).unwrap_err();
            assert!(
                matches!(dist_error(&e), Some(DistError::Desync { rank: 1, .. })),
                "{name}: {e}"
            );
            // the peer's wait is bounded and typed
            let pe = peer.join().unwrap().unwrap_err();
            assert!(dist_error(&pe).is_some(), "{name}: {pe}");
        }
        // a gather contribution larger than the coordinator's shard 0
        let port = free_port();
        let mut w0 = world(0, 2, port);
        w0.timeout = Duration::from_secs(2);
        let mut w1 = w0.clone();
        w1.rank = 1;
        w1.local_rank = 1;
        let peer = std::thread::spawn(move || {
            let mut c = TcpCollective::connect(&w1, &CancelFlag::new()).unwrap();
            let _ = c.all_gather(&[1.0, 2.0], &Partition::whole(2));
        });
        let mut c = TcpCollective::connect(&w0, &CancelFlag::new()).unwrap();
        let e = c.all_gather(&[1.0], &Partition::whole(2)).unwrap_err();
        assert!(
            matches!(dist_error(&e), Some(DistError::Desync { rank: 1, .. })),
            "{e}"
        );
        peer.join().unwrap();
        // barriers with different tags
        let port = free_port();
        let mut w0 = world(0, 2, port);
        w0.timeout = Duration::from_secs(2);
        let mut w1 = w0.clone();
        w1.rank = 1;
        w1.local_rank = 1;
        let peer = std::thread::spawn(move || {
            let mut c = TcpCollective::connect(&w1, &CancelFlag::new()).unwrap();
            let _ = c.barrier("export");
        });
        let mut c = TcpCollective::connect(&w0, &CancelFlag::new()).unwrap();
        let e = c.barrier("checkpoint").unwrap_err();
        assert!(
            matches!(dist_error(&e), Some(DistError::Desync { .. })),
            "{e}"
        );
        peer.join().unwrap();
    }

    /// Hybrid sharding's replicate-group reduction: each group reduces its own row shard, so the
    /// groups' lengths differ (shard 0's rows are the longest) — valid, only a mismatch WITHIN a
    /// group is a desync.
    #[test]
    fn sub_group_reductions_may_differ_in_length_across_groups() {
        let port = free_port();
        let handles: Vec<_> = (0..4)
            .map(|r| {
                std::thread::spawn(move || {
                    let mut c =
                        TcpCollective::connect(&world(r, 4, port), &CancelFlag::new()).unwrap();
                    let groups = Partition(vec![0, 1, 0, 1]);
                    let mut v = vec![r as f32; if r % 2 == 0 { 3 } else { 2 }];
                    c.all_reduce(&mut v, ReduceOp::Sum, Precision::Fp32, &groups)
                        .map(|_| v)
                })
            })
            .collect();
        let out: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        assert_eq!(out[0].as_ref().unwrap(), &vec![2.0; 3]);
        assert_eq!(out[1].as_ref().unwrap(), &vec![4.0; 2]);
        assert_eq!(out[2].as_ref().unwrap(), &vec![2.0; 3]);
        assert_eq!(out[3].as_ref().unwrap(), &vec![4.0; 2]);
    }

    // ---- sc-25690: cancellable, separately bounded rendezvous; polled waits; keepalive ----

    fn cancelled() -> CancelFlag {
        let c = CancelFlag::new();
        c.cancel();
        c
    }

    /// A cancelled run never sits out the rendezvous: neither the coordinator waiting for ranks
    /// nor a rank retrying its connect. The bounds are long enough that only an honoured cancel
    /// can end the wait as `Canceled` (an ignored one ends as `RankLost` at the bound).
    #[test]
    fn a_cancelled_rendezvous_is_typed_canceled_on_every_side() {
        let port = free_port();
        let mut w0 = world(0, 2, port);
        w0.rendezvous_timeout = Duration::from_secs(5);
        let e = TcpCollective::connect(&w0, &cancelled()).unwrap_err();
        assert!(matches!(e, Error::Canceled), "coordinator: {e}");
        let mut w1 = w0.clone();
        w1.rank = 1;
        w1.local_rank = 1;
        let e = TcpCollective::connect(&w1, &cancelled()).unwrap_err();
        assert!(matches!(e, Error::Canceled), "connecting rank: {e}");
        // A rank that joined and waits for its admission answer: a stand-in coordinator that
        // accepted it and read its hello (so the rank is provably in the admission wait) before
        // the cancel trips.
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = l.local_addr().unwrap().port();
        let mut w1 = world(1, 3, port);
        w1.rendezvous_timeout = Duration::from_secs(5);
        let flag = CancelFlag::new();
        let rank1 = {
            let (w1, flag) = (w1.clone(), flag.clone());
            std::thread::spawn(move || TcpCollective::connect(&w1, &flag))
        };
        let (mut s, _) = l.accept().unwrap();
        configure(&s).unwrap();
        let hello = recv_frame(&mut s, &test_pace(), 0, 0, 1, "rendezvous", HELLO_MAX).unwrap();
        assert!(String::from_utf8_lossy(&hello).contains("\"rank\":1"));
        flag.cancel();
        let e = rank1.join().unwrap().unwrap_err();
        assert!(matches!(e, Error::Canceled), "admitted-wait rank: {e}");
        drop(s);
    }

    /// The coordinator's wait for one accepted connection's hello honours a cancel too.
    #[test]
    fn a_cancel_during_a_silent_hello_is_not_a_dropped_stranger() {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = l.local_addr().unwrap().port();
        let silent = std::thread::spawn(move || {
            let s = dial(port);
            std::thread::sleep(Duration::from_millis(800));
            drop(s);
        });
        let (mut s, _) = l.accept().unwrap();
        configure(&s).unwrap();
        let pace = cancelled_pace();
        let w = world(0, 2, port);
        let peers: Vec<Option<TcpStream>> = vec![None, None];
        assert!(matches!(
            admit(&mut s, &pace, &w, &peers),
            Err(Rejection::Canceled)
        ));
        silent.join().unwrap();
    }

    /// `dist` options naming the rendezvous bound and the cancel grace (seconds).
    fn world_options(rendezvous: u64, grace: u64) -> JsonValue {
        json!({"world": {"rendezvous_timeout_secs": rendezvous, "cancel_grace_secs": grace}})
    }

    /// The rendezvous has its own bound: a rank that never starts is reported at
    /// `rendezvous_timeout`, not after the (5400 s by default) collective timeout.
    #[test]
    fn the_rendezvous_is_bounded_separately_from_the_collective_timeout() {
        let port = free_port();
        let mut w = world(0, 2, port);
        w.timeout = Duration::from_secs(3600);
        w.rendezvous_timeout = Duration::from_millis(400);
        let e = TcpCollective::connect(&w, &CancelFlag::new()).unwrap_err();
        assert!(
            matches!(dist_error(&e), Some(DistError::RankLost { op, detail, .. })
                if op == "rendezvous" && detail.contains("400ms")),
            "{e}"
        );
        // configured constants, read as their printed values (no clock is involved)
        assert_eq!(format!("{:?}", World::single().rendezvous_timeout), "600s");
        let cfg = world_options(30, 7);
        let parsed = parse_dist(Some(&cfg), false, true).unwrap();
        assert_eq!(format!("{:?}", parsed.world.rendezvous_timeout), "30s");
        assert_eq!(format!("{:?}", parsed.world.cancel_grace), "7s");
        let zero = world_options(0, 120);
        assert!(parse_dist(Some(&zero), false, true).is_err());
        // a zero grace would end the cancel agreement's own wait at its first idle poll
        let no_grace = world_options(30, 0);
        assert!(parse_dist(Some(&no_grace), false, true).is_err());
    }

    /// A cancel pending in a collective wait lets the peers arrive within the grace (the
    /// cooperative cancel still completes its collective agreement), and ends a wait on a peer
    /// that never answers as `Canceled` once the grace passed: never the 5400 s bound.
    #[test]
    fn a_pending_cancel_ends_a_stalled_collective_after_its_grace() {
        // grace honoured: the peer arrives late but within it
        let port = free_port();
        let mut w0 = world(0, 2, port);
        w0.timeout = Duration::from_secs(3600);
        w0.cancel_grace = Duration::from_secs(30);
        let mut w1 = w0.clone();
        w1.rank = 1;
        w1.local_rank = 1;
        let peer = std::thread::spawn(move || {
            let mut c = TcpCollective::connect(&w1, &CancelFlag::new()).unwrap();
            std::thread::sleep(Duration::from_millis(600));
            let mut v = [1.0f32];
            c.all_reduce(&mut v, ReduceOp::Max, Precision::Fp32, &Partition::whole(2))
                .map(|_| v[0])
        });
        let flag = CancelFlag::new();
        let mut c = TcpCollective::connect(&w0, &flag).unwrap();
        flag.cancel();
        let mut v = [0.0f32];
        c.all_reduce(&mut v, ReduceOp::Max, Precision::Fp32, &Partition::whole(2))
            .unwrap();
        assert_eq!(v, [1.0]);
        assert_eq!(peer.join().unwrap().unwrap(), 1.0);
        // grace passed: a peer that never answers
        let port = free_port();
        let mut w0 = world(0, 2, port);
        w0.timeout = Duration::from_secs(3600);
        w0.cancel_grace = Duration::from_millis(300);
        let mut w1 = w0.clone();
        w1.rank = 1;
        w1.local_rank = 1;
        let (stop_tx, stop_rx) = std::sync::mpsc::channel::<()>();
        let peer = std::thread::spawn(move || {
            let c = TcpCollective::connect(&w1, &CancelFlag::new()).unwrap();
            let _ = stop_rx.recv(); // alive, connected, silent
            drop(c);
        });
        let flag = CancelFlag::new();
        let mut c = TcpCollective::connect(&w0, &flag).unwrap();
        flag.cancel();
        let e = c.barrier("stalled").unwrap_err();
        assert!(matches!(e, Error::Canceled), "{e}");
        stop_tx.send(()).unwrap();
        peer.join().unwrap();
    }

    /// A frame whose bytes arrive in several pieces is reassembled whole: the polled
    /// wait keeps what arrived before a call would have blocked.
    #[test]
    fn a_frame_arriving_in_pieces_is_read_whole() {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = l.local_addr().unwrap().port();
        let sender = std::thread::spawn(move || {
            let mut s = dial(port);
            let mut frame = vec![OP_BARRIER];
            frame.extend(7u64.to_le_bytes());
            frame.extend(5u64.to_le_bytes());
            frame.extend(b"hello");
            for piece in [&frame[..5], &frame[5..19], &frame[19..]] {
                s.write_all(piece).unwrap();
                s.flush().unwrap();
                std::thread::sleep(Duration::from_millis(300));
            }
        });
        let (mut s, _) = l.accept().unwrap();
        configure(&s).unwrap();
        let got = recv_frame(&mut s, &test_pace(), OP_BARRIER, 7, 1, "x", 64).unwrap();
        assert_eq!(got, b"hello");
        sender.join().unwrap();
    }

    /// Every transport socket carries TCP keepalive, so a peer that vanishes without a reset is
    /// detected by the kernel's probes rather than the collective timeout.
    #[cfg(unix)]
    #[test]
    fn transport_sockets_carry_tcp_keepalive() {
        use std::os::fd::AsRawFd;
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = l.local_addr().unwrap().port();
        let s = TcpStream::connect(("127.0.0.1", port)).unwrap();
        configure(&s).unwrap();
        let get = |level: libc::c_int, name: libc::c_int| -> libc::c_int {
            let mut v: libc::c_int = 0;
            let mut len = std::mem::size_of::<libc::c_int>() as libc::socklen_t;
            // SAFETY: a valid socket and a pointer/length to one live c_int.
            let r = unsafe {
                libc::getsockopt(
                    s.as_raw_fd(),
                    level,
                    name,
                    (&mut v as *mut libc::c_int).cast(),
                    &mut len,
                )
            };
            assert_eq!(r, 0);
            v
        };
        assert_ne!(get(libc::SOL_SOCKET, libc::SO_KEEPALIVE), 0);
        // probe count and interval (seconds) as configured: 6 probes, 10 apart
        let probes = (
            get(libc::IPPROTO_TCP, libc::TCP_KEEPCNT),
            get(libc::IPPROTO_TCP, libc::TCP_KEEPINTVL),
        );
        assert_eq!(probes, (6, 10));
    }

    /// sc-25690: only `WouldBlock` retries; on a non-blocking socket `TimedOut` is keepalive
    /// declaring the peer dead.
    #[test]
    fn a_keepalive_timeout_is_a_lost_peer_not_a_retry() {
        use std::io::{Error as IoError, ErrorKind};
        assert!(would_block(&IoError::from(ErrorKind::WouldBlock)));
        assert!(!would_block(&IoError::from(ErrorKind::TimedOut)));
        assert!(!would_block(&IoError::from(ErrorKind::ConnectionReset)));
    }

    /// sc-25690: keepalive timing is best effort; only `SO_KEEPALIVE` itself is required.
    #[test]
    fn keepalive_timing_failures_are_not_fatal() {
        let opts = keepalive::Opts {
            sol_socket: 1,
            so_keepalive: 2,
            ipproto_tcp: 3,
            idle: 4,
            interval: 5,
            probes: 6,
        };
        let calls = std::cell::RefCell::new(Vec::new());
        let old_windows = |level: i32, name: i32, v: u32| -> std::io::Result<()> {
            calls.borrow_mut().push((level, name, v));
            if level == 3 {
                Err(std::io::Error::from(std::io::ErrorKind::InvalidInput))
            } else {
                Ok(())
            }
        };
        assert!(!keepalive::apply(&old_windows, &opts).unwrap());
        assert_eq!(calls.borrow()[0], (1, 2, 1), "SO_KEEPALIVE first");
        let modern = |_: i32, _: i32, _: u32| -> std::io::Result<()> { Ok(()) };
        assert!(keepalive::apply(&modern, &opts).unwrap());
        let none = |level: i32, _: i32, _: u32| -> std::io::Result<()> {
            if level == 1 {
                Err(std::io::Error::from(std::io::ErrorKind::Unsupported))
            } else {
                Ok(())
            }
        };
        assert!(keepalive::apply(&none, &opts).is_err());
    }

    /// Two ranks of a fresh loopback world: `(rank 0's collective, rank 1's thread)`; rank 1 runs
    /// `peer` on its collective.
    fn pair<T: Send + 'static>(
        w0: World,
        flag0: &CancelFlag,
        peer: impl FnOnce(&mut TcpCollective) -> T + Send + 'static,
    ) -> (TcpCollective, std::thread::JoinHandle<T>) {
        let mut w1 = w0.clone();
        w1.rank = 1;
        w1.local_rank = 1;
        let h = std::thread::spawn(move || {
            let mut c = TcpCollective::connect(&w1, &CancelFlag::new()).unwrap();
            peer(&mut c)
        });
        (TcpCollective::connect(&w0, flag0).unwrap(), h)
    }

    /// sc-25690: a disarmed collective (the cancel agreement decided, or a checkpoint / export
    /// write in progress) waits for a slow writer past the cancel grace.
    #[test]
    fn a_disarmed_collective_outwaits_the_cancel_grace() {
        let mut w0 = world(0, 2, free_port());
        w0.timeout = Duration::from_secs(3600);
        w0.cancel_grace = Duration::from_millis(200);
        let flag = CancelFlag::new();
        let (mut c, peer) = pair(w0, &flag, |c| {
            // the writing rank: busy for well past the grace before it reaches the barrier
            std::thread::sleep(Duration::from_millis(1200));
            c.barrier("checkpoint")
        });
        flag.cancel();
        assert!(c.set_cancel_armed(false), "transports start armed");
        c.barrier("checkpoint").unwrap();
        peer.join().unwrap().unwrap();
        assert!(!c.set_cancel_armed(true));
    }

    /// sc-25690: the grace is one budget for the whole transport, not renewed per frame or per
    /// collective: a peer that answers every collective just inside the grace still ends the waits
    /// once the pending cancel is older than the grace.
    #[test]
    fn the_cancel_grace_is_not_renewed_per_collective() {
        let mut w0 = world(0, 2, free_port());
        w0.timeout = Duration::from_secs(3600);
        w0.cancel_grace = Duration::from_millis(700);
        let flag = CancelFlag::new();
        let (mut c, peer) = pair(w0, &flag, |c| {
            for _ in 0..4 {
                std::thread::sleep(Duration::from_millis(400));
                if c.barrier("slow").is_err() {
                    break;
                }
            }
        });
        flag.cancel();
        let outcome = (0..4).try_for_each(|_| c.barrier("slow"));
        assert!(matches!(outcome, Err(Error::Canceled)), "{outcome:?}");
        drop(c);
        peer.join().unwrap();
    }
}
