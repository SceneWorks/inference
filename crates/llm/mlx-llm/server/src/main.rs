//! Example OpenAI-compatible chat server on the mlx-llm engine (story 7174).
//!
//! ```text
//! cargo run --release -p mlx-llm-server -- --model <snapshot_dir> [--port 8080] [--quant q4|q8]
//! ```
//!
//! Serves `POST /v1/chat/completions` (streaming SSE or buffered JSON), `GET /v1/models`, and a
//! health check, for a single model loaded through the **backend-neutral** `core_llm` contract and
//! the explicit MLX provider catalog. The HTTP serving path speaks only the `TextLlm` contract.
//!
//! This is a *reference*, deliberately minimal: one model on one serving thread (MLX's Metal device
//! is single-threaded — see the engine's `.cargo/config.toml`), `Connection: close`, no auth.
//! Requests that arrive while the engine is busy are decoded together on the next round through
//! `TextLlm::generate_batch` (continuous batching on the MLX provider, each request with its own
//! `kv_compression` opt-in and `kv_cache` report — sc-20681). A production gateway (multi-model,
//! auth, Anthropic/Ollama compat) is the separate server-app project, not this example.
//!
//! ```text
//! curl -N http://localhost:8080/v1/chat/completions \
//!   -H 'content-type: application/json' \
//!   -d '{"model":"local","stream":true,"messages":[{"role":"user","content":"Hi!"}]}'
//! ```

mod http;
mod openai;

use std::io::{self, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use mlx_llm::core_llm::{
    self, CancelFlag, Error as CoreError, LoadSpec, Quantize, StreamEvent, TextLlm,
};

/// How long a connected peer may stay silent before its connection is dropped (F-022). The server
/// is single-threaded, so a peer that connects and sends nothing (a stray `nc`) would otherwise
/// block `read_line` forever and wedge every subsequent client. 10 seconds is generous for any
/// legitimate client writing a request (even by hand over a slow link) while bounding how long one
/// idle connection can monopolise the serving thread.
const READ_TIMEOUT: Duration = Duration::from_secs(10);
/// Maximum wall time allowed to receive one request. This is independent of model execution time.
const REQUEST_DEADLINE: Duration = Duration::from_secs(30);
/// Maximum response-write lifetime, starting lazily at the first emitted byte. The OpenAI adapter's
/// default output is 512 tokens; ten minutes permits even a sub-1-token/s local model to complete
/// while still bounding a streaming client that continues accepting only intermittent progress.
const RESPONSE_DEADLINE: Duration = Duration::from_secs(10 * 60);
/// Maximum time spent waiting for a peer to accept response bytes. A client that submits a valid
/// request and then stops reading is dropped so the serial accept loop can serve the next client.
const WRITE_TIMEOUT: Duration = Duration::from_secs(10);
const INTERNAL_ERROR_MESSAGE: &str = "internal server error";

#[derive(Clone, Copy)]
struct ConnectionLimits {
    read_idle: Duration,
    request_total: Duration,
    write: Duration,
    response_total: Duration,
}

impl ConnectionLimits {
    const PRODUCTION: Self = Self {
        read_idle: READ_TIMEOUT,
        request_total: REQUEST_DEADLINE,
        write: WRITE_TIMEOUT,
        response_total: RESPONSE_DEADLINE,
    };
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

/// Parsed CLI configuration.
struct Args {
    model: String,
    host: String,
    port: u16,
    quantize: Option<Quantize>,
    provider: Option<String>,
}

fn parse_args() -> Result<Args, String> {
    let mut model = None;
    let mut host = "127.0.0.1".to_string();
    let mut port = 8080u16;
    let mut quantize = None;
    let mut provider = None;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let mut next = || args.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--model" | "-m" => model = Some(next()?),
            "--host" => host = next()?,
            "--port" | "-p" => port = next()?.parse().map_err(|_| "invalid --port".to_string())?,
            "--provider" => provider = Some(next()?),
            "--quant" => {
                quantize = Some(match next()?.as_str() {
                    "q4" => Quantize::Q4,
                    "q8" => Quantize::Q8,
                    other => return Err(format!("unknown --quant {other:?} (expected q4|q8)")),
                })
            }
            "-h" | "--help" => {
                println!("usage: mlx-llm-server --model <dir> [--host 127.0.0.1] [--port 8080] [--quant q4|q8] [--provider <id>]");
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    Ok(Args {
        model: model.ok_or("missing required --model <snapshot_dir>")?,
        host,
        port,
        quantize,
        provider,
    })
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;
    let registry = mlx_llm::text_registry()?;

    // Use the requested provider id, else default to a bundled *text* (non-vision) provider. Several
    // may be present (e.g. a VLM captioner alongside the generic text model), so don't just grab the
    // first. The catalog is explicit and contains no process-global discovery state.
    let provider_id = match args.provider {
        Some(id) => id,
        None => {
            let descriptors = || registry.registrations().map(|r| (r.descriptor)());
            descriptors()
                .find(|d| !d.capabilities.supports_vision)
                .or_else(|| descriptors().next())
                .ok_or("no TextLlm provider registered")?
                .id
        }
    };
    eprintln!(
        "loading model from {} via provider '{provider_id}' …",
        args.model
    );
    let spec = LoadSpec {
        source: args.model.clone(),
        projector_source: None,
        quantize: args.quantize,
        cuda_graphs: None,
    };
    let provider = registry.load_textllm(&provider_id, &spec)?;

    // A friendly default model name for responses (the snapshot dir's basename).
    let default_model = std::path::Path::new(&args.model)
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| provider_id.clone());

    let listener = TcpListener::bind((args.host.as_str(), args.port))?;
    let addr = listener.local_addr()?;
    eprintln!("mlx-llm-server listening on http://{addr}  (model: {default_model})");

    serve(
        &listener,
        provider.as_ref(),
        &default_model,
        ConnectionLimits::PRODUCTION,
    );
    Ok(())
}

/// Most connections one accept round serves together (sc-20681).
const MAX_BATCH: usize = 8;

/// The accept loop. Each round takes one connection and every other connection already waiting
/// (up to [`MAX_BATCH`]); their chat completions run as one batch through
/// [`TextLlm::generate_batch`] — continuous batching on a backend that implements it — so
/// requests that arrive while the engine is busy decode together. A per-connection error
/// (including a read timeout) drops that connection only — the loop always continues serving
/// subsequent clients.
fn serve(
    listener: &TcpListener,
    provider: &dyn TextLlm,
    default_model: &str,
    limits: ConnectionLimits,
) {
    for stream in listener.incoming() {
        match stream {
            Ok(first) => {
                let mut streams = vec![first];
                drain_pending(listener, &mut streams);
                let mut chats = Vec::new();
                for stream in streams {
                    match handle_connection(stream, provider, default_model, limits) {
                        Ok(Some(chat)) => chats.push(chat),
                        Ok(None) => {}
                        Err(e) => eprintln!("connection error: {e}"),
                    }
                }
                if let Err(e) = run_chats(provider, chats) {
                    eprintln!("connection error: {e}");
                }
            }
            Err(e) => eprintln!("accept error: {e}"),
        }
    }
}

/// Accept every connection already waiting on `listener`, up to [`MAX_BATCH`] in all, without
/// blocking for new ones.
fn drain_pending(listener: &TcpListener, streams: &mut Vec<TcpStream>) {
    if listener.set_nonblocking(true).is_err() {
        return;
    }
    while streams.len() < MAX_BATCH {
        match listener.accept() {
            // An accepted socket may inherit the listener's non-blocking mode.
            Ok((stream, _)) => match stream.set_nonblocking(false) {
                Ok(()) => streams.push(stream),
                Err(e) => eprintln!("accept error: {e}"),
            },
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => break,
            Err(e) => {
                eprintln!("accept error: {e}");
                break;
            }
        }
    }
    if let Err(e) = listener.set_nonblocking(false) {
        eprintln!("listener error: {e}");
    }
}

/// A parsed, validated chat completion waiting for the engine.
struct ChatJob {
    writer: DeadlineWriter,
    req: core_llm::TextLlmRequest,
    cancel: CancelFlag,
    stream: bool,
    id: String,
    model: String,
    created: u64,
}

/// Read one request on a connection. A chat completion comes back as a [`ChatJob`] for the
/// engine; every other route (and any invalid request) is answered here and the connection closed
/// (`Connection: close`).
fn handle_connection(
    stream: TcpStream,
    provider: &dyn TextLlm,
    default_model: &str,
    limits: ConnectionLimits,
) -> io::Result<Option<ChatJob>> {
    let request_deadline = Instant::now() + limits.request_total;
    let read_stream = stream.try_clone()?;
    let mut reader = BufReader::new(DeadlineReader::new(
        read_stream,
        limits.read_idle,
        request_deadline,
    ));
    let mut writer = DeadlineWriter::new(stream, limits.write, limits.response_total);
    let req = match http::read_request(&mut reader) {
        Ok(Some(req)) => req,
        Ok(None) => return Ok(None), // idle disconnect
        // Read timeout (F-022): the peer went silent mid-request — treat it as a dropped
        // connection, not an error worth replying to (the peer isn't reading anyway).
        Err(e)
            if matches!(
                e.kind(),
                io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
            ) =>
        {
            return Ok(None);
        }
        Err(e) => {
            let status = http::error_status(&e);
            write_json(
                &mut writer,
                status,
                &openai::error_body(&e.to_string(), "invalid_request"),
            )?;
            return Ok(None);
        }
    };

    match (req.method.as_str(), req.path.as_str()) {
        ("POST", "/v1/chat/completions") => parse_chat(writer, provider, &req.body, default_model),
        ("GET", "/v1/models") => write_json(
            &mut writer,
            200,
            &openai::models_list(default_model, unix_secs()),
        )
        .map(|()| None),
        ("GET", "/" | "/health") => write_text(&mut writer, 200, "ok").map(|()| None),
        _ => write_json(
            &mut writer,
            404,
            &openai::error_body("not found", "not_found"),
        )
        .map(|()| None),
    }
}

/// A socket reader with both an idle timeout and an absolute request-receive deadline.
///
/// `TcpStream::set_read_timeout` alone is a per-read idle bound: a slow-loris peer can reset it by
/// periodically sending a byte. Before every OS read, this wrapper caps the timeout at the time
/// remaining on the fixed deadline and returns `TimedOut` once that deadline has elapsed.
struct DeadlineReader {
    stream: TcpStream,
    idle_timeout: Duration,
    deadline: Instant,
}

impl DeadlineReader {
    fn new(stream: TcpStream, idle_timeout: Duration, deadline: Instant) -> Self {
        Self {
            stream,
            idle_timeout,
            deadline,
        }
    }
}

/// A response writer with both an idle timeout and an absolute response deadline.
///
/// The deadline starts lazily on the first write, so request parsing and non-streaming model compute
/// do not consume the response budget. Every JSON, text, and SSE response uses this writer. Capping
/// each socket timeout by the fixed remaining duration prevents partial write progress from
/// extending the response indefinitely.
struct DeadlineWriter {
    stream: TcpStream,
    idle_timeout: Duration,
    total_timeout: Duration,
    deadline: Option<Instant>,
}

impl DeadlineWriter {
    fn new(stream: TcpStream, idle_timeout: Duration, total_timeout: Duration) -> Self {
        Self {
            stream,
            idle_timeout,
            total_timeout,
            deadline: None,
        }
    }

    fn prepare_write(&mut self) -> io::Result<()> {
        let deadline = *self
            .deadline
            .get_or_insert_with(|| Instant::now() + self.total_timeout);
        let remaining = deadline
            .checked_duration_since(Instant::now())
            .ok_or_else(|| io::Error::new(io::ErrorKind::TimedOut, "response deadline exceeded"))?;
        self.stream
            .set_write_timeout(Some(self.idle_timeout.min(remaining)))
    }
}

impl Write for DeadlineWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.prepare_write()?;
        self.stream.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.prepare_write()?;
        self.stream.flush()
    }
}

impl Read for DeadlineReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let remaining = self
            .deadline
            .checked_duration_since(Instant::now())
            .ok_or_else(|| io::Error::new(io::ErrorKind::TimedOut, "request deadline exceeded"))?;
        self.stream
            .set_read_timeout(Some(self.idle_timeout.min(remaining)))?;
        self.stream.read(buf)
    }
}

/// Handle a chat completion: parse → validate → stream SSE or return one JSON body.
fn parse_chat(
    mut writer: DeadlineWriter,
    provider: &dyn TextLlm,
    body: &[u8],
    default_model: &str,
) -> io::Result<Option<ChatJob>> {
    let stream = &mut writer;
    let chat: openai::ChatRequest = match serde_json::from_slice(body) {
        Ok(c) => c,
        Err(e) => {
            return write_json(
                stream,
                400,
                &openai::error_body(&e.to_string(), "invalid_request"),
            )
            .map(|()| None)
        }
    };
    let model = chat
        .model
        .clone()
        .unwrap_or_else(|| default_model.to_string());
    let want_stream = chat.stream;

    let mut req = match chat.into_text_llm_request() {
        Ok(r) => r,
        Err(msg) => {
            return write_json(stream, 400, &openai::error_body(&msg, "invalid_request"))
                .map(|()| None)
        }
    };
    // Reject anything outside the provider's declared surface before sending any 200.
    if let Err(e) = provider.validate(&req) {
        return write_json(
            stream,
            400,
            &openai::error_body(&e.to_string(), "invalid_request"),
        )
        .map(|()| None);
    }

    let cancel = CancelFlag::new();
    req.cancel = cancel.clone();
    Ok(Some(ChatJob {
        writer,
        req,
        cancel,
        stream: want_stream,
        id: completion_id(),
        model,
        created: unix_secs(),
    }))
}

/// Run the round's chat completions: one on its own as before, several together through
/// [`TextLlm::generate_batch`].
fn run_chats(provider: &dyn TextLlm, mut chats: Vec<ChatJob>) -> io::Result<()> {
    match chats.len() {
        0 => Ok(()),
        1 => run_chat(provider, chats.pop().expect("one chat")),
        _ => run_chat_batch(provider, chats),
    }
}

/// One chat completion: stream SSE or return one JSON body.
fn run_chat(provider: &dyn TextLlm, mut job: ChatJob) -> io::Result<()> {
    let (stream, req, cancel, id, model, created) = (
        &mut job.writer,
        &job.req,
        &job.cancel,
        job.id.as_str(),
        job.model.as_str(),
        job.created,
    );
    if job.stream {
        stream_chat(stream, provider, req, cancel, id, model, created)
    } else {
        match provider.complete(req) {
            Ok(out) => {
                let finish = out
                    .finish_reason
                    .map(openai::finish_reason_str)
                    .unwrap_or("stop");
                let body = openai::completion(
                    id,
                    model,
                    created,
                    &out.text,
                    finish,
                    out.usage.prompt_tokens,
                    out.usage.generated_tokens,
                    out.kv_cache.as_ref(),
                );
                write_json(stream, 200, &body)
            }
            Err(CoreError::Canceled) => Ok(()), // client vanished mid-generation
            Err(e) => write_json(stream, 500, &server_error_body(&e)),
        }
    }
}

/// Several chat completions decoded together. Streaming clients get their SSE headers and role
/// chunk first, then their own content chunks as the batch decodes; a client that disconnects
/// cancels only its own request. Each finishes with its own final chunk (or JSON body), carrying
/// its own `kv_cache` report.
fn run_chat_batch(provider: &dyn TextLlm, mut jobs: Vec<ChatJob>) -> io::Result<()> {
    let mut disconnected = vec![false; jobs.len()];
    for (job, gone) in jobs.iter_mut().zip(disconnected.iter_mut()) {
        if job.stream
            && (start_sse(&mut job.writer).is_err()
                || sse(
                    &mut job.writer,
                    &openai::role_chunk(&job.id, &job.model, job.created),
                )
                .is_err())
        {
            job.cancel.cancel();
            *gone = true;
        }
    }
    let reqs = jobs.iter().map(|job| job.req.clone()).collect::<Vec<_>>();
    let results = provider.generate_batch(&reqs, &mut |i, event| {
        let job = &mut jobs[i];
        if !job.stream || disconnected[i] {
            return;
        }
        if let StreamEvent::Token { text, .. } = event {
            if !text.is_empty()
                && sse(
                    &mut job.writer,
                    &openai::content_chunk(&job.id, &job.model, job.created, &text),
                )
                .is_err()
            {
                job.cancel.cancel();
                disconnected[i] = true;
            }
        }
    });
    for ((mut job, result), gone) in jobs.into_iter().zip(results).zip(disconnected) {
        if gone {
            continue;
        }
        let outcome = if job.stream {
            finish_sse(&mut job.writer, &job.id, &job.model, job.created, result)
        } else {
            match result {
                Ok(out) => {
                    let finish = out
                        .finish_reason
                        .map(openai::finish_reason_str)
                        .unwrap_or("stop");
                    let body = openai::completion(
                        &job.id,
                        &job.model,
                        job.created,
                        &out.text,
                        finish,
                        out.usage.prompt_tokens,
                        out.usage.generated_tokens,
                        out.kv_cache.as_ref(),
                    );
                    write_json(&mut job.writer, 200, &body)
                }
                Err(CoreError::Canceled) => Ok(()),
                Err(e) => write_json(&mut job.writer, 500, &server_error_body(&e)),
            }
        };
        if let Err(e) = outcome {
            eprintln!("connection error: {e}");
        }
    }
    Ok(())
}

/// The SSE response head.
fn start_sse(stream: &mut DeadlineWriter) -> io::Result<()> {
    stream.write_all(
        b"HTTP/1.1 200 OK\r\n\
          Content-Type: text/event-stream\r\n\
          Cache-Control: no-cache\r\n\
          Connection: close\r\n\
          X-Accel-Buffering: no\r\n\r\n",
    )
}

/// The end of an SSE response: the final chunk (with the `kv_cache` report) or an error chunk,
/// then `[DONE]`. Nothing more for a cancelled request.
fn finish_sse(
    stream: &mut DeadlineWriter,
    id: &str,
    model: &str,
    created: u64,
    result: core_llm::Result<core_llm::TextLlmOutput>,
) -> io::Result<()> {
    match result {
        Ok(out) => {
            let finish = out
                .finish_reason
                .map(openai::finish_reason_str)
                .unwrap_or("stop");
            let _ = sse(
                stream,
                &openai::final_chunk(id, model, created, finish, out.kv_cache.as_ref()),
            );
        }
        Err(CoreError::Canceled) => return Ok(()),
        Err(e) => {
            let _ = sse(stream, &server_error_body(&e));
        }
    }
    let _ = stream.write_all(b"data: [DONE]\n\n");
    let _ = stream.flush();
    Ok(())
}

/// Stream a chat completion as Server-Sent Events. A failed write (client disconnected) trips the
/// request's [`CancelFlag`], so the decode loop stops promptly — i.e. **cancel disconnects the
/// stream** and frees the engine.
fn stream_chat(
    stream: &mut DeadlineWriter,
    provider: &dyn TextLlm,
    req: &core_llm::TextLlmRequest,
    cancel: &CancelFlag,
    id: &str,
    model: &str,
    created: u64,
) -> io::Result<()> {
    start_sse(stream)?;
    // If even the role chunk can't be written, the client is already gone.
    if sse(stream, &openai::role_chunk(id, model, created)).is_err() {
        cancel.cancel();
        return Ok(());
    }

    let mut disconnected = false;
    let result = {
        let mut sink = |ev: StreamEvent| {
            if disconnected {
                return;
            }
            if let StreamEvent::Token { text, .. } = ev {
                if !text.is_empty()
                    && sse(stream, &openai::content_chunk(id, model, created, &text)).is_err()
                {
                    cancel.cancel();
                    disconnected = true;
                }
            }
        };
        provider.generate(req, &mut sink)
    };

    if disconnected {
        return Ok(()); // socket is dead; nothing more to send
    }
    finish_sse(stream, id, model, created, result)
}

/// Write one SSE event (`data: <payload>\n\n`) and flush it so the client sees it immediately.
fn sse(w: &mut impl Write, data: &str) -> io::Result<()> {
    write!(w, "data: {data}\n\n")?;
    w.flush()
}

/// Write a fixed-length JSON response with the given status.
fn write_json(stream: &mut DeadlineWriter, status: u16, body: &str) -> io::Result<()> {
    write_response(stream, status, "application/json", body.as_bytes())
}

/// Write a fixed-length plain-text response.
fn write_text(stream: &mut DeadlineWriter, status: u16, body: &str) -> io::Result<()> {
    write_response(stream, status, "text/plain; charset=utf-8", body.as_bytes())
}

/// Keep backend diagnostics (which may contain local paths or model details) on the server side.
/// The reference server may be exposed with `--host`, so 500 responses are generic on every bind.
fn server_error_body(error: &CoreError) -> String {
    eprintln!("generation error: {error}");
    openai::error_body(INTERNAL_ERROR_MESSAGE, "server_error")
}

fn write_response(
    stream: &mut DeadlineWriter,
    status: u16,
    content_type: &str,
    body: &[u8],
) -> io::Result<()> {
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        431 => "Request Header Fields Too Large",
        500 => "Internal Server Error",
        _ => "OK",
    };
    let head = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(head.as_bytes())?;
    stream.write_all(body)?;
    stream.flush()
}

/// Seconds since the Unix epoch (the OpenAI `created` field).
fn unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// A per-process-monotonic completion id (`chatcmpl-…`).
fn completion_id() -> String {
    static N: AtomicU64 = AtomicU64::new(0);
    format!("chatcmpl-{:012}", N.fetch_add(1, Ordering::Relaxed))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;
    use std::net::SocketAddr;

    /// The tests below only exercise routes that never touch the provider (`/health`, parse
    /// errors), so every method is unreachable.
    struct StubLlm;
    impl TextLlm for StubLlm {
        fn descriptor(&self) -> &core_llm::TextLlmDescriptor {
            unreachable!("tests never invoke the provider")
        }
        fn validate(&self, _: &core_llm::TextLlmRequest) -> core_llm::Result<()> {
            unreachable!("tests never invoke the provider")
        }
        fn generate(
            &self,
            _: &core_llm::TextLlmRequest,
            _: &mut dyn FnMut(StreamEvent),
        ) -> core_llm::Result<core_llm::TextLlmOutput> {
            unreachable!("tests never invoke the provider")
        }
    }

    /// Run the real [`serve`] loop on an ephemeral port; returns the address to connect to.
    fn spawn_server(read_timeout: Duration) -> SocketAddr {
        spawn_server_with(
            "test-model".into(),
            ConnectionLimits {
                read_idle: read_timeout,
                request_total: Duration::from_secs(30),
                write: Duration::from_secs(30),
                response_total: Duration::from_secs(30),
            },
        )
    }

    fn spawn_server_with(default_model: String, limits: ConnectionLimits) -> SocketAddr {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            let stub = StubLlm;
            serve(&listener, &stub, &default_model, limits);
        });
        addr
    }

    /// Issue `GET /health` and return the whole response. The generous-but-bounded client read
    /// timeout keeps a regression from hanging the test binary.
    fn get_health(addr: SocketAddr) -> String {
        let mut s = TcpStream::connect(addr).unwrap();
        s.set_read_timeout(Some(Duration::from_secs(30))).unwrap();
        s.write_all(b"GET /health HTTP/1.1\r\n\r\n").unwrap();
        let mut resp = String::new();
        s.read_to_string(&mut resp).unwrap();
        resp
    }

    /// F-022: a peer that connects and sends nothing must not wedge the single-threaded server —
    /// it times out, is dropped without a response, and the next client is served.
    #[test]
    fn silent_connection_times_out_and_next_client_is_served() {
        let addr = spawn_server(Duration::from_millis(200));
        let mut silent = TcpStream::connect(addr).unwrap();
        silent
            .set_read_timeout(Some(Duration::from_secs(30)))
            .unwrap();

        // Served only after the silent peer times out (the server is strictly serial).
        let resp = get_health(addr);
        assert!(
            resp.starts_with("HTTP/1.1 200"),
            "unexpected response: {resp:?}"
        );
        assert!(resp.ends_with("ok"), "unexpected response: {resp:?}");

        // The silent connection was dropped (clean EOF), not answered.
        let mut buf = [0u8; 16];
        assert_eq!(silent.read(&mut buf).unwrap(), 0);
    }

    /// F-022: silence *mid-request* (partial headers, then nothing) is also treated as a dropped
    /// connection, and the loop continues serving subsequent clients.
    #[test]
    fn mid_request_silence_times_out_and_next_client_is_served() {
        let addr = spawn_server(Duration::from_millis(200));
        let mut stalled = TcpStream::connect(addr).unwrap();
        // A valid request line and a header fragment with no terminator, then silence.
        stalled
            .write_all(b"GET /health HTTP/1.1\r\nHost: x")
            .unwrap();

        let resp = get_health(addr);
        assert!(
            resp.starts_with("HTTP/1.1 200"),
            "unexpected response: {resp:?}"
        );
    }

    /// sc-12548: successful reads do not reset the request's wall-clock deadline. A peer that
    /// trickles bytes faster than the idle timeout is still dropped, then the next client is served.
    #[test]
    fn trickling_request_hits_receive_deadline_and_next_client_is_served() {
        let addr = spawn_server_with(
            "test-model".into(),
            ConnectionLimits {
                read_idle: Duration::from_secs(1),
                request_total: Duration::from_millis(120),
                write: Duration::from_secs(30),
                response_total: Duration::from_secs(30),
            },
        );
        let mut trickle = TcpStream::connect(addr).unwrap();
        trickle.write_all(b"G").unwrap();
        for _ in 0..4 {
            std::thread::sleep(Duration::from_millis(30));
            // The final write may race the server closing at the deadline; either outcome proves
            // the peer cannot keep the request alive by resetting the idle timeout.
            if trickle.write_all(b"E").is_err() {
                break;
            }
        }

        let resp = get_health(addr);
        assert!(
            resp.starts_with("HTTP/1.1 200"),
            "unexpected response: {resp:?}"
        );
    }

    /// sc-12548: a peer that requests a response larger than the socket buffer and never reads is
    /// dropped at the absolute deadline; the serial accept loop then serves the queued request.
    #[test]
    fn blocked_write_times_out_and_next_client_is_served() {
        let addr = spawn_server_with(
            "x".repeat(16 * 1024 * 1024),
            ConnectionLimits {
                read_idle: Duration::from_secs(30),
                request_total: Duration::from_secs(30),
                // Longer than the response deadline: this pins that cumulative progress cannot
                // reset the bound and accidentally reduce it to SO_SNDTIMEO semantics.
                write: Duration::from_secs(10),
                response_total: Duration::from_millis(200),
            },
        );
        let mut blocked = TcpStream::connect(addr).unwrap();
        blocked
            .write_all(b"GET /v1/models HTTP/1.1\r\n\r\n")
            .unwrap();

        let started = Instant::now();
        let resp = get_health(addr);
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "blocked response outlived its configured connection deadline"
        );
        assert!(
            resp.starts_with("HTTP/1.1 200"),
            "unexpected response: {resp:?}"
        );
        drop(blocked);
    }

    /// Exercise the bounded writer directly: output larger than loopback socket buffers must return
    /// a timeout within the absolute deadline even though the kernel accepts partial progress.
    #[test]
    fn bounded_writer_observes_absolute_deadline_on_blocked_output() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let _non_reader = TcpStream::connect(addr).unwrap();
        let (server, _) = listener.accept().unwrap();
        let total = Duration::from_millis(150);
        let mut writer = DeadlineWriter::new(server, Duration::from_secs(10), total);

        let started = Instant::now();
        let err = writer.write_all(&vec![b'x'; 32 * 1024 * 1024]).unwrap_err();
        assert!(
            matches!(
                err.kind(),
                io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
            ),
            "unexpected blocked-write error: {err}"
        );
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "bounded writer did not honor its absolute deadline"
        );
    }

    /// Model compute before the first response byte does not consume the response-write budget.
    #[test]
    fn response_deadline_starts_on_first_write() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let mut client = TcpStream::connect(addr).unwrap();
        client
            .set_read_timeout(Some(Duration::from_secs(1)))
            .unwrap();
        let (server, _) = listener.accept().unwrap();
        let mut writer =
            DeadlineWriter::new(server, Duration::from_secs(1), Duration::from_millis(50));

        std::thread::sleep(Duration::from_millis(100));
        writer.write_all(b"ok").unwrap();
        let mut got = [0u8; 2];
        client.read_exact(&mut got).unwrap();
        assert_eq!(&got, b"ok");
    }

    /// JSON, plain text, and SSE all route through the same deadline-aware writer.
    #[test]
    fn every_response_mode_uses_the_bounded_writer() {
        fn expired_writer() -> DeadlineWriter {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let addr = listener.local_addr().unwrap();
            let _client = TcpStream::connect(addr).unwrap();
            let (server, _) = listener.accept().unwrap();
            let mut writer =
                DeadlineWriter::new(server, Duration::from_secs(1), Duration::from_secs(1));
            writer.deadline = Some(Instant::now() - Duration::from_millis(1));
            writer
        }

        let mut json = expired_writer();
        assert_eq!(
            write_json(&mut json, 200, "{}").unwrap_err().kind(),
            io::ErrorKind::TimedOut
        );
        let mut text = expired_writer();
        assert_eq!(
            write_text(&mut text, 200, "ok").unwrap_err().kind(),
            io::ErrorKind::TimedOut
        );
        let mut event = expired_writer();
        assert_eq!(
            sse(&mut event, "{}").unwrap_err().kind(),
            io::ErrorKind::TimedOut
        );
    }

    /// F-006, end to end: a no-newline flood gets a 431 response (not OOM), and the server keeps
    /// serving. Exactly `MAX_LINE + 1` bytes so the server consumes the whole flood before
    /// responding — no unread bytes to turn the close into a RST.
    #[test]
    fn request_line_flood_gets_431_and_server_keeps_serving() {
        let addr = spawn_server(Duration::from_secs(30));
        let mut flood = TcpStream::connect(addr).unwrap();
        flood
            .set_read_timeout(Some(Duration::from_secs(30)))
            .unwrap();
        flood
            .write_all(&vec![b'A'; http::MAX_LINE as usize + 1])
            .unwrap();
        let mut resp = String::new();
        flood.read_to_string(&mut resp).unwrap();
        assert!(
            resp.starts_with("HTTP/1.1 431"),
            "unexpected response: {resp:?}"
        );

        let resp2 = get_health(addr);
        assert!(
            resp2.starts_with("HTTP/1.1 200"),
            "unexpected response: {resp2:?}"
        );
    }

    /// Records each `generate_batch` call's size; answers every request with its own index.
    struct BatchingLlm {
        batches: std::sync::Arc<std::sync::Mutex<Vec<usize>>>,
    }
    impl TextLlm for BatchingLlm {
        fn descriptor(&self) -> &core_llm::TextLlmDescriptor {
            unreachable!("the server never asks for the descriptor")
        }
        fn validate(&self, _: &core_llm::TextLlmRequest) -> core_llm::Result<()> {
            Ok(())
        }
        fn generate(
            &self,
            _: &core_llm::TextLlmRequest,
            _: &mut dyn FnMut(StreamEvent),
        ) -> core_llm::Result<core_llm::TextLlmOutput> {
            unreachable!("concurrent requests go through the batch")
        }
        fn generate_batch(
            &self,
            reqs: &[core_llm::TextLlmRequest],
            on_event: &mut dyn FnMut(usize, StreamEvent),
        ) -> Vec<core_llm::Result<core_llm::TextLlmOutput>> {
            self.batches.lock().unwrap().push(reqs.len());
            (0..reqs.len())
                .map(|i| {
                    let text = format!("batched-{i}");
                    on_event(
                        i,
                        StreamEvent::Token {
                            id: 1,
                            text: text.clone(),
                            index: 0,
                            channel: core_llm::Channel::Content,
                        },
                    );
                    Ok(core_llm::TextLlmOutput {
                        text,
                        finish_reason: Some(core_llm::FinishReason::Stop),
                        kv_cache: Some(core_llm::KvCacheReport::dense(
                            core_llm::KvCacheFallbackReason::PolicyDisabled,
                            None,
                        )),
                        ..Default::default()
                    })
                })
                .collect()
        }
    }

    /// sc-20681: chat completions waiting together (here two, one streamed and one buffered,
    /// both connected before the server's next accept) run as one `generate_batch` call, and each
    /// client gets its own answer and `kv_cache` report.
    #[test]
    fn concurrent_chat_requests_run_as_one_batch() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let send = |stream: bool| {
            let body =
                format!(r#"{{"messages":[{{"role":"user","content":"hi"}}],"stream":{stream}}}"#);
            let mut client = TcpStream::connect(addr).unwrap();
            client
                .set_read_timeout(Some(Duration::from_secs(30)))
                .unwrap();
            write!(
                client,
                "POST /v1/chat/completions HTTP/1.1\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            )
            .unwrap();
            client
        };
        let mut buffered = send(false);
        let mut streamed = send(true);
        let batches = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let recorded = batches.clone();
        std::thread::spawn(move || {
            let provider = BatchingLlm { batches: recorded };
            serve(
                &listener,
                &provider,
                "test-model",
                ConnectionLimits::PRODUCTION,
            );
        });
        let mut buffered_response = String::new();
        buffered.read_to_string(&mut buffered_response).unwrap();
        let mut streamed_response = String::new();
        streamed.read_to_string(&mut streamed_response).unwrap();
        assert_eq!(*batches.lock().unwrap(), vec![2]);
        assert!(
            buffered_response.starts_with("HTTP/1.1 200")
                && buffered_response.contains("batched-0")
                && buffered_response.contains("policy_disabled"),
            "{buffered_response}"
        );
        assert!(
            streamed_response.contains("text/event-stream")
                && streamed_response.contains("batched-1")
                && streamed_response.contains("policy_disabled")
                && streamed_response.ends_with("data: [DONE]\n\n"),
            "{streamed_response}"
        );
    }

    #[test]
    fn internal_error_body_does_not_expose_backend_detail() {
        let secret = "/Users/private/models/checkpoint.safetensors";
        let body = server_error_body(&CoreError::Msg(secret.into()));
        assert!(body.contains(INTERNAL_ERROR_MESSAGE));
        assert!(!body.contains(secret));
    }
}
