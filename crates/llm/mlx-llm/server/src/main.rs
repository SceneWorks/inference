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
//! This is a *reference*, deliberately minimal: one model, one request at a time (MLX's Metal device
//! is single-threaded — see the engine's `.cargo/config.toml`), `Connection: close`, no auth. A
//! production gateway (multi-model, auth, batching across requests, Anthropic/Ollama compat) is the
//! separate server-app project, not this example.
//!
//! ```text
//! curl -N http://localhost:8080/v1/chat/completions \
//!   -H 'content-type: application/json' \
//!   -d '{"model":"local","stream":true,"messages":[{"role":"user","content":"Hi!"}]}'
//! ```

mod http;
mod openai;

use std::io::{self, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use mlx_llm::core_llm::{
    self, CancelFlag, Error as CoreError, LoadSpec, Quantize, StreamEvent, TextLlm,
};

/// How long a connected peer may stay silent before its connection is dropped (F-022). The server
/// is single-threaded, so a peer that connects and sends nothing (a stray `nc`) would otherwise
/// block `read_line` forever and wedge every subsequent client. 10 seconds is generous for any
/// legitimate client writing a request (even by hand over a slow link) while bounding how long one
/// idle connection can monopolise the serving thread.
const READ_TIMEOUT: Duration = Duration::from_secs(10);
const INTERNAL_ERROR_MESSAGE: &str = "internal server error";

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
        quantize: args.quantize,
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

    serve(&listener, provider.as_ref(), &default_model, READ_TIMEOUT);
    Ok(())
}

/// The serial accept loop. A per-connection error (including a read timeout) drops that connection
/// only — the loop always continues serving subsequent clients.
fn serve(
    listener: &TcpListener,
    provider: &dyn TextLlm,
    default_model: &str,
    read_timeout: Duration,
) {
    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                // F-022: bound how long a silent peer can hold the single serving thread.
                if let Err(e) = stream.set_read_timeout(Some(read_timeout)) {
                    eprintln!("connection error: {e}");
                    continue;
                }
                if let Err(e) = handle_connection(stream, provider, default_model) {
                    eprintln!("connection error: {e}");
                }
            }
            Err(e) => eprintln!("accept error: {e}"),
        }
    }
}

/// Serve one request on a connection, then close it (`Connection: close`).
fn handle_connection(
    mut stream: TcpStream,
    provider: &dyn TextLlm,
    default_model: &str,
) -> io::Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let req = match http::read_request(&mut reader) {
        Ok(Some(req)) => req,
        Ok(None) => return Ok(()), // idle disconnect
        // Read timeout (F-022): the peer went silent mid-request — treat it as a dropped
        // connection, not an error worth replying to (the peer isn't reading anyway).
        Err(e)
            if matches!(
                e.kind(),
                io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
            ) =>
        {
            return Ok(());
        }
        Err(e) => {
            let status = http::error_status(&e);
            return write_json(
                &mut stream,
                status,
                &openai::error_body(&e.to_string(), "invalid_request"),
            );
        }
    };

    match (req.method.as_str(), req.path.as_str()) {
        ("POST", "/v1/chat/completions") => {
            handle_chat(&mut stream, provider, &req.body, default_model)
        }
        ("GET", "/v1/models") => write_json(
            &mut stream,
            200,
            &openai::models_list(default_model, unix_secs()),
        ),
        ("GET", "/" | "/health") => write_text(&mut stream, 200, "ok"),
        _ => write_json(
            &mut stream,
            404,
            &openai::error_body("not found", "not_found"),
        ),
    }
}

/// Handle a chat completion: parse → validate → stream SSE or return one JSON body.
fn handle_chat(
    stream: &mut TcpStream,
    provider: &dyn TextLlm,
    body: &[u8],
    default_model: &str,
) -> io::Result<()> {
    let chat: openai::ChatRequest = match serde_json::from_slice(body) {
        Ok(c) => c,
        Err(e) => {
            return write_json(
                stream,
                400,
                &openai::error_body(&e.to_string(), "invalid_request"),
            )
        }
    };
    let model = chat
        .model
        .clone()
        .unwrap_or_else(|| default_model.to_string());
    let want_stream = chat.stream;

    let mut req = match chat.into_text_llm_request() {
        Ok(r) => r,
        Err(msg) => return write_json(stream, 400, &openai::error_body(&msg, "invalid_request")),
    };
    // Reject anything outside the provider's declared surface before sending any 200.
    if let Err(e) = provider.validate(&req) {
        return write_json(
            stream,
            400,
            &openai::error_body(&e.to_string(), "invalid_request"),
        );
    }

    let cancel = CancelFlag::new();
    req.cancel = cancel.clone();
    let id = completion_id();
    let created = unix_secs();

    if want_stream {
        stream_chat(stream, provider, &req, &cancel, &id, &model, created)
    } else {
        match provider.complete(&req) {
            Ok(out) => {
                let finish = out
                    .finish_reason
                    .map(openai::finish_reason_str)
                    .unwrap_or("stop");
                let body = openai::completion(
                    &id,
                    &model,
                    created,
                    &out.text,
                    finish,
                    out.usage.prompt_tokens,
                    out.usage.generated_tokens,
                );
                write_json(stream, 200, &body)
            }
            Err(CoreError::Canceled) => Ok(()), // client vanished mid-generation
            Err(e) => write_json(stream, 500, &server_error_body(&e)),
        }
    }
}

/// Stream a chat completion as Server-Sent Events. A failed write (client disconnected) trips the
/// request's [`CancelFlag`], so the decode loop stops promptly — i.e. **cancel disconnects the
/// stream** and frees the engine.
fn stream_chat(
    stream: &mut TcpStream,
    provider: &dyn TextLlm,
    req: &core_llm::TextLlmRequest,
    cancel: &CancelFlag,
    id: &str,
    model: &str,
    created: u64,
) -> io::Result<()> {
    stream.write_all(
        b"HTTP/1.1 200 OK\r\n\
          Content-Type: text/event-stream\r\n\
          Cache-Control: no-cache\r\n\
          Connection: close\r\n\
          X-Accel-Buffering: no\r\n\r\n",
    )?;
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
    match result {
        Ok(out) => {
            let finish = out
                .finish_reason
                .map(openai::finish_reason_str)
                .unwrap_or("stop");
            let _ = sse(stream, &openai::final_chunk(id, model, created, finish));
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

/// Write one SSE event (`data: <payload>\n\n`) and flush it so the client sees it immediately.
fn sse(w: &mut impl Write, data: &str) -> io::Result<()> {
    write!(w, "data: {data}\n\n")?;
    w.flush()
}

/// Write a fixed-length JSON response with the given status.
fn write_json(stream: &mut TcpStream, status: u16, body: &str) -> io::Result<()> {
    write_response(stream, status, "application/json", body.as_bytes())
}

/// Write a fixed-length plain-text response.
fn write_text(stream: &mut TcpStream, status: u16, body: &str) -> io::Result<()> {
    write_response(stream, status, "text/plain; charset=utf-8", body.as_bytes())
}

/// Keep backend diagnostics (which may contain local paths or model details) on the server side.
/// The reference server may be exposed with `--host`, so 500 responses are generic on every bind.
fn server_error_body(error: &CoreError) -> String {
    eprintln!("generation error: {error}");
    openai::error_body(INTERNAL_ERROR_MESSAGE, "server_error")
}

fn write_response(
    stream: &mut TcpStream,
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
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            let stub = StubLlm;
            serve(&listener, &stub, "test-model", read_timeout);
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

    #[test]
    fn internal_error_body_does_not_expose_backend_detail() {
        let secret = "/Users/private/models/checkpoint.safetensors";
        let body = server_error_body(&CoreError::Msg(secret.into()));
        assert!(body.contains(INTERNAL_ERROR_MESSAGE));
        assert!(!body.contains(secret));
    }
}
