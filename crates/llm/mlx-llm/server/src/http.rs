//! A minimal HTTP/1.1 request reader — just enough to serve a single JSON/SSE endpoint without an
//! HTTP-framework dependency. Reads the request line, headers, and a `Content-Length` body; the
//! server replies `Connection: close`, so one request per connection (no keep-alive/chunked-body
//! handling, which a real gateway would add).

use std::io::{self, BufRead, Read};

/// Reject absurd request bodies up front (a hostile or buggy `Content-Length`).
const MAX_BODY: usize = 16 * 1024 * 1024;

/// Cap on the request line and on each header line, in bytes including the terminator. 8 KiB
/// matches common server defaults (Apache `LimitRequestLine`, nginx `large_client_header_buffers`).
/// Without this, a peer streaming bytes with no `\n` grows one `String` without bound (F-006).
pub(crate) const MAX_LINE: u64 = 8 * 1024;

/// Cap on header-loop lines, including the terminating blank line. The effective field cap is 99.
/// Together with [`MAX_LINE`] this bounds total header bytes below 800 KiB and guarantees the loop
/// terminates even on an endless header stream (F-006).
const MAX_HEADERS: usize = 100;

/// A parsed request: method, path (query stripped), and the raw body bytes.
#[derive(Debug, PartialEq, Eq)]
pub struct Request {
    pub method: String,
    pub path: String,
    pub body: Vec<u8>,
}

/// Read one request from `reader`. `Ok(None)` means the peer closed before sending anything (a clean
/// idle disconnect). A malformed request line or an over-cap body is an [`io::Error`]; cap
/// violations carry [`io::ErrorKind::FileTooLarge`] so [`error_status`] can map them to 431.
pub fn read_request(reader: &mut impl BufRead) -> io::Result<Option<Request>> {
    let mut line = String::new();
    if read_line_capped(reader, &mut line)? == 0 {
        return Ok(None); // peer closed with no request
    }
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or_default().to_string();
    let target = parts.next().unwrap_or_default().to_string();
    if method.is_empty() || target.is_empty() {
        return Err(invalid("malformed request line"));
    }
    // Strip any query string for routing.
    let path = target.split('?').next().unwrap_or("").to_string();

    // Headers until a blank line; we only need Content-Length. Both the header count and each
    // line's length are capped, so a hostile peer cannot spin this loop or grow `h` unboundedly.
    let mut content_length = 0usize;
    let mut header_count = 0usize;
    loop {
        header_count += 1;
        if header_count > MAX_HEADERS {
            return Err(too_large("too many headers"));
        }
        let mut h = String::new();
        if read_line_capped(reader, &mut h)? == 0 {
            return Err(invalid("unexpected EOF in headers"));
        }
        let h = h.trim_end();
        if h.is_empty() {
            break;
        }
        if let Some((name, value)) = h.split_once(':') {
            if name.trim().eq_ignore_ascii_case("content-length") {
                content_length = value
                    .trim()
                    .parse()
                    .map_err(|_| invalid("bad Content-Length"))?;
            }
        }
    }
    if content_length > MAX_BODY {
        return Err(invalid("request body too large"));
    }

    let mut body = vec![0u8; content_length];
    reader.read_exact(&mut body)?;
    Ok(Some(Request { method, path, body }))
}

/// Read one line, refusing to buffer more than [`MAX_LINE`] bytes. The `Take` wrapper caps how much
/// the `read_line` below can pull *through* the `BufReader` per call (buffered bytes are preserved
/// across calls, since `Take` merely delegates to the underlying `BufRead`). A fresh `Take` is
/// created per line so each line gets the full budget.
fn read_line_capped(reader: &mut impl BufRead, line: &mut String) -> io::Result<usize> {
    // `MAX_LINE + 1` so a line of exactly MAX_LINE bytes (terminator included) still parses, and
    // anything longer is detected by having read past the cap without finding `\n`.
    let n = reader.by_ref().take(MAX_LINE + 1).read_line(line)?;
    if n as u64 > MAX_LINE {
        return Err(too_large("header line too long"));
    }
    Ok(n)
}

/// The HTTP status a `read_request` error should map to: 431 (Request Header Fields Too Large) for
/// request-line/header cap violations, 400 for everything else.
pub fn error_status(e: &io::Error) -> u16 {
    if e.kind() == io::ErrorKind::FileTooLarge {
        431
    } else {
        400
    }
}

fn invalid(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

fn too_large(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::FileTooLarge, msg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn parses_post_with_body() {
        let raw = "POST /v1/chat/completions?x=1 HTTP/1.1\r\n\
                   Host: localhost\r\n\
                   Content-Type: application/json\r\n\
                   Content-Length: 14\r\n\
                   \r\n\
                   {\"hello\":\"hi\"}";
        let mut c = Cursor::new(raw.as_bytes());
        let req = read_request(&mut c).unwrap().unwrap();
        assert_eq!(req.method, "POST");
        assert_eq!(req.path, "/v1/chat/completions"); // query stripped
        assert_eq!(req.body, b"{\"hello\":\"hi\"}");
    }

    #[test]
    fn parses_get_without_body() {
        let mut c = Cursor::new(&b"GET /v1/models HTTP/1.1\r\nHost: x\r\n\r\n"[..]);
        let req = read_request(&mut c).unwrap().unwrap();
        assert_eq!(req.method, "GET");
        assert_eq!(req.path, "/v1/models");
        assert!(req.body.is_empty());
    }

    #[test]
    fn content_length_case_insensitive() {
        let mut c = Cursor::new(&b"POST / HTTP/1.1\r\ncontent-length: 2\r\n\r\nhi"[..]);
        let req = read_request(&mut c).unwrap().unwrap();
        assert_eq!(req.body, b"hi");
    }

    #[test]
    fn empty_stream_is_none() {
        let mut c = Cursor::new(&b""[..]);
        assert_eq!(read_request(&mut c).unwrap(), None);
    }

    #[test]
    fn oversized_body_rejected() {
        let raw = format!(
            "POST / HTTP/1.1\r\nContent-Length: {}\r\n\r\n",
            MAX_BODY + 1
        );
        let mut c = Cursor::new(raw.into_bytes());
        let err = read_request(&mut c).unwrap_err();
        assert_eq!(error_status(&err), 400);
    }

    /// F-006: an endless byte stream with no `\n` must be rejected after at most [`MAX_LINE`]
    /// bytes — this returning at all (instead of hanging/OOM) is the point of the test.
    #[test]
    fn no_newline_flood_rejected_as_431() {
        let mut r = std::io::BufReader::new(std::io::repeat(b'A'));
        let err = read_request(&mut r).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::FileTooLarge);
        assert_eq!(error_status(&err), 431);
    }

    /// F-006: an over-long header line (after a valid request line) is rejected, not buffered.
    #[test]
    fn overlong_header_line_rejected_as_431() {
        let mut raw = b"GET / HTTP/1.1\r\nX-Flood: ".to_vec();
        raw.extend(std::iter::repeat_n(b'a', 2 * MAX_LINE as usize));
        raw.extend_from_slice(b"\r\n\r\n");
        let mut c = Cursor::new(raw);
        let err = read_request(&mut c).unwrap_err();
        assert_eq!(error_status(&err), 431);
    }

    /// F-006: an *endless* header stream must terminate the header loop at [`MAX_HEADERS`].
    #[test]
    fn endless_header_flood_rejected_as_431() {
        /// Yields `GET / HTTP/1.1\r\n` once, then `X-H: y\r\n` forever.
        struct HeaderFlood {
            sent_request_line: bool,
            pos: usize,
        }
        impl std::io::Read for HeaderFlood {
            fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
                const REQUEST_LINE: &[u8] = b"GET / HTTP/1.1\r\n";
                const HEADER: &[u8] = b"X-H: y\r\n";
                if !self.sent_request_line {
                    buf[..REQUEST_LINE.len()].copy_from_slice(REQUEST_LINE);
                    self.sent_request_line = true;
                    return Ok(REQUEST_LINE.len());
                }
                let mut n = 0;
                for b in buf.iter_mut() {
                    *b = HEADER[self.pos];
                    self.pos = (self.pos + 1) % HEADER.len();
                    n += 1;
                }
                Ok(n)
            }
        }
        let mut r = std::io::BufReader::new(HeaderFlood {
            sent_request_line: false,
            pos: 0,
        });
        let err = read_request(&mut r).unwrap_err();
        assert_eq!(error_status(&err), 431);
    }

    /// A request line of exactly [`MAX_LINE`] bytes (terminator included) still parses — the cap
    /// rejects only lines *beyond* the limit, so legitimate requests are byte-identical.
    #[test]
    fn request_line_at_cap_still_parses() {
        let target_len = MAX_LINE as usize - "GET  HTTP/1.1\r\n".len();
        let target: String = std::iter::once('/')
            .chain(std::iter::repeat_n('x', target_len - 1))
            .collect();
        let request_line = format!("GET {target} HTTP/1.1\r\n");
        assert_eq!(request_line.len() as u64, MAX_LINE);
        let mut c = Cursor::new(format!("{request_line}\r\n").into_bytes());
        let req = read_request(&mut c).unwrap().unwrap();
        assert_eq!(req.path, target);
    }

    /// Exactly 99 header fields plus the terminating blank line still parse.
    #[test]
    fn header_count_at_cap_still_parses() {
        let mut raw = b"GET / HTTP/1.1\r\n".to_vec();
        // MAX_HEADERS counts every header-loop line read, including the terminating blank line.
        for _ in 0..MAX_HEADERS - 1 {
            raw.extend_from_slice(b"X-H: y\r\n");
        }
        raw.extend_from_slice(b"\r\n");
        let mut c = Cursor::new(raw);
        assert!(read_request(&mut c).unwrap().is_some());
    }
}
