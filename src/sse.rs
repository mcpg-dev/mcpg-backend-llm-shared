//! SSE event-boundary parsing helpers.
//!
//! Three provider adapters (OpenAI, Anthropic, Gemini) consume
//! Server-Sent Events from upstream HTTP responses. The framing rule
//! — events end at `\n\n` or `\r\n\r\n` — is identical across them.
//! Lifting it here removes the per-adapter duplication.

/// Find the end-of-event boundary (`\n\n` or `\r\n\r\n`) in `buf`.
/// Returns the index where the event payload ends — boundary
/// excluded. Caller drains `..idx` to consume the payload, then
/// strips the boundary bytes via [`strip_boundary_prefix`].
///
/// Recognizes both LF-only (`\n\n`) and CRLF (`\r\n\r\n`) framings.
/// The CRLF check runs first because OpenAI / Anthropic / Gemini all
/// emit CRLF in their HTTPS responses today; the LF-only path is
/// belt-and-suspenders for HTTP/2 frames stripped of their `\r`.
pub fn find_event_boundary(buf: &[u8]) -> Option<usize> {
    if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
        return Some(pos);
    }
    buf.windows(2).position(|w| w == b"\n\n")
}

/// After draining up to the boundary index, strip the boundary
/// bytes themselves from the head of `buf`. Returns the number of
/// bytes consumed (2 or 4).
pub fn strip_boundary_prefix(buf: &mut Vec<u8>) -> usize {
    if buf.starts_with(b"\r\n\r\n") {
        buf.drain(..4);
        4
    } else if buf.starts_with(b"\n\n") {
        buf.drain(..2);
        2
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_lf_boundary() {
        let buf = b"data: hello\n\ndata: next\n\n";
        assert_eq!(find_event_boundary(buf), Some(11));
    }

    #[test]
    fn finds_crlf_boundary() {
        let buf = b"data: hello\r\n\r\ndata: next\r\n\r\n";
        assert_eq!(find_event_boundary(buf), Some(11));
    }

    #[test]
    fn returns_none_when_no_boundary() {
        let buf = b"data: hello";
        assert_eq!(find_event_boundary(buf), None);
    }

    #[test]
    fn strip_consumes_correct_byte_count() {
        let mut buf = b"\r\n\r\nrest".to_vec();
        assert_eq!(strip_boundary_prefix(&mut buf), 4);
        assert_eq!(buf, b"rest");

        let mut buf = b"\n\nrest".to_vec();
        assert_eq!(strip_boundary_prefix(&mut buf), 2);
        assert_eq!(buf, b"rest");

        // Wrong prefix → no consumption.
        let mut buf = b"data".to_vec();
        assert_eq!(strip_boundary_prefix(&mut buf), 0);
        assert_eq!(buf, b"data");
    }

    #[test]
    fn crlf_preferred_over_lf_when_both_present() {
        // Synthetic stream with embedded CRLF before LF-LF — should
        // pick the CRLF boundary first since it's earlier and the
        // canonical SSE shape.
        let buf = b"data\r\n\r\nmore\n\n";
        assert_eq!(find_event_boundary(buf), Some(4));
    }
}
