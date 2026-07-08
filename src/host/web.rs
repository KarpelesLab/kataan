//! §4.3 Web platform globals — the WHATWG / WHATWG-adjacent browser globals a
//! script expects from a modern JS runtime, built on the §4.0 embedding API
//! ([`Interp::register_global_fn`] / [`Interp::register_global_constructor`] /
//! [`Ctx`]). Installed via [`install`].
//!
//! Implemented here: `TextEncoder` / `TextDecoder`, `atob` / `btoa`, `URL` /
//! `URLSearchParams`, `structuredClone`, `performance`, and a full-formatting
//! `console`.
//!
//! Simplifications (deliberate, noted inline): URL host parsing does no IDNA /
//! punycode (hosts are stored roughly as given, ASCII-lowercased for special
//! schemes) and its path normalization covers the common `.`/`..` cases;
//! `url.searchParams` returns a fresh (non-live) `URLSearchParams`;
//! `TextDecoder` supports utf-8 / utf-16le / utf-16be; `structuredClone` cycle
//! detection keys on raw heap handles (correct for a single non-GC-relocating
//! clone).

use alloc::format;
use alloc::rc::Rc;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use core::cell::RefCell;
use std::collections::HashMap;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::heap::Handle;
use crate::nanbox::{NanBox, Unpacked};
use crate::nbexec::{Ctx, Interp};

/// Install the §4.3 web-platform globals into `interp`.
pub fn install(interp: &mut Interp<'_>) {
    install_encoding(interp);
    install_base64(interp);
    install_url(interp);
    install_structured_clone(interp);
    install_performance(interp);
    install_console(interp);
}

// ---------------------------------------------------------------------------
// small shared helpers
// ---------------------------------------------------------------------------

/// `args[i]` or `undefined`.
fn arg(args: &[NanBox], i: usize) -> NanBox {
    args.get(i).copied().unwrap_or_else(NanBox::undefined)
}

fn is_nullish(v: NanBox) -> bool {
    matches!(v.unpack(), Unpacked::Undefined | Unpacked::Null)
}

/// Read a named global (`globalThis[name]`), `undefined` on any failure.
fn global_get(cx: &mut Ctx, name: &str) -> NanBox {
    let g = cx.global();
    cx.get(g, name).unwrap_or_else(|_| NanBox::undefined())
}

/// `Object.prototype.toString.call(v)` → the inner builtin tag, e.g. `"Map"`,
/// `"Array"`, `"Uint8Array"`, `"Object"`. Empty string on failure.
fn builtin_tag(cx: &mut Ctx, v: NanBox) -> String {
    let obj = global_get(cx, "Object");
    let proto = cx
        .get(obj, "prototype")
        .unwrap_or_else(|_| NanBox::undefined());
    let ts = cx
        .get(proto, "toString")
        .unwrap_or_else(|_| NanBox::undefined());
    let Ok(r) = cx.call(ts, v, &[]) else {
        return String::new();
    };
    let s = cx.to_string(r).unwrap_or_default();
    // "[object Xxx]" → "Xxx"
    s.strip_prefix("[object ")
        .and_then(|s| s.strip_suffix(']'))
        .map(ToString::to_string)
        .unwrap_or(s)
}

/// Attach a host method (`register_fn`) to `target`'s own properties.
fn add_method<F>(interp: &mut Interp<'_>, target: NanBox, name: &str, length: u32, f: F)
where
    F: FnMut(&mut Ctx<'_, '_>, NanBox, &[NanBox]) -> Result<NanBox, NanBox> + 'static,
{
    let m = interp.register_fn(name, length, f);
    if let Some(h) = target.as_handle().map(Handle::from_raw) {
        interp.realm_mut().set_property(h, name, m);
    }
}

/// The `.prototype` object of a constructor value, if any.
fn prototype_of(interp: &Interp<'_>, ctor: NanBox) -> Option<NanBox> {
    let h = ctor.as_handle().map(Handle::from_raw)?;
    interp.realm().get_property(h, "prototype")
}

/// Define a getter/setter accessor `name` on `proto`. `set` is a no-op-returning
/// closure for read-only accessors.
fn add_accessor<G, S>(interp: &mut Interp<'_>, proto: NanBox, name: &str, get: G, set: S)
where
    G: FnMut(&mut Ctx<'_, '_>, NanBox, &[NanBox]) -> Result<NanBox, NanBox> + 'static,
    S: FnMut(&mut Ctx<'_, '_>, NanBox, &[NanBox]) -> Result<NanBox, NanBox> + 'static,
{
    let g = interp.register_fn(&format!("get {name}"), 0, get);
    let s = interp.register_fn(&format!("set {name}"), 1, set);
    if let Some(h) = proto.as_handle().map(Handle::from_raw) {
        interp.realm_mut().define_accessor(h, name, g, s);
    }
}

/// Build a JS value carrying an `InvalidCharacterError`-style name.
fn named_error(cx: &mut Ctx, name: &str, message: &str) -> NanBox {
    let e = cx.error(message);
    let n = cx.string(name);
    cx.set(e, "name", n);
    e
}

// ===========================================================================
// TextEncoder / TextDecoder
// ===========================================================================

fn install_encoding(interp: &mut Interp<'_>) {
    // --- TextEncoder ---
    let enc_ctor = interp.register_global_constructor("TextEncoder", 0, |cx, this, _args| {
        let e = cx.string("utf-8");
        cx.set(this, "encoding", e);
        Ok(this)
    });
    if let Some(proto) = prototype_of(interp, enc_ctor) {
        add_method(interp, proto, "encode", 1, |cx, _this, args| {
            let s = if is_nullish(arg(args, 0)) {
                String::new()
            } else {
                cx.to_string(arg(args, 0))?
            };
            Ok(bytes_to_u8array(cx, s.as_bytes()))
        });
        add_method(interp, proto, "encodeInto", 2, |cx, _this, args| {
            let src = cx.to_string(arg(args, 0))?;
            let dest = arg(args, 1);
            let cap = cx
                .get(dest, "length")
                .ok()
                .and_then(|l| cx.to_number(l).ok())
                .unwrap_or(0.0) as usize;
            let mut written = 0usize;
            let mut read = 0usize; // UTF-16 code units consumed
            for ch in src.chars() {
                let mut buf = [0u8; 4];
                let bytes = ch.encode_utf8(&mut buf).as_bytes();
                if written + bytes.len() > cap {
                    break;
                }
                for &b in bytes {
                    let _ =
                        cx.set_property(dest, &written.to_string(), NanBox::number(f64::from(b)));
                    written += 1;
                }
                read += ch.len_utf16();
            }
            let out = cx.new_object();
            let r = NanBox::number(read as f64);
            let w = NanBox::number(written as f64);
            cx.set(out, "read", r);
            cx.set(out, "written", w);
            Ok(out)
        });
    }

    // --- TextDecoder ---
    let dec_ctor = interp.register_global_constructor("TextDecoder", 0, |cx, this, args| {
        let label = if is_nullish(arg(args, 0)) {
            String::from("utf-8")
        } else {
            cx.to_string(arg(args, 0))?
        };
        let Some(encoding) = normalize_encoding_label(&label) else {
            return Err(cx.range_error(&format!("The encoding label \"{label}\" is invalid.")));
        };
        let opts = arg(args, 1);
        let read_bool = |cx: &mut Ctx, key: &str| -> bool {
            if is_nullish(opts) {
                return false;
            }
            cx.get(opts, key).map(|v| cx.to_boolean(v)).unwrap_or(false)
        };
        let fatal = read_bool(cx, "fatal");
        let ignore_bom = read_bool(cx, "ignoreBOM");
        let enc_v = cx.string(encoding);
        cx.set(this, "encoding", enc_v);
        cx.set(this, "fatal", NanBox::boolean(fatal));
        cx.set(this, "ignoreBOM", NanBox::boolean(ignore_bom));
        Ok(this)
    });
    if let Some(proto) = prototype_of(interp, dec_ctor) {
        add_method(interp, proto, "decode", 1, |cx, this, args| {
            let bytes = read_bytes(cx, arg(args, 0));
            let encoding = cx
                .get(this, "encoding")
                .ok()
                .and_then(|v| cx.to_string(v).ok())
                .unwrap_or_else(|| String::from("utf-8"));
            let fatal = cx
                .get(this, "fatal")
                .map(|v| cx.to_boolean(v))
                .unwrap_or(false);
            let ignore_bom = cx
                .get(this, "ignoreBOM")
                .map(|v| cx.to_boolean(v))
                .unwrap_or(false);
            match decode_bytes(&bytes, &encoding, fatal, ignore_bom) {
                Ok(s) => Ok(cx.string(&s)),
                Err(msg) => {
                    let e = cx.type_error(&msg);
                    let n = cx.string("TypeError");
                    cx.set(e, "name", n);
                    Err(e)
                }
            }
        });
    }
}

fn normalize_encoding_label(label: &str) -> Option<&'static str> {
    match label.trim().to_ascii_lowercase().as_str() {
        "utf-8" | "utf8" | "unicode-1-1-utf-8" | "unicode11utf8" | "unicode20utf8"
        | "x-unicode20utf8" => Some("utf-8"),
        "utf-16le" | "utf-16" | "ucs-2" | "csunicode" | "unicode" | "unicodefeff"
        | "iso-10646-ucs-2" => Some("utf-16le"),
        "utf-16be" | "unicodefffe" => Some("utf-16be"),
        _ => None,
    }
}

fn decode_bytes(
    bytes: &[u8],
    encoding: &str,
    fatal: bool,
    ignore_bom: bool,
) -> Result<String, String> {
    match encoding {
        "utf-16le" | "utf-16be" => {
            let be = encoding == "utf-16be";
            let mut units: Vec<u16> = Vec::with_capacity(bytes.len() / 2);
            for pair in bytes.chunks(2) {
                if pair.len() < 2 {
                    if fatal {
                        return Err(String::from("The encoded data was not valid."));
                    }
                    break;
                }
                let u = if be {
                    u16::from_be_bytes([pair[0], pair[1]])
                } else {
                    u16::from_le_bytes([pair[0], pair[1]])
                };
                units.push(u);
            }
            let mut start = 0;
            if !ignore_bom && units.first() == Some(&0xFEFF) {
                start = 1;
            }
            if fatal {
                String::from_utf16(&units[start..])
                    .map_err(|_| String::from("The encoded data was not valid."))
            } else {
                Ok(String::from_utf16_lossy(&units[start..]))
            }
        }
        _ => {
            // utf-8
            let mut data = bytes;
            if !ignore_bom && data.starts_with(&[0xEF, 0xBB, 0xBF]) {
                data = &data[3..];
            }
            if fatal {
                core::str::from_utf8(data)
                    .map(ToString::to_string)
                    .map_err(|_| String::from("The encoded data was not valid."))
            } else {
                Ok(String::from_utf8_lossy(data).into_owned())
            }
        }
    }
}

/// Build a `Uint8Array` holding `bytes` (via the JS `Uint8Array` constructor).
fn bytes_to_u8array(cx: &mut Ctx, bytes: &[u8]) -> NanBox {
    let ctor = global_get(cx, "Uint8Array");
    let nums: Vec<NanBox> = bytes
        .iter()
        .map(|&b| NanBox::number(f64::from(b)))
        .collect();
    let arr = cx.new_array(nums);
    cx.construct(ctor, &[arr]).unwrap_or(arr)
}

/// Read the raw bytes of any BufferSource (ArrayBuffer / typed array / DataView)
/// by normalizing to a `Uint8Array` and reading each element.
fn read_bytes(cx: &mut Ctx, v: NanBox) -> Vec<u8> {
    if is_nullish(v) {
        return Vec::new();
    }
    let u8ctor = global_get(cx, "Uint8Array");
    let view = if cx.has(v, "buffer") {
        // A typed-array view or DataView: wrap its viewed region.
        let buffer = cx.get(v, "buffer").unwrap_or_else(|_| NanBox::undefined());
        let off = cx
            .get(v, "byteOffset")
            .ok()
            .and_then(|x| cx.to_number(x).ok())
            .unwrap_or(0.0);
        let len = cx
            .get(v, "byteLength")
            .ok()
            .and_then(|x| cx.to_number(x).ok())
            .unwrap_or(0.0);
        cx.construct(u8ctor, &[buffer, NanBox::number(off), NanBox::number(len)])
            .unwrap_or(v)
    } else {
        // Assume an ArrayBuffer.
        cx.construct(u8ctor, &[v]).unwrap_or(v)
    };
    let n = cx
        .get(view, "length")
        .ok()
        .and_then(|l| cx.to_number(l).ok())
        .unwrap_or(0.0) as usize;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let b = cx
            .get(view, &i.to_string())
            .ok()
            .and_then(|e| cx.to_number(e).ok())
            .unwrap_or(0.0);
        out.push(b as u8);
    }
    out
}

// ===========================================================================
// atob / btoa
// ===========================================================================

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn b64_encode(data: &[u8]) -> String {
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0];
        let b1 = chunk.get(1).copied().unwrap_or(0);
        let b2 = chunk.get(2).copied().unwrap_or(0);
        out.push(B64[(b0 >> 2) as usize] as char);
        out.push(B64[(((b0 & 0x03) << 4) | (b1 >> 4)) as usize] as char);
        if chunk.len() > 1 {
            out.push(B64[(((b1 & 0x0f) << 2) | (b2 >> 6)) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(B64[(b2 & 0x3f) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

fn b64_val(c: u8) -> Option<u8> {
    match c {
        b'A'..=b'Z' => Some(c - b'A'),
        b'a'..=b'z' => Some(c - b'a' + 26),
        b'0'..=b'9' => Some(c - b'0' + 52),
        b'+' => Some(62),
        b'/' => Some(63),
        _ => None,
    }
}

fn b64_decode(input: &str) -> Result<Vec<u8>, ()> {
    // Strip ASCII whitespace (HTML "forgiving base64").
    let filtered: Vec<u8> = input
        .bytes()
        .filter(|b| !matches!(b, b' ' | b'\t' | b'\n' | b'\r' | 0x0c))
        .collect();
    let mut data = &filtered[..];
    // Drop up to two trailing '='.
    let mut pad = 0;
    while pad < 2 && data.last() == Some(&b'=') {
        data = &data[..data.len() - 1];
        pad += 1;
    }
    if data.contains(&b'=') {
        return Err(());
    }
    if data.len() % 4 == 1 {
        return Err(());
    }
    let mut vals = Vec::with_capacity(data.len());
    for &b in data {
        vals.push(b64_val(b).ok_or(())?);
    }
    let mut out = Vec::with_capacity(vals.len() * 3 / 4);
    for chunk in vals.chunks(4) {
        match chunk.len() {
            4 => {
                out.push((chunk[0] << 2) | (chunk[1] >> 4));
                out.push((chunk[1] << 4) | (chunk[2] >> 2));
                out.push((chunk[2] << 6) | chunk[3]);
            }
            3 => {
                out.push((chunk[0] << 2) | (chunk[1] >> 4));
                out.push((chunk[1] << 4) | (chunk[2] >> 2));
            }
            2 => {
                out.push((chunk[0] << 2) | (chunk[1] >> 4));
            }
            _ => return Err(()),
        }
    }
    Ok(out)
}

fn install_base64(interp: &mut Interp<'_>) {
    interp.register_global_fn("btoa", 1, |cx, _this, args| {
        let s = cx.to_string(arg(args, 0))?;
        let mut bytes = Vec::with_capacity(s.len());
        for ch in s.chars() {
            let code = ch as u32;
            if code > 0xFF {
                return Err(named_error(
                    cx,
                    "InvalidCharacterError",
                    "The string to be encoded contains characters outside of the Latin1 range.",
                ));
            }
            bytes.push(code as u8);
        }
        let out = b64_encode(&bytes);
        Ok(cx.string(&out))
    });

    interp.register_global_fn("atob", 1, |cx, _this, args| {
        let s = cx.to_string(arg(args, 0))?;
        match b64_decode(&s) {
            Ok(bytes) => {
                let out: String = bytes.iter().map(|&b| char::from(b)).collect();
                Ok(cx.string(&out))
            }
            Err(()) => Err(named_error(
                cx,
                "InvalidCharacterError",
                "The string to be decoded is not correctly encoded.",
            )),
        }
    });
}

// ===========================================================================
// URL / URLSearchParams
// ===========================================================================

#[derive(Clone, Default)]
struct UrlState {
    scheme: String,
    username: String,
    password: String,
    host: String,
    port: String,
    path: String,
    query: Option<String>,
    fragment: Option<String>,
    special: bool,
    has_authority: bool,
}

fn is_special(scheme: &str) -> bool {
    matches!(scheme, "http" | "https" | "ws" | "wss" | "ftp" | "file")
}

fn default_port(scheme: &str) -> Option<&'static str> {
    match scheme {
        "http" | "ws" => Some("80"),
        "https" | "wss" => Some("443"),
        "ftp" => Some("21"),
        _ => None,
    }
}

impl UrlState {
    fn href(&self) -> String {
        let mut s = format!("{}:", self.scheme);
        if self.has_authority {
            s.push_str("//");
            if !self.username.is_empty() || !self.password.is_empty() {
                s.push_str(&self.username);
                if !self.password.is_empty() {
                    s.push(':');
                    s.push_str(&self.password);
                }
                s.push('@');
            }
            s.push_str(&self.host);
            if !self.port.is_empty() {
                s.push(':');
                s.push_str(&self.port);
            }
        }
        s.push_str(&self.path);
        if let Some(q) = &self.query {
            s.push('?');
            s.push_str(q);
        }
        if let Some(f) = &self.fragment {
            s.push('#');
            s.push_str(f);
        }
        s
    }

    fn host_str(&self) -> String {
        if self.port.is_empty() {
            self.host.clone()
        } else {
            format!("{}:{}", self.host, self.port)
        }
    }

    fn origin(&self) -> String {
        if self.special && self.scheme != "file" && !self.host.is_empty() {
            let mut s = format!("{}://{}", self.scheme, self.host);
            if !self.port.is_empty() {
                s.push(':');
                s.push_str(&self.port);
            }
            s
        } else {
            String::from("null")
        }
    }
}

/// Split a `path[?query][#fragment]` tail.
fn split_pqf(s: &str) -> (String, Option<String>, Option<String>) {
    let (before_frag, frag) = match s.find('#') {
        Some(i) => (&s[..i], Some(s[i + 1..].to_string())),
        None => (s, None),
    };
    let (path, query) = match before_frag.find('?') {
        Some(i) => (&before_frag[..i], Some(before_frag[i + 1..].to_string())),
        None => (before_frag, None),
    };
    (path.to_string(), query, frag)
}

/// Collapse `.`/`..` segments, preserving leading and trailing slashes.
fn normalize_path(path: &str) -> String {
    if path.is_empty() {
        return String::new();
    }
    let absolute = path.starts_with('/');
    let segs: Vec<&str> = path.split('/').collect();
    let mut out: Vec<&str> = Vec::new();
    let last = segs.len() - 1;
    for (i, seg) in segs.iter().enumerate() {
        match *seg {
            "." => {
                if i == last {
                    out.push("");
                }
            }
            ".." => {
                if !out.is_empty() && !out.last().unwrap().is_empty() {
                    out.pop();
                } else if out.is_empty() && !absolute {
                    out.push("..");
                }
                if i == last {
                    out.push("");
                }
            }
            "" => {
                // Trailing slash (or leading empty from an absolute prefix, which
                // the `absolute` prefix below re-adds). Interior `//` collapses.
                if i == last && i != 0 {
                    out.push("");
                }
            }
            other => out.push(other),
        }
    }
    let joined = out.join("/");
    if absolute {
        format!("/{}", joined.trim_start_matches('/'))
    } else {
        joined
    }
}

fn parse_host_port(hp: &str, st: &mut UrlState) {
    if let Some(rest) = hp.strip_prefix('[') {
        // IPv6 literal — keep brackets, no normalization.
        if let Some(end) = rest.find(']') {
            st.host = format!("[{}]", &rest[..end]);
            let tail = &rest[end + 1..];
            if let Some(p) = tail.strip_prefix(':') {
                set_port(p, st);
            }
        } else {
            st.host = format!("[{rest}");
        }
        return;
    }
    match hp.rfind(':') {
        Some(i) => {
            st.host = normalize_host(&hp[..i], st.special);
            set_port(&hp[i + 1..], st);
        }
        None => st.host = normalize_host(hp, st.special),
    }
}

fn normalize_host(h: &str, special: bool) -> String {
    if special {
        h.to_ascii_lowercase()
    } else {
        h.to_string()
    }
}

fn set_port(p: &str, st: &mut UrlState) {
    if p.is_empty() {
        st.port = String::new();
        return;
    }
    // Keep only if all digits; drop a default port.
    if p.bytes().all(|b| b.is_ascii_digit()) {
        // Normalize leading zeros (spec parses as integer).
        let n: u64 = p.parse().unwrap_or(0);
        let norm = n.to_string();
        if default_port(&st.scheme) == Some(norm.as_str()) {
            st.port = String::new();
        } else {
            st.port = norm;
        }
    }
}

fn parse_authority_and_path(s: &str, st: &mut UrlState) {
    let end = s
        .find(|c| c == '/' || c == '?' || c == '#' || (st.special && c == '\\'))
        .unwrap_or(s.len());
    let authority = &s[..end];
    let remainder = &s[end..];
    let (userinfo, hostport) = match authority.rfind('@') {
        Some(i) => (Some(&authority[..i]), &authority[i + 1..]),
        None => (None, authority),
    };
    if let Some(ui) = userinfo {
        match ui.find(':') {
            Some(j) => {
                st.username = ui[..j].to_string();
                st.password = ui[j + 1..].to_string();
            }
            None => st.username = ui.to_string(),
        }
    }
    parse_host_port(hostport, st);
    let normalized_remainder: String = remainder.replace('\\', "/");
    let src = if st.special {
        normalized_remainder.as_str()
    } else {
        remainder
    };
    let (p, q, f) = split_pqf(src);
    st.path = if p.is_empty() {
        if st.special {
            String::from("/")
        } else {
            String::new()
        }
    } else {
        normalize_path(&p)
    };
    st.query = q;
    st.fragment = f;
}

/// Parse `input`, resolving against `base` when it is not absolute.
fn parse_url(input: &str, base: Option<&UrlState>) -> Result<UrlState, String> {
    let trimmed = input.trim_matches(|c: char| c <= ' ');
    let cleaned: String = trimmed
        .chars()
        .filter(|&c| c != '\t' && c != '\n' && c != '\r')
        .collect();

    if let Some(st) = parse_absolute(&cleaned) {
        return Ok(st);
    }
    if let Some(base) = base {
        return resolve_relative(&cleaned, base);
    }
    Err(format!("Invalid URL: {input}"))
}

fn parse_absolute(input: &str) -> Option<UrlState> {
    let colon = input.find(':')?;
    let scheme = &input[..colon];
    if scheme.is_empty() || !scheme.as_bytes()[0].is_ascii_alphabetic() {
        return None;
    }
    if !scheme
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'+' | b'-' | b'.'))
    {
        return None;
    }
    let scheme = scheme.to_ascii_lowercase();
    let special = is_special(&scheme);
    let rest = &input[colon + 1..];
    let mut st = UrlState {
        scheme,
        special,
        ..UrlState::default()
    };
    if special {
        let after = rest.trim_start_matches(['/', '\\']);
        st.has_authority = true;
        parse_authority_and_path(after, &mut st);
    } else if let Some(after) = rest.strip_prefix("//") {
        st.has_authority = true;
        parse_authority_and_path(after, &mut st);
    } else {
        // Opaque path (e.g. `mailto:`, `data:`).
        let (p, q, f) = split_pqf(rest);
        st.path = p;
        st.query = q;
        st.fragment = f;
    }
    Some(st)
}

fn resolve_relative(input: &str, base: &UrlState) -> Result<UrlState, String> {
    let mut st = base.clone();
    if input.is_empty() {
        st.fragment = None;
        return Ok(st);
    }
    let first = input.as_bytes()[0];
    match first {
        b'#' => {
            st.fragment = Some(input[1..].to_string());
            Ok(st)
        }
        b'?' => {
            st.query = Some(input[1..].to_string());
            st.fragment = None;
            Ok(st)
        }
        _ if input.starts_with("//") && base.special => {
            // protocol-relative
            let mut new = UrlState {
                scheme: base.scheme.clone(),
                special: base.special,
                has_authority: true,
                ..UrlState::default()
            };
            parse_authority_and_path(&input[2..], &mut new);
            Ok(new)
        }
        b'/' => {
            let (p, q, f) = split_pqf(input);
            st.path = normalize_path(&p);
            st.query = q;
            st.fragment = f;
            Ok(st)
        }
        b'\\' if base.special => {
            let repl = input.replace('\\', "/");
            let (p, q, f) = split_pqf(&repl);
            st.path = normalize_path(&p);
            st.query = q;
            st.fragment = f;
            Ok(st)
        }
        _ => {
            let (p, q, f) = split_pqf(input);
            let base_dir = match base.path.rfind('/') {
                Some(i) => &base.path[..=i],
                None => "/",
            };
            let merged = format!("{base_dir}{p}");
            st.path = normalize_path(&merged);
            st.query = q;
            st.fragment = f;
            Ok(st)
        }
    }
}

// --- URLSearchParams application/x-www-form-urlencoded helpers ---

fn form_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b'%' if i + 2 < bytes.len() => {
                let hi = (bytes[i + 1] as char).to_digit(16);
                let lo = (bytes[i + 2] as char).to_digit(16);
                if let (Some(hi), Some(lo)) = (hi, lo) {
                    out.push((hi * 16 + lo) as u8);
                    i += 3;
                } else {
                    out.push(bytes[i]);
                    i += 1;
                }
            }
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn form_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b' ' => out.push('+'),
            b'*' | b'-' | b'.' | b'_' => out.push(b as char),
            b if b.is_ascii_alphanumeric() => out.push(b as char),
            b => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

fn parse_query_pairs(s: &str) -> Vec<(String, String)> {
    let s = s.strip_prefix('?').unwrap_or(s);
    if s.is_empty() {
        return Vec::new();
    }
    s.split('&')
        .filter(|p| !p.is_empty())
        .map(|p| match p.find('=') {
            Some(i) => (form_decode(&p[..i]), form_decode(&p[i + 1..])),
            None => (form_decode(p), String::new()),
        })
        .collect()
}

fn serialize_pairs(pairs: &[(String, String)]) -> String {
    pairs
        .iter()
        .map(|(k, v)| format!("{}={}", form_encode(k), form_encode(v)))
        .collect::<Vec<_>>()
        .join("&")
}

/// Build an array iterator (a proper iterator object) over `items`.
fn array_iterator(cx: &mut Ctx, iter_key: &str, items: Vec<NanBox>) -> Result<NanBox, NanBox> {
    let arr = cx.new_array(items);
    let itfn = cx.get(arr, iter_key)?;
    cx.call(itfn, arr, &[])
}

fn usp_pairs(cx: &Ctx, this: NanBox) -> Vec<(String, String)> {
    cx.native_state::<Vec<(String, String)>>(this)
        .cloned()
        .unwrap_or_default()
}

fn install_url(interp: &mut Interp<'_>) {
    // Precompute the Symbol.iterator storage key so search-params can hand back
    // genuine iterators.
    let iter_sym = interp.well_known_symbol("iterator");
    let iter_key = interp.member_key(iter_sym);

    // --- URLSearchParams ---
    let usp_ctor = interp.register_global_constructor("URLSearchParams", 0, |cx, this, args| {
        let init = arg(args, 0);
        let pairs: Vec<(String, String)> = if is_nullish(init) {
            Vec::new()
        } else if let Some(existing) = cx.native_state::<Vec<(String, String)>>(init) {
            existing.clone()
        } else if cx.is_array(init) {
            let n = cx.array_len(init).unwrap_or(0);
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let pair = cx.array_get(init, i);
                let k = cx.array_get(pair, 0);
                let v = cx.array_get(pair, 1);
                out.push((cx.to_string(k)?, cx.to_string(v)?));
            }
            out
        } else if cx.type_of(init) == "object" {
            let keys = cx.own_keys(init);
            let mut out = Vec::with_capacity(keys.len());
            for k in keys {
                let v = cx.get(init, &k)?;
                out.push((k, cx.to_string(v)?));
            }
            out
        } else {
            parse_query_pairs(&cx.to_string(init)?)
        };
        cx.set_native_state(this, pairs);
        Ok(this)
    });

    if let Some(proto) = prototype_of(interp, usp_ctor) {
        add_method(interp, proto, "get", 1, |cx, this, args| {
            let key = cx.to_string(arg(args, 0))?;
            let pairs = usp_pairs(cx, this);
            match pairs.iter().find(|(k, _)| *k == key) {
                Some((_, v)) => Ok(cx.string(v)),
                None => Ok(NanBox::null()),
            }
        });
        add_method(interp, proto, "getAll", 1, |cx, this, args| {
            let key = cx.to_string(arg(args, 0))?;
            let pairs = usp_pairs(cx, this);
            let vals: Vec<NanBox> = pairs
                .iter()
                .filter(|(k, _)| *k == key)
                .map(|(_, v)| cx.string(v))
                .collect();
            Ok(cx.new_array(vals))
        });
        add_method(interp, proto, "has", 1, |cx, this, args| {
            let key = cx.to_string(arg(args, 0))?;
            let has_val = !is_nullish(arg(args, 1));
            let val = if has_val {
                Some(cx.to_string(arg(args, 1))?)
            } else {
                None
            };
            let pairs = usp_pairs(cx, this);
            let found = pairs
                .iter()
                .any(|(k, v)| *k == key && val.as_ref().map(|vv| vv == v).unwrap_or(true));
            Ok(NanBox::boolean(found))
        });
        add_method(interp, proto, "append", 2, |cx, this, args| {
            let k = cx.to_string(arg(args, 0))?;
            let v = cx.to_string(arg(args, 1))?;
            let mut pairs = usp_pairs(cx, this);
            pairs.push((k, v));
            cx.set_native_state(this, pairs);
            Ok(NanBox::undefined())
        });
        add_method(interp, proto, "set", 2, |cx, this, args| {
            let k = cx.to_string(arg(args, 0))?;
            let v = cx.to_string(arg(args, 1))?;
            let mut pairs = usp_pairs(cx, this);
            if let Some(pos) = pairs.iter().position(|(kk, _)| *kk == k) {
                pairs[pos].1 = v;
                // Keep the (now-updated) first occurrence; drop later duplicates.
                let mut seen = false;
                pairs.retain(|(kk, _)| {
                    if *kk == k {
                        if seen {
                            false
                        } else {
                            seen = true;
                            true
                        }
                    } else {
                        true
                    }
                });
            } else {
                pairs.push((k, v));
            }
            cx.set_native_state(this, pairs);
            Ok(NanBox::undefined())
        });
        add_method(interp, proto, "delete", 1, |cx, this, args| {
            let k = cx.to_string(arg(args, 0))?;
            let has_val = !is_nullish(arg(args, 1));
            let val = if has_val {
                Some(cx.to_string(arg(args, 1))?)
            } else {
                None
            };
            let mut pairs = usp_pairs(cx, this);
            pairs.retain(|(kk, vv)| !(*kk == k && val.as_ref().map(|v| v == vv).unwrap_or(true)));
            cx.set_native_state(this, pairs);
            Ok(NanBox::undefined())
        });
        add_method(interp, proto, "sort", 0, |cx, this, _args| {
            let mut pairs = usp_pairs(cx, this);
            pairs.sort_by(|a, b| a.0.cmp(&b.0));
            cx.set_native_state(this, pairs);
            Ok(NanBox::undefined())
        });
        add_method(interp, proto, "toString", 0, |cx, this, _args| {
            let pairs = usp_pairs(cx, this);
            Ok(cx.string(&serialize_pairs(&pairs)))
        });
        add_method(interp, proto, "forEach", 1, |cx, this, args| {
            let cb = arg(args, 0);
            let this_arg = arg(args, 1);
            let pairs = usp_pairs(cx, this);
            for (k, v) in pairs {
                let kv = cx.string(&k);
                let vv = cx.string(&v);
                cx.call(cb, this_arg, &[vv, kv, this])?;
            }
            Ok(NanBox::undefined())
        });

        // entries / keys / values return genuine iterators.
        let ik = iter_key.clone();
        add_method(interp, proto, "entries", 0, move |cx, this, _args| {
            let pairs = usp_pairs(cx, this);
            let items: Vec<NanBox> = pairs
                .iter()
                .map(|(k, v)| {
                    let kv = cx.string(k);
                    let vv = cx.string(v);
                    cx.new_array(vec![kv, vv])
                })
                .collect();
            array_iterator(cx, &ik, items)
        });
        let ik = iter_key.clone();
        add_method(interp, proto, "keys", 0, move |cx, this, _args| {
            let pairs = usp_pairs(cx, this);
            let items: Vec<NanBox> = pairs.iter().map(|(k, _)| cx.string(k)).collect();
            array_iterator(cx, &ik, items)
        });
        let ik = iter_key.clone();
        add_method(interp, proto, "values", 0, move |cx, this, _args| {
            let pairs = usp_pairs(cx, this);
            let items: Vec<NanBox> = pairs.iter().map(|(_, v)| cx.string(v)).collect();
            array_iterator(cx, &ik, items)
        });

        // [Symbol.iterator] === entries
        let ik = iter_key.clone();
        let sym_iter = interp.register_fn("[Symbol.iterator]", 0, move |cx, this, _args| {
            let pairs = usp_pairs(cx, this);
            let items: Vec<NanBox> = pairs
                .iter()
                .map(|(k, v)| {
                    let kv = cx.string(k);
                    let vv = cx.string(v);
                    cx.new_array(vec![kv, vv])
                })
                .collect();
            array_iterator(cx, &ik, items)
        });
        if let Some(ph) = proto.as_handle().map(Handle::from_raw) {
            interp.realm_mut().set_property(ph, &iter_key, sym_iter);
        }
    }

    // --- URL ---
    let url_ctor = interp.register_global_constructor("URL", 1, |cx, this, args| {
        let url = cx.to_string(arg(args, 0))?;
        let base_arg = arg(args, 1);
        let base_state = if is_nullish(base_arg) {
            None
        } else {
            let bs = cx.to_string(base_arg)?;
            match parse_url(&bs, None) {
                Ok(s) => Some(s),
                Err(_) => return Err(cx.type_error(&format!("Invalid base URL: {bs}"))),
            }
        };
        match parse_url(&url, base_state.as_ref()) {
            Ok(st) => {
                cx.set_native_state(this, st);
                Ok(this)
            }
            Err(msg) => Err(cx.type_error(&msg)),
        }
    });

    if let Some(proto) = prototype_of(interp, url_ctor) {
        url_accessor(interp, proto, "href", UrlState::href, |st, v| {
            if let Ok(new) = parse_url(v, None) {
                *st = new;
            }
        });
        url_accessor(
            interp,
            proto,
            "protocol",
            |st| format!("{}:", st.scheme),
            |st, v| {
                let s = v.trim_end_matches(':').to_ascii_lowercase();
                if !s.is_empty() {
                    st.special = is_special(&s);
                    st.scheme = s;
                }
            },
        );
        url_accessor(
            interp,
            proto,
            "hostname",
            |st| st.host.clone(),
            |st, v| st.host = normalize_host(v, st.special),
        );
        url_accessor(interp, proto, "host", UrlState::host_str, |st, v| {
            parse_host_port(v, st);
        });
        url_accessor(
            interp,
            proto,
            "port",
            |st| st.port.clone(),
            |st, v| set_port(v, st),
        );
        url_accessor(
            interp,
            proto,
            "pathname",
            |st| st.path.clone(),
            |st, v| {
                st.path = if st.has_authority && !v.starts_with('/') && !v.is_empty() {
                    format!("/{v}")
                } else {
                    v.to_string()
                };
            },
        );
        url_accessor(
            interp,
            proto,
            "search",
            |st| match &st.query {
                Some(q) if !q.is_empty() => format!("?{q}"),
                _ => String::new(),
            },
            |st, v| {
                let v = v.strip_prefix('?').unwrap_or(v);
                st.query = if v.is_empty() {
                    None
                } else {
                    Some(v.to_string())
                };
            },
        );
        url_accessor(
            interp,
            proto,
            "hash",
            |st| match &st.fragment {
                Some(f) if !f.is_empty() => format!("#{f}"),
                _ => String::new(),
            },
            |st, v| {
                let v = v.strip_prefix('#').unwrap_or(v);
                st.fragment = if v.is_empty() {
                    None
                } else {
                    Some(v.to_string())
                };
            },
        );
        url_accessor(
            interp,
            proto,
            "username",
            |st| st.username.clone(),
            |st, v| st.username = v.to_string(),
        );
        url_accessor(
            interp,
            proto,
            "password",
            |st| st.password.clone(),
            |st, v| st.password = v.to_string(),
        );

        // origin — read-only.
        add_accessor(
            interp,
            proto,
            "origin",
            |cx, this, _args| {
                let s = cx
                    .native_state::<UrlState>(this)
                    .map(UrlState::origin)
                    .unwrap_or_default();
                Ok(cx.string(&s))
            },
            |_cx, _this, _args| Ok(NanBox::undefined()),
        );

        // searchParams — read-only, returns a fresh (non-live) URLSearchParams.
        add_accessor(
            interp,
            proto,
            "searchParams",
            |cx, this, _args| {
                let q = cx
                    .native_state::<UrlState>(this)
                    .and_then(|st| st.query.clone())
                    .unwrap_or_default();
                let ctor = global_get(cx, "URLSearchParams");
                let qs = cx.string(&q);
                cx.construct(ctor, &[qs])
            },
            |_cx, _this, _args| Ok(NanBox::undefined()),
        );

        add_method(interp, proto, "toString", 0, |cx, this, _args| {
            let s = cx
                .native_state::<UrlState>(this)
                .map(UrlState::href)
                .unwrap_or_default();
            Ok(cx.string(&s))
        });
        add_method(interp, proto, "toJSON", 0, |cx, this, _args| {
            let s = cx
                .native_state::<UrlState>(this)
                .map(UrlState::href)
                .unwrap_or_default();
            Ok(cx.string(&s))
        });
    }
}

/// Install a `name` getter/setter on the URL prototype backed by `get`/`set`
/// over the instance's [`UrlState`].
fn url_accessor(
    interp: &mut Interp<'_>,
    proto: NanBox,
    name: &str,
    get: fn(&UrlState) -> String,
    set: fn(&mut UrlState, &str),
) {
    add_accessor(
        interp,
        proto,
        name,
        move |cx, this, _args| {
            let s = cx
                .native_state::<UrlState>(this)
                .map(get)
                .unwrap_or_default();
            Ok(cx.string(&s))
        },
        move |cx, this, args| {
            let v = cx.to_string(arg(args, 0))?;
            let mut st = cx
                .native_state::<UrlState>(this)
                .cloned()
                .unwrap_or_default();
            set(&mut st, &v);
            cx.set_native_state(this, st);
            Ok(NanBox::undefined())
        },
    );
}

// ===========================================================================
// structuredClone
// ===========================================================================

fn install_structured_clone(interp: &mut Interp<'_>) {
    interp.register_global_fn("structuredClone", 1, |cx, _this, args| {
        let mut memo: Vec<(u64, NanBox)> = Vec::new();
        clone_value(cx, arg(args, 0), &mut memo)
    });
}

fn data_clone_error(cx: &mut Ctx, what: &str) -> NanBox {
    named_error(
        cx,
        "DataCloneError",
        &format!("{what} could not be cloned."),
    )
}

fn clone_value(cx: &mut Ctx, v: NanBox, memo: &mut Vec<(u64, NanBox)>) -> Result<NanBox, NanBox> {
    match cx.type_of(v) {
        "undefined" | "boolean" | "number" | "string" | "bigint" => Ok(v),
        "symbol" => Err(data_clone_error(cx, "A Symbol object")),
        "function" => Err(data_clone_error(cx, "A function")),
        _ => {
            if matches!(v.unpack(), Unpacked::Null) {
                return Ok(v);
            }
            let raw = match v.as_handle() {
                Some(r) => r,
                None => return Ok(v),
            };
            if let Some((_, cloned)) = memo.iter().find(|(k, _)| *k == raw) {
                return Ok(*cloned);
            }
            let tag = builtin_tag(cx, v);
            match tag.as_str() {
                "Array" => {
                    let n = cx.array_len(v).unwrap_or(0);
                    let out = cx.new_array(Vec::new());
                    memo.push((raw, out));
                    for i in 0..n {
                        let e = cx.array_get(v, i);
                        let ce = clone_value(cx, e, memo)?;
                        cx.array_set(out, i, ce);
                    }
                    Ok(out)
                }
                "Date" => {
                    let ctor = global_get(cx, "Date");
                    let get_time = cx.get(v, "getTime")?;
                    let t = cx.call(get_time, v, &[])?;
                    let d = cx.construct(ctor, &[t])?;
                    memo.push((raw, d));
                    Ok(d)
                }
                "RegExp" => {
                    let ctor = global_get(cx, "RegExp");
                    let source = cx.get(v, "source")?;
                    let flags = cx.get(v, "flags")?;
                    let r = cx.construct(ctor, &[source, flags])?;
                    memo.push((raw, r));
                    Ok(r)
                }
                "Map" => {
                    let ctor = global_get(cx, "Map");
                    let m = cx.construct(ctor, &[])?;
                    memo.push((raw, m));
                    let set_fn = cx.get(m, "set")?;
                    for (k, val) in map_entries(cx, v)? {
                        let ck = clone_value(cx, k, memo)?;
                        let cv = clone_value(cx, val, memo)?;
                        cx.call(set_fn, m, &[ck, cv])?;
                    }
                    Ok(m)
                }
                "Set" => {
                    let ctor = global_get(cx, "Set");
                    let s = cx.construct(ctor, &[])?;
                    memo.push((raw, s));
                    let add_fn = cx.get(s, "add")?;
                    for val in set_values(cx, v)? {
                        let cv = clone_value(cx, val, memo)?;
                        cx.call(add_fn, s, &[cv])?;
                    }
                    Ok(s)
                }
                "ArrayBuffer" => {
                    let slice = cx.get(v, "slice")?;
                    let copy = cx.call(slice, v, &[NanBox::number(0.0)])?;
                    memo.push((raw, copy));
                    Ok(copy)
                }
                t if t.ends_with("Array") || t == "DataView" => {
                    // Typed array / DataView: clone the buffer, rebuild the view.
                    let ctor = cx.get(v, "constructor")?;
                    let buffer = cx.get(v, "buffer")?;
                    let cloned_buffer = clone_value(cx, buffer, memo)?;
                    let off = cx.get(v, "byteOffset")?;
                    let result = if t == "DataView" {
                        let len = cx.get(v, "byteLength")?;
                        cx.construct(ctor, &[cloned_buffer, off, len])?
                    } else {
                        let len = cx.get(v, "length")?;
                        cx.construct(ctor, &[cloned_buffer, off, len])?
                    };
                    memo.push((raw, result));
                    Ok(result)
                }
                _ => {
                    // Plain object (best-effort for anything else): copy own
                    // enumerable string keys.
                    let out = cx.new_object();
                    memo.push((raw, out));
                    for k in cx.own_keys(v) {
                        let val = cx.get(v, &k)?;
                        let cv = clone_value(cx, val, memo)?;
                        cx.set(out, &k, cv);
                    }
                    Ok(out)
                }
            }
        }
    }
}

/// `[...map]` → the `[k, v]` entries (via `Array.from`).
fn map_entries(cx: &mut Ctx, m: NanBox) -> Result<Vec<(NanBox, NanBox)>, NanBox> {
    let arr = array_from(cx, m)?;
    let n = cx.array_len(arr).unwrap_or(0);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let pair = cx.array_get(arr, i);
        out.push((cx.array_get(pair, 0), cx.array_get(pair, 1)));
    }
    Ok(out)
}

fn set_values(cx: &mut Ctx, s: NanBox) -> Result<Vec<NanBox>, NanBox> {
    let arr = array_from(cx, s)?;
    let n = cx.array_len(arr).unwrap_or(0);
    Ok((0..n).map(|i| cx.array_get(arr, i)).collect())
}

fn array_from(cx: &mut Ctx, iterable: NanBox) -> Result<NanBox, NanBox> {
    let array = global_get(cx, "Array");
    let from = cx.get(array, "from")?;
    cx.call(from, array, &[iterable])
}

// ===========================================================================
// performance
// ===========================================================================

struct PerfEntry {
    name: String,
    entry_type: &'static str,
    start_time: f64,
    duration: f64,
}

#[derive(Default)]
struct PerfState {
    entries: Vec<PerfEntry>,
}

fn install_performance(interp: &mut Interp<'_>) {
    let start = Instant::now();
    let time_origin = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64() * 1000.0)
        .unwrap_or(0.0);
    let state = Rc::new(RefCell::new(PerfState::default()));

    let perf = {
        let h = interp.realm_mut().new_object();
        NanBox::handle(h.to_raw())
    };

    // now()
    let start_now = start;
    add_method(interp, perf, "now", 0, move |cx, _this, _args| {
        let ms = start_now.elapsed().as_secs_f64() * 1000.0;
        Ok(cx.number(ms))
    });

    // timeOrigin (data property)
    if let Some(ph) = perf.as_handle().map(Handle::from_raw) {
        interp
            .realm_mut()
            .set_property(ph, "timeOrigin", NanBox::number(time_origin));
    }

    // mark(name)
    let st = state.clone();
    let mark_start = start;
    add_method(interp, perf, "mark", 1, move |cx, _this, args| {
        let name = cx.to_string(arg(args, 0))?;
        let start_time = mark_start.elapsed().as_secs_f64() * 1000.0;
        st.borrow_mut().entries.push(PerfEntry {
            name: name.clone(),
            entry_type: "mark",
            start_time,
            duration: 0.0,
        });
        Ok(perf_entry_object(cx, &name, "mark", start_time, 0.0))
    });

    // measure(name, startMark?, endMark?)
    let st = state.clone();
    let measure_start = start;
    add_method(interp, perf, "measure", 1, move |cx, _this, args| {
        let name = cx.to_string(arg(args, 0))?;
        let now = measure_start.elapsed().as_secs_f64() * 1000.0;
        let resolve_mark = |cx: &mut Ctx, v: NanBox, fallback: f64| -> f64 {
            if is_nullish(v) {
                return fallback;
            }
            if cx.type_of(v) == "number" {
                return cx.to_number(v).unwrap_or(fallback);
            }
            let name = cx.to_string(v).unwrap_or_default();
            st.borrow()
                .entries
                .iter()
                .rev()
                .find(|e| e.name == name)
                .map(|e| e.start_time)
                .unwrap_or(fallback)
        };
        let start_time = resolve_mark(cx, arg(args, 1), 0.0);
        let end_time = resolve_mark(cx, arg(args, 2), now);
        let duration = end_time - start_time;
        st.borrow_mut().entries.push(PerfEntry {
            name: name.clone(),
            entry_type: "measure",
            start_time,
            duration,
        });
        Ok(perf_entry_object(
            cx, &name, "measure", start_time, duration,
        ))
    });

    // getEntriesByName(name, type?)
    let st = state.clone();
    add_method(
        interp,
        perf,
        "getEntriesByName",
        1,
        move |cx, _this, args| {
            let name = cx.to_string(arg(args, 0))?;
            let ty = if is_nullish(arg(args, 1)) {
                None
            } else {
                Some(cx.to_string(arg(args, 1))?)
            };
            let matched: Vec<(String, &'static str, f64, f64)> = st
                .borrow()
                .entries
                .iter()
                .filter(|e| {
                    e.name == name && ty.as_ref().map(|t| t == e.entry_type).unwrap_or(true)
                })
                .map(|e| (e.name.clone(), e.entry_type, e.start_time, e.duration))
                .collect();
            let objs: Vec<NanBox> = matched
                .into_iter()
                .map(|(n, t, s, d)| perf_entry_object(cx, &n, t, s, d))
                .collect();
            Ok(cx.new_array(objs))
        },
    );

    // getEntriesByType(type)
    let st = state.clone();
    add_method(
        interp,
        perf,
        "getEntriesByType",
        1,
        move |cx, _this, args| {
            let ty = cx.to_string(arg(args, 0))?;
            let matched: Vec<(String, &'static str, f64, f64)> = st
                .borrow()
                .entries
                .iter()
                .filter(|e| e.entry_type == ty)
                .map(|e| (e.name.clone(), e.entry_type, e.start_time, e.duration))
                .collect();
            let objs: Vec<NanBox> = matched
                .into_iter()
                .map(|(n, t, s, d)| perf_entry_object(cx, &n, t, s, d))
                .collect();
            Ok(cx.new_array(objs))
        },
    );

    // clearMarks(name?)
    let st = state.clone();
    add_method(interp, perf, "clearMarks", 0, move |cx, _this, args| {
        let name = if is_nullish(arg(args, 0)) {
            None
        } else {
            Some(cx.to_string(arg(args, 0))?)
        };
        st.borrow_mut().entries.retain(|e| {
            e.entry_type != "mark" || name.as_ref().map(|n| n != &e.name).unwrap_or(false)
        });
        Ok(NanBox::undefined())
    });

    // clearMeasures(name?)
    let st = state.clone();
    add_method(interp, perf, "clearMeasures", 0, move |cx, _this, args| {
        let name = if is_nullish(arg(args, 0)) {
            None
        } else {
            Some(cx.to_string(arg(args, 0))?)
        };
        st.borrow_mut().entries.retain(|e| {
            e.entry_type != "measure" || name.as_ref().map(|n| n != &e.name).unwrap_or(false)
        });
        Ok(NanBox::undefined())
    });

    interp.declare_global("performance", perf);
}

fn perf_entry_object(
    cx: &mut Ctx,
    name: &str,
    entry_type: &str,
    start_time: f64,
    duration: f64,
) -> NanBox {
    let o = cx.new_object();
    let n = cx.string(name);
    cx.set(o, "name", n);
    let t = cx.string(entry_type);
    cx.set(o, "entryType", t);
    cx.set(o, "startTime", NanBox::number(start_time));
    cx.set(o, "duration", NanBox::number(duration));
    o
}

// ===========================================================================
// console
// ===========================================================================

#[derive(Default)]
struct ConsoleState {
    indent: usize,
    counts: HashMap<String, u64>,
    timers: HashMap<String, Instant>,
}

fn install_console(interp: &mut Interp<'_>) {
    // Capture the engine's native `console.log` so our formatting still lands in
    // `interp.output()`. Persist it so it survives across calls and GC.
    let native_log = interp
        .global_object()
        .and_then(|g| interp.realm().get_property(g, "console"))
        .and_then(|c| c.as_handle().map(Handle::from_raw))
        .and_then(|ch| interp.realm().get_property(ch, "log"))
        .unwrap_or_else(NanBox::undefined);
    let log_idx = interp.persist(native_log);

    let state = Rc::new(RefCell::new(ConsoleState::default()));

    let console = {
        let h = interp.realm_mut().new_object();
        NanBox::handle(h.to_raw())
    };

    // The printing methods (log/info/warn/error/debug) share a formatter and emit
    // via the persisted native logger.
    for name in ["log", "info", "warn", "error", "debug"] {
        let st = state.clone();
        add_method(interp, console, name, 0, move |cx, _this, args| {
            let line = format_console(cx, args);
            emit(cx, log_idx, st.borrow().indent, &line);
            Ok(NanBox::undefined())
        });
    }

    // dir(obj) — inspect a single value.
    let st = state.clone();
    add_method(interp, console, "dir", 1, move |cx, _this, args| {
        let s = inspect(cx, arg(args, 0), 4, false, &mut Vec::new());
        emit(cx, log_idx, st.borrow().indent, &s);
        Ok(NanBox::undefined())
    });

    // trace(...) — no real stack; prefix "Trace:".
    let st = state.clone();
    add_method(interp, console, "trace", 0, move |cx, _this, args| {
        let line = format_console(cx, args);
        let indent = st.borrow().indent;
        emit(cx, log_idx, indent, &format!("Trace: {line}"));
        Ok(NanBox::undefined())
    });

    // assert(cond, ...msg)
    let st = state.clone();
    add_method(interp, console, "assert", 0, move |cx, _this, args| {
        let cond = cx.to_boolean(arg(args, 0));
        if !cond {
            let rest: &[NanBox] = if args.len() > 1 { &args[1..] } else { &[] };
            let msg = format_console(cx, rest);
            let indent = st.borrow().indent;
            let line = if msg.is_empty() {
                String::from("Assertion failed")
            } else {
                format!("Assertion failed: {msg}")
            };
            emit(cx, log_idx, indent, &line);
        }
        Ok(NanBox::undefined())
    });

    // table(data) — minimal: fall back to inspection.
    let st = state.clone();
    add_method(interp, console, "table", 1, move |cx, _this, args| {
        let s = inspect(cx, arg(args, 0), 4, false, &mut Vec::new());
        emit(cx, log_idx, st.borrow().indent, &s);
        Ok(NanBox::undefined())
    });

    // group / groupCollapsed (print label, then indent)
    for name in ["group", "groupCollapsed"] {
        let st = state.clone();
        add_method(interp, console, name, 0, move |cx, _this, args| {
            let indent = st.borrow().indent;
            if !args.is_empty() {
                let line = format_console(cx, args);
                emit(cx, log_idx, indent, &line);
            }
            st.borrow_mut().indent += 1;
            Ok(NanBox::undefined())
        });
    }

    // groupEnd
    let st = state.clone();
    add_method(interp, console, "groupEnd", 0, move |_cx, _this, _args| {
        let mut s = st.borrow_mut();
        if s.indent > 0 {
            s.indent -= 1;
        }
        Ok(NanBox::undefined())
    });

    // count(label?) / countReset(label?)
    let st = state.clone();
    add_method(interp, console, "count", 0, move |cx, _this, args| {
        let label = if is_nullish(arg(args, 0)) {
            String::from("default")
        } else {
            cx.to_string(arg(args, 0))?
        };
        let (indent, n) = {
            let mut s = st.borrow_mut();
            let c = s.counts.entry(label.clone()).or_insert(0);
            *c += 1;
            let n = *c;
            (s.indent, n)
        };
        emit(cx, log_idx, indent, &format!("{label}: {n}"));
        Ok(NanBox::undefined())
    });
    let st = state.clone();
    add_method(interp, console, "countReset", 0, move |cx, _this, args| {
        let label = if is_nullish(arg(args, 0)) {
            String::from("default")
        } else {
            cx.to_string(arg(args, 0))?
        };
        st.borrow_mut().counts.insert(label, 0);
        Ok(NanBox::undefined())
    });

    // time(label?) / timeEnd(label?) / timeLog(label?)
    let st = state.clone();
    add_method(interp, console, "time", 0, move |cx, _this, args| {
        let label = if is_nullish(arg(args, 0)) {
            String::from("default")
        } else {
            cx.to_string(arg(args, 0))?
        };
        st.borrow_mut().timers.insert(label, Instant::now());
        Ok(NanBox::undefined())
    });
    let st = state.clone();
    add_method(interp, console, "timeEnd", 0, move |cx, _this, args| {
        let label = if is_nullish(arg(args, 0)) {
            String::from("default")
        } else {
            cx.to_string(arg(args, 0))?
        };
        let (indent, elapsed) = {
            let mut s = st.borrow_mut();
            let e = s
                .timers
                .remove(&label)
                .map(|t| t.elapsed().as_secs_f64() * 1000.0);
            (s.indent, e)
        };
        if let Some(ms) = elapsed {
            emit(cx, log_idx, indent, &format!("{label}: {ms:.3}ms"));
        }
        Ok(NanBox::undefined())
    });
    let st = state.clone();
    add_method(interp, console, "timeLog", 0, move |cx, _this, args| {
        let label = if is_nullish(arg(args, 0)) {
            String::from("default")
        } else {
            cx.to_string(arg(args, 0))?
        };
        let (indent, elapsed) = {
            let s = st.borrow();
            (
                s.indent,
                s.timers
                    .get(&label)
                    .map(|t| t.elapsed().as_secs_f64() * 1000.0),
            )
        };
        if let Some(ms) = elapsed {
            emit(cx, log_idx, indent, &format!("{label}: {ms:.3}ms"));
        }
        Ok(NanBox::undefined())
    });

    // `declare_global` overwrites the engine's minimal console.
    interp.declare_global("console", console);
}

/// Write `line` (indented) through the persisted native `console.log`.
fn emit(cx: &mut Ctx, log_idx: u32, indent: usize, line: &str) {
    let prefixed = if indent > 0 {
        format!("{}{}", "  ".repeat(indent), line)
    } else {
        line.to_string()
    };
    let logv = cx.persistent(log_idx);
    let s = cx.string(&prefixed);
    let _ = cx.call(logv, NanBox::undefined(), &[s]);
}

/// Format a console call's arguments: `%`-substitution when the first arg is a
/// format string, else the args inspected and space-joined.
fn format_console(cx: &mut Ctx, args: &[NanBox]) -> String {
    if args.is_empty() {
        return String::new();
    }
    let first = args[0];
    if cx.type_of(first) == "string" {
        let fmt = cx.to_string(first).unwrap_or_default();
        if fmt.contains('%') {
            return format_substitute(cx, &fmt, &args[1..]);
        }
    }
    let mut parts: Vec<String> = Vec::with_capacity(args.len());
    for &a in args {
        parts.push(inspect(cx, a, 2, true, &mut Vec::new()));
    }
    parts.join(" ")
}

fn format_substitute(cx: &mut Ctx, fmt: &str, args: &[NanBox]) -> String {
    let mut out = String::new();
    let mut ai = 0usize;
    let chars: Vec<char> = fmt.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '%' && i + 1 < chars.len() {
            let spec = chars[i + 1];
            match spec {
                '%' => {
                    out.push('%');
                    i += 2;
                    continue;
                }
                's' | 'd' | 'i' | 'f' | 'o' | 'O' | 'j' | 'c' => {
                    if ai >= args.len() {
                        // No arg — emit the directive literally.
                        out.push('%');
                        out.push(spec);
                        i += 2;
                        continue;
                    }
                    let a = args[ai];
                    ai += 1;
                    match spec {
                        's' => out.push_str(&cx.to_string(a).unwrap_or_default()),
                        'd' | 'i' => {
                            if cx.type_of(a) == "symbol" {
                                out.push_str("NaN");
                            } else {
                                let n = cx.to_number(a).unwrap_or(f64::NAN);
                                if n.is_nan() {
                                    out.push_str("NaN");
                                } else {
                                    out.push_str(&(n.trunc() as i64).to_string());
                                }
                            }
                        }
                        'f' => {
                            let n = cx.to_number(a).unwrap_or(f64::NAN);
                            out.push_str(&number_string(n));
                        }
                        'j' => {
                            let json = global_get(cx, "JSON");
                            let s = cx
                                .get(json, "stringify")
                                .ok()
                                .and_then(|f| cx.call(f, NanBox::undefined(), &[a]).ok())
                                .and_then(|r| cx.to_string(r).ok())
                                .unwrap_or_else(|| String::from("undefined"));
                            out.push_str(&s);
                        }
                        'o' | 'O' => out.push_str(&inspect(cx, a, 4, false, &mut Vec::new())),
                        'c' => { /* CSS directive — consume, output nothing */ }
                        _ => unreachable!(),
                    }
                    i += 2;
                    continue;
                }
                _ => {
                    out.push('%');
                    i += 1;
                    continue;
                }
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    // Trailing (non-substituted) args, space-joined.
    for &a in &args[ai..] {
        out.push(' ');
        out.push_str(&inspect(cx, a, 2, true, &mut Vec::new()));
    }
    out
}

/// Node-ish value inspection. `top` prints bare strings unquoted; nested strings
/// are single-quoted. `depth` caps recursion.
fn inspect(cx: &mut Ctx, v: NanBox, depth: i32, top: bool, seen: &mut Vec<u64>) -> String {
    match cx.type_of(v) {
        "undefined" => String::from("undefined"),
        "boolean" => {
            if cx.to_boolean(v) {
                String::from("true")
            } else {
                String::from("false")
            }
        }
        "number" => number_string(cx.to_number(v).unwrap_or(f64::NAN)),
        "bigint" => format!("{}n", cx.to_string(v).unwrap_or_default()),
        "string" => {
            let s = cx.to_string(v).unwrap_or_default();
            if top {
                s
            } else {
                format!("'{}'", s.replace('\'', "\\'"))
            }
        }
        "symbol" => cx.to_string(v).unwrap_or_else(|_| String::from("Symbol()")),
        "function" => {
            let name = cx
                .get(v, "name")
                .ok()
                .and_then(|n| cx.to_string(n).ok())
                .unwrap_or_default();
            if name.is_empty() {
                String::from("[Function (anonymous)]")
            } else {
                format!("[Function: {name}]")
            }
        }
        _ => {
            if matches!(v.unpack(), Unpacked::Null) {
                return String::from("null");
            }
            let raw = v.as_handle().unwrap_or(0);
            if seen.contains(&raw) {
                return String::from("[Circular *1]");
            }
            let tag = builtin_tag(cx, v);
            match tag.as_str() {
                "Array" => inspect_array(cx, v, depth, raw, seen),
                "Date" | "RegExp" | "Error" => cx.to_string(v).unwrap_or_default(),
                "Map" => inspect_collection(cx, v, depth, raw, seen, true),
                "Set" => inspect_collection(cx, v, depth, raw, seen, false),
                "Promise" => String::from("Promise { <state> }"),
                "ArrayBuffer" => {
                    let n = cx
                        .get(v, "byteLength")
                        .ok()
                        .and_then(|b| cx.to_number(b).ok())
                        .unwrap_or(0.0);
                    format!("ArrayBuffer {{ byteLength: {} }}", n as u64)
                }
                t if t.ends_with("Array") => inspect_typed_array(cx, v, &tag, depth, raw, seen),
                _ => inspect_object(cx, v, depth, raw, seen),
            }
        }
    }
}

fn inspect_array(cx: &mut Ctx, v: NanBox, depth: i32, raw: u64, seen: &mut Vec<u64>) -> String {
    if depth < 0 {
        return String::from("[Array]");
    }
    let n = cx.array_len(v).unwrap_or(0);
    if n == 0 {
        return String::from("[]");
    }
    seen.push(raw);
    let mut parts = Vec::with_capacity(n);
    for i in 0..n {
        let e = cx.array_get(v, i);
        parts.push(inspect(cx, e, depth - 1, false, seen));
    }
    seen.pop();
    format!("[ {} ]", parts.join(", "))
}

fn inspect_typed_array(
    cx: &mut Ctx,
    v: NanBox,
    tag: &str,
    depth: i32,
    raw: u64,
    seen: &mut Vec<u64>,
) -> String {
    let n = cx
        .get(v, "length")
        .ok()
        .and_then(|l| cx.to_number(l).ok())
        .unwrap_or(0.0) as usize;
    if n == 0 {
        return format!("{tag}(0) []");
    }
    if depth < 0 {
        return format!("[{tag}]");
    }
    seen.push(raw);
    let mut parts = Vec::with_capacity(n);
    for i in 0..n {
        let e = cx
            .get(v, &i.to_string())
            .unwrap_or_else(|_| NanBox::undefined());
        parts.push(inspect(cx, e, depth - 1, false, seen));
    }
    seen.pop();
    format!("{tag}({n}) [ {} ]", parts.join(", "))
}

fn inspect_collection(
    cx: &mut Ctx,
    v: NanBox,
    depth: i32,
    raw: u64,
    seen: &mut Vec<u64>,
    is_map: bool,
) -> String {
    let kind = if is_map { "Map" } else { "Set" };
    let size = cx
        .get(v, "size")
        .ok()
        .and_then(|s| cx.to_number(s).ok())
        .unwrap_or(0.0) as usize;
    if size == 0 {
        return format!("{kind}(0) {{}}");
    }
    if depth < 0 {
        return format!("[{kind}]");
    }
    seen.push(raw);
    let body = if is_map {
        let entries = map_entries(cx, v).unwrap_or_default();
        entries
            .into_iter()
            .map(|(k, val)| {
                let ks = inspect(cx, k, depth - 1, false, seen);
                let vs = inspect(cx, val, depth - 1, false, seen);
                format!("{ks} => {vs}")
            })
            .collect::<Vec<_>>()
            .join(", ")
    } else {
        let values = set_values(cx, v).unwrap_or_default();
        values
            .into_iter()
            .map(|val| inspect(cx, val, depth - 1, false, seen))
            .collect::<Vec<_>>()
            .join(", ")
    };
    seen.pop();
    format!("{kind}({size}) {{ {body} }}")
}

fn inspect_object(cx: &mut Ctx, v: NanBox, depth: i32, raw: u64, seen: &mut Vec<u64>) -> String {
    if depth < 0 {
        return String::from("[Object]");
    }
    let keys = cx.own_keys(v);
    if keys.is_empty() {
        return String::from("{}");
    }
    seen.push(raw);
    let mut parts = Vec::with_capacity(keys.len());
    for k in keys {
        let val = cx.get(v, &k).unwrap_or_else(|_| NanBox::undefined());
        let vs = inspect(cx, val, depth - 1, false, seen);
        let key_disp = if is_ident(&k) {
            k
        } else {
            format!("'{}'", k.replace('\'', "\\'"))
        };
        parts.push(format!("{key_disp}: {vs}"));
    }
    seen.pop();
    format!("{{ {} }}", parts.join(", "))
}

fn is_ident(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c == '_' || c == '$' || c.is_ascii_alphabetic() => {}
        _ => return false,
    }
    chars.all(|c| c == '_' || c == '$' || c.is_ascii_alphanumeric())
}

/// Render a JS number the way the engine's `String(n)` would for the common
/// integer / finite cases (used by `%f` and number inspection).
fn number_string(n: f64) -> String {
    if n.is_nan() {
        String::from("NaN")
    } else if n.is_infinite() {
        if n < 0.0 {
            String::from("-Infinity")
        } else {
            String::from("Infinity")
        }
    } else if n == 0.0 {
        String::from("0")
    } else if n.fract() == 0.0 && n.abs() < 1e21 {
        format!("{}", n as i64)
    } else {
        format!("{n}")
    }
}
