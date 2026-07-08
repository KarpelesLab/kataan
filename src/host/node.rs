//! Node-compat builtins (ROADMAP §4.5): the pure, no-I/O parts of Node's core
//! module surface — `Buffer`, `path`, `os`, `util`, `querystring` — plus a
//! minimal `process` global and a `require('node:...')` shim.
//!
//! Built entirely on the §4.0 embedding API ([`Interp::register_fn`] /
//! [`Interp::register_global_constructor`] / [`Ctx`]). No engine internals are
//! reached from the host closures; everything a builtin needs (`Uint8Array`,
//! `Reflect.construct`, `JSON`, `Array.from`, `Object.prototype.toString`, …) is
//! read back out of the realm's own globals at call time.
//!
//! ## `Buffer` — a real `Uint8Array` subclass
//!
//! `Buffer` is installed as a plain function whose `prototype`'s `[[Prototype]]`
//! is `Uint8Array.prototype`, and every instance is produced by
//! `Reflect.construct(Uint8Array, [...], Buffer)` — so a `Buffer` genuinely *is*
//! a byte-backed `Uint8Array` (index access, `.length`, `instanceof Uint8Array`
//! all work) and the Buffer-specific methods live once on `Buffer.prototype`,
//! with **no per-instance property pollution**. `.slice`/`.subarray` share the
//! backing store like Node; the byte-producing statics copy.
//!
//! ## Exposure
//!
//! `path`/`os`/`util`/`querystring` are installed as globals (CommonJS `require`
//! is not implemented yet — ROADMAP §4.2), and additionally reachable through a
//! `require('node:path' | 'path' | …)` / `require('buffer').Buffer` shim.
//! `Buffer` and `process` are genuine globals.

use crate::heap::Handle;
use crate::nanbox::NanBox;
use crate::nbexec::{Ctx, Interp};
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

/// Install the §4.5 Node-compat surface into `interp`.
pub fn install(interp: &mut Interp<'_>) {
    install_buffer(interp);
    install_path(interp);
    install_os(interp);
    install_util(interp);
    install_querystring(interp);
    install_process(interp);
    install_require(interp);
}

// ---------------------------------------------------------------------------
// small install-time helpers (only a `&mut Interp` — no `Ctx` yet)
// ---------------------------------------------------------------------------

/// A fresh namespace object handle.
fn new_ns(interp: &mut Interp<'_>) -> Handle {
    interp.realm_mut().new_object()
}

/// A heap string value.
fn str_val(interp: &mut Interp<'_>, s: &str) -> NanBox {
    NanBox::handle(interp.realm_mut().new_string(s).to_raw())
}

/// Register a host function and install it as a property `name` of `obj`.
fn set_fn<F>(interp: &mut Interp<'_>, obj: Handle, name: &str, length: u32, f: F)
where
    F: FnMut(&mut Ctx<'_, '_>, NanBox, &[NanBox]) -> Result<NanBox, NanBox> + 'static,
{
    let v = interp.register_fn(name, length, f);
    interp.realm_mut().set_property(obj, name, v);
}

/// Install a string-valued property.
fn set_str(interp: &mut Interp<'_>, obj: Handle, name: &str, s: &str) {
    let v = str_val(interp, s);
    interp.realm_mut().set_property(obj, name, v);
}

/// Install a number-valued property.
fn set_num(interp: &mut Interp<'_>, obj: Handle, name: &str, n: f64) {
    interp
        .realm_mut()
        .set_property(obj, name, NanBox::number(n));
}

/// Bind `handle` as a global named `name`.
fn declare(interp: &mut Interp<'_>, name: &str, handle: Handle) {
    interp.declare_global(name, NanBox::handle(handle.to_raw()));
}

// ---------------------------------------------------------------------------
// runtime arg helpers (inside host closures — only `Ctx`)
// ---------------------------------------------------------------------------

#[inline]
fn arg(args: &[NanBox], i: usize) -> NanBox {
    args.get(i).copied().unwrap_or(NanBox::undefined())
}

/// A required string arg (coerced), defaulting to `""` when absent/undefined.
fn str_arg(cx: &mut Ctx<'_, '_>, args: &[NanBox], i: usize) -> Result<String, NanBox> {
    let v = arg(args, i);
    if v.is_undefined() {
        Ok(String::new())
    } else {
        cx.to_string(v)
    }
}

/// An optional numeric arg with a fallback for absent/undefined.
fn num_arg(cx: &mut Ctx<'_, '_>, args: &[NanBox], i: usize, default: f64) -> f64 {
    match args.get(i).copied() {
        Some(v) if !v.is_undefined() => cx.to_number(v).unwrap_or(default),
        _ => default,
    }
}

// ===========================================================================
// Buffer
// ===========================================================================

/// The supported `Buffer`/string encodings.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Enc {
    Utf8,
    Hex,
    Base64,
    Base64Url,
    Latin1,
    Ascii,
}

impl Enc {
    fn parse(s: &str) -> Option<Enc> {
        let l = s.to_ascii_lowercase();
        Some(match l.as_str() {
            "utf8" | "utf-8" | "" => Enc::Utf8,
            "hex" => Enc::Hex,
            "base64" => Enc::Base64,
            "base64url" => Enc::Base64Url,
            "latin1" | "binary" => Enc::Latin1,
            "ascii" => Enc::Ascii,
            // ucs2/utf16le fall back to latin1-ish handling elsewhere; unknown → None.
            _ => return None,
        })
    }
}

/// Parse an encoding from an optional string arg (default UTF-8).
fn enc_arg(cx: &mut Ctx<'_, '_>, v: NanBox) -> Result<Enc, NanBox> {
    if v.is_undefined() || v.is_null() {
        return Ok(Enc::Utf8);
    }
    let s = cx.to_string(v)?;
    Enc::parse(&s).ok_or_else(|| cx.type_error(&format!("Unknown encoding: {s}")))
}

fn hexval(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

fn hex_decode(s: &str) -> Vec<u8> {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len() / 2);
    let mut i = 0;
    while i + 1 < b.len() + 1 && i + 1 < b.len() {
        match (hexval(b[i]), hexval(b[i + 1])) {
            (Some(h), Some(l)) => out.push((h << 4) | l),
            _ => break,
        }
        i += 2;
    }
    out
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push_str(&format!("{b:02x}"));
    }
    out
}

fn b64_encode(data: &[u8], url: bool) -> String {
    const STD: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    const URL: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let alpha = if url { URL } else { STD };
    let mut out = String::new();
    for chunk in data.chunks(3) {
        let b0 = chunk[0];
        let b1 = *chunk.get(1).unwrap_or(&0);
        let b2 = *chunk.get(2).unwrap_or(&0);
        let n = (u32::from(b0) << 16) | (u32::from(b1) << 8) | u32::from(b2);
        out.push(alpha[((n >> 18) & 63) as usize] as char);
        out.push(alpha[((n >> 12) & 63) as usize] as char);
        if chunk.len() >= 2 {
            out.push(alpha[((n >> 6) & 63) as usize] as char);
        } else if !url {
            out.push('=');
        }
        if chunk.len() >= 3 {
            out.push(alpha[(n & 63) as usize] as char);
        } else if !url {
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
        b'+' | b'-' => Some(62),
        b'/' | b'_' => Some(63),
        _ => None,
    }
}

fn b64_decode(s: &str) -> Vec<u8> {
    let vals: Vec<u8> = s.bytes().filter_map(b64_val).collect();
    let mut out = Vec::with_capacity(vals.len() * 3 / 4);
    for chunk in vals.chunks(4) {
        if chunk.len() < 2 {
            break;
        }
        let c2 = u32::from(*chunk.get(2).unwrap_or(&0));
        let c3 = u32::from(*chunk.get(3).unwrap_or(&0));
        let n = (u32::from(chunk[0]) << 18) | (u32::from(chunk[1]) << 12) | (c2 << 6) | c3;
        out.push((n >> 16) as u8);
        if chunk.len() >= 3 {
            out.push((n >> 8) as u8);
        }
        if chunk.len() >= 4 {
            out.push(n as u8);
        }
    }
    out
}

/// Decode a string into bytes under `enc`.
fn decode_string(s: &str, enc: Enc) -> Vec<u8> {
    match enc {
        Enc::Utf8 => s.as_bytes().to_vec(),
        Enc::Hex => hex_decode(s),
        Enc::Base64 | Enc::Base64Url => b64_decode(s),
        Enc::Latin1 => s.chars().map(|c| (c as u32 & 0xff) as u8).collect(),
        Enc::Ascii => s.chars().map(|c| (c as u32 & 0x7f) as u8).collect(),
    }
}

/// Encode bytes into a string under `enc`.
fn encode_bytes(bytes: &[u8], enc: Enc) -> String {
    match enc {
        Enc::Utf8 => String::from_utf8_lossy(bytes).into_owned(),
        Enc::Hex => hex_encode(bytes),
        Enc::Base64 => b64_encode(bytes, false),
        Enc::Base64Url => b64_encode(bytes, true),
        Enc::Latin1 => bytes.iter().map(|&b| char::from(b)).collect(),
        Enc::Ascii => bytes.iter().map(|&b| char::from(b & 0x7f)).collect(),
    }
}

/// `(Reflect.construct, Uint8Array, Buffer)` read out of the realm globals.
fn buffer_ctors(cx: &mut Ctx<'_, '_>) -> Result<(NanBox, NanBox, NanBox), NanBox> {
    let g = cx.global();
    let u8 = cx.get(g, "Uint8Array")?;
    let bctor = cx.get(g, "Buffer")?;
    let reflect = cx.get(g, "Reflect")?;
    let construct = cx.get(reflect, "construct")?;
    Ok((construct, u8, bctor))
}

/// Buffer methods whose names collide with a `%TypedArray%.prototype` native.
///
/// The interpreter's method-call fast path (`expr.rs`) only prefers a
/// prototype-resolved method over the built-in typed-array dispatch when the
/// *receiver itself* owns a property of that name; a shadowing method on
/// `Buffer.prototype` alone is bypassed. So each instance gets these as own
/// properties (the same function objects as on `Buffer.prototype`). The
/// non-colliding methods (`write`/`equals`/`copy`/`readUInt8`/…) resolve
/// through the prototype normally.
///
/// **Divergences:** (1) these names appear as enumerable own keys of a Buffer;
/// (2) `subarray` is *not* included — the engine force-synthesizes the
/// typed-array `subarray` even over an own property, so `buf.subarray(...)`
/// always returns a plain `Uint8Array` view (still shares memory, but without
/// the Buffer methods). `Buffer.prototype.subarray` remains usable via
/// `.call`, and `slice` (the Buffer-returning variant) works normally.
const OWN_OVERRIDES: [&str; 3] = ["toString", "slice", "fill"];

/// Copy the collision-prone methods from `Buffer.prototype` onto `buf` as own
/// properties, so `buf.toString()` (etc.) dispatch to the Buffer versions.
fn attach_overrides(cx: &mut Ctx<'_, '_>, buf: NanBox) -> Result<(), NanBox> {
    let g = cx.global();
    let bctor = cx.get(g, "Buffer")?;
    let bproto = cx.get(bctor, "prototype")?;
    for name in OWN_OVERRIDES {
        let m = cx.get(bproto, name)?;
        cx.set(buf, name, m);
    }
    Ok(())
}

/// Build a `Buffer` (a `Uint8Array` whose `[[Prototype]]` is `Buffer.prototype`)
/// that owns a fresh copy of `bytes`.
fn make_buffer(cx: &mut Ctx<'_, '_>, bytes: &[u8]) -> Result<NanBox, NanBox> {
    let nums: Vec<NanBox> = bytes
        .iter()
        .map(|&b| NanBox::number(f64::from(b)))
        .collect();
    let inner = cx.new_array(nums);
    let (construct, u8, bctor) = buffer_ctors(cx)?;
    let arglist = cx.new_array(vec![inner]);
    let u = cx.undefined();
    let buf = cx.call(construct, u, &[u8, arglist, bctor])?;
    attach_overrides(cx, buf)?;
    Ok(buf)
}

/// Build a `Buffer` *view* sharing an existing `ArrayBuffer`'s memory
/// (`new Uint8Array(buffer, byteOffset, length)` with newTarget `Buffer`).
fn make_buffer_view(
    cx: &mut Ctx<'_, '_>,
    buffer: NanBox,
    off: usize,
    len: usize,
) -> Result<NanBox, NanBox> {
    let (construct, u8, bctor) = buffer_ctors(cx)?;
    let arglist = cx.new_array(vec![
        buffer,
        NanBox::number(off as f64),
        NanBox::number(len as f64),
    ]);
    let u = cx.undefined();
    let buf = cx.call(construct, u, &[u8, arglist, bctor])?;
    attach_overrides(cx, buf)?;
    Ok(buf)
}

/// Read every byte of a `Buffer`/`Uint8Array`/array-like `this` value.
fn read_bytes(cx: &mut Ctx<'_, '_>, v: NanBox) -> Vec<u8> {
    let len = cx
        .get(v, "length")
        .ok()
        .map(|l| cx.to_number(l).unwrap_or(0.0).max(0.0) as usize)
        .unwrap_or(0);
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let e = cx.get(v, &i.to_string()).unwrap_or(NanBox::undefined());
        out.push(cx.to_number(e).unwrap_or(0.0) as u8);
    }
    out
}

/// Read property `key` of `obj` and coerce it to a number (avoids the nested
/// `cx.to_number(cx.get(...)?)?` double-borrow).
fn get_num(cx: &mut Ctx<'_, '_>, obj: NanBox, key: &str) -> Result<f64, NanBox> {
    let v = cx.get(obj, key)?;
    cx.to_number(v)
}

/// Write `byte` at index `i` of the typed-array `buf` (routes through `[[Set]]`).
fn write_byte(cx: &mut Ctx<'_, '_>, buf: NanBox, i: usize, byte: u8) {
    let _ = cx.set_property(buf, &i.to_string(), NanBox::number(f64::from(byte)));
}

/// Whether `v` looks like an `ArrayBuffer` (has `byteLength`, no `length`).
fn is_array_buffer(cx: &mut Ctx<'_, '_>, v: NanBox) -> bool {
    cx.is_object(v) && !cx.has(v, "length") && cx.has(v, "byteLength")
}

fn install_buffer(interp: &mut Interp<'_>) {
    // `Buffer` itself: a constructor whose plain call / `new` both yield a proper
    // byte-backed buffer (Node's deprecated `new Buffer(...)` path).
    let buffer = interp.register_global_constructor("Buffer", 3, |cx, _this, args| {
        let a0 = arg(args, 0);
        if a0.is_number() {
            let n = cx.to_number(a0)? as usize;
            return make_buffer(cx, &vec![0u8; n]);
        }
        buffer_from(cx, args)
    });
    let bctor = buffer
        .as_handle()
        .map(Handle::from_raw)
        .expect("Buffer handle");

    // Rewire `Buffer.prototype`'s [[Prototype]] to `Uint8Array.prototype`, so a
    // Buffer inherits the whole typed-array surface beneath its own methods.
    if let Some((bproto, u8proto)) = buffer_proto_pair(interp, bctor) {
        interp.realm_mut().set_object_proto(bproto, Some(u8proto));
        install_buffer_proto(interp, bproto);
        // A non-enumerable brand for `Buffer.isBuffer`.
        interp
            .realm_mut()
            .set_hidden_property(bproto, "\u{0}isBuffer", NanBox::boolean(true));
    }

    // Statics.
    set_fn(interp, bctor, "from", 3, |cx, _t, args| {
        buffer_from(cx, args)
    });
    set_fn(interp, bctor, "alloc", 3, |cx, _t, args| {
        let n = num_arg(cx, args, 0, 0.0).max(0.0) as usize;
        let mut bytes = vec![0u8; n];
        // Optional fill (number, or string in the given encoding).
        let fill = arg(args, 1);
        if !fill.is_undefined() && n > 0 {
            if fill.is_number() {
                let v = cx.to_number(fill)? as u8;
                bytes.iter_mut().for_each(|b| *b = v);
            } else {
                let enc = enc_arg(cx, arg(args, 2))?;
                let fb = decode_string(&cx.to_string(fill)?, enc);
                if !fb.is_empty() {
                    for (i, b) in bytes.iter_mut().enumerate() {
                        *b = fb[i % fb.len()];
                    }
                }
            }
        }
        make_buffer(cx, &bytes)
    });
    set_fn(interp, bctor, "allocUnsafe", 1, |cx, _t, args| {
        // We cannot expose uninitialized memory safely, so this zero-fills like
        // `alloc` (a deliberate divergence — the bytes are defined, not garbage).
        let n = num_arg(cx, args, 0, 0.0).max(0.0) as usize;
        make_buffer(cx, &vec![0u8; n])
    });
    set_fn(interp, bctor, "allocUnsafeSlow", 1, |cx, _t, args| {
        let n = num_arg(cx, args, 0, 0.0).max(0.0) as usize;
        make_buffer(cx, &vec![0u8; n])
    });
    set_fn(interp, bctor, "concat", 2, |cx, _t, args| {
        let list = arg(args, 0);
        let n = cx.array_len(list).unwrap_or(0);
        let mut all: Vec<u8> = Vec::new();
        for i in 0..n {
            let item = cx.array_get(list, i);
            all.extend_from_slice(&read_bytes(cx, item));
        }
        // Optional totalLength: truncate or zero-pad.
        if let Some(total) = args.get(1).copied().filter(|v| !v.is_undefined()) {
            let t = cx.to_number(total)?.max(0.0) as usize;
            all.resize(t, 0);
        }
        make_buffer(cx, &all)
    });
    set_fn(interp, bctor, "isBuffer", 1, |cx, _t, args| {
        let v = arg(args, 0);
        Ok(NanBox::boolean(
            cx.is_object(v) && cx.has(v, "\u{0}isBuffer"),
        ))
    });
    set_fn(interp, bctor, "isEncoding", 1, |cx, _t, args| {
        let s = str_arg(cx, args, 0)?;
        Ok(NanBox::boolean(Enc::parse(&s).is_some()))
    });
    set_fn(interp, bctor, "byteLength", 2, |cx, _t, args| {
        let v = arg(args, 0);
        if cx.is_object(v) {
            // A Buffer / typed array / array-like: its element count.
            let len = read_bytes(cx, v).len();
            return Ok(NanBox::number(len as f64));
        }
        let enc = enc_arg(cx, arg(args, 1))?;
        let s = cx.to_string(v)?;
        Ok(NanBox::number(decode_string(&s, enc).len() as f64))
    });
}

/// `(Buffer.prototype, Uint8Array.prototype)` handles, if both are present.
fn buffer_proto_pair(interp: &Interp<'_>, bctor: Handle) -> Option<(Handle, Handle)> {
    let g = interp.global_object()?;
    let u8ctor = interp
        .realm()
        .get_property(g, "Uint8Array")?
        .as_handle()
        .map(Handle::from_raw)?;
    let u8proto = interp
        .realm()
        .get_property(u8ctor, "prototype")?
        .as_handle()
        .map(Handle::from_raw)?;
    let bproto = interp
        .realm()
        .get_property(bctor, "prototype")?
        .as_handle()
        .map(Handle::from_raw)?;
    Some((bproto, u8proto))
}

/// The shared `Buffer.from(...)` implementation (also the constructor body).
fn buffer_from(cx: &mut Ctx<'_, '_>, args: &[NanBox]) -> Result<NanBox, NanBox> {
    let a0 = arg(args, 0);
    match cx.type_of(a0) {
        "string" => {
            let enc = enc_arg(cx, arg(args, 1))?;
            let bytes = decode_string(&cx.to_string(a0)?, enc);
            make_buffer(cx, &bytes)
        }
        _ if is_array_buffer(cx, a0) => {
            // Share the backing store (Node semantics for `Buffer.from(arrayBuffer)`).
            let byte_len = get_num(cx, a0, "byteLength")? as usize;
            let off = num_arg(cx, args, 1, 0.0).max(0.0) as usize;
            let len = match args.get(2).copied().filter(|v| !v.is_undefined()) {
                Some(l) => cx.to_number(l)?.max(0.0) as usize,
                None => byte_len.saturating_sub(off),
            };
            make_buffer_view(cx, a0, off, len)
        }
        _ if cx.is_object(a0) => {
            // Array / typed array / Buffer / array-like: copy the bytes.
            let bytes = read_bytes(cx, a0);
            make_buffer(cx, &bytes)
        }
        _ => Err(cx.type_error(
            "The first argument must be of type string, Buffer, ArrayBuffer, Array, or Array-like",
        )),
    }
}

/// Normalize an optional (possibly negative) index against `len`.
fn norm_index(
    cx: &mut Ctx<'_, '_>,
    args: &[NanBox],
    i: usize,
    len: usize,
    default: usize,
) -> usize {
    match args.get(i).copied().filter(|v| !v.is_undefined()) {
        Some(v) => {
            let n = cx.to_number(v).unwrap_or(0.0);
            if n < 0.0 {
                (len as f64 + n).max(0.0) as usize
            } else {
                (n as usize).min(len)
            }
        }
        None => default,
    }
}

fn install_buffer_proto(interp: &mut Interp<'_>, proto: Handle) {
    set_fn(interp, proto, "toString", 3, |cx, this, args| {
        let bytes = read_bytes(cx, this);
        let enc = enc_arg(cx, arg(args, 0))?;
        let start = norm_index(cx, args, 1, bytes.len(), 0);
        let end = norm_index(cx, args, 2, bytes.len(), bytes.len());
        let slice = if start < end && end <= bytes.len() {
            &bytes[start..end]
        } else {
            &[][..]
        };
        Ok(cx.string(&encode_bytes(slice, enc)))
    });
    set_fn(interp, proto, "toJSON", 0, |cx, this, _args| {
        let bytes = read_bytes(cx, this);
        let data: Vec<NanBox> = bytes
            .iter()
            .map(|&b| NanBox::number(f64::from(b)))
            .collect();
        let arr = cx.new_array(data);
        let o = cx.new_object();
        let ty = cx.string("Buffer");
        cx.set(o, "type", ty);
        cx.set(o, "data", arr);
        Ok(o)
    });
    // `slice`/`subarray` share the backing store (like Node's Buffer.slice).
    for name in ["slice", "subarray"] {
        set_fn(interp, proto, name, 2, |cx, this, args| {
            let len = get_num(cx, this, "length")?.max(0.0) as usize;
            let start = norm_index(cx, args, 0, len, 0);
            let end = norm_index(cx, args, 1, len, len);
            let end = end.max(start);
            let buffer = cx.get(this, "buffer")?;
            let base_off = get_num(cx, this, "byteOffset")?.max(0.0) as usize;
            make_buffer_view(cx, buffer, base_off + start, end - start)
        });
    }
    set_fn(interp, proto, "equals", 1, |cx, this, args| {
        let a = read_bytes(cx, this);
        let b = read_bytes(cx, arg(args, 0));
        Ok(NanBox::boolean(a == b))
    });
    set_fn(interp, proto, "write", 4, |cx, this, args| {
        let s = str_arg(cx, args, 0)?;
        let len = get_num(cx, this, "length")?.max(0.0) as usize;
        let offset = num_arg(cx, args, 1, 0.0).max(0.0) as usize;
        // write(string[, offset[, length]][, encoding]) — encoding may be arg1 or 3.
        let enc = if args.get(1).is_some_and(|v| cx.type_of(*v) == "string") {
            enc_arg(cx, arg(args, 1))?
        } else {
            enc_arg(cx, arg(args, 3))?
        };
        let max = num_arg(cx, args, 2, (len.saturating_sub(offset)) as f64).max(0.0) as usize;
        let bytes = decode_string(&s, enc);
        let n = bytes.len().min(max).min(len.saturating_sub(offset));
        for (k, &b) in bytes.iter().take(n).enumerate() {
            write_byte(cx, this, offset + k, b);
        }
        Ok(NanBox::number(n as f64))
    });
    set_fn(interp, proto, "fill", 3, |cx, this, args| {
        let len = get_num(cx, this, "length")?.max(0.0) as usize;
        let fill = arg(args, 0);
        let bytes: Vec<u8> = if fill.is_number() {
            vec![cx.to_number(fill)? as u8]
        } else {
            let enc = enc_arg(cx, arg(args, 3))?;
            decode_string(&cx.to_string(fill)?, enc)
        };
        let start = norm_index(cx, args, 1, len, 0);
        let end = norm_index(cx, args, 2, len, len);
        if !bytes.is_empty() {
            for i in start..end.min(len) {
                write_byte(cx, this, i, bytes[(i - start) % bytes.len()]);
            }
        }
        Ok(this)
    });
    set_fn(interp, proto, "copy", 4, |cx, this, args| {
        let src = read_bytes(cx, this);
        let target = arg(args, 0);
        let target_len = get_num(cx, target, "length")?.max(0.0) as usize;
        let target_start = num_arg(cx, args, 1, 0.0).max(0.0) as usize;
        let src_start = num_arg(cx, args, 2, 0.0).max(0.0) as usize;
        let src_end = num_arg(cx, args, 3, src.len() as f64).max(0.0) as usize;
        let src_end = src_end.min(src.len());
        let mut n = 0usize;
        let mut i = src_start;
        while i < src_end && target_start + n < target_len {
            write_byte(cx, target, target_start + n, src[i]);
            i += 1;
            n += 1;
        }
        Ok(NanBox::number(n as f64))
    });

    // A representative set of fixed-width numeric accessors.
    set_fn(interp, proto, "readUInt8", 1, |cx, this, args| {
        let off = num_arg(cx, args, 0, 0.0) as usize;
        let b = cx.get(this, &off.to_string())?;
        Ok(NanBox::number(cx.to_number(b).unwrap_or(0.0)))
    });
    set_fn(interp, proto, "readInt8", 1, |cx, this, args| {
        let off = num_arg(cx, args, 0, 0.0) as usize;
        let b = get_num(cx, this, &off.to_string())? as u8;
        Ok(NanBox::number(f64::from(b as i8)))
    });
    set_fn(interp, proto, "writeUInt8", 2, |cx, this, args| {
        let val = cx.to_number(arg(args, 0))? as u8;
        let off = num_arg(cx, args, 1, 0.0) as usize;
        write_byte(cx, this, off, val);
        Ok(NanBox::number((off + 1) as f64))
    });
    set_fn(interp, proto, "writeInt8", 2, |cx, this, args| {
        let val = cx.to_number(arg(args, 0))? as i32 as u8;
        let off = num_arg(cx, args, 1, 0.0) as usize;
        write_byte(cx, this, off, val);
        Ok(NanBox::number((off + 1) as f64))
    });
    // 16-bit.
    set_fn(interp, proto, "readUInt16LE", 1, |cx, this, args| {
        read_uint(cx, this, args, 2, true)
    });
    set_fn(interp, proto, "readUInt16BE", 1, |cx, this, args| {
        read_uint(cx, this, args, 2, false)
    });
    set_fn(interp, proto, "writeUInt16LE", 2, |cx, this, args| {
        write_uint(cx, this, args, 2, true)
    });
    set_fn(interp, proto, "writeUInt16BE", 2, |cx, this, args| {
        write_uint(cx, this, args, 2, false)
    });
    // 32-bit.
    set_fn(interp, proto, "readUInt32LE", 1, |cx, this, args| {
        read_uint(cx, this, args, 4, true)
    });
    set_fn(interp, proto, "readUInt32BE", 1, |cx, this, args| {
        read_uint(cx, this, args, 4, false)
    });
    set_fn(interp, proto, "writeUInt32LE", 2, |cx, this, args| {
        write_uint(cx, this, args, 4, true)
    });
    set_fn(interp, proto, "writeUInt32BE", 2, |cx, this, args| {
        write_uint(cx, this, args, 4, false)
    });
}

fn read_uint(
    cx: &mut Ctx<'_, '_>,
    this: NanBox,
    args: &[NanBox],
    width: usize,
    le: bool,
) -> Result<NanBox, NanBox> {
    let off = num_arg(cx, args, 0, 0.0) as usize;
    let mut val: u64 = 0;
    for k in 0..width {
        let b = get_num(cx, this, &(off + k).to_string())? as u64 & 0xff;
        let shift = if le { k } else { width - 1 - k } * 8;
        val |= b << shift;
    }
    Ok(NanBox::number(val as f64))
}

fn write_uint(
    cx: &mut Ctx<'_, '_>,
    this: NanBox,
    args: &[NanBox],
    width: usize,
    le: bool,
) -> Result<NanBox, NanBox> {
    let val = cx.to_number(arg(args, 0))? as u64;
    let off = num_arg(cx, args, 1, 0.0) as usize;
    for k in 0..width {
        let shift = if le { k } else { width - 1 - k } * 8;
        write_byte(cx, this, off + k, ((val >> shift) & 0xff) as u8);
    }
    Ok(NanBox::number((off + width) as f64))
}

// ===========================================================================
// path (POSIX)
// ===========================================================================

/// The process working directory (`/` when `std` is unavailable).
fn cwd() -> String {
    #[cfg(feature = "std")]
    {
        std::env::current_dir()
            .ok()
            .and_then(|p| p.to_str().map(ToString::to_string))
            .unwrap_or_else(|| "/".to_string())
    }
    #[cfg(not(feature = "std"))]
    {
        "/".to_string()
    }
}

fn path_normalize(p: &str) -> String {
    if p.is_empty() {
        return ".".to_string();
    }
    let is_abs = p.starts_with('/');
    let trailing = p.ends_with('/');
    let mut parts: Vec<&str> = Vec::new();
    for seg in p.split('/') {
        match seg {
            "" | "." => {}
            ".." => {
                if !parts.is_empty() && *parts.last().unwrap() != ".." {
                    parts.pop();
                } else if !is_abs {
                    parts.push("..");
                }
            }
            s => parts.push(s),
        }
    }
    let mut res = parts.join("/");
    if res.is_empty() && !is_abs {
        res = ".".to_string();
    }
    if is_abs {
        res = format!("/{res}");
    } else if trailing && !res.is_empty() && !res.ends_with('/') {
        res.push('/');
    }
    if is_abs && trailing && res.len() > 1 && !res.ends_with('/') {
        res.push('/');
    }
    res
}

fn path_join(parts: &[String]) -> String {
    let joined = parts
        .iter()
        .filter(|s| !s.is_empty())
        .cloned()
        .collect::<Vec<_>>()
        .join("/");
    if joined.is_empty() {
        ".".to_string()
    } else {
        path_normalize(&joined)
    }
}

fn path_resolve(parts: &[String]) -> String {
    let mut resolved = String::new();
    let mut abs = false;
    for p in parts.iter().rev() {
        if p.is_empty() {
            continue;
        }
        resolved = if resolved.is_empty() {
            p.clone()
        } else {
            format!("{p}/{resolved}")
        };
        if p.starts_with('/') {
            abs = true;
            break;
        }
    }
    if !abs {
        let c = cwd();
        resolved = if resolved.is_empty() {
            c
        } else {
            format!("{c}/{resolved}")
        };
    }
    let mut n = path_normalize(&resolved);
    if !n.starts_with('/') {
        n = format!("/{n}");
    }
    if n.len() > 1 && n.ends_with('/') {
        n.truncate(n.len() - 1);
    }
    n
}

fn path_dirname(p: &str) -> String {
    if p.is_empty() {
        return ".".to_string();
    }
    let bytes = p.as_bytes();
    let mut end = p.len();
    while end > 1 && bytes[end - 1] == b'/' {
        end -= 1;
    }
    let s = &p[..end];
    match s.rfind('/') {
        None => ".".to_string(),
        Some(0) => "/".to_string(),
        Some(i) => s[..i].to_string(),
    }
}

fn path_basename(p: &str, ext: Option<&str>) -> String {
    let bytes = p.as_bytes();
    let mut end = p.len();
    while end > 1 && bytes[end - 1] == b'/' {
        end -= 1;
    }
    let s = &p[..end];
    let base = match s.rfind('/') {
        Some(i) => &s[i + 1..],
        None => s,
    };
    match ext {
        Some(e) if !e.is_empty() && base.ends_with(e) && base != e => {
            base[..base.len() - e.len()].to_string()
        }
        _ => base.to_string(),
    }
}

fn path_extname(p: &str) -> String {
    let base = path_basename(p, None);
    match base.rfind('.') {
        Some(0) | None => String::new(),
        Some(i) => base[i..].to_string(),
    }
}

fn path_relative(from: &str, to: &str) -> String {
    let f = path_resolve(&[from.to_string()]);
    let t = path_resolve(&[to.to_string()]);
    let fp: Vec<&str> = f.split('/').filter(|s| !s.is_empty()).collect();
    let tp: Vec<&str> = t.split('/').filter(|s| !s.is_empty()).collect();
    let mut i = 0;
    while i < fp.len() && i < tp.len() && fp[i] == tp[i] {
        i += 1;
    }
    let mut out: Vec<String> = Vec::new();
    for _ in i..fp.len() {
        out.push("..".to_string());
    }
    for seg in &tp[i..] {
        out.push((*seg).to_string());
    }
    out.join("/")
}

fn install_path(interp: &mut Interp<'_>) {
    let path = new_ns(interp);
    set_fn(interp, path, "join", 2, |cx, _t, args| {
        let mut parts = Vec::with_capacity(args.len());
        for &a in args {
            parts.push(cx.to_string(a)?);
        }
        Ok(cx.string(&path_join(&parts)))
    });
    set_fn(interp, path, "resolve", 2, |cx, _t, args| {
        let mut parts = Vec::with_capacity(args.len());
        for &a in args {
            parts.push(cx.to_string(a)?);
        }
        Ok(cx.string(&path_resolve(&parts)))
    });
    set_fn(interp, path, "normalize", 1, |cx, _t, args| {
        let p = str_arg(cx, args, 0)?;
        Ok(cx.string(&path_normalize(&p)))
    });
    set_fn(interp, path, "dirname", 1, |cx, _t, args| {
        let p = str_arg(cx, args, 0)?;
        Ok(cx.string(&path_dirname(&p)))
    });
    set_fn(interp, path, "basename", 2, |cx, _t, args| {
        let p = str_arg(cx, args, 0)?;
        let ext = match args.get(1).copied().filter(|v| !v.is_undefined()) {
            Some(v) => Some(cx.to_string(v)?),
            None => None,
        };
        Ok(cx.string(&path_basename(&p, ext.as_deref())))
    });
    set_fn(interp, path, "extname", 1, |cx, _t, args| {
        let p = str_arg(cx, args, 0)?;
        Ok(cx.string(&path_extname(&p)))
    });
    set_fn(interp, path, "isAbsolute", 1, |cx, _t, args| {
        let p = str_arg(cx, args, 0)?;
        Ok(NanBox::boolean(p.starts_with('/')))
    });
    set_fn(interp, path, "relative", 2, |cx, _t, args| {
        let from = str_arg(cx, args, 0)?;
        let to = str_arg(cx, args, 1)?;
        Ok(cx.string(&path_relative(&from, &to)))
    });
    set_fn(interp, path, "parse", 1, |cx, _t, args| {
        let p = str_arg(cx, args, 0)?;
        let root = if p.starts_with('/') { "/" } else { "" };
        let dir = path_dirname(&p);
        let base = path_basename(&p, None);
        let ext = path_extname(&p);
        let name = base[..base.len() - ext.len()].to_string();
        let o = cx.new_object();
        let rv = cx.string(root);
        cx.set(o, "root", rv);
        let dv = cx.string(&dir);
        cx.set(o, "dir", dv);
        let bv = cx.string(&base);
        cx.set(o, "base", bv);
        let ev = cx.string(&ext);
        cx.set(o, "ext", ev);
        let nv = cx.string(&name);
        cx.set(o, "name", nv);
        Ok(o)
    });
    set_fn(interp, path, "format", 1, |cx, _t, args| {
        let o = arg(args, 0);
        let get = |cx: &mut Ctx<'_, '_>, k: &str| -> Result<String, NanBox> {
            let v = cx.get(o, k)?;
            if v.is_undefined() || v.is_null() {
                Ok(String::new())
            } else {
                cx.to_string(v)
            }
        };
        let mut base = get(cx, "base")?;
        if base.is_empty() {
            base = format!("{}{}", get(cx, "name")?, get(cx, "ext")?);
        }
        let mut dir = get(cx, "dir")?;
        if dir.is_empty() {
            dir = get(cx, "root")?;
        }
        let out = if dir.is_empty() {
            base
        } else if dir.ends_with('/') {
            format!("{dir}{base}")
        } else {
            format!("{dir}/{base}")
        };
        Ok(cx.string(&out))
    });
    set_str(interp, path, "sep", "/");
    set_str(interp, path, "delimiter", ":");
    // `path.posix` self-reference (win32 deferred).
    interp
        .realm_mut()
        .set_property(path, "posix", NanBox::handle(path.to_raw()));
    declare(interp, "path", path);
}

// ===========================================================================
// os
// ===========================================================================

fn os_platform() -> &'static str {
    #[cfg(feature = "std")]
    {
        match std::env::consts::OS {
            "macos" => "darwin",
            "windows" => "win32",
            other => match other {
                "linux" => "linux",
                "freebsd" => "freebsd",
                "openbsd" => "openbsd",
                "android" => "android",
                _ => "linux",
            },
        }
    }
    #[cfg(not(feature = "std"))]
    {
        "linux"
    }
}

fn os_arch() -> &'static str {
    #[cfg(feature = "std")]
    {
        match std::env::consts::ARCH {
            "x86_64" => "x64",
            "x86" => "ia32",
            "aarch64" => "arm64",
            "arm" => "arm",
            other => other,
        }
    }
    #[cfg(not(feature = "std"))]
    {
        "x64"
    }
}

fn os_type() -> &'static str {
    match os_platform() {
        "darwin" => "Darwin",
        "win32" => "Windows_NT",
        _ => "Linux",
    }
}

fn env_var(_name: &str) -> Option<String> {
    #[cfg(feature = "std")]
    {
        std::env::var(_name).ok()
    }
    #[cfg(not(feature = "std"))]
    {
        None
    }
}

fn install_os(interp: &mut Interp<'_>) {
    let os = new_ns(interp);
    set_fn(interp, os, "platform", 0, |cx, _t, _a| {
        Ok(cx.string(os_platform()))
    });
    set_fn(interp, os, "arch", 0, |cx, _t, _a| Ok(cx.string(os_arch())));
    set_fn(interp, os, "type", 0, |cx, _t, _a| Ok(cx.string(os_type())));
    set_fn(
        interp,
        os,
        "release",
        0,
        |cx, _t, _a| Ok(cx.string("0.0.0")),
    );
    set_fn(interp, os, "version", 0, |cx, _t, _a| {
        Ok(cx.string("kataan"))
    });
    set_fn(interp, os, "homedir", 0, |cx, _t, _a| {
        let h = env_var("HOME")
            .or_else(|| env_var("USERPROFILE"))
            .unwrap_or_else(|| "/".to_string());
        Ok(cx.string(&h))
    });
    set_fn(interp, os, "tmpdir", 0, |cx, _t, _a| {
        let t = env_var("TMPDIR")
            .or_else(|| env_var("TMP"))
            .or_else(|| env_var("TEMP"))
            .unwrap_or_else(|| "/tmp".to_string());
        Ok(cx.string(&t))
    });
    set_fn(interp, os, "hostname", 0, |cx, _t, _a| {
        let h = env_var("HOSTNAME")
            .or_else(|| env_var("COMPUTERNAME"))
            .unwrap_or_else(|| "localhost".to_string());
        Ok(cx.string(&h))
    });
    set_fn(interp, os, "endianness", 0, |cx, _t, _a| {
        Ok(cx.string(if cfg!(target_endian = "big") {
            "BE"
        } else {
            "LE"
        }))
    });
    set_fn(interp, os, "totalmem", 0, |_cx, _t, _a| {
        Ok(NanBox::number(8.0 * 1024.0 * 1024.0 * 1024.0))
    });
    set_fn(interp, os, "freemem", 0, |_cx, _t, _a| {
        Ok(NanBox::number(4.0 * 1024.0 * 1024.0 * 1024.0))
    });
    set_fn(interp, os, "uptime", 0, |_cx, _t, _a| {
        Ok(NanBox::number(0.0))
    });
    set_fn(interp, os, "loadavg", 0, |cx, _t, _a| {
        Ok(cx.new_array(vec![
            NanBox::number(0.0),
            NanBox::number(0.0),
            NanBox::number(0.0),
        ]))
    });
    set_fn(interp, os, "cpus", 0, |cx, _t, _a| {
        let n = cpu_count();
        let mut arr = Vec::with_capacity(n);
        for _ in 0..n {
            let o = cx.new_object();
            let model = cx.string("kataan virtual CPU");
            cx.set(o, "model", model);
            cx.set(o, "speed", NanBox::number(2400.0));
            let times = cx.new_object();
            for k in ["user", "nice", "sys", "idle", "irq"] {
                cx.set(times, k, NanBox::number(0.0));
            }
            cx.set(o, "times", times);
            arr.push(o);
        }
        Ok(cx.new_array(arr))
    });
    let eol = if cfg!(target_os = "windows") {
        "\r\n"
    } else {
        "\n"
    };
    set_str(interp, os, "EOL", eol);
    set_num(interp, os, "constants", 0.0); // placeholder; real constants deferred.
    declare(interp, "os", os);
}

fn cpu_count() -> usize {
    #[cfg(feature = "std")]
    {
        std::thread::available_parallelism()
            .map(core::num::NonZeroUsize::get)
            .unwrap_or(1)
    }
    #[cfg(not(feature = "std"))]
    {
        1
    }
}

// ===========================================================================
// util
// ===========================================================================

/// `Object.prototype.toString.call(v)` stripped to the tag word (`"Array"`,
/// `"Map"`, `"Date"`, `"Object"`, …).
fn obj_tag(cx: &mut Ctx<'_, '_>, v: NanBox) -> String {
    let g = cx.global();
    let tag = (|| -> Result<String, NanBox> {
        let obj = cx.get(g, "Object")?;
        let proto = cx.get(obj, "prototype")?;
        let ts = cx.get(proto, "toString")?;
        let r = cx.call(ts, v, &[])?;
        cx.to_string(r)
    })()
    .unwrap_or_default();
    tag.strip_prefix("[object ")
        .and_then(|s| s.strip_suffix(']'))
        .map(ToString::to_string)
        .unwrap_or(tag)
}

/// Whether an object key needs quoting when printed by `inspect`.
fn needs_quote(k: &str) -> bool {
    k.is_empty()
        || !k.chars().enumerate().all(|(i, c)| {
            c == '_' || c == '$' || c.is_ascii_alphabetic() || (i > 0 && c.is_ascii_digit())
        })
}

/// A depth-limited debug rendering (`util.inspect`), tracking visited objects to
/// break cycles.
fn inspect(cx: &mut Ctx<'_, '_>, v: NanBox, depth: i32, seen: &mut Vec<u64>) -> String {
    match cx.type_of(v) {
        "undefined" => "undefined".to_string(),
        "boolean" | "number" | "bigint" => cx.to_string(v).unwrap_or_default(),
        "symbol" => cx.to_string(v).unwrap_or_default(),
        "string" => format!(
            "'{}'",
            cx.to_string(v).unwrap_or_default().replace('\'', "\\'")
        ),
        "function" => {
            let name = cx.get(v, "name").ok().and_then(|n| cx.to_string(n).ok());
            match name.filter(|n| !n.is_empty()) {
                Some(n) => format!("[Function: {n}]"),
                None => "[Function (anonymous)]".to_string(),
            }
        }
        _ => {
            if v.is_null() {
                return "null".to_string();
            }
            let handle = v.as_handle().unwrap_or(0);
            if seen.contains(&handle) {
                return "[Circular *1]".to_string();
            }
            let tag = obj_tag(cx, v);
            if cx.is_array(v) {
                if depth < 0 {
                    return "[Array]".to_string();
                }
                seen.push(handle);
                let n = cx.array_len(v).unwrap_or(0);
                let items: Vec<String> = (0..n)
                    .map(|i| {
                        let e = cx.array_get(v, i);
                        inspect(cx, e, depth - 1, seen)
                    })
                    .collect();
                seen.pop();
                if items.is_empty() {
                    "[]".to_string()
                } else {
                    format!("[ {} ]", items.join(", "))
                }
            } else if tag == "Map" {
                inspect_collection(cx, v, depth, seen, handle, true)
            } else if tag == "Set" {
                inspect_collection(cx, v, depth, seen, handle, false)
            } else if tag == "Date" {
                // Prefer the ISO rendering, like Node's inspect.
                (|| -> Result<String, NanBox> {
                    let iso = cx.get(v, "toISOString")?;
                    let r = cx.call(iso, v, &[])?;
                    cx.to_string(r)
                })()
                .unwrap_or_else(|_| cx.to_string(v).unwrap_or_default())
            } else if tag == "RegExp" || tag == "Error" {
                cx.to_string(v).unwrap_or_default()
            } else {
                if depth < 0 {
                    return "[Object]".to_string();
                }
                seen.push(handle);
                let keys = cx.own_keys(v);
                let entries: Vec<String> = keys
                    .into_iter()
                    .map(|k| {
                        let val = cx.get(v, &k).unwrap_or(NanBox::undefined());
                        let vs = inspect(cx, val, depth - 1, seen);
                        let ks = if needs_quote(&k) {
                            format!("'{}'", k.replace('\'', "\\'"))
                        } else {
                            k
                        };
                        format!("{ks}: {vs}")
                    })
                    .collect();
                seen.pop();
                let prefix = if tag != "Object" && !tag.is_empty() {
                    format!("{tag} ")
                } else {
                    String::new()
                };
                if entries.is_empty() {
                    format!("{prefix}{{}}")
                } else {
                    format!("{prefix}{{ {} }}", entries.join(", "))
                }
            }
        }
    }
}

/// Render a `Map`/`Set` by reflecting its entries through `Array.from`.
fn inspect_collection(
    cx: &mut Ctx<'_, '_>,
    v: NanBox,
    depth: i32,
    seen: &mut Vec<u64>,
    handle: u64,
    is_map: bool,
) -> String {
    let size = cx
        .get(v, "size")
        .ok()
        .and_then(|s| cx.to_number(s).ok())
        .unwrap_or(0.0) as usize;
    let label = if is_map { "Map" } else { "Set" };
    if depth < 0 {
        return format!("[{label}]");
    }
    // `Array.from(collection)` → values (Set) or [k, v] pairs (Map).
    let pairs = (|| -> Result<NanBox, NanBox> {
        let g = cx.global();
        let arr_ctor = cx.get(g, "Array")?;
        let from = cx.get(arr_ctor, "from")?;
        cx.call(from, NanBox::undefined(), &[v])
    })()
    .unwrap_or(NanBox::undefined());
    seen.push(handle);
    let n = cx.array_len(pairs).unwrap_or(0);
    let items: Vec<String> = (0..n)
        .map(|i| {
            let entry = cx.array_get(pairs, i);
            if is_map {
                let k = cx.array_get(entry, 0);
                let val = cx.array_get(entry, 1);
                let ks = inspect(cx, k, depth - 1, seen);
                let vs = inspect(cx, val, depth - 1, seen);
                format!("{ks} => {vs}")
            } else {
                inspect(cx, entry, depth - 1, seen)
            }
        })
        .collect();
    seen.pop();
    if items.is_empty() {
        format!("{label}(0) {{}}")
    } else {
        format!("{label}({size}) {{ {} }}", items.join(", "))
    }
}

/// A single `%`-less argument rendered for `util.format` / `console`.
fn format_value(cx: &mut Ctx<'_, '_>, v: NanBox) -> String {
    if cx.type_of(v) == "string" {
        cx.to_string(v).unwrap_or_default()
    } else {
        let mut seen = Vec::new();
        inspect(cx, v, 2, &mut seen)
    }
}

/// `util.format(fmt, ...args)` — the `%s/%d/%i/%f/%j/%o/%O/%c/%%` mini-language.
fn util_format(cx: &mut Ctx<'_, '_>, args: &[NanBox]) -> Result<String, NanBox> {
    if args.is_empty() {
        return Ok(String::new());
    }
    let first = args[0];
    let mut out = String::new();
    let mut next = 1usize;
    if cx.type_of(first) == "string" {
        let fmt = cx.to_string(first)?;
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
                    's' | 'd' | 'i' | 'f' | 'j' | 'o' | 'O' | 'c' => {
                        if spec == 'c' {
                            // CSS directive — consumed, produces nothing.
                            if next < args.len() {
                                next += 1;
                            }
                            i += 2;
                            continue;
                        }
                        if next >= args.len() {
                            out.push('%');
                            out.push(spec);
                            i += 2;
                            continue;
                        }
                        let a = args[next];
                        next += 1;
                        let s = match spec {
                            's' => {
                                if cx.type_of(a) == "string" || a.is_number() {
                                    cx.to_string(a).unwrap_or_default()
                                } else {
                                    let mut seen = Vec::new();
                                    inspect(cx, a, 2, &mut seen)
                                }
                            }
                            'd' | 'i' => {
                                let n = cx.to_number(a).unwrap_or(f64::NAN);
                                if n.is_nan() {
                                    "NaN".to_string()
                                } else {
                                    format!("{}", n.trunc() as i64)
                                }
                            }
                            'f' => {
                                let n = cx.to_number(a).unwrap_or(f64::NAN);
                                cx.to_string(NanBox::number(n)).unwrap_or_default()
                            }
                            'j' => json_stringify(cx, a).unwrap_or_else(|| "undefined".to_string()),
                            'o' | 'O' => {
                                let mut seen = Vec::new();
                                inspect(cx, a, 2, &mut seen)
                            }
                            _ => String::new(),
                        };
                        out.push_str(&s);
                        i += 2;
                        continue;
                    }
                    _ => {}
                }
            }
            out.push(chars[i]);
            i += 1;
        }
    } else {
        out.push_str(&format_value(cx, first));
    }
    // Trailing args are appended space-separated.
    for &a in &args[next..] {
        out.push(' ');
        out.push_str(&format_value(cx, a));
    }
    Ok(out)
}

/// `JSON.stringify(v)` via the realm's own JSON (returns `None` for `undefined`).
fn json_stringify(cx: &mut Ctx<'_, '_>, v: NanBox) -> Option<String> {
    let g = cx.global();
    let json = cx.get(g, "JSON").ok()?;
    let stringify = cx.get(json, "stringify").ok()?;
    let r = cx.call(stringify, NanBox::undefined(), &[v]).ok()?;
    if r.is_undefined() {
        None
    } else {
        cx.to_string(r).ok()
    }
}

fn install_util(interp: &mut Interp<'_>) {
    let util = new_ns(interp);
    set_fn(interp, util, "format", 1, |cx, _t, args| {
        let s = util_format(cx, args)?;
        Ok(cx.string(&s))
    });
    set_fn(interp, util, "inspect", 2, |cx, _t, args| {
        let v = arg(args, 0);
        // Optional `{ depth }`.
        let depth = match args.get(1).copied().filter(|o| cx.is_object(*o)) {
            Some(o) => {
                let d = cx.get(o, "depth")?;
                if d.is_null() {
                    i32::MAX
                } else if d.is_undefined() {
                    2
                } else {
                    cx.to_number(d)? as i32
                }
            }
            None => 2,
        };
        let mut seen = Vec::new();
        let s = inspect(cx, v, depth, &mut seen);
        Ok(cx.string(&s))
    });
    set_fn(interp, util, "inherits", 2, |cx, _t, args| {
        // Re-link the existing `ctor.prototype` under `superCtor.prototype`
        // (modern Node semantics — preserves the prototype object `new` uses).
        let ctor = arg(args, 0);
        let super_ctor = arg(args, 1);
        let g = cx.global();
        let obj = cx.get(g, "Object")?;
        let set_proto = cx.get(obj, "setPrototypeOf")?;
        let ctor_proto = cx.get(ctor, "prototype")?;
        let super_proto = cx.get(super_ctor, "prototype")?;
        cx.set(ctor, "super_", super_ctor);
        let u = cx.undefined();
        cx.call(set_proto, u, &[ctor_proto, super_proto])?;
        Ok(NanBox::undefined())
    });
    set_fn(interp, util, "deprecate", 3, |_cx, _t, args| {
        // Divergence: returns the original function unwrapped (no one-shot warning),
        // since a per-call wrapper cannot be minted from a host closure.
        Ok(arg(args, 0))
    });
    // `util.promisify` — see the generic invoker/callback pair below.
    install_promisify(interp, util);
    // `util.types.*`.
    let types = new_ns(interp);
    for (name, tag) in [
        ("isDate", "Date"),
        ("isRegExp", "RegExp"),
        ("isMap", "Map"),
        ("isSet", "Set"),
        ("isWeakMap", "WeakMap"),
        ("isWeakSet", "WeakSet"),
        ("isPromise", "Promise"),
        ("isArrayBuffer", "ArrayBuffer"),
        ("isProxy", "Proxy"),
    ] {
        let want = tag.to_string();
        set_fn(interp, types, name, 1, move |cx, _t, args| {
            let v = arg(args, 0);
            Ok(NanBox::boolean(cx.is_object(v) && obj_tag(cx, v) == want))
        });
    }
    set_fn(interp, types, "isNativeError", 1, |cx, _t, args| {
        let v = arg(args, 0);
        Ok(NanBox::boolean(
            cx.is_object(v) && obj_tag(cx, v) == "Error",
        ))
    });
    set_fn(interp, types, "isTypedArray", 1, |cx, _t, args| {
        let v = arg(args, 0);
        // A typed array has both `length` and a `buffer`.
        Ok(NanBox::boolean(
            cx.is_object(v)
                && cx.has(v, "length")
                && cx.has(v, "buffer")
                && cx.has(v, "byteOffset"),
        ))
    });
    interp
        .realm_mut()
        .set_property(util, "types", NanBox::handle(types.to_raw()));
    declare(interp, "util", util);
}

/// Install `util.promisify` plus its shared, pre-registered invoker + callback.
///
/// `promisify(fn)` returns `invoker.bind(undefined, fn)`. Calling that runs
/// `invoker(fn, ...args)`, which builds a deferred promise, binds the token into
/// the generic Node-style callback, calls `fn(...args, callback)`, and hands the
/// promise back. The callback settles the deferred from the pinned
/// `[resolve, reject]` pair.
fn install_promisify(interp: &mut Interp<'_>, util: Handle) {
    // The generic `(token, err, value)` callback, stashed as a hidden util slot.
    let cb = interp.register_fn("nodeUtilPromisifyCallback", 3, |cx, _t, args| {
        let token = cx.to_number(arg(args, 0))? as u32;
        let pair = cx.persistent(token);
        let resolve = cx.array_get(pair, 0);
        let reject = cx.array_get(pair, 1);
        let err = arg(args, 1);
        let u = cx.undefined();
        if err.is_null() || err.is_undefined() {
            cx.call(resolve, u, &[arg(args, 2)])?;
        } else {
            cx.call(reject, u, &[err])?;
        }
        cx.release_persistent(token);
        Ok(NanBox::undefined())
    });
    interp
        .realm_mut()
        .set_hidden_property(util, "\u{0}promisifyCb", cb);

    // The generic invoker: `(fn, ...args)` → Promise.
    let invoker = interp.register_fn("nodeUtilPromisifyInvoker", 1, |cx, _t, args| {
        let fn_v = arg(args, 0);
        let real = &args[1..];
        let (promise, token) = cx.deferred()?;
        // Bind the token into the shared callback.
        let g = cx.global();
        let util = cx.get(g, "util")?;
        let cb = cx.get(util, "\u{0}promisifyCb")?;
        let bind = cx.get(cb, "bind")?;
        let u = cx.undefined();
        let bound = cx.call(bind, cb, &[u, NanBox::number(f64::from(token))])?;
        let mut call_args = real.to_vec();
        call_args.push(bound);
        // If `fn` throws synchronously, reject the promise.
        if let Err(e) = cx.call(fn_v, u, &call_args) {
            let pair = cx.persistent(token);
            let reject = cx.array_get(pair, 1);
            let _ = cx.call(reject, u, &[e]);
            cx.release_persistent(token);
        }
        Ok(promise)
    });
    let invoker_idx = interp.persist(invoker);
    interp
        .realm_mut()
        .set_hidden_property(util, "\u{0}promisifyInvoker", invoker);

    set_fn(interp, util, "promisify", 1, move |cx, _t, args| {
        let fn_v = arg(args, 0);
        if !cx.is_callable(fn_v) {
            return Err(cx.type_error("The \"original\" argument must be of type function"));
        }
        let invoker = cx.persistent(invoker_idx);
        let bind = cx.get(invoker, "bind")?;
        let u = cx.undefined();
        cx.call(bind, invoker, &[u, fn_v])
    });
}

// ===========================================================================
// querystring
// ===========================================================================

fn qs_escape(s: &str) -> String {
    let mut out = String::new();
    for &b in s.as_bytes() {
        if b.is_ascii_alphanumeric()
            || matches!(
                b,
                b'-' | b'_' | b'.' | b'!' | b'~' | b'*' | b'\'' | b'(' | b')'
            )
        {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

fn qs_unescape(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%'
            && i + 2 < b.len()
            && let (Some(h), Some(l)) = (hexval(b[i + 1]), hexval(b[i + 2]))
        {
            out.push((h << 4) | l);
            i += 3;
            continue;
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn install_querystring(interp: &mut Interp<'_>) {
    let qs = new_ns(interp);
    set_fn(interp, qs, "escape", 1, |cx, _t, args| {
        let s = str_arg(cx, args, 0)?;
        Ok(cx.string(&qs_escape(&s)))
    });
    set_fn(interp, qs, "unescape", 1, |cx, _t, args| {
        let s = str_arg(cx, args, 0)?;
        Ok(cx.string(&qs_unescape(&s)))
    });
    set_fn(interp, qs, "parse", 3, |cx, _t, args| {
        let s = str_arg(cx, args, 0)?;
        let sep = opt_sep(cx, args, 1, "&")?;
        let eq = opt_sep(cx, args, 2, "=")?;
        let out = cx.new_object();
        if !s.is_empty() {
            for pair in s.split(sep.as_str()) {
                if pair.is_empty() {
                    continue;
                }
                let (k, v) = match pair.split_once(eq.as_str()) {
                    Some((k, v)) => (k, v),
                    None => (pair, ""),
                };
                let key = qs_unescape(&k.replace('+', " "));
                let val = qs_unescape(&v.replace('+', " "));
                let val_box = cx.string(&val);
                // Repeated keys accumulate into an array.
                if cx.has_own(out, &key) {
                    let existing = cx.get(out, &key)?;
                    if cx.is_array(existing) {
                        let n = cx.array_len(existing).unwrap_or(0);
                        cx.array_set(existing, n, val_box);
                    } else {
                        let arr = cx.new_array(vec![existing, val_box]);
                        cx.set(out, &key, arr);
                    }
                } else {
                    cx.set(out, &key, val_box);
                }
            }
        }
        Ok(out)
    });
    set_fn(interp, qs, "stringify", 3, |cx, _t, args| {
        let obj = arg(args, 0);
        let sep = opt_sep(cx, args, 1, "&")?;
        let eq = opt_sep(cx, args, 2, "=")?;
        if !cx.is_object(obj) {
            return Ok(cx.string(""));
        }
        let mut parts: Vec<String> = Vec::new();
        for key in cx.own_keys(obj) {
            let ek = qs_escape(&key);
            let v = cx.get(obj, &key)?;
            if cx.is_array(v) {
                let n = cx.array_len(v).unwrap_or(0);
                for i in 0..n {
                    let e = cx.array_get(v, i);
                    let ev = qs_escape(&cx.to_string(e)?);
                    parts.push(format!("{ek}{eq}{ev}"));
                }
            } else {
                let sv = if v.is_undefined() || v.is_null() {
                    String::new()
                } else {
                    qs_escape(&cx.to_string(v)?)
                };
                parts.push(format!("{ek}{eq}{sv}"));
            }
        }
        Ok(cx.string(&parts.join(sep.as_str())))
    });
    // `querystring.decode`/`encode` aliases.
    if let Some(parse) = interp.realm().get_property(qs, "parse") {
        interp.realm_mut().set_property(qs, "decode", parse);
    }
    if let Some(stringify) = interp.realm().get_property(qs, "stringify") {
        interp.realm_mut().set_property(qs, "encode", stringify);
    }
    declare(interp, "querystring", qs);
}

fn opt_sep(
    cx: &mut Ctx<'_, '_>,
    args: &[NanBox],
    i: usize,
    default: &str,
) -> Result<String, NanBox> {
    match args
        .get(i)
        .copied()
        .filter(|v| !v.is_undefined() && !v.is_null())
    {
        Some(v) => cx.to_string(v),
        None => Ok(default.to_string()),
    }
}

// ===========================================================================
// process (additive: fetch-or-create the global)
// ===========================================================================

fn install_process(interp: &mut Interp<'_>) {
    let g = interp.global_object().expect("global");
    // Fetch-or-create — the timers agent also augments `process` (nextTick), so we
    // never overwrite an existing object.
    let proc = match interp
        .realm()
        .get_property(g, "process")
        .and_then(|v| v.as_handle().map(Handle::from_raw))
    {
        Some(h) => h,
        None => {
            let h = interp.realm_mut().new_object();
            declare(interp, "process", h);
            h
        }
    };
    set_str(interp, proc, "platform", os_platform());
    set_str(interp, proc, "arch", os_arch());
    set_str(interp, proc, "version", "v20.0.0");
    set_num(interp, proc, "pid", 1.0);

    // `process.versions`.
    let versions = interp.realm_mut().new_object();
    set_str(interp, versions, "node", "20.0.0");
    set_str(interp, versions, "kataan", env!("CARGO_PKG_VERSION"));
    interp
        .realm_mut()
        .set_property(proc, "versions", NanBox::handle(versions.to_raw()));

    // `process.argv` — ["node", ...].
    let argv = interp.realm_mut().new_array(vec![]);
    let node = str_val(interp, "node");
    interp.realm_mut().set_element(argv, 0, node);
    interp
        .realm_mut()
        .set_property(proc, "argv", NanBox::handle(argv.to_raw()));

    // `process.env` — a snapshot object.
    let env = interp.realm_mut().new_object();
    #[cfg(feature = "std")]
    {
        for (k, v) in std::env::vars() {
            let vv = str_val(interp, &v);
            interp.realm_mut().set_property(env, &k, vv);
        }
    }
    interp
        .realm_mut()
        .set_property(proc, "env", NanBox::handle(env.to_raw()));

    set_fn(interp, proc, "cwd", 0, |cx, _t, _a| Ok(cx.string(&cwd())));
}

// ===========================================================================
// require('node:...') shim
// ===========================================================================

fn install_require(interp: &mut Interp<'_>) {
    // Only add a `require` if none exists (a real CJS loader would own it).
    let g = interp.global_object().expect("global");
    if interp
        .realm()
        .get_property(g, "require")
        .is_some_and(|v| !v.is_undefined())
    {
        return;
    }
    interp.register_global_fn("require", 1, |cx, _t, args| {
        let spec = str_arg(cx, args, 0)?;
        let name = spec.strip_prefix("node:").unwrap_or(&spec);
        let g = cx.global();
        match name {
            "path" | "os" | "util" | "querystring" | "process" => cx.get(g, name),
            "buffer" => {
                // `require('buffer')` → a module namespace exposing `Buffer`.
                let m = cx.new_object();
                let b = cx.get(g, "Buffer")?;
                cx.set(m, "Buffer", b);
                Ok(m)
            }
            other => Err(cx.error(&format!("Cannot find module '{other}'"))),
        }
    });
}
