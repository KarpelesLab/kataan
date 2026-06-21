//! The ES2025 `uint8array-base64` proposal: base64/hex codecs for `Uint8Array`.
//!
//! Six methods — `Uint8Array.prototype.{toBase64,toHex,setFromBase64,setFromHex}`
//! and the statics `Uint8Array.{fromBase64,fromHex}` — implemented as a pure
//! byte↔string codec (the spec's `FromBase64`/`FromHex`/`DecodeBase64Chunk`
//! abstract operations) plus thin dispatch wrappers that read the option bag,
//! validate the receiver, and materialize the result.

use super::*;

/// The standard base64 alphabet (`alphabet: "base64"`).
const B64_STD: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
/// The URL-safe base64 alphabet (`alphabet: "base64url"`): `-`/`_` for `+`/`/`.
const B64_URL: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

/// Which base64 alphabet a decode/encode uses.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum B64Alphabet {
    /// `"base64"` — `+` and `/` for the last two symbols.
    Standard,
    /// `"base64url"` — `-` and `_` for the last two symbols.
    Url,
}

/// `lastChunkHandling` for base64 decode: how the trailing (possibly partial)
/// chunk is treated.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum LastChunk {
    /// Default: a 2/3-char final chunk decodes; a single leftover char is an
    /// error; padding optional; overflow bits ignored.
    Loose,
    /// Padding required; overflow bits must be zero; no trailing chars.
    Strict,
    /// A partial trailing chunk lacking enough chars (no padding) is left
    /// undecoded — `read` reflects only the fully-decoded portion.
    StopBeforePartial,
}

/// The outcome of a `FromBase64`/`FromHex` decode: the bytes produced, the count
/// of input code units consumed, and whether an error occurred *after* writing
/// those bytes (used by `setFrom*`, which writes valid output before throwing).
pub(crate) struct DecodeResult {
    /// Decoded bytes (capped at `maxLength`).
    pub bytes: Vec<u8>,
    /// Input UTF-16 code units consumed to produce `bytes`.
    pub read: usize,
    /// `true` if the input is malformed past the decoded portion.
    pub error: bool,
}

/// Whether a code unit is ASCII whitespace per the spec's `SkipAsciiWhitespace`
/// (TAB, LF, FF, CR, SPACE).
fn is_ascii_ws(c: u8) -> bool {
    matches!(c, b'\t' | b'\n' | b'\x0C' | b'\r' | b' ')
}

/// The spec `SkipAsciiWhitespace(string, index)`: the first index ≥ `index` whose
/// code unit is not ASCII whitespace (or the end of `units`).
fn skip_ascii_ws(units: &[u8], mut index: usize) -> usize {
    while index < units.len() && is_ascii_ws(units[index]) {
        index += 1;
    }
    index
}

/// The 0..=63 value of a base64 symbol in `alphabet`, or `None` if it is not a
/// symbol of that alphabet.
fn b64_value(c: u8, alphabet: B64Alphabet) -> Option<u8> {
    match c {
        b'A'..=b'Z' => Some(c - b'A'),
        b'a'..=b'z' => Some(c - b'a' + 26),
        b'0'..=b'9' => Some(c - b'0' + 52),
        b'+' if alphabet == B64Alphabet::Standard => Some(62),
        b'/' if alphabet == B64Alphabet::Standard => Some(63),
        b'-' if alphabet == B64Alphabet::Url => Some(62),
        b'_' if alphabet == B64Alphabet::Url => Some(63),
        _ => None,
    }
}

/// The spec `FromBase64(string, alphabet, lastChunkHandling, maxLength)`.
///
/// Decodes the base64 `units` (UTF-16 code units, but base64/whitespace are all
/// ASCII so a non-ASCII unit is simply not a base64 symbol) into bytes, stopping
/// once `max_len` bytes are produced. Returns the decoded bytes, the count of
/// units read to produce them, and whether the remaining input is malformed
/// (`error`, surfaced as a `SyntaxError` by the caller — after the valid bytes
/// are kept). On a hard error the partial output and the read offset *before*
/// the offending chunk are returned so `setFromBase64` writes up to the error.
pub(crate) fn from_base64(
    units: &[u8],
    alphabet: B64Alphabet,
    last_chunk: LastChunk,
    max_len: usize,
) -> DecodeResult {
    let length = units.len();
    let mut bytes = Vec::new();
    if max_len == 0 {
        return DecodeResult { bytes, read: 0, error: false };
    }
    // The current chunk of base64 *values* being accumulated, the offset where
    // the next char is read, and `read` — the offset after the last *complete*
    // 4-symbol chunk (the resumption point a streaming `setFromBase64` reports).
    let mut chunk = [0u8; 4];
    let mut chunk_len = 0usize;
    let mut index = 0usize;
    let mut read = 0usize;
    loop {
        index = skip_ascii_ws(units, index);
        if index == length {
            // End of input. A non-empty trailing chunk is the final (partial) one.
            if chunk_len > 0 {
                match last_chunk {
                    LastChunk::StopBeforePartial => {
                        return DecodeResult { bytes, read, error: false };
                    }
                    LastChunk::Loose => {
                        if chunk_len == 1 {
                            // A single leftover symbol can never form a byte.
                            return DecodeResult { bytes, read, error: true };
                        }
                        // Decode the unpadded 2/3-symbol chunk (extra bits ignored).
                        let decoded = decode_chunk(&chunk[..chunk_len], false).unwrap();
                        bytes.extend_from_slice(&decoded[..chunk_len - 1]);
                    }
                    LastChunk::Strict => {
                        // Missing padding is malformed in strict mode.
                        return DecodeResult { bytes, read, error: true };
                    }
                }
            }
            return DecodeResult { bytes, read: length, error: false };
        }
        let c = units[index];
        index += 1;
        if c == b'=' {
            // Padding is only legal once a chunk holds 2 or 3 symbols.
            if chunk_len < 2 {
                return DecodeResult { bytes, read, error: true };
            }
            index = skip_ascii_ws(units, index);
            if chunk_len == 2 {
                // A 2-symbol chunk needs a second `=`.
                if index == length {
                    if last_chunk == LastChunk::StopBeforePartial {
                        return DecodeResult { bytes, read, error: false };
                    }
                    return DecodeResult { bytes, read, error: true };
                }
                if units[index] == b'=' {
                    index = skip_ascii_ws(units, index + 1);
                }
            }
            // Any non-whitespace content after the padding is malformed.
            if index < length {
                return DecodeResult { bytes, read, error: true };
            }
            // `strict` requires the chunk's overflow bits to be zero.
            let throw_extra = last_chunk == LastChunk::Strict;
            match decode_chunk(&chunk[..chunk_len], throw_extra) {
                // A 2/3-symbol chunk yields `chunk_len - 1` bytes (1 or 2).
                Some(decoded) => bytes.extend_from_slice(&decoded[..chunk_len - 1]),
                None => return DecodeResult { bytes, read, error: true },
            }
            return DecodeResult { bytes, read: length, error: false };
        }
        let Some(v) = b64_value(c, alphabet) else {
            // A non-base64, non-whitespace symbol is malformed.
            return DecodeResult { bytes, read, error: true };
        };
        // Stop before a chunk whose decoded bytes would overflow `max_len` (a
        // 2-symbol chunk needs 1 free byte, a 3-symbol chunk needs 2).
        let remaining = max_len - bytes.len();
        if (remaining == 1 && chunk_len == 2) || (remaining == 2 && chunk_len == 3) {
            return DecodeResult { bytes, read, error: false };
        }
        chunk[chunk_len] = v;
        chunk_len += 1;
        if chunk_len == 4 {
            // A complete chunk decodes to 3 bytes.
            let decoded = decode_chunk(&chunk, false).unwrap();
            bytes.extend_from_slice(&decoded);
            chunk_len = 0;
            read = index;
            if bytes.len() == max_len {
                return DecodeResult { bytes, read, error: false };
            }
        }
    }
}

/// The spec `DecodeBase64Chunk(chunk, throwOnExtraBits)`: a chunk of 2, 3, or 4
/// base64 *values* decodes to 1, 2, or 3 bytes. When `throw_on_extra` is set, a
/// short chunk whose unused low bits are non-zero is rejected (`None`). A
/// 4-value chunk never has extra bits.
fn decode_chunk(chunk: &[u8], throw_on_extra: bool) -> Option<[u8; 3]> {
    match chunk.len() {
        2 => {
            // 12 bits → 1 byte; the low 4 bits of the second symbol are unused.
            if throw_on_extra && (chunk[1] & 0b1111) != 0 {
                return None;
            }
            let b0 = (chunk[0] << 2) | (chunk[1] >> 4);
            Some([b0, 0, 0])
        }
        3 => {
            // 18 bits → 2 bytes; the low 2 bits of the third symbol are unused.
            if throw_on_extra && (chunk[2] & 0b11) != 0 {
                return None;
            }
            let b0 = (chunk[0] << 2) | (chunk[1] >> 4);
            let b1 = (chunk[1] << 4) | (chunk[2] >> 2);
            Some([b0, b1, 0])
        }
        4 => {
            let b0 = (chunk[0] << 2) | (chunk[1] >> 4);
            let b1 = (chunk[1] << 4) | (chunk[2] >> 2);
            let b2 = (chunk[2] << 6) | chunk[3];
            Some([b0, b1, b2])
        }
        _ => None,
    }
}

/// Encodes `bytes` to a base64 string with `alphabet`, dropping the trailing `=`
/// padding when `omit_padding` is set.
pub(crate) fn to_base64(bytes: &[u8], alphabet: B64Alphabet, omit_padding: bool) -> Vec<u8> {
    let tbl = match alphabet {
        B64Alphabet::Standard => B64_STD,
        B64Alphabet::Url => B64_URL,
    };
    let mut out = Vec::with_capacity(bytes.len().div_ceil(3) * 4);
    let mut chunks = bytes.chunks_exact(3);
    for c in &mut chunks {
        let n = (u32::from(c[0]) << 16) | (u32::from(c[1]) << 8) | u32::from(c[2]);
        out.push(tbl[(n >> 18) as usize & 63]);
        out.push(tbl[(n >> 12) as usize & 63]);
        out.push(tbl[(n >> 6) as usize & 63]);
        out.push(tbl[n as usize & 63]);
    }
    let rem = chunks.remainder();
    match rem.len() {
        1 => {
            let n = u32::from(rem[0]) << 16;
            out.push(tbl[(n >> 18) as usize & 63]);
            out.push(tbl[(n >> 12) as usize & 63]);
            if !omit_padding {
                out.push(b'=');
                out.push(b'=');
            }
        }
        2 => {
            let n = (u32::from(rem[0]) << 16) | (u32::from(rem[1]) << 8);
            out.push(tbl[(n >> 18) as usize & 63]);
            out.push(tbl[(n >> 12) as usize & 63]);
            out.push(tbl[(n >> 6) as usize & 63]);
            if !omit_padding {
                out.push(b'=');
            }
        }
        _ => {}
    }
    out
}

/// The hex value (0..=15) of an ASCII hex digit (either case), or `None`.
fn hex_value(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

/// The spec `FromHex(string, maxLength)`: decodes hex `units` into bytes, two
/// units per byte, stopping at `max_len` bytes. An odd length is malformed (no
/// bytes written); a non-hex unit is malformed at that pair (bytes up to it are
/// kept, for `setFromHex`'s write-up-to-error behavior).
pub(crate) fn from_hex(units: &[u8], max_len: usize) -> DecodeResult {
    let mut bytes = Vec::new();
    if !units.len().is_multiple_of(2) {
        // Odd length: nothing is written.
        return DecodeResult { bytes, read: 0, error: true };
    }
    let mut i = 0usize;
    while i + 1 < units.len() {
        if bytes.len() == max_len {
            return DecodeResult { bytes, read: i, error: false };
        }
        let (Some(hi), Some(lo)) = (hex_value(units[i]), hex_value(units[i + 1])) else {
            return DecodeResult { bytes, read: i, error: true };
        };
        bytes.push((hi << 4) | lo);
        i += 2;
    }
    DecodeResult { bytes, read: i, error: false }
}

/// Lowercase hex encoding of `bytes`, two chars per byte.
pub(crate) fn to_hex(bytes: &[u8]) -> Vec<u8> {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = Vec::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(HEX[(b >> 4) as usize]);
        out.push(HEX[(b & 0xF) as usize]);
    }
    out
}

impl<'a> Interp<'a> {
    /// The spec `ValidateUint8Array(this)`: returns the receiver handle if it is a
    /// `Uint8Array` (element kind 1), else a `TypeError`. Per spec this runs
    /// *before* the options getters and does **not** check for detachment — that
    /// check is deferred to the byte read (so option-getter side effects, which
    /// may detach the buffer, are still observed). `op` names the method for the
    /// error message.
    fn validate_uint8array(&mut self, op: &str) -> Result<Handle, ExecError> {
        let this = self.this_val;
        match this.as_handle().map(Handle::from_raw) {
            Some(h) if self.realm.typed_kind(h) == Some(1) => Ok(h),
            _ => Err(self.type_error(&alloc::format!(
                "Uint8Array.prototype.{op} called on a non-Uint8Array object"
            ))),
        }
    }

    /// The spec `GetUint8ArrayBytes`/byte-access witness for an already-validated
    /// `Uint8Array` `h`: rejects a detached/out-of-bounds buffer (a `TypeError`),
    /// then returns its backing-bytes handle, byte offset, and live length.
    fn uint8array_bytes_view(
        &mut self,
        h: Handle,
        op: &str,
    ) -> Result<(Handle, usize, usize), ExecError> {
        if self.typed_array_detached(h) {
            return Err(self.type_error(&alloc::format!(
                "Uint8Array.prototype.{op} called on a detached ArrayBuffer"
            )));
        }
        let buffer = self.realm.typed_buffer(h).unwrap();
        let offset = self.realm.typed_byte_offset(h).unwrap();
        let len = self.realm.typed_len(h).unwrap();
        Ok((buffer, offset, len))
    }

    /// A snapshot copy of the `len` bytes of the `Uint8Array` view backed by
    /// `buffer` at `offset` (the codec reads from this copy so options-getter
    /// side effects on the live array are observed before, not during, encoding).
    fn uint8array_snapshot(&self, buffer: Handle, offset: usize, len: usize) -> Vec<u8> {
        self.realm
            .bytes_at(buffer)
            .map(|b| b.get(offset..offset + len).unwrap_or(&[]).to_vec())
            .unwrap_or_default()
    }

    /// Reads the `alphabet` option from `options` (Get then validate): `undefined`
    /// → `Standard`; the strings `"base64"`/`"base64url"` map to the alphabet; a
    /// non-undefined value of any other type or string is a `TypeError`. A
    /// non-object non-undefined `options` is itself a `TypeError`.
    fn read_alphabet_option(&mut self, options: NanBox) -> Result<B64Alphabet, ExecError> {
        if matches!(options.unpack(), Unpacked::Undefined) {
            return Ok(B64Alphabet::Standard);
        }
        let Some(oh) = options.as_handle().map(Handle::from_raw).filter(|_| self.is_object_value(options)) else {
            return Err(self.type_error("options is not an object"));
        };
        let v = self.read_member(oh, "alphabet")?;
        if matches!(v.unpack(), Unpacked::Undefined) {
            return Ok(B64Alphabet::Standard);
        }
        // The value must be a primitive string (no coercion): a String *wrapper*
        // object or any non-string is a TypeError.
        match self.opt_string(v).as_deref() {
            Some("base64") => Ok(B64Alphabet::Standard),
            Some("base64url") => Ok(B64Alphabet::Url),
            _ => Err(self.type_error("alphabet must be \"base64\" or \"base64url\"")),
        }
    }

    /// Reads the `lastChunkHandling` option (Get then validate): `undefined` →
    /// `Loose`; `"loose"`/`"strict"`/`"stop-before-partial"` map accordingly; any
    /// other value (including a non-string) is a `TypeError`.
    fn read_last_chunk_option(&mut self, options: NanBox) -> Result<LastChunk, ExecError> {
        if matches!(options.unpack(), Unpacked::Undefined) {
            return Ok(LastChunk::Loose);
        }
        let Some(oh) = options.as_handle().map(Handle::from_raw).filter(|_| self.is_object_value(options)) else {
            return Err(self.type_error("options is not an object"));
        };
        let v = self.read_member(oh, "lastChunkHandling")?;
        if matches!(v.unpack(), Unpacked::Undefined) {
            return Ok(LastChunk::Loose);
        }
        match self.opt_string(v).as_deref() {
            Some("loose") => Ok(LastChunk::Loose),
            Some("strict") => Ok(LastChunk::Strict),
            Some("stop-before-partial") => Ok(LastChunk::StopBeforePartial),
            _ => Err(self.type_error(
                "lastChunkHandling must be \"loose\", \"strict\", or \"stop-before-partial\"",
            )),
        }
    }

    /// The WTF-8 bytes of `v` only when it is a *primitive string* value (a String
    /// wrapper object or any non-string yields `None`) — used for option values
    /// that the spec validates without ToString coercion. Returned lossily as a
    /// `String` for the ASCII keyword comparisons above.
    fn opt_string(&self, v: NanBox) -> Option<String> {
        v.as_handle()
            .map(Handle::from_raw)
            .and_then(|h| self.realm.string_value(h))
    }

    /// The base64 input argument as WTF-8 bytes only when it is a *primitive
    /// string* (per spec — no ToString coercion); any other value yields `None`,
    /// which the caller surfaces as a `TypeError`.
    fn require_string_arg(&self, v: NanBox) -> Option<Vec<u8>> {
        v.as_handle()
            .map(Handle::from_raw)
            .and_then(|h| self.realm.string_bytes(h))
    }

    /// Builds a fresh `Uint8Array` (kind 1) of `bytes` whose `[[Prototype]]` is
    /// `%Uint8Array.prototype%` — used by the `fromBase64`/`fromHex` statics,
    /// which ignore their receiver (never invoking a subclass constructor).
    fn new_uint8array_from(&mut self, bytes: Vec<u8>) -> NanBox {
        let buf = self.make_array_buffer_from_bytes(&bytes);
        let bytes_h = self.array_buffer_bytes(buf).unwrap();
        let view = self.realm.new_typed_array(bytes_h, buf, 0, bytes.len(), 1);
        if let Some(proto) = self.intrinsic_proto("Uint8Array") {
            self.realm.set_native_proto(view, proto);
        }
        NanBox::handle(view.to_raw())
    }

    /// A `SyntaxError` with `message` — the malformed-input error for the codecs
    /// (also used by the module loader/linker for parse/link-phase errors).
    pub(crate) fn syntax_error(&mut self, message: &str) -> ExecError {
        let m = self.new_str(message);
        ExecError::Throw(self.make_error(N_SYNTAX_ERROR, Some(m)))
    }

    /// Builds the ordinary `{ read, written }` result object (own enumerable data
    /// properties) returned by `setFromBase64`/`setFromHex`.
    fn read_written_record(&mut self, read: usize, written: usize) -> NanBox {
        let obj = self.realm.new_object();
        self.realm.set_property(obj, "read", NanBox::number(read as f64));
        self.realm
            .set_property(obj, "written", NanBox::number(written as f64));
        NanBox::handle(obj.to_raw())
    }

    /// `Uint8Array.prototype.toBase64(options?)` (N_UINT8_TO_BASE64).
    pub(crate) fn uint8_to_base64(&mut self, options: NanBox) -> Result<NanBox, ExecError> {
        // Validate the receiver (no detach check); read `alphabet` (its getter may
        // run arbitrary code / detach the buffer) then `omitPadding`; only *then*
        // take the byte witness (which rejects a detached buffer) and encode.
        let h = self.validate_uint8array("toBase64")?;
        let alphabet = self.read_alphabet_option(options)?;
        let omit_padding = if matches!(options.unpack(), Unpacked::Undefined) {
            false
        } else {
            let oh = options.as_handle().map(Handle::from_raw).unwrap();
            let v = self.read_member(oh, "omitPadding")?;
            self.realm.truthy(v)
        };
        let (buffer, offset, len) = self.uint8array_bytes_view(h, "toBase64")?;
        let snapshot = self.uint8array_snapshot(buffer, offset, len);
        let out = to_base64(&snapshot, alphabet, omit_padding);
        Ok(self.new_str_bytes(out))
    }

    /// `Uint8Array.prototype.toHex()` (N_UINT8_TO_HEX). No options.
    pub(crate) fn uint8_to_hex(&mut self) -> Result<NanBox, ExecError> {
        let h = self.validate_uint8array("toHex")?;
        let (buffer, offset, len) = self.uint8array_bytes_view(h, "toHex")?;
        let snapshot = self.uint8array_snapshot(buffer, offset, len);
        let out = to_hex(&snapshot);
        Ok(self.new_str_bytes(out))
    }

    /// `Uint8Array.prototype.setFromBase64(string, options?)` (N_UINT8_SET_FROM_BASE64).
    pub(crate) fn uint8_set_from_base64(
        &mut self,
        string: NanBox,
        options: NanBox,
    ) -> Result<NanBox, ExecError> {
        // Validate the receiver and the string argument, read `alphabet` then
        // `lastChunkHandling` (their getters may run code / detach), then take the
        // byte witness (rejecting a detached buffer) before decoding into it.
        let h = self.validate_uint8array("setFromBase64")?;
        let Some(units) = self.require_string_arg(string) else {
            return Err(self.type_error("setFromBase64 requires a string argument"));
        };
        let alphabet = self.read_alphabet_option(options)?;
        let last_chunk = self.read_last_chunk_option(options)?;
        let (buffer, offset, len) = self.uint8array_bytes_view(h, "setFromBase64")?;
        let r = from_base64(&units, alphabet, last_chunk, len);
        // Write the decoded bytes into the live view (capped at its length).
        let written = r.bytes.len();
        if let Some(store) = self.realm.bytes_at_mut(buffer) {
            let dst = &mut store[offset..offset + len];
            dst[..written].copy_from_slice(&r.bytes);
        }
        if r.error {
            return Err(self.syntax_error("malformed base64 input"));
        }
        Ok(self.read_written_record(r.read, written))
    }

    /// `Uint8Array.prototype.setFromHex(string)` (N_UINT8_SET_FROM_HEX).
    pub(crate) fn uint8_set_from_hex(&mut self, string: NanBox) -> Result<NanBox, ExecError> {
        // No options/getters here, so the byte witness (detach check) can be taken
        // immediately after validating the receiver and the string argument.
        let h = self.validate_uint8array("setFromHex")?;
        let Some(units) = self.require_string_arg(string) else {
            return Err(self.type_error("setFromHex requires a string argument"));
        };
        let (buffer, offset, len) = self.uint8array_bytes_view(h, "setFromHex")?;
        let r = from_hex(&units, len);
        let written = r.bytes.len();
        if let Some(store) = self.realm.bytes_at_mut(buffer) {
            let dst = &mut store[offset..offset + len];
            dst[..written].copy_from_slice(&r.bytes);
        }
        if r.error {
            return Err(self.syntax_error("malformed hex input"));
        }
        Ok(self.read_written_record(r.read, written))
    }

    /// `Uint8Array.fromBase64(string, options?)` (N_UINT8_FROM_BASE64). Static —
    /// ignores its receiver; the result is always a `%Uint8Array.prototype%` view.
    pub(crate) fn uint8_from_base64(
        &mut self,
        string: NanBox,
        options: NanBox,
    ) -> Result<NanBox, ExecError> {
        let Some(units) = self.require_string_arg(string) else {
            return Err(self.type_error("fromBase64 requires a string argument"));
        };
        let alphabet = self.read_alphabet_option(options)?;
        let last_chunk = self.read_last_chunk_option(options)?;
        // No maxLength cap: decode the whole input (usize::MAX stands in for +∞).
        let r = from_base64(&units, alphabet, last_chunk, usize::MAX);
        if r.error {
            return Err(self.syntax_error("malformed base64 input"));
        }
        Ok(self.new_uint8array_from(r.bytes))
    }

    /// `Uint8Array.fromHex(string)` (N_UINT8_FROM_HEX). Static — ignores its
    /// receiver; the result is always a `%Uint8Array.prototype%` view.
    pub(crate) fn uint8_from_hex(&mut self, string: NanBox) -> Result<NanBox, ExecError> {
        let Some(units) = self.require_string_arg(string) else {
            return Err(self.type_error("fromHex requires a string argument"));
        };
        let r = from_hex(&units, usize::MAX);
        if r.error {
            return Err(self.syntax_error("malformed hex input"));
        }
        Ok(self.new_uint8array_from(r.bytes))
    }
}
