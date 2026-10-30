//! WTF-8 — the string-storage codec (`ROADMAP.md` §3, item 4; milestone B1).
//!
//! JavaScript strings are sequences of UTF-16 *code units*, and a DOMString may
//! contain **lone surrogates**: `"\uD800".length === 1`, its
//! `charCodeAt(0) === 0xD800`. Strict UTF-8 (Rust `str`) cannot represent those
//! surrogate code points, so `String::from_utf16_lossy` collapses them to the
//! replacement character U+FFFD — data loss that breaks round-tripping.
//!
//! [WTF-8](https://simonsapin.github.io/wtf-8/) ("Wobbly Transformation Format
//! − 8 bit") is UTF-8 generalized to also encode the surrogate code points
//! U+D800..=U+DFFF, each as the natural 3-byte form `ED A0..BF 80..BF`. A WTF-8
//! string therefore round-trips every UTF-16 sequence, lone surrogates and all,
//! while a string with *no* surrogates is byte-identical to its UTF-8 — which is
//! the overwhelmingly common case, so [`as_str`](crate::wtf8::as_str) hands
//! those back for free.
//!
//! This module is a pure, safe `alloc`-only codec over `&[u8]`: no `unsafe`, no
//! `std`. (Storage moves to WTF-8 bytes in `rope`/`atom`; the surrogate-correct
//! string *operations* in `nbexec` are a later milestone — this is foundation.)

use alloc::string::String;
use alloc::vec::Vec;

/// The first code point that needs a 2-byte UTF-8 / WTF-8 encoding.
const TWO_BYTE_MIN: u32 = 0x80;
/// The first code point that needs a 3-byte encoding.
const THREE_BYTE_MIN: u32 = 0x800;
/// The first code point that needs a 4-byte encoding.
const FOUR_BYTE_MIN: u32 = 0x1_0000;
/// The highest valid Unicode code point.
const CODE_POINT_MAX: u32 = 0x10_FFFF;

/// Inclusive range of the UTF-16 surrogate code points.
const SURROGATE_MIN: u32 = 0xD800;
const SURROGATE_MAX: u32 = 0xDFFF;
const HIGH_SURROGATE_MAX: u32 = 0xDBFF;
const LOW_SURROGATE_MIN: u32 = 0xDC00;

/// Whether `cp` is a UTF-16 surrogate code point (U+D800..=U+DFFF).
#[must_use]
#[inline]
pub const fn is_surrogate(cp: u32) -> bool {
    cp >= SURROGATE_MIN && cp <= SURROGATE_MAX
}

/// Decodes the 3-byte WTF-8 surrogate sequence at `b` (`ED A0..BF 80..BF`) to its
/// code point, or `None` if `b` is not exactly such a sequence. Used to detect a
/// surrogate at a concatenation boundary.
#[inline]
fn decode_surrogate3(b: &[u8]) -> Option<u32> {
    if b.len() != 3 || b[0] != 0xED || b[1] & 0xC0 != 0x80 || b[2] & 0xC0 != 0x80 {
        return None;
    }
    let cp =
        (u32::from(b[0] & 0x0F) << 12) | (u32::from(b[1] & 0x3F) << 6) | u32::from(b[2] & 0x3F);
    is_surrogate(cp).then_some(cp)
}

/// If `bytes` ends with a lone **high** surrogate code point (`U+D800..=U+DBFF`,
/// its 3-byte WTF-8 form), returns it. Used to pair a boundary surrogate with a
/// following low surrogate on concatenation.
#[must_use]
pub fn trailing_high_surrogate(bytes: &[u8]) -> Option<u16> {
    let n = bytes.len();
    if n < 3 {
        return None;
    }
    let cp = decode_surrogate3(&bytes[n - 3..])?;
    (SURROGATE_MIN..=HIGH_SURROGATE_MAX)
        .contains(&cp)
        .then_some(cp as u16)
}

/// If `bytes` begins with a lone **low** surrogate code point (`U+DC00..=U+DFFF`,
/// its 3-byte WTF-8 form), returns it.
#[must_use]
pub fn leading_low_surrogate(bytes: &[u8]) -> Option<u16> {
    if bytes.len() < 3 {
        return None;
    }
    let cp = decode_surrogate3(&bytes[..3])?;
    (LOW_SURROGATE_MIN..=SURROGATE_MAX)
        .contains(&cp)
        .then_some(cp as u16)
}

/// WTF-8-encodes a single code point — **including** a lone surrogate — and
/// appends it to `out`.
///
/// A scalar value is encoded exactly as UTF-8. A surrogate code point
/// (U+D800..=U+DFFF) is encoded as its natural 3-byte form (`ED A0..BF 80..BF`),
/// which strict UTF-8 forbids — this is the one place WTF-8 diverges. A value
/// past U+10FFFF is clamped to U+FFFD (callers that must reject it do so before
/// calling).
pub fn encode_code_point(cp: u32, out: &mut Vec<u8>) {
    let cp = if cp > CODE_POINT_MAX { 0xFFFD } else { cp };
    if cp < TWO_BYTE_MIN {
        out.push(cp as u8);
    } else if cp < THREE_BYTE_MIN {
        out.push((0xC0 | (cp >> 6)) as u8);
        out.push((0x80 | (cp & 0x3F)) as u8);
    } else if cp < FOUR_BYTE_MIN {
        // Covers the BMP, surrogate code points included (no exclusion here —
        // that exclusion is exactly what separates WTF-8 from strict UTF-8).
        out.push((0xE0 | (cp >> 12)) as u8);
        out.push((0x80 | ((cp >> 6) & 0x3F)) as u8);
        out.push((0x80 | (cp & 0x3F)) as u8);
    } else {
        out.push((0xF0 | (cp >> 18)) as u8);
        out.push((0x80 | ((cp >> 12) & 0x3F)) as u8);
        out.push((0x80 | ((cp >> 6) & 0x3F)) as u8);
        out.push((0x80 | (cp & 0x3F)) as u8);
    }
}

/// WTF-8-encodes a single UTF-16 code unit and appends it to `out`. A BMP scalar
/// becomes its normal 1–3-byte UTF-8; a lone surrogate becomes its WTF-8 3-byte
/// form. (Pairing of adjacent surrogates is [`from_utf16`]'s job, not this one's
/// — a unit in isolation has no partner.)
pub fn encode_utf16_unit(u: u16, out: &mut Vec<u8>) {
    encode_code_point(u32::from(u), out);
}

/// Encodes a slice of UTF-16 code units to WTF-8.
///
/// Adjacent high+low surrogate pairs combine into the astral scalar they denote
/// (encoded as 4 bytes), exactly like UTF-16 decoding. A surrogate with no valid
/// partner is preserved as a surrogate *code point* (3 bytes) rather than
/// replaced — the key difference from [`String::from_utf16_lossy`], which would
/// emit U+FFFD.
#[must_use]
pub fn from_utf16(units: &[u16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(units.len());
    push_utf16(&mut out, units);
    out
}

/// Appends the WTF-8 encoding of `units` to `out` as its own string (no
/// joining with what `out` ends with — see [`append_utf16`]).
fn push_utf16(out: &mut Vec<u8>, units: &[u16]) {
    // An ASCII prefix (often the whole input) is one byte per unit.
    let ascii = units.iter().position(|&u| u >= 0x80).unwrap_or(units.len());
    out.extend(units[..ascii].iter().map(|&u| u as u8));
    let mut i = ascii;
    while i < units.len() {
        let u = u32::from(units[i]);
        if (SURROGATE_MIN..=HIGH_SURROGATE_MAX).contains(&u) && i + 1 < units.len() {
            let lo = u32::from(units[i + 1]);
            if (LOW_SURROGATE_MIN..=SURROGATE_MAX).contains(&lo) {
                // A well-formed pair → the astral scalar value.
                let cp = 0x1_0000 + ((u - SURROGATE_MIN) << 10) + (lo - LOW_SURROGATE_MIN);
                encode_code_point(cp, out);
                i += 2;
                continue;
            }
        }
        // A scalar, or a lone surrogate kept as a surrogate code point.
        encode_code_point(u, out);
        i += 1;
    }
}

/// [`append`]`(out, &from_utf16(units))` without the intermediate buffer:
/// the string concatenation of `out` and the code units `units` (a lone high
/// surrogate ending `out` pairs with a lone low one starting `units`).
pub fn append_utf16(out: &mut Vec<u8>, units: &[u16]) {
    if units
        .first()
        .is_some_and(|&u| (LOW_SURROGATE_MIN..=SURROGATE_MAX).contains(&u32::from(u)))
        && trailing_high_surrogate(out).is_some()
    {
        append(out, &from_utf16(units));
        return;
    }
    push_utf16(out, units);
}

/// Decodes the next code point from `bytes` starting at `i`, returning the code
/// point (a scalar value, or a lone surrogate code point) and its byte length.
///
/// Assumes `bytes` is well-formed WTF-8 (as produced by this module and stored
/// in ropes/atoms); a malformed lead byte is reported as a 1-byte U+FFFD so the
/// iterators always make forward progress rather than looping or panicking.
pub fn decode_code_point(bytes: &[u8], i: usize) -> (u32, usize) {
    let b0 = bytes[i];
    if b0 < 0x80 {
        return (u32::from(b0), 1);
    }
    let cont = |k: usize| -> u32 { u32::from(bytes[i + k] & 0x3F) };
    if b0 < 0xE0 {
        if i + 1 < bytes.len() {
            return ((u32::from(b0 & 0x1F) << 6) | cont(1), 2);
        }
    } else if b0 < 0xF0 {
        if i + 2 < bytes.len() {
            return ((u32::from(b0 & 0x0F) << 12) | (cont(1) << 6) | cont(2), 3);
        }
    } else if i + 3 < bytes.len() {
        return (
            (u32::from(b0 & 0x07) << 18) | (cont(1) << 12) | (cont(2) << 6) | cont(3),
            4,
        );
    }
    // Truncated / malformed: yield a replacement and advance one byte.
    (0xFFFD, 1)
}

/// An iterator over the **code points** of a WTF-8 slice: scalar values plus any
/// stored lone surrogate code points (yielded as themselves). This is the
/// `for…of` / `codePointAt` view — astral characters are single code points.
/// Appends `add` to `out` in **canonical** WTF-8: if `out` ends with a lone high
/// surrogate and `add` opens with a lone low surrogate, the pair is recombined
/// into the 4-byte astral form rather than left as two unpaired encodings.
///
/// Two byte sequences that encode the same UTF-16 units must be byte-equal, or
/// they compare unequal, iterate as a different number of code points, and stop
/// being well-formed. Anywhere pieces of a string are concatenated — a `replace`
/// result assembled from per-match fragments, for instance — has to go through
/// this rather than a plain `extend_from_slice`. `Rope::concat` applies the same
/// rule for the string type.
pub fn append(out: &mut Vec<u8>, add: &[u8]) {
    if let Some(hi) = trailing_high_surrogate(out)
        && let Some(lo) = leading_low_surrogate(add)
    {
        out.truncate(out.len() - 3); // drop the lone high surrogate
        let cp = 0x1_0000 + ((u32::from(hi) - 0xD800) << 10) + (u32::from(lo) - 0xDC00);
        encode_code_point(cp, out);
        out.extend_from_slice(&add[3..]); // drop the lone low surrogate
        return;
    }
    out.extend_from_slice(add);
}

/// An iterator over the **code points** of a WTF-8 slice: scalar values plus any
/// stored lone surrogate code points (yielded as themselves). This is the
/// `for…of` / `codePointAt` view — astral characters are single code points.
pub fn code_points(bytes: &[u8]) -> CodePoints<'_> {
    CodePoints { bytes, i: 0 }
}

/// Iterator returned by [`code_points`].
#[derive(Clone)]
pub struct CodePoints<'a> {
    bytes: &'a [u8],
    i: usize,
}

impl Iterator for CodePoints<'_> {
    type Item = u32;
    fn next(&mut self) -> Option<u32> {
        if self.i >= self.bytes.len() {
            return None;
        }
        let (cp, n) = decode_code_point(self.bytes, self.i);
        self.i += n;
        Some(cp)
    }
}

/// An iterator over the **UTF-16 code units** of a WTF-8 slice — the JS string
/// view. A BMP code point (scalar or lone surrogate) yields one unit; an astral
/// scalar yields its surrogate pair (two units).
pub fn utf16_units(bytes: &[u8]) -> Utf16Units<'_> {
    Utf16Units {
        inner: code_points(bytes),
        pending_low: None,
    }
}

/// Iterator returned by [`utf16_units`].
#[derive(Clone)]
pub struct Utf16Units<'a> {
    inner: CodePoints<'a>,
    /// The low surrogate buffered after emitting the high half of an astral pair.
    pending_low: Option<u16>,
}

impl Iterator for Utf16Units<'_> {
    type Item = u16;
    fn next(&mut self) -> Option<u16> {
        if let Some(lo) = self.pending_low.take() {
            return Some(lo);
        }
        let cp = self.inner.next()?;
        if cp >= FOUR_BYTE_MIN {
            // Astral scalar → high + low surrogate; buffer the low half.
            let v = cp - FOUR_BYTE_MIN;
            let high = SURROGATE_MIN + (v >> 10);
            let low = LOW_SURROGATE_MIN + (v & 0x3FF);
            self.pending_low = Some(low as u16);
            Some(high as u16)
        } else {
            // BMP scalar or a lone surrogate code point: one unit, as-is.
            Some(cp as u16)
        }
    }
}

/// Orders two WTF-8 strings by **UTF-16 code unit**, the JS string order (the
/// default `sort` comparator, `<`). Byte order agrees with it except where an
/// astral character (a `0xF0..` lead byte, whose units are surrogates) meets a
/// code point at or above U+D800 (`0xED..` lead), so the units are compared only
/// when the first differing bytes are in that range.
#[must_use]
pub fn cmp_utf16(a: &[u8], b: &[u8]) -> core::cmp::Ordering {
    match a.iter().zip(b).position(|(x, y)| x != y) {
        Some(i) if a[i] >= 0xED || b[i] >= 0xED => utf16_units(a).cmp(utf16_units(b)),
        Some(i) => a[i].cmp(&b[i]),
        None => a.len().cmp(&b.len()),
    }
}

/// The number of UTF-16 code units in `bytes` — JavaScript's `String.length`.
#[must_use]
pub fn utf16_len(bytes: &[u8]) -> usize {
    // Astral code points (4-byte forms) count as two units; everything else as
    // one. In well-formed WTF-8 every code point has exactly one lead byte
    // (anything but `10xxxxxx`) and a 4-byte form's lead is `>= 0xF0`, so the
    // count is a per-byte sum with no decoding — taken eight bytes at a time:
    // in each byte, bit 7 clear means ASCII, `bit7 & !bit6` a continuation
    // byte, and bits 7..4 all set a 4-byte lead.
    const HI: u64 = 0x8080_8080_8080_8080;
    let (words, rest) = bytes.as_chunks::<8>();
    let mut n = 0usize;
    for w in words {
        let x = u64::from_le_bytes(*w);
        if x & HI == 0 {
            n += 8;
            continue;
        }
        let cont = x & !(x << 1) & HI;
        let four = x & (x << 1) & (x << 2) & (x << 3) & HI;
        n += 8 - cont.count_ones() as usize + four.count_ones() as usize;
    }
    n + rest
        .iter()
        .map(|&b| usize::from(b & 0xC0 != 0x80) + usize::from(b >= 0xF0))
        .sum::<usize>()
}

/// The UTF-16 code unit at index `i` (0-based) — JavaScript's `charCodeAt`.
/// Returns `None` when `i` is out of range.
#[must_use]
pub fn utf16_index(bytes: &[u8], i: usize) -> Option<u16> {
    utf16_units(bytes).nth(i)
}

/// The UTF-16 unit at index `i`, scanning forward from a known character
/// boundary — byte offset `byte`, which starts unit `unit` (`unit <= i`; pass
/// `(0, 0)` to scan from the start). Also returns the boundary of the
/// character holding unit `i`, so a caller walking a string in order can resume
/// there and stay linear overall. `None` past the end.
#[must_use]
pub fn utf16_index_from(
    bytes: &[u8],
    mut byte: usize,
    mut unit: usize,
    i: usize,
) -> Option<(u16, usize, usize)> {
    while byte < bytes.len() {
        let (cp, n) = decode_code_point(bytes, byte);
        if cp >= FOUR_BYTE_MIN {
            if i < unit + 2 {
                let v = cp - FOUR_BYTE_MIN;
                let u = if i == unit {
                    0xD800 | (v >> 10)
                } else {
                    0xDC00 | (v & 0x3FF)
                };
                return Some((u as u16, byte, unit));
            }
            unit += 2;
        } else {
            if i == unit {
                return Some((cp as u16, byte, unit));
            }
            unit += 1;
        }
        byte += n;
    }
    None
}

/// Returns the byte offset in `bytes` for UTF-16 unit `unit`.
///
/// `unit` may equal [`utf16_len`] (the past-the-end position), which maps to
/// `bytes.len()`. A `unit` that falls in the *middle* of a surrogate pair (the
/// low half of an astral character) cannot be a byte boundary — splitting the
/// 4-byte sequence is impossible — so it is rounded to a whole-character edge:
/// `round_up == false` (a slice *start*) rounds down to the astral char's start;
/// `round_up == true` (a slice *end*) rounds up to its end. This keeps astral
/// characters whole.
fn byte_offset_of_unit(bytes: &[u8], unit: usize, round_up: bool) -> usize {
    if unit == 0 {
        return 0;
    }
    let mut units_seen = 0;
    let mut i = 0;
    while i < bytes.len() {
        if units_seen >= unit {
            return i;
        }
        let (cp, n) = decode_code_point(bytes, i);
        let w = if cp >= FOUR_BYTE_MIN { 2 } else { 1 };
        // The requested unit lands inside this (astral) character: round to a
        // whole-character edge so the surrogate pair is never split.
        if units_seen + w > unit {
            return if round_up { i + n } else { i };
        }
        units_seen += w;
        i += n;
    }
    bytes.len()
}

/// Extracts the WTF-8 bytes for the UTF-16 unit range `[start_unit, end_unit)`.
///
/// Indices are clamped to `[0, utf16_len]`, and `start > end` yields an empty
/// slice (mirroring JS `slice`-style clamping). A boundary that falls **inside**
/// an astral character's surrogate pair splits it, exactly like JavaScript: e.g.
/// `"💩".slice(0, 1)` yields the lone leading surrogate. The astral character's
/// halves are re-encoded as lone surrogate code points (3 bytes each); non-astral
/// characters are copied byte-for-byte.
/// [`slice_utf16`], with the caller's knowledge of whether `bytes` is all-ASCII.
///
/// When it is, a UTF-16 unit index *is* a byte index, so the slice is a plain
/// sub-slice — no length scan and no walk to find each boundary. Without this,
/// `slice`/`substring`/`substr` were O(n) per call however short the result, so
/// slicing in a loop was quadratic. `ascii` must be accurate; pass `false` when
/// unknown (`Rope::is_ascii` computes it in O(1) from memoized lengths).
#[must_use]
pub fn slice_utf16_ascii(bytes: &[u8], start_unit: usize, end_unit: usize, ascii: bool) -> Vec<u8> {
    if !ascii {
        return slice_utf16(bytes, start_unit, end_unit);
    }
    let s = start_unit.min(bytes.len());
    let e = end_unit.min(bytes.len());
    if s >= e {
        return Vec::new();
    }
    bytes[s..e].to_vec()
}

/// Extracts the WTF-8 bytes for the UTF-16 unit range `[start_unit, end_unit)`.
///
/// Indices are clamped to `[0, utf16_len]`, and `start > end` yields an empty
/// slice (mirroring JS `slice`-style clamping). A boundary that falls **inside**
/// an astral character's surrogate pair splits it, exactly like JavaScript.
/// Prefer [`slice_utf16_ascii`] when the caller knows the string is ASCII.
#[must_use]
pub fn slice_utf16(bytes: &[u8], start_unit: usize, end_unit: usize) -> Vec<u8> {
    let len = utf16_len(bytes);
    let s = start_unit.min(len);
    let e = end_unit.min(len);
    if s >= e {
        return Vec::new();
    }
    // Fast path: neither boundary splits an astral pair — a plain byte sub-slice.
    let start_byte = byte_offset_of_unit(bytes, s, false);
    let end_byte = byte_offset_of_unit(bytes, e, true);
    if unit_at_char_boundary(bytes, s) && unit_at_char_boundary(bytes, e) {
        return bytes[start_byte..end_byte].to_vec();
    }
    // Slow path: a boundary lands mid-pair, so re-encode the touched units,
    // splitting the boundary astral character(s) into lone surrogates.
    let mut out = Vec::new();
    let mut units_seen = 0usize;
    let mut i = 0usize;
    while i < bytes.len() && units_seen < e {
        let (cp, n) = decode_code_point(bytes, i);
        if cp >= FOUR_BYTE_MIN {
            let v = cp - 0x1_0000;
            let hi = (0xD800 + (v >> 10)) as u16;
            let lo = (LOW_SURROGATE_MIN + (v & 0x3FF)) as u16;
            if units_seen >= s && units_seen < e {
                encode_utf16_unit(hi, &mut out);
            }
            if units_seen + 1 >= s && units_seen + 1 < e {
                encode_utf16_unit(lo, &mut out);
            }
            units_seen += 2;
        } else {
            if units_seen >= s && units_seen < e {
                out.extend_from_slice(&bytes[i..i + n]);
            }
            units_seen += 1;
        }
        i += n;
    }
    out
}

/// Whether UTF-16 `unit` falls on a whole-character boundary of `bytes` (i.e. it
/// does **not** land on the low half of an astral surrogate pair). The endpoints
/// `0` and `utf16_len` are boundaries.
/// Where the UTF-16 unit range `[start, end)` of a WTF-8 string lies in its
/// bytes (see [`utf16_range`]): the whole characters `bytes[start..end]`,
/// plus a lone low surrogate before them (`lead`) and a lone high surrogate
/// after them (`trail`) for a boundary that falls inside an astral
/// character, between its surrogate halves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnitRange {
    /// The first whole character's byte offset.
    pub start: usize,
    /// The byte offset just past the last whole character.
    pub end: usize,
    /// The low surrogate unit that begins the range, when it starts inside
    /// an astral character.
    pub lead: Option<u16>,
    /// The high surrogate unit that ends the range, when it ends inside an
    /// astral character.
    pub trail: Option<u16>,
}

/// Locates the UTF-16 unit range `[start_unit, end_unit)` of `bytes`
/// (`start_unit < end_unit <= total_units`, the string's UTF-16 length) in
/// one forward pass that starts at `hint`: a `(byte, unit)` position known
/// to be a character boundary (`(0, 0)` when none is known; ignored when
/// past `start_unit`). An `end_unit` equal to `total_units` is the end of
/// the bytes without a scan, so the cost is the distance from the hint to
/// `start_unit`, plus the range's length unless it runs to the end.
#[must_use]
pub fn utf16_range(
    bytes: &[u8],
    hint: (usize, usize),
    start_unit: usize,
    end_unit: usize,
    total_units: usize,
) -> UnitRange {
    let (mut i, mut u) = if hint.1 <= start_unit && hint.0 <= bytes.len() {
        hint
    } else {
        (0, 0)
    };
    let mut lead = None;
    while u < start_unit && i < bytes.len() {
        let (cp, n) = decode_code_point(bytes, i);
        i += n;
        if cp >= FOUR_BYTE_MIN {
            if u + 1 == start_unit {
                lead = Some((LOW_SURROGATE_MIN + ((cp - 0x1_0000) & 0x3FF)) as u16);
            }
            u += 2;
        } else {
            u += 1;
        }
    }
    let start = i;
    if end_unit >= total_units {
        return UnitRange {
            start,
            end: bytes.len(),
            lead,
            trail: None,
        };
    }
    let mut trail = None;
    while u < end_unit && i < bytes.len() {
        let (cp, n) = decode_code_point(bytes, i);
        if cp >= FOUR_BYTE_MIN {
            if u + 1 == end_unit {
                trail = Some((0xD800 + ((cp - 0x1_0000) >> 10)) as u16);
                break;
            }
            u += 2;
        } else {
            u += 1;
        }
        i += n;
    }
    UnitRange {
        start,
        end: i,
        lead,
        trail,
    }
}

fn unit_at_char_boundary(bytes: &[u8], unit: usize) -> bool {
    if unit == 0 {
        return true;
    }
    let mut units_seen = 0usize;
    let mut i = 0usize;
    while i < bytes.len() {
        if units_seen == unit {
            return true;
        }
        let (cp, n) = decode_code_point(bytes, i);
        let w = if cp >= FOUR_BYTE_MIN { 2 } else { 1 };
        if units_seen < unit && unit < units_seen + w {
            return false; // lands on the low half of this astral pair
        }
        units_seen += w;
        i += n;
    }
    true
}

/// Returns `Some(&str)` when `bytes` is valid UTF-8 — i.e. contains no surrogate
/// code points, the overwhelmingly common case — so callers can use the bytes
/// directly without re-decoding. Returns `None` when a lone surrogate is present.
#[must_use]
pub fn as_str(bytes: &[u8]) -> Option<&str> {
    core::str::from_utf8(bytes).ok()
}

/// Whether `bytes` is plain UTF-8 (no surrogate code points). Equivalent to
/// `as_str(bytes).is_some()` but without borrowing.
#[must_use]
pub fn is_utf8(bytes: &[u8]) -> bool {
    core::str::from_utf8(bytes).is_ok()
}

/// Whether the string is **well-formed** UTF-16 (ECMAScript
/// `String.prototype.isWellFormed`): every high surrogate is immediately
/// followed by a low surrogate and there is no lone low surrogate. Unlike
/// [`is_utf8`], this scans the UTF-16 *code-unit* sequence, so a valid surrogate
/// pair that happens to span two WTF-8 leaves (e.g. `"\uD83D" + "\uDCA9"`) is
/// recognized as well-formed.
#[must_use]
pub fn is_well_formed_utf16(bytes: &[u8]) -> bool {
    let mut expect_low = false;
    for u in utf16_units(bytes) {
        let is_high = (0xD800..=0xDBFF).contains(&u);
        let is_low = (0xDC00..=0xDFFF).contains(&u);
        if expect_low {
            if !is_low {
                return false; // a high surrogate not followed by a low one
            }
            expect_low = false;
        } else if is_low {
            return false; // a lone low surrogate
        } else if is_high {
            expect_low = true;
        }
    }
    !expect_low
}

/// Decodes WTF-8 to a real Rust `String`, replacing each lone surrogate with the
/// replacement character U+FFFD. For sinks that require valid UTF-8 (printing,
/// `&str` interop) and accept the lossy mapping.
#[must_use]
pub fn to_string_lossy(bytes: &[u8]) -> String {
    // Fast path: already valid UTF-8 (no surrogates) → no recoding.
    if let Some(s) = as_str(bytes) {
        return String::from(s);
    }
    let mut out = String::with_capacity(bytes.len());
    for cp in code_points(bytes) {
        // `char::from_u32` is `None` exactly for surrogate code points.
        out.push(char::from_u32(cp).unwrap_or('\u{FFFD}'));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn utf16_len_word_scan_matches_per_byte_count() {
        // Every alignment of ASCII, 2-, 3- and 4-byte forms and lone surrogates.
        let mut s = Vec::new();
        for cp in [
            0x41, 0x7F, 0xE9, 0x7FF, 0x20AC, 0xD800, 0xDFFF, 0xFFFF, 0x1F600, 0x10FFFF,
        ] {
            encode_code_point(cp, &mut s);
            for start in 0..s.len().min(9) {
                let b = &s[start..];
                let want: usize = b
                    .iter()
                    .map(|&c| usize::from(c & 0xC0 != 0x80) + usize::from(c >= 0xF0))
                    .sum();
                assert_eq!(utf16_len(b), want);
            }
        }
        assert_eq!(utf16_len("a\u{1F600}bcdefgh\u{E9}".as_bytes()), 11);
    }

    fn enc(cp: u32) -> Vec<u8> {
        let mut v = Vec::new();
        encode_code_point(cp, &mut v);
        v
    }

    #[test]
    fn bmp_scalar_is_plain_utf8() {
        // ASCII, 2-byte, and 3-byte BMP scalars match Rust's UTF-8 exactly.
        assert_eq!(enc(u32::from(b'A')), b"A");
        assert_eq!(enc(0xE9), "é".as_bytes()); // U+00E9
        assert_eq!(enc(0x4E2D), "中".as_bytes()); // U+4E2D
        for s in ["A", "é", "中", "hello world"] {
            assert_eq!(as_str(s.as_bytes()), Some(s));
            assert_eq!(utf16_len(s.as_bytes()), s.encode_utf16().count());
        }
    }

    #[test]
    fn astral_is_one_code_point_two_units() {
        // 😀 U+1F600 = surrogate pair D83D DE00.
        let grin = "😀";
        let bytes = grin.as_bytes().to_vec();
        assert_eq!(enc(0x1_F600), bytes);
        assert_eq!(utf16_len(&bytes), 2);
        let units: Vec<u16> = utf16_units(&bytes).collect();
        assert_eq!(units, vec![0xD83D, 0xDE00]);
        let cps: Vec<u32> = code_points(&bytes).collect();
        assert_eq!(cps, vec![0x1_F600]);
    }

    #[test]
    fn lone_high_surrogate_round_trips() {
        // "\uD800": one unit, charCodeAt(0)==0xD800; must survive storage.
        let bytes = from_utf16(&[0xD800]);
        assert_eq!(bytes, vec![0xED, 0xA0, 0x80]); // WTF-8 3-byte form
        assert_eq!(utf16_len(&bytes), 1);
        assert_eq!(utf16_index(&bytes, 0), Some(0xD800));
        let units: Vec<u16> = utf16_units(&bytes).collect();
        assert_eq!(units, vec![0xD800]);
        // Round-trips back through from_utf16.
        assert_eq!(from_utf16(&units), bytes);
        // Not valid UTF-8 → fast path declines.
        assert_eq!(as_str(&bytes), None);
        // Code points view yields the surrogate code point itself.
        let cps: Vec<u32> = code_points(&bytes).collect();
        assert_eq!(cps, vec![0xD800]);
    }

    #[test]
    fn lone_low_surrogate_round_trips() {
        let bytes = from_utf16(&[0xDC00]);
        assert_eq!(bytes, vec![0xED, 0xB0, 0x80]);
        assert_eq!(utf16_len(&bytes), 1);
        assert_eq!(utf16_index(&bytes, 0), Some(0xDC00));
        assert_eq!(from_utf16(&utf16_units(&bytes).collect::<Vec<_>>()), bytes);
    }

    #[test]
    fn from_utf16_pairs_adjacent_surrogates() {
        // A well-formed pair becomes the astral scalar (4 bytes).
        let bytes = from_utf16(&[0xD83D, 0xDE00]);
        assert_eq!(bytes, "😀".as_bytes());
        // A high surrogate followed by a non-low is kept lone, then the next.
        let mixed = from_utf16(&[0xD800, u32::from(b'x') as u16]);
        assert_eq!(mixed, [&[0xED, 0xA0, 0x80][..], b"x"].concat());
        assert_eq!(utf16_len(&mixed), 2);
        // A low surrogate with no preceding high is kept lone.
        let lone_low_first = from_utf16(&[0xDC00, 0x41]);
        assert_eq!(lone_low_first, [&[0xED, 0xB0, 0x80][..], b"A"].concat());
    }

    #[test]
    fn encode_utf16_unit_matches_code_point() {
        let mut a = Vec::new();
        encode_utf16_unit(0xD800, &mut a);
        assert_eq!(a, enc(0xD800));
        let mut b = Vec::new();
        encode_utf16_unit(0x41, &mut b);
        assert_eq!(b, b"A");
    }

    #[test]
    fn slice_across_astral_char_is_boundary_correct() {
        // "a😀b": units = [a, D83D, DE00, b], length 4.
        let bytes = from_utf16(&[0x61, 0xD83D, 0xDE00, 0x62]);
        assert_eq!(bytes, "a😀b".as_bytes());
        assert_eq!(utf16_len(&bytes), 4);
        // Whole astral char.
        assert_eq!(slice_utf16(&bytes, 1, 3), "😀".as_bytes());
        // Leading "a".
        assert_eq!(slice_utf16(&bytes, 0, 1), b"a");
        // Trailing "b".
        assert_eq!(slice_utf16(&bytes, 3, 4), b"b");
        // A boundary that lands on the low half (unit 2) SPLITS the astral char,
        // exactly like JS `"a😀b".slice(1, 2)` → the lone leading surrogate D83D.
        assert_eq!(slice_utf16(&bytes, 1, 2), enc(0xD83D));
        // [2,3) starts inside the pair → the lone trailing surrogate DE00.
        assert_eq!(slice_utf16(&bytes, 2, 3), enc(0xDE00));
        // Out-of-range / inverted clamps to empty.
        assert_eq!(slice_utf16(&bytes, 3, 1), b"");
        assert_eq!(slice_utf16(&bytes, 10, 20), b"");
        // Full slice == original.
        assert_eq!(slice_utf16(&bytes, 0, 4), bytes);
    }

    #[test]
    fn slice_preserves_lone_surrogates() {
        // "x\uD800y": units [x, D800, y].
        let bytes = from_utf16(&[0x78, 0xD800, 0x79]);
        assert_eq!(utf16_len(&bytes), 3);
        let mid = slice_utf16(&bytes, 1, 2);
        assert_eq!(mid, vec![0xED, 0xA0, 0x80]);
        assert_eq!(utf16_index(&mid, 0), Some(0xD800));
    }

    #[test]
    fn to_string_lossy_replaces_surrogates() {
        let bytes = from_utf16(&[0x61, 0xD800, 0x62]);
        assert_eq!(to_string_lossy(&bytes), "a\u{FFFD}b");
        // Pure UTF-8 is unchanged (fast path).
        assert_eq!(to_string_lossy("héllo".as_bytes()), "héllo");
        assert_eq!(to_string_lossy("😀".as_bytes()), "😀");
    }

    #[test]
    fn utf16_index_out_of_range() {
        let bytes = from_utf16(&[0x61, 0xD83D, 0xDE00]);
        assert_eq!(utf16_index(&bytes, 0), Some(0x61));
        assert_eq!(utf16_index(&bytes, 1), Some(0xD83D));
        assert_eq!(utf16_index(&bytes, 2), Some(0xDE00));
        assert_eq!(utf16_index(&bytes, 3), None);
    }

    #[test]
    fn over_max_code_point_clamps() {
        assert_eq!(enc(0x11_0000), "\u{FFFD}".as_bytes());
    }

    #[test]
    fn is_surrogate_predicate() {
        assert!(is_surrogate(0xD800));
        assert!(is_surrogate(0xDFFF));
        assert!(!is_surrogate(0xD7FF));
        assert!(!is_surrogate(0xE000));
        assert!(!is_surrogate(0x1_F600));
    }

    #[test]
    fn empty_slice() {
        assert_eq!(utf16_len(b""), 0);
        assert_eq!(utf16_units(b"").count(), 0);
        assert_eq!(code_points(b"").count(), 0);
        assert_eq!(slice_utf16(b"", 0, 0), b"");
        assert_eq!(as_str(b""), Some(""));
    }
}
