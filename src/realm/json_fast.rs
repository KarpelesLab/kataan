//! The heap-side steps of `JSON.stringify`'s fast path (see
//! `Interp::json_ser_got`): writing a value that needs no user code straight
//! from its cell, and classifying plain objects and arrays.
//!
//! Nothing here runs user code or allocates a cell, so a caller that holds a
//! value read from the heap across these calls sees the heap unchanged.

use super::Realm;
use crate::cell::Cell;
use crate::heap::Handle;
use crate::nanbox::{NanBox, Unpacked};
use crate::shape::Shape;
use alloc::rc::Rc;
use alloc::vec::Vec;

/// The bytes [`json_quote_wtf8_into`] must look at: the C0 controls, `"`,
/// `\` and `0xED` (the lead byte of a surrogate's WTF-8 encoding, and of
/// some ordinary scalars).
const JSON_QUOTE_SPECIAL: [bool; 256] = {
    let mut t = [false; 256];
    let mut i = 0;
    while i < 0x20 {
        t[i] = true;
        i += 1;
    }
    t[b'"' as usize] = true;
    t[b'\\' as usize] = true;
    t[0xED] = true;
    t
};

/// Quotes and escapes WTF-8 bytes as a JSON string literal onto `out`
/// (`QuoteJSONString`): `"`, `\` and the C0 controls escaped, a **lone
/// surrogate** (well-formed JSON per the spec) as `\uXXXX`, and every other
/// code point as itself. Its WTF-8 bytes are then its UTF-8 bytes, so the
/// runs between the bytes that need a look are copied as they are.
pub(crate) fn json_quote_wtf8_into(out: &mut Vec<u8>, bytes: &[u8]) {
    // A short string with nothing to escape (the usual key or value):
    // checked and copied into a fixed buffer in one pass, then appended as
    // a copy of known size (a few moves, not a call) cut to length.
    if bytes.len() <= SHORT_TEXT - 2 {
        let mut pad = [0u8; SHORT_TEXT];
        pad[0] = b'"';
        let mut clean = true;
        for (d, &b) in pad[1..].iter_mut().zip(bytes) {
            clean &= !JSON_QUOTE_SPECIAL[usize::from(b)];
            *d = b;
        }
        if clean {
            pad[bytes.len() + 1] = b'"';
            push_padded(out, &pad, bytes.len() + 2);
            return;
        }
    }
    out.reserve(bytes.len() + 2);
    out.push(b'"');
    let (mut start, mut i) = (0, 0);
    while i < bytes.len() {
        let b = bytes[i];
        if !JSON_QUOTE_SPECIAL[usize::from(b)] {
            i += 1;
            continue;
        }
        let esc: &[u8] = match b {
            b'"' => b"\\\"",
            b'\\' => b"\\\\",
            0x08 => b"\\b",
            0x0C => b"\\f",
            0x0A => b"\\n",
            0x0D => b"\\r",
            0x09 => b"\\t",
            0xED => {
                // `ED A0..BF xx` encodes a surrogate (U+D800..U+DFFF), which
                // WTF-8 holds only unpaired; `ED 80..9F xx` is a scalar.
                match (bytes.get(i + 1), bytes.get(i + 2)) {
                    (Some(&b1), Some(&b2)) if b1 >= 0xA0 => {
                        out.extend_from_slice(&bytes[start..i]);
                        let cp = 0xD000 | (u32::from(b1 & 0x3F) << 6) | u32::from(b2 & 0x3F);
                        push_json_unicode_escape(out, cp);
                        i += 3;
                        start = i;
                    }
                    _ => i += 1,
                }
                continue;
            }
            _ => {
                out.extend_from_slice(&bytes[start..i]);
                push_json_unicode_escape(out, u32::from(b));
                i += 1;
                start = i;
                continue;
            }
        };
        out.extend_from_slice(&bytes[start..i]);
        out.extend_from_slice(esc);
        i += 1;
        start = i;
    }
    out.extend_from_slice(&bytes[start..]);
    out.push(b'"');
}

/// The size of the fixed buffers short texts are staged in.
pub(crate) const SHORT_TEXT: usize = 32;

/// Appends `pad[..len]` to `out` as a copy of all of `pad` (whose size is
/// known, so the copy is a few moves rather than a call) cut back to `len`.
#[inline]
pub(crate) fn push_padded(out: &mut Vec<u8>, pad: &[u8; SHORT_TEXT], len: usize) {
    let start = out.len();
    out.extend_from_slice(pad);
    out.truncate(start + len);
}

/// Appends `\u` and the four lowercase hex digits of the code unit `cp`.
fn push_json_unicode_escape(out: &mut Vec<u8>, cp: u32) {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    out.extend_from_slice(&[
        b'\\',
        b'u',
        HEX[(cp >> 12) as usize & 15],
        HEX[(cp >> 8) as usize & 15],
        HEX[(cp >> 4) as usize & 15],
        HEX[cp as usize & 15],
    ]);
}

impl Realm {
    /// Appends the JSON text of `v` to `out` and returns `true` when `v` is
    /// a value whose serialization consults nothing (no `toJSON`, no
    /// wrapper): a Number (`null` when not finite), a Boolean, `null`, or a
    /// flat string primitive. Returns `false`, having written nothing, for
    /// anything else (a rope is left for the caller to flatten), and for a
    /// string whose quoted text would take `out` past `max` bytes (the
    /// caller's general path throws the length error).
    #[inline]
    pub(crate) fn json_write_prim(&self, out: &mut Vec<u8>, v: NanBox, max: usize) -> bool {
        match v.unpack() {
            Unpacked::Number(n) => {
                if n.is_finite() {
                    super::builtins_fast::push_number_bytes(out, n);
                } else {
                    out.extend_from_slice(b"null");
                }
                true
            }
            Unpacked::Bool(b) => {
                out.extend_from_slice(if b { b"true" } else { b"false" });
                true
            }
            Unpacked::Null => {
                out.extend_from_slice(b"null");
                true
            }
            Unpacked::Handle(raw) => self.json_write_flat_string(out, raw, max),
            _ => false,
        }
    }

    /// [`json_write_prim`](Self::json_write_prim) for a heap value: a flat
    /// string primitive (no auxiliary object, so `typeof` is "string").
    fn json_write_flat_string(&self, out: &mut Vec<u8>, raw: u64, max: usize) -> bool {
        if let Some(Cell::Str(r)) = self.heap.get(Handle::from_raw(raw))
            && let Some(bytes) = r.as_leaf_bytes()
            && out.len() + bytes.len() + 2 <= max
            && !self.aux_props.contains_key(&raw)
        {
            json_quote_wtf8_into(out, bytes);
            return true;
        }
        false
    }

    /// For an ordinary object whose own properties are exactly its shape's
    /// enumerable data keys (see
    /// [`plain_keyed_shape`](Self::plain_keyed_shape)): its shape and its
    /// `[[Prototype]]`, from one heap read.
    #[inline]
    pub(crate) fn json_plain_object(&self, h: Handle) -> Option<(&Rc<Shape>, Option<Handle>)> {
        let Some(Cell::Object(o)) = self.heap.get(h) else {
            return None;
        };
        if !o.keys_follow_shape() || self.is_host_exotic(h) {
            return None;
        }
        Some((o.shaped_shape()?, o.proto()))
    }

    /// For an array with no auxiliary object (so no named own property, no
    /// accessor or attribute override on an index, no element past the dense
    /// store) that is not host-exotic: its `[[Prototype]]` (`Some(None)` for
    /// `null`). `None` for anything else.
    pub(crate) fn json_plain_array(&self, h: Handle) -> Option<Option<Handle>> {
        if !matches!(self.heap.get(h), Some(Cell::Array(_)))
            || self.aux_props.contains_key(&h.to_raw())
            || self.is_host_exotic(h)
        {
            return None;
        }
        Some(self.object_proto(h))
    }

    /// Copies the values of `slots` of the object at `h` into `out` while it
    /// still has `shape` with its keys following it; `false` (nothing
    /// trustworthy written) otherwise.
    pub(crate) fn json_slots_under(
        &self,
        h: Handle,
        shape: &Rc<Shape>,
        slots: &[u32],
        out: &mut [NanBox],
    ) -> bool {
        let Some(Cell::Object(o)) = self.heap.get(h) else {
            return false;
        };
        if !o.keys_follow_shape() || !o.shaped_shape().is_some_and(|s| Rc::ptr_eq(s, shape)) {
            return false;
        }
        for (d, &s) in out.iter_mut().zip(slots) {
            match o.slot(s) {
                Some(v) => *d = v,
                None => return false,
            }
        }
        true
    }

    /// Writes the elements `from..` of the plain array at `h` (see
    /// [`json_plain_array`](Self::json_plain_array), re-checked here) for as
    /// long as each is a present [`json_write_prim`](Self::json_write_prim)
    /// value, below `len`, each after a `,` but the first element of the
    /// array, and while `out` stays within `max` bytes. Returns the index of
    /// the first element not written.
    pub(crate) fn json_write_array_run(
        &self,
        h: Handle,
        from: usize,
        len: usize,
        out: &mut Vec<u8>,
        max: usize,
    ) -> usize {
        let Some(Cell::Array(a)) = self.heap.get(h) else {
            return from;
        };
        if self.aux_props.contains_key(&h.to_raw()) {
            return from;
        }
        let elems: &[NanBox] = a;
        let end = len.min(elems.len());
        let mut i = from;
        while i < end {
            let v = elems[i];
            let mark = out.len();
            if i > 0 {
                out.push(b',');
            }
            if v.is_hole() || !self.json_write_prim(out, v, max) || out.len() > max {
                out.truncate(mark);
                break;
            }
            i += 1;
        }
        i
    }

    /// Whether the ordinary `[[Get]]` of the named key `key` from `start`
    /// finds nothing: every link an ordinary object, array, RegExp or
    /// collection (none host-exotic) not owning `key`, at most eight deep.
    /// Every link is then watched (see
    /// [`watch_link`](Self::watch_link)), so the answer holds while the
    /// prototype epoch read after this call is current. `false` when the
    /// walk cannot show it.
    pub(crate) fn prove_chain_lacks(&mut self, start: Option<Handle>, key: &str) -> bool {
        let mut links: [(u64, Option<u64>); 8] = [(0, None); 8];
        let mut n = 0;
        let mut cur = start;
        while let Some(h) = cur {
            if n == links.len() {
                return false;
            }
            let Some((aux, _, next)) = self.key_link(h) else {
                return false;
            };
            if self.has_own(h, key) {
                return false;
            }
            links[n] = (h.to_raw(), aux);
            n += 1;
            cur = next;
        }
        for &(raw, aux) in &links[..n] {
            self.watch_link(raw, aux);
        }
        links[..n]
            .iter()
            .all(|&(raw, aux)| self.link_watched(raw, aux))
    }
}
