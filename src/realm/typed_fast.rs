//! Typed-array element reads and Number writes for a view that lies wholly
//! inside its buffer.
//!
//! `ta[i]` and `ta[i] = n` reached [`Realm::typed_get`] / [`Realm::typed_set`]
//! through several separate checks, each its own heap lookup: the view's
//! length, whether it is out of bounds (which also searched the set of
//! length-tracking views), its element kind, the buffer for the bytes, and
//! for a write the general `ToNumber` → modular reduction (an `fmod`) →
//! encode into a scratch array → byte-by-byte copy. A sudoku solver that keeps
//! its candidate masks in `Uint16Array`s spent most of its time there.
//!
//! When the view's extent `[byteOffset, byteOffset + length × size)` fits in
//! the buffer's current bytes, the view is in bounds whichever kind of view it
//! is (a fixed-length view is out of bounds only when that extent passes the
//! end; a length-tracking one only when its offset does), so an index below
//! `length` names bytes that exist and `IsValidIntegerIndex` holds. These
//! helpers handle exactly that case with two lookups and answer `None` /
//! `false` for anything else (BigInt kinds, Float16, a value that needs real
//! `ToNumber` or modular reduction, a view not wholly in its buffer) so the
//! general path keeps every other rule.

use super::{Handle, NanBox, Realm, typed_elem_size};
use crate::cell::Cell;
use crate::nbexec::decode_typed_element;

/// The element kinds these helpers write (all but Float16 = 11 and the BigInt
/// kinds 9 / 10).
const fn fast_kind(kind: u8) -> bool {
    kind <= 8
}

/// `n` as an exact integer when it is one with magnitude below 2^53 (so the
/// `i64` round trip is exact): the modular `ToInt8`..`ToUint32` of such an `n`
/// is its low bits. `None` for NaN, ±Infinity, a fraction, or a huge value.
fn exact_int(n: f64) -> Option<i64> {
    if !(-9_007_199_254_740_992.0..9_007_199_254_740_992.0).contains(&n) {
        return None;
    }
    let t = n as i64;
    (t as f64 == n).then_some(t)
}

/// Where element `i` lives: `(buffer, start byte, extent end, kind)`, when `i`
/// is below the view's length and the kind is one [`fast_kind`] accepts. The
/// caller checks `end` against the buffer's current length.
#[inline]
fn slot(cell: &Cell, i: usize) -> Option<(Handle, usize, usize, u8)> {
    let (buffer, off, len, kind) = cell.as_typed_array()?;
    if i >= len || !fast_kind(kind) {
        return None;
    }
    let size = typed_elem_size(kind);
    let end = len.checked_mul(size)?.checked_add(off)?;
    Some((buffer, off + i * size, end, kind))
}

impl Realm {
    /// `view[i]` for an in-range `i` of a non-BigInt view wholly inside its
    /// buffer; `None` sends the read down [`Realm::typed_get`]'s general path.
    #[inline]
    pub(crate) fn typed_get_fast(&self, handle: Handle, i: usize) -> Option<NanBox> {
        let (buffer, start, end, kind) = slot(self.heap.get(handle)?, i)?;
        let bytes = self.bytes_at(buffer)?;
        if end > bytes.len() {
            return None;
        }
        let slice = bytes.get(start..start + typed_elem_size(kind))?;
        Some(NanBox::number(decode_typed_element(kind, slice)))
    }

    /// `view[i] = n` (`TypedArraySetElement` with a Number) for an in-range `i`
    /// of a view wholly inside its buffer, when `n` needs no rounding beyond
    /// what the kind's own cast does exactly: an exact integer below 2^53 for
    /// the integer kinds (stored as its low bits, clamped for Uint8Clamped),
    /// any Number for Float32 / Float64. `false` (nothing written) for anything
    /// else.
    pub(crate) fn typed_set_fast(&mut self, handle: Handle, i: usize, n: f64) -> bool {
        let Some((buffer, start, end, kind)) = self.heap.get(handle).and_then(|c| slot(c, i))
        else {
            return false;
        };
        let mut enc = [0u8; 8];
        let size = match kind {
            7 => {
                enc[..4].copy_from_slice(&(n as f32).to_le_bytes());
                4
            }
            8 => {
                enc = n.to_le_bytes();
                8
            }
            _ => {
                let Some(t) = exact_int(n) else {
                    return false;
                };
                let w = if kind == 2 { t.clamp(0, 255) } else { t } as u32;
                enc[..4].copy_from_slice(&w.to_le_bytes());
                typed_elem_size(kind)
            }
        };
        let Some(bytes) = self.bytes_at_mut(buffer) else {
            return false;
        };
        if end > bytes.len() {
            return false;
        }
        let Some(dst) = bytes.get_mut(start..start + size) else {
            return false;
        };
        dst.copy_from_slice(&enc[..size]);
        true
    }
}
