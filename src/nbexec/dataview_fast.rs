//! `DataView.prototype.get*` / `set*` without the by-name dispatch.
//!
//! A first-class DataView accessor is a bound native that re-dispatches
//! through `call_method` by its name, which walks the whole by-name cascade
//! (a probe for every other receiver kind) before it reaches the DataView arm
//! in `method_dispatch`; a `dv.getInt32(off, true)` cost about 4,000
//! instructions, so a byte-code interpreter keeping its memory in a DataView
//! ran at 150 ns per load.
//!
//! [`dataview_fast`] is that arm's work for the calls where none of its steps
//! can run code or throw: a Number offset that is already a valid index (an
//! integer in `[0, 2**53)`, so `ToIndex` is the identity), a Number value for
//! a set (so `ToNumber` is the identity), a buffer that is neither detached
//! nor immutable, a view in bounds, and an access inside it. It reads the same
//! hidden slots in the same way and stores the same bits. Anything else
//! answers `None` and the caller takes the general path, which raises the
//! spec's errors in the spec's order. BigInt and Float16 accessors are not
//! served here.

use super::{
    ARRAY_BUFFER_BYTES, ARRAY_BUFFER_DETACHED, ARRAY_BUFFER_IMMUTABLE, DATA_VIEW_BUF,
    DATA_VIEW_LEN, DATA_VIEW_OFF, trunc_toward_zero,
};
use crate::heap::Handle;
use crate::nanbox::NanBox;
use crate::realm::Realm;

/// One `DataView` accessor served by [`dataview_fast`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct DvAccess {
    /// A `set*` (else a `get*`).
    pub set: bool,
    /// The element size in bytes: 1, 2, 4 or 8.
    pub size: u8,
    /// A signed integer type.
    pub signed: bool,
    /// `Float32` / `Float64`.
    pub float: bool,
}

/// The accessor named `name` (`getInt8` … `setFloat64`), or `None` for any
/// other name, a BigInt accessor or a Float16 one.
pub(crate) fn dv_access_for(name: &[u8]) -> Option<DvAccess> {
    let (set, ty) = match name {
        [b'g', b'e', b't', rest @ ..] => (false, rest),
        [b's', b'e', b't', rest @ ..] => (true, rest),
        _ => return None,
    };
    let (size, signed, float) = match ty {
        b"Int8" => (1, true, false),
        b"Uint8" => (1, false, false),
        b"Int16" => (2, true, false),
        b"Uint16" => (2, false, false),
        b"Int32" => (4, true, false),
        b"Uint32" => (4, false, false),
        b"Float32" => (4, false, true),
        b"Float64" => (8, false, true),
        _ => return None,
    };
    Some(DvAccess {
        set,
        size,
        signed,
        float,
    })
}

/// A handle-valued hidden slot `key` of `h`.
fn slot_handle(realm: &Realm, h: Handle, key: &str) -> Option<Handle> {
    realm
        .get_property(h, key)
        .and_then(NanBox::as_handle)
        .map(Handle::from_raw)
}

/// `dv.<acc>(offset, …rest)` when no step of `GetViewValue` / `SetViewValue`
/// can run code or throw (see the module docs); `None` means "take the general
/// path", having changed nothing. `rest` is `[littleEndian]` for a get and
/// `[value, littleEndian]` for a set. A set answers `undefined`.
pub(crate) fn dataview_fast(
    realm: &mut Realm,
    recv: NanBox,
    acc: DvAccess,
    offset: NanBox,
    rest: [NanBox; 2],
) -> Option<NanBox> {
    let h = Handle::from_raw(recv.as_handle()?);
    let buf = slot_handle(realm, h, DATA_VIEW_BUF)?;
    if acc.set && realm.get_property(buf, ARRAY_BUFFER_IMMUTABLE).is_some() {
        return None;
    }
    // ToIndex(requestIndex) is the identity on an integral Number in range.
    let off = offset.as_number()?;
    if !(0.0..9_007_199_254_740_992.0).contains(&off) || (off as u64) as f64 != off {
        return None;
    }
    let (value, le) = if acc.set {
        (Some(rest[0].as_number()?), realm.truthy(rest[1]))
    } else {
        (None, realm.truthy(rest[0]))
    };
    if realm.get_property(buf, ARRAY_BUFFER_DETACHED).is_some() {
        return None;
    }
    let bh = slot_handle(realm, buf, ARRAY_BUFFER_BYTES)?;
    let base = realm
        .get_property(h, DATA_VIEW_OFF)
        .and_then(|n| n.as_number())
        .unwrap_or(0.0) as usize;
    let total = realm.bytes_len(bh).unwrap_or(0);
    let recorded = realm
        .get_property(h, DATA_VIEW_LEN)
        .and_then(|n| n.as_number())
        .map(|n| n as usize);
    let oob = match recorded {
        Some(len) => base.checked_add(len).is_none_or(|end| end > total),
        None => base > total,
    };
    if oob {
        return None;
    }
    let view_len = recorded
        .unwrap_or(total.saturating_sub(base))
        .min(total.saturating_sub(base));
    let size = usize::from(acc.size);
    let r = usize::try_from(off as u64).ok()?;
    if !r.checked_add(size).is_some_and(|end| end <= view_len) {
        return None;
    }
    let abs = base.checked_add(r)?;
    if abs.checked_add(size)? > total {
        return None;
    }
    if let Some(v) = value {
        let bits: u64 = if acc.float {
            if size == 4 {
                u64::from((v as f32).to_bits())
            } else {
                v.to_bits()
            }
        } else if v.is_finite() {
            // The integer types take the value modulo 2^(8*size), as the
            // general path does.
            const TWO32: f64 = 4_294_967_296.0;
            let m = trunc_toward_zero(v) % TWO32;
            let t = (if m < 0.0 { m + TWO32 } else { m }) as u64;
            t & ((1u64 << (8 * size)) - 1)
        } else {
            0
        };
        let bytes = realm.bytes_at_mut(bh)?;
        let dst = bytes.get_mut(abs..abs + size)?;
        for (i, slot) in dst.iter_mut().enumerate() {
            let shift = if le { i } else { size - 1 - i };
            *slot = ((bits >> (8 * shift)) & 0xff) as u8;
        }
        return Some(NanBox::undefined());
    }
    let src = realm.bytes_at(bh)?.get(abs..abs + size)?;
    let mut bits: u64 = 0;
    for (i, &b) in src.iter().enumerate() {
        let shift = if le { i } else { size - 1 - i };
        bits |= u64::from(b) << (8 * shift);
    }
    let n = if acc.float {
        if size == 4 {
            f64::from(f32::from_bits(bits as u32))
        } else {
            f64::from_bits(bits)
        }
    } else if acc.signed && bits & (1 << (8 * size - 1)) != 0 {
        (bits as i64 - (1i64 << (8 * size))) as f64
    } else {
        bits as f64
    };
    Some(NanBox::number(n))
}
