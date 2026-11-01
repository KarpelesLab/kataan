//! String built-ins over ropes: the forms of `padStart`/`padEnd`, `trim*`,
//! `startsWith`/`endsWith` and `repeat` that do not flatten or copy their
//! receiver.
//!
//! Flattening a `Concat` costs its whole length, and so does copying a flat
//! receiver into a fresh result; done per call, a loop that grows a string
//! through one of these methods (`s = s.padEnd(s.length + 1)`) or reads it
//! between appends (`s += x; if (s.endsWith(y)) …`) was quadratic. Here a
//! padded result is the receiver joined to its filler (an O(1) rope
//! concatenation), a trimmed one shares the receiver's bytes or subtrees,
//! and an affix test compares only the window it names, found by walking
//! the tree. Each helper returns `None` when its walk would be long, and the
//! caller then flattens and takes the flat path, so the cost is bounded
//! either way.

use alloc::vec::Vec;

use crate::rope::Rope;

/// `padStart`/`padEnd` (`at_start`) of `s` to `target` UTF-16 units with the
/// filler `pad` (WTF-8 bytes, repeated and cut to a unit boundary): `s` itself
/// when it is already long enough or `pad` is empty, else `s` joined to the
/// filler without copying `s`. `None` when the result would pass the string
/// length cap (the caller throws the `RangeError`).
pub(crate) fn pad_rope(s: &Rope, target: usize, pad: &[u8], at_start: bool) -> Option<Rope> {
    let len = s.utf16_len();
    if len >= target || pad.is_empty() {
        return Some(s.clone());
    }
    let filler = Rope::from_wtf8(pad_filler(target - len, pad));
    // (The join pairs a lone surrogate at the seam, as `+` does.)
    if at_start {
        filler.try_concat(s)
    } else {
        s.try_concat(&filler)
    }
}

/// `need` UTF-16 units of `pad` (non-empty WTF-8) repeated, cut by code unit
/// (which may leave a lone surrogate), as WTF-8 bytes.
pub(crate) fn pad_filler(need: usize, pad: &[u8]) -> Vec<u8> {
    let pad_units: Vec<u16> = crate::wtf8::utf16_units(pad).collect();
    if pad_units.is_empty() {
        return Vec::new();
    }
    let units: Vec<u16> = pad_units.iter().copied().cycle().take(need).collect();
    crate::wtf8::from_utf16(&units)
}

/// `TrimString(s, where)` (`start`/`end` say which ends) sharing `s`'s bytes
/// (a flat `s`) or subtrees (a `Concat`), never copying the kept part. `None`
/// for a `Concat` whose walk would be long (flatten it and call again: a flat
/// string always succeeds).
pub(crate) fn trim_rope(s: &Rope, start: bool, end: bool) -> Option<Rope> {
    if !s.is_concat() {
        let bytes = s.as_leaf_bytes()?;
        let kept = super::trim_wtf8(bytes, start, end);
        if kept.len() == bytes.len() {
            return Some(s.clone());
        }
        let b0 = kept.as_ptr() as usize - bytes.as_ptr() as usize;
        return s.share_byte_range(b0, b0 + kept.len());
    }
    let total = s.utf16_len();
    // White space is BMP, one unit per character: each end's trimmed run is
    // counted leaf by leaf until a leaf keeps something.
    let edge = |from_right: bool| -> Option<usize> {
        let mut units = 0usize;
        let whole = s.walk_edge_leaves(from_right, |leaf| {
            let kept = super::trim_wtf8(leaf, !from_right, from_right);
            let cut = leaf.len() - kept.len();
            units += if from_right {
                crate::wtf8::utf16_len(&leaf[leaf.len() - cut..])
            } else {
                crate::wtf8::utf16_len(&leaf[..cut])
            };
            kept.is_empty()
        });
        whole.then_some(units)
    };
    let lead = if start { edge(false)? } else { 0 };
    if lead >= total {
        return Some(Rope::new());
    }
    let trail = if end { edge(true)? } else { 0 };
    if lead == 0 && trail == 0 {
        return Some(s.clone());
    }
    s.slice_units_tree(lead, total - trail)
}

/// Whether the UTF-16 units of the `Concat` `s` at `[at, at + |pat|)` are
/// those of `pat` (`startsWith`/`endsWith` at a clamped position), comparing
/// only that window. `None` for a flat `s` or a long walk (flatten and use the
/// flat test).
pub(crate) fn window_is(s: &Rope, at: usize, pat: &[u8], pat_units: usize) -> Option<bool> {
    if !s.is_concat() {
        return None;
    }
    let total = s.utf16_len();
    if pat_units > total || at > total - pat_units {
        return Some(false);
    }
    if pat_units == 0 {
        return Some(true);
    }
    // Both sides canonical WTF-8, so equal units are equal bytes.
    let w = s.slice_units_tree(at, at + pat_units)?;
    Some(match w.as_leaf_bytes() {
        Some(b) => b == pat,
        None => w.len() == pat.len() && w.materialize_bytes() == pat,
    })
}

/// The shortest flat block [`repeat_rope`] doubles: a shorter receiver is
/// first repeated flat up to this, so the result has few nodes to flatten.
const REPEAT_BLOCK: usize = 1024;

/// `n` copies of the WTF-8 `one`, flat. A string that starts with a lone low
/// surrogate and ends with a lone high one pairs them at every seam (as `+`
/// does), so the copies are joined rather than byte-repeated then.
fn repeat_flat(one: &[u8], n: usize) -> Rope {
    if crate::wtf8::trailing_high_surrogate(one).is_none()
        || crate::wtf8::leading_low_surrogate(one).is_none()
    {
        return Rope::from_wtf8(one.repeat(n));
    }
    let piece = Rope::from_bytes(one);
    let mut out = Rope::new();
    for _ in 0..n {
        out = out.concat(&piece);
    }
    Rope::from_wtf8(out.materialize_bytes())
}

/// `s.repeat(n)`, its byte length `n * |s|` already checked against the cap:
/// `s` itself for `n == 1`; a short result flat; a long one as a rope of
/// O(log n) nodes over a flat block (built by doubling), so neither `s` nor
/// the result is copied per repetition. `None` only if a join passes the
/// rope's own length cap.
pub(crate) fn repeat_rope(s: &Rope, n: usize) -> Option<Rope> {
    match n {
        0 => return Some(Rope::new()),
        1 => return Some(s.clone()),
        _ => {}
    }
    let total = s.len().checked_mul(n)?;
    if total < 4 * REPEAT_BLOCK {
        return Some(repeat_flat(&s.materialize_bytes(), n));
    }
    // A flat block of `k` copies (`k` = 1, `s` itself, for a long receiver).
    let k = REPEAT_BLOCK.div_ceil(s.len().max(1)).clamp(1, n);
    let block = if k == 1 {
        s.clone()
    } else {
        repeat_flat(&s.materialize_bytes(), k)
    };
    // `n = q * k + r`: `q` blocks by binary doubling, then `r` single copies.
    let (mut q, r) = (n / k, n % k);
    let mut out = Rope::new();
    let mut pow = block;
    while q > 0 {
        if q & 1 == 1 {
            out = out.try_concat(&pow)?;
        }
        q >>= 1;
        if q > 0 {
            pow = pow.try_concat(&pow)?;
        }
    }
    for _ in 0..r {
        out = out.try_concat(s)?;
    }
    Some(out)
}
