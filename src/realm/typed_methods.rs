//! The lean form of a few `%TypedArray%.prototype` methods for a memoized
//! call site (`nbvm::native_call`): `reverse`, `indexOf` / `lastIndexOf` /
//! `includes`, `fill` and `set`, on a view whose bytes lie wholly inside an
//! attached buffer, of a non-BigInt kind, with Number (or absent) arguments
//! (whose ToNumber / ToIntegerOrInfinity run no code). Under those
//! conditions each one is its specification algorithm over the bytes, with
//! nothing observable but the result and the written elements. Anything
//! else (`None`, having done nothing) is the host's method, which owns every
//! error.

use super::Realm;
use crate::cell::Cell;
use crate::heap::Handle;
use crate::nanbox::NanBox;
use crate::nbexec::{ARRAY_BUFFER_DETACHED, ARRAY_BUFFER_IMMUTABLE, decode_typed_element};

/// Which search a [`Realm::ta_search`] is.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum TaSearch {
    IndexOf,
    LastIndexOf,
    Includes,
}

/// A `%TypedArray%.prototype` method served by [`Realm::ta_method`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum TaMethod {
    Reverse,
    Search(TaSearch),
    Fill,
    Set,
}

impl TaMethod {
    /// The method a `%TypedArray%.prototype` method name is.
    pub(crate) fn of(name: &[u8]) -> Option<Self> {
        Some(match name {
            b"reverse" => Self::Reverse,
            b"indexOf" => Self::Search(TaSearch::IndexOf),
            b"lastIndexOf" => Self::Search(TaSearch::LastIndexOf),
            b"includes" => Self::Search(TaSearch::Includes),
            b"fill" => Self::Fill,
            b"set" => Self::Set,
            _ => return None,
        })
    }
}

/// A view's geometry: its bytes cell, byte offset, element count, kind and
/// element size.
#[derive(Clone, Copy)]
struct Region {
    bytes: Handle,
    off: usize,
    len: usize,
    kind: u8,
    size: usize,
}

/// ToIntegerOrInfinity of a Number argument (`undefined` is `default`);
/// `None` for anything else (its conversion may run code).
fn int_arg(v: Option<NanBox>, default: f64) -> Option<f64> {
    let Some(v) = v else {
        return Some(default);
    };
    if v.is_undefined() {
        // ToIntegerOrInfinity(undefined) is 0 (an explicit `undefined`).
        return Some(0.0);
    }
    let n = v.as_number()?;
    // (A double of magnitude 2^52 or more, or infinite, is integral; below
    // that the `i64` conversion truncates toward zero exactly.)
    const BIG: f64 = 4_503_599_627_370_496.0;
    Some(if n.is_nan() {
        0.0
    } else if (-BIG..=BIG).contains(&n) {
        n as i64 as f64
    } else {
        n
    })
}

/// A relative index (`fill`'s `start` / `end`) clamped into `0..=len`.
fn rel_index(n: f64, len: usize) -> usize {
    if n < 0.0 {
        let k = len as f64 + n;
        if k < 0.0 { 0 } else { k as usize }
    } else if n > len as f64 {
        len
    } else {
        n as usize
    }
}

impl Realm {
    /// The geometry of the typed array `h` when it is a fixed-length,
    /// non-BigInt view lying wholly inside an attached buffer; `None`
    /// otherwise.
    fn ta_region(&self, h: Handle) -> Option<Region> {
        let (bytes, off, len, kind) = self.heap.get(h)?.as_typed_array()?;
        if !super::typed_fast::fast_kind(kind) || self.length_tracking_views.contains(&h.to_raw()) {
            return None;
        }
        let size = super::typed_elem_size(kind);
        let end = len.checked_mul(size)?.checked_add(off)?;
        if end > self.bytes_len(bytes)? {
            return None;
        }
        let buf = self.typed_array_object(h)?;
        if self.get_property(buf, ARRAY_BUFFER_DETACHED).is_some() {
            return None;
        }
        Some(Region {
            bytes,
            off,
            len,
            kind,
            size,
        })
    }

    /// Whether the typed array `h`'s buffer is immutable.
    fn ta_immutable(&self, h: Handle) -> bool {
        self.typed_array_object(h)
            .is_some_and(|b| self.get_property(b, ARRAY_BUFFER_IMMUTABLE).is_some())
    }

    /// `m` on the typed array `h` with arguments `args` (see the module docs);
    /// its result, or `None`, having done nothing.
    pub(crate) fn ta_method(&mut self, h: Handle, m: TaMethod, args: &[NanBox]) -> Option<NanBox> {
        let r = self.ta_region(h)?;
        let arg = |i: usize| args.get(i).copied();
        match m {
            TaMethod::Reverse => {
                if self.ta_immutable(h) {
                    return None;
                }
                let bytes = self.bytes_at_mut(r.bytes)?;
                let view = bytes.get_mut(r.off..r.off + r.len * r.size)?;
                let (mut lo, mut hi) = (0, r.len);
                while lo + 1 < hi {
                    hi -= 1;
                    for b in 0..r.size {
                        view.swap(lo * r.size + b, hi * r.size + b);
                    }
                    lo += 1;
                }
                Some(NanBox::handle(h.to_raw()))
            }
            TaMethod::Search(s) => {
                let target = arg(0).unwrap_or(NanBox::undefined());
                let found = self.ta_search(r, s, target, arg(1))?;
                Some(match s {
                    TaSearch::Includes => NanBox::boolean(found.is_some()),
                    _ => NanBox::number(found.map_or(-1.0, |i| i as f64)),
                })
            }
            TaMethod::Fill => {
                let value = arg(0)?.as_number()?;
                let start = int_arg(arg(1), 0.0)?;
                // (An `undefined` end is the length.)
                let end = int_arg(arg(2).filter(|v| !v.is_undefined()), r.len as f64)?;
                if self.ta_immutable(h) {
                    return None;
                }
                let (s, e) = (rel_index(start, r.len), rel_index(end, r.len));
                self.typed_fill_range(h, NanBox::number(value), s, e);
                Some(NanBox::handle(h.to_raw()))
            }
            TaMethod::Set => {
                let offset = int_arg(arg(1), 0.0)?;
                if !(0.0..=r.len as f64).contains(&offset) || self.ta_immutable(h) {
                    return None;
                }
                let offset = offset as usize;
                let src = Handle::from_raw(arg(0)?.as_handle()?);
                if self.heap.get(src).and_then(Cell::as_typed_array).is_some() {
                    // A typed-array source of a non-BigInt kind, in bounds.
                    let s = self.ta_region(src)?;
                    if offset.checked_add(s.len)? > r.len {
                        return None;
                    }
                    if !self.typed_set_same_kind(h, src, offset) {
                        let elems = self.elements_vec(src)?;
                        self.typed_set_from_numbers(h, offset, &elems);
                    }
                    return Some(NanBox::undefined());
                }
                // A plain dense array of Numbers: each `Get` reads its element.
                if !self.is_array(src)
                    || self.is_host_exotic(src)
                    || self.array_has_index_overrides(src)
                    || self.proto_index_accessor_dirty()
                {
                    return None;
                }
                let elems = self.array_elements(src)?;
                if self.array_length(src) != Some(elems.len())
                    || offset.checked_add(elems.len())? > r.len
                    || elems.iter().any(|v| !v.is_number())
                {
                    return None;
                }
                let elems: alloc::vec::Vec<NanBox> = elems.to_vec();
                self.typed_set_from_numbers(h, offset, &elems);
                Some(NanBox::undefined())
            }
        }
    }

    /// `indexOf` / `lastIndexOf` / `includes` of the Number `target` (any
    /// other value matches no element of a non-BigInt view, except
    /// `undefined` for `includes` on an empty range: never, since every
    /// element is a Number) from `from` over the view `r`.
    fn ta_search(
        &self,
        r: Region,
        s: TaSearch,
        target: NanBox,
        from: Option<NanBox>,
    ) -> Option<Option<usize>> {
        let len = r.len;
        if len == 0 {
            return Some(None);
        }
        let bytes = self.bytes_at(r.bytes)?.get(r.off..r.off + len * r.size)?;
        let at = |k: usize| decode_typed_element(r.kind, &bytes[k * r.size..(k + 1) * r.size]);
        let t = target.as_number();
        let matches = |x: f64| match t {
            Some(t) if s == TaSearch::Includes && t.is_nan() => x.is_nan(),
            Some(t) => x == t,
            None => false,
        };
        // A non-Number target is still ToIntegerOrInfinity's `fromIndex`
        // first: only Number (or absent) ones are served.
        if s == TaSearch::LastIndexOf {
            let n = int_arg(from, (len - 1) as f64)?;
            if n == f64::NEG_INFINITY {
                return Some(None);
            }
            let k = if n >= 0.0 {
                if n >= (len - 1) as f64 {
                    len - 1
                } else {
                    n as usize
                }
            } else {
                let k = len as f64 + n;
                if k < 0.0 {
                    return Some(None);
                }
                k as usize
            };
            return Some((0..=k).rev().find(|&i| matches(at(i))));
        }
        let n = int_arg(from, 0.0)?;
        if n == f64::INFINITY {
            return Some(None);
        }
        let k = rel_index(n, len);
        Some((k..len).find(|&i| matches(at(i))))
    }
}
