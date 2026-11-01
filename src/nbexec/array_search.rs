//! `Array.prototype.indexOf` / `lastIndexOf` / `includes` over a plain
//! array's dense store, scanned in place.
//!
//! The caller has established that the receiver is a real array whose
//! `length` is its dense store's, with no per-index side table (so a present
//! slot *is* its `[[Get]]`), that no index accessor was ever installed on the
//! default prototype chain, and that the position argument was coerced
//! without running code. Present slots are then compared straight off the
//! slice. A hole is the only index whose answer can come from elsewhere: when
//! the prototype chain owns no index at all (checked once, at the first hole)
//! a hole is simply absent (`HasProperty` false, `[[Get]]` `undefined`), so
//! the scan steps over it in place too; otherwise that one index is probed
//! with the real `HasProperty` / `[[Get]]` (which may run an inherited getter,
//! after which every index is re-read live and an index past a shrunken store
//! is probed the same way).

use super::*;

/// Which of the three searches (their equality and hole rules differ).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum ArraySearch {
    /// `indexOf`: ascending, `===`, absent indices skipped.
    IndexOf,
    /// `lastIndexOf`: descending, `===`, absent indices skipped.
    LastIndexOf,
    /// `includes`: ascending, SameValueZero, an absent index reads `undefined`.
    Includes,
}

/// What one in-place run of the scan stopped at.
enum Stop {
    /// The index of a matching element.
    Match(usize),
    /// An index that must be probed through the object model.
    Probe(usize),
    /// The range is exhausted without a match.
    Done,
}

impl Interp {
    /// Searches the dense array `handle` for `target` over `lo..hi`
    /// (descending from `hi - 1` for `lastIndexOf`); the index found, if any.
    pub(crate) fn array_dense_search(
        &mut self,
        handle: Handle,
        lo: usize,
        hi: usize,
        kind: ArraySearch,
        target: NanBox,
    ) -> Result<Option<usize>, ExecError> {
        let desc = kind == ArraySearch::LastIndexOf;
        // The next index to visit (descending: one past it, so `lo` ends it).
        let mut cur = if desc { hi } else { lo };
        // Whether holes may be stepped over in place (the chain owns no index).
        let mut chain_free: Option<bool> = None;
        loop {
            let skip_holes = chain_free == Some(true);
            let stop = {
                let elems = self.realm.array_elements(handle).unwrap_or(&[]);
                scan(
                    &self.realm,
                    elems,
                    cur,
                    lo,
                    hi,
                    desc,
                    kind,
                    target,
                    skip_holes,
                )
            };
            let i = match stop {
                Stop::Match(i) => return Ok(Some(i)),
                Stop::Done => return Ok(None),
                Stop::Probe(i) => i,
            };
            if chain_free.is_none() {
                let free = (self.realm.array_chain_index_free(handle)
                    || self.realm.proto_chain_index_free(handle))
                    && self
                        .realm
                        .array_elements(handle)
                        .is_some_and(|a| i < a.len());
                chain_free = Some(free);
                if free {
                    // `includes` reads the hole as `undefined`.
                    if kind == ArraySearch::Includes && target.is_undefined() {
                        return Ok(Some(i));
                    }
                    cur = if desc { i } else { i + 1 };
                    continue;
                }
            }
            let key = alloc::format!("{i}");
            let hit = if kind == ArraySearch::Includes {
                let v = self.read_member(handle, &key)?;
                self.realm.same_value_zero(v, target)
            } else if self.has_property(handle, &key) {
                let v = self.read_member(handle, &key)?;
                self.realm.strict_equals(v, target)
            } else {
                false
            };
            if hit {
                return Ok(Some(i));
            }
            cur = if desc { i } else { i + 1 };
        }
    }
}

/// Scans `elems` from `cur` (ascending to `hi`, or descending from `cur - 1`
/// to `lo`) and stops at the first match, or at the first index that needs a
/// probe: a hole (unless `skip_holes`, where a hole is an absent index), or an
/// index past the store (it shrank under an inherited getter).
#[allow(clippy::too_many_arguments)]
fn scan(
    realm: &Realm,
    elems: &[NanBox],
    cur: usize,
    lo: usize,
    hi: usize,
    desc: bool,
    kind: ArraySearch,
    target: NanBox,
    skip_holes: bool,
) -> Stop {
    let prim = target.as_handle().is_none();
    let nan = kind == ArraySearch::Includes && target.as_number().is_some_and(f64::is_nan);
    let hole_matches = kind == ArraySearch::Includes && target.is_undefined();
    let eq = |e: NanBox| {
        if prim {
            e.strict_equals(target) || (nan && e.as_number().is_some_and(f64::is_nan))
        } else {
            realm.strict_equals(e, target)
        }
    };
    let visit = |i: usize| -> Option<Stop> {
        let Some(&e) = elems.get(i) else {
            return Some(Stop::Probe(i));
        };
        if e.is_hole() {
            if !skip_holes {
                return Some(Stop::Probe(i));
            }
            return hole_matches.then_some(Stop::Match(i));
        }
        eq(e).then_some(Stop::Match(i))
    };
    if desc {
        for i in (lo..cur).rev() {
            if let Some(s) = visit(i) {
                return s;
            }
        }
    } else {
        for i in cur..hi {
            if let Some(s) = visit(i) {
                return s;
            }
        }
    }
    Stop::Done
}
