//! The string `switch` dispatch: `switch (d) { case "a": … case "b": … }`
//! with every case test a string literal (see `Op::SwitchStr`).
//!
//! `d === "a"` holds exactly when `d` is a string whose contents equal the
//! literal's (strict equality compares strings by their code units, which
//! for a string held as WTF-8 is its bytes — what `Realm::strict_equals`
//! compares); a literal's evaluation has no effect. So the case whose test
//! first equals `d` is the first case with `d`'s contents, found here by a
//! search over the literals sorted by length and bytes instead of one
//! comparison per case.

use super::{Cell, NanBox, Realm, rope_bytes};
use alloc::vec::Vec;
use core::cmp::Ordering;

/// The order a string switch table is sorted in: by length, then bytes.
#[must_use]
pub fn switch_key_order(a: &[u8], b: &[u8]) -> Ordering {
    a.len().cmp(&b.len()).then_with(|| a.cmp(b))
}

impl Realm {
    /// The index stored with the entry of `table` (sorted by
    /// [`switch_key_order`], keys distinct) whose key is the contents of the
    /// string `v`; `None` when `v` is not a string or no key matches.
    #[must_use]
    pub fn string_switch_index(&self, v: NanBox, table: &[(Vec<u8>, u32)]) -> Option<u32> {
        let h = v.as_handle()?;
        let r = self
            .heap
            .get(crate::heap::Handle::from_raw(h))
            .and_then(Cell::as_str)?;
        // The keys of the string's length (a run of the sorted table); no
        // need to look at the bytes (or flatten a rope) when there are none.
        let len = r.len();
        let lo = table.partition_point(|(k, _)| k.len() < len);
        let hi = lo + table[lo..].partition_point(|(k, _)| k.len() == len);
        if lo == hi {
            return None;
        }
        let bytes = rope_bytes(r);
        table[lo..hi]
            .iter()
            .find(|(k, _)| **k == *bytes)
            .map(|(_, i)| *i)
    }
}
