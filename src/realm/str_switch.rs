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

/// A [`SwitchMemo`]'s ways: each a rope it looked up, with the answer.
type SwitchWays = [(Option<crate::rope::Rope>, u32); SWITCH_MEMO_WAYS];

/// A string `switch` site's memo of its last few discriminants (see
/// [`Realm::string_switch_index_memo`]): each the rope of a string it looked
/// up, with the answer, at one of two ways chosen by the rope's address
/// (two-choice: a direct-mapped memo let two hot discriminants that
/// happened to share a way evict each other on every lookup, so a
/// program's speed depended on where its literals were allocated). A rope is
/// immutable and the memo keeps it alive, so a string cell holding that very
/// rope (`Rope::ptr_eq`) has those contents and that answer. A program
/// dispatches on a few literal strings (each literal site's evaluations share
/// one cell), so the memo answers almost every lookup without comparing
/// bytes. Only short strings are kept.
/// (Boxed: the op that holds it stays small.)
///
/// Compiled code reads the memo through `jit`, a plain mirror of the ways:
/// way `w` holds the address of `ways[w]`'s rope (`Rope::addr`, `0` while
/// empty) and its answer as the helper returns it (a Number's bits, `-1`
/// for none), written together with the way, so an address there is always
/// that of a rope the memo keeps alive.
#[derive(Default)]
pub struct SwitchMemo {
    ways: alloc::boxed::Box<core::cell::RefCell<SwitchWays>>,
    jit: alloc::boxed::Box<[SwitchJitWay; SWITCH_MEMO_WAYS]>,
}

/// One way of [`SwitchMemo`]'s compiled-code mirror (see there).
#[repr(C)]
#[derive(Default)]
pub(super) struct SwitchJitWay {
    addr: core::cell::Cell<u64>,
    bits: core::cell::Cell<u64>,
}

/// How compiled code finds a rope's ways in a [`SwitchMemo`]: the address
/// times this, its top five bits the first way and the next five the second
/// (see [`Realm::string_switch_index_memo`]).
pub const SWITCH_MEMO_HASH: u64 = 0x9E37_79B9_7F4A_7C15;

impl SwitchMemo {
    /// The address of the compiled-code mirror's first way (see
    /// [`SwitchJitWay`]: 16 bytes each, the rope address then the answer's
    /// bits).
    #[must_use]
    pub fn jit_ways_addr(&self) -> usize {
        self.jit.as_ptr() as usize
    }
}

/// How many discriminants a [`SwitchMemo`] remembers.
const SWITCH_MEMO_WAYS: usize = 32;

/// The longest string a [`SwitchMemo`] keeps (bytes).
const SWITCH_MEMO_MAX: usize = 64;

/// "No case matched", as a memoized answer.
const SWITCH_NONE: u32 = u32::MAX;

impl Clone for SwitchMemo {
    /// An empty memo: a copy of the op (inlining, splicing) starts afresh.
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl core::fmt::Debug for SwitchMemo {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("SwitchMemo")
    }
}

/// The order a string switch table is sorted in: by length, then bytes.
#[must_use]
pub fn switch_key_order(a: &[u8], b: &[u8]) -> Ordering {
    a.len().cmp(&b.len()).then_with(|| a.cmp(b))
}

impl Realm {
    /// Whether the string `v` has the contents `lit` (WTF-8 bytes, as
    /// strict equality compares strings); `None` when `v` is not a string.
    #[must_use]
    pub fn string_equals_bytes(&self, v: NanBox, lit: &[u8]) -> Option<bool> {
        let h = v.as_handle()?;
        let r = self
            .heap
            .get(crate::heap::Handle::from_raw(h))
            .and_then(Cell::as_str)?;
        Some(r.len() == lit.len() && *rope_bytes(r) == *lit)
    }

    /// [`string_switch_index`](Self::string_switch_index), answered from the
    /// site's `memo` when it holds `v`'s rope, else looked up and noted there.
    #[must_use]
    pub fn string_switch_index_memo(
        &self,
        v: NanBox,
        table: &[(Vec<u8>, u32)],
        memo: &SwitchMemo,
    ) -> Option<u32> {
        let h = v.as_handle()?;
        let r = self
            .heap
            .get(crate::heap::Handle::from_raw(h))
            .and_then(Cell::as_str)?;
        // (Fibonacci hashing of the address: nodes allocated a fixed stride
        // apart still spread over the ways. Its top bits pick one way,
        // the next ones the other.)
        let h = (r.addr() as u64).wrapping_mul(SWITCH_MEMO_HASH);
        const _: () = assert!(
            SWITCH_MEMO_WAYS == 32,
            "compiled code takes five bits a way"
        );
        let w1 = (h >> 59) as usize % SWITCH_MEMO_WAYS;
        let w2 = (h >> 54) as usize % SWITCH_MEMO_WAYS;
        {
            let ways = memo.ways.borrow();
            for w in [w1, w2] {
                if let (Some(k), i) = &ways[w]
                    && k.ptr_eq(r)
                {
                    return (*i != SWITCH_NONE).then_some(*i);
                }
            }
        }
        let i = self.string_switch_index(v, table);
        if r.len() <= SWITCH_MEMO_MAX {
            let mut ways = memo.ways.borrow_mut();
            // (The first way if free, else the second if free, else the
            // first again.)
            let w = if ways[w1].0.is_some() && ways[w2].0.is_none() {
                w2
            } else {
                w1
            };
            ways[w] = (Some(r.clone()), i.unwrap_or(SWITCH_NONE));
            let j = &memo.jit[w];
            j.addr.set(r.addr() as u64);
            j.bits
                .set(NanBox::number(i.map_or(-1.0, f64::from)).to_bits());
        }
        i
    }

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
        if len <= 8 {
            // A short key by one word, without a `memcmp` call per candidate.
            let w = word8(&bytes);
            return table[lo..hi]
                .iter()
                .find(|(k, _)| word8(k) == w)
                .map(|(_, i)| *i);
        }
        table[lo..hi]
            .iter()
            .find(|(k, _)| **k == *bytes)
            .map(|(_, i)| *i)
    }
}

/// The bytes of a string of at most 8 bytes as one zero-padded word (two
/// such strings of one length are equal exactly when their words are).
#[inline]
fn word8(b: &[u8]) -> u64 {
    let mut w = [0u8; 8];
    for (d, s) in w.iter_mut().zip(b) {
        *d = *s;
    }
    u64::from_le_bytes(w)
}
