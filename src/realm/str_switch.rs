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

#[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
use super::text_memo::TEXT_MAX;
use super::text_memo::Text;
use super::{Cell, NanBox, Realm, rope_bytes};
use alloc::vec::Vec;
use core::cmp::Ordering;

/// A string `switch` site's memo of the discriminants it has seen (see
/// [`Realm::string_switch_index_memo`]): each way a short string's **text**
/// (its bytes as three padded words and its length, `text_memo::Text`) with
/// the answer for it, at one of two ways picked by the text's hash.
///
/// Keyed by text, not by the string's address: any String with that text
/// hits (a discriminant read from many AST nodes, each holding its own copy
/// of `"+"`, takes one way, not one per copy), and which discriminants share
/// a way depends on their texts only, never on where they were allocated. (A
/// memo keyed by the rope's address let hot discriminants that happened to
/// share ways evict each other, so a program's speed moved with heap
/// layout.) A text names its contents exactly, so a way holds no reference
/// to any string. Only flat strings of at most `text_memo::TEXT_MAX` bytes
/// are kept.
/// (Boxed: the op that holds it stays small.)
///
/// Compiled code reads the ways in place ([`SwitchMemoLayout`]): a String
/// held in its rope node (`Node::Small`, bytes past its length zero) has its
/// text's words at its buffer's start.
pub struct SwitchMemo {
    ways: alloc::boxed::Box<[SwitchWay; SWITCH_MEMO_WAYS]>,
}

/// One way of a [`SwitchMemo`]: a text and its answer as the helper returns
/// it (a Number's bits, `-1` for none). Empty while the text's length is
/// [`EMPTY_LEN`] (no text has it).
/// (Padded to a power of two, so compiled code finds a way by a shift, and
/// each way is one cache line.)
#[repr(C, align(64))]
struct SwitchWay {
    text: core::cell::Cell<Text>,
    bits: core::cell::Cell<u64>,
}

/// A [`SwitchWay`]'s length while it is empty.
const EMPTY_LEN: u32 = u32::MAX;

impl Default for SwitchMemo {
    fn default() -> Self {
        Self {
            ways: alloc::boxed::Box::new(core::array::from_fn(|_| SwitchWay {
                text: core::cell::Cell::new(Text {
                    words: [0; 3],
                    len: EMPTY_LEN,
                }),
                bits: core::cell::Cell::new(0),
            })),
        }
    }
}

/// Where compiled code finds a [`SwitchMemo`]'s ways and how it picks them:
/// a way's text words, length and answer, the way stride's log2, and the
/// hash's multiplier ([`switch_hash`]), whose top five bits are the first way
/// and the next five the second. (A text longer than any way holds,
/// `text_memo::TEXT_MAX`, matches no way's length.)
#[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
#[derive(Clone, Copy, Debug)]
pub(crate) struct SwitchMemoLayout {
    pub(crate) words: i32,
    pub(crate) len: i32,
    pub(crate) bits: i32,
    pub(crate) size_log2: u8,
    pub(crate) mul: u64,
}

/// Derives [`SwitchMemoLayout`].
#[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
#[must_use]
pub(crate) fn jit_switch_memo_layout() -> SwitchMemoLayout {
    use core::mem::{offset_of, size_of};
    let i = |v: usize| i32::try_from(v).expect("switch memo offset fits in i32");
    // (A `Cell` is its value: the way's fields are where their values are.)
    let text = offset_of!(SwitchWay, text);
    const _: () =
        assert!(size_of::<SwitchWay>().is_power_of_two() && TEXT_MAX < EMPTY_LEN as usize);
    SwitchMemoLayout {
        words: i(text + offset_of!(Text, words)),
        len: i(text + offset_of!(Text, len)),
        bits: i(offset_of!(SwitchWay, bits)),
        size_log2: size_of::<SwitchWay>().trailing_zeros() as u8,
        mul: SWITCH_HASH_MUL,
    }
}

impl SwitchMemo {
    /// The address of the first way (see [`SwitchMemoLayout`]).
    #[must_use]
    pub fn jit_ways_addr(&self) -> usize {
        self.ways.as_ptr() as usize
    }
}

/// The hash that picks a text's ways in a [`SwitchMemo`]: its first word
/// and its length, mixed (compiled code computes it the same way,
/// `crate::jit::key_probe`). Texts that differ only past their first eight
/// bytes share their ways, which costs only an eviction.
#[inline]
fn switch_hash(t: &Text) -> u64 {
    (t.words[0] ^ u64::from(t.len)).wrapping_mul(SWITCH_HASH_MUL)
}

/// [`switch_hash`]'s multiplier (Fibonacci hashing: its top bits depend on
/// every bit of the word).
const SWITCH_HASH_MUL: u64 = 0x9E37_79B9_7F4A_7C15;

/// How many discriminants a [`SwitchMemo`] remembers.
const SWITCH_MEMO_WAYS: usize = 32;

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
    /// site's `memo` when it holds `v`'s text, else looked up and noted there.
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
        let Some(t) = r.as_leaf_bytes().and_then(Text::of) else {
            return self.string_switch_index(v, table);
        };
        // (The hash's top five bits pick one way, the next five the other.)
        let hash = switch_hash(&t);
        const _: () = assert!(
            SWITCH_MEMO_WAYS == 32,
            "compiled code takes five bits a way"
        );
        let w1 = (hash >> 59) as usize % SWITCH_MEMO_WAYS;
        let w2 = (hash >> 54) as usize % SWITCH_MEMO_WAYS;
        for w in [w1, w2] {
            let way = &memo.ways[w];
            if way.text.get() == t {
                let i = NanBox::from_bits(way.bits.get()).as_number()?;
                return (i >= 0.0).then_some(i as u32);
            }
        }
        let i = self.string_switch_index(v, table);
        // (The first way if free, else the second if free, else the first
        // again.)
        let free = |w: usize| memo.ways[w].text.get().len == EMPTY_LEN;
        let w = if !free(w1) && free(w2) { w2 } else { w1 };
        let way = &memo.ways[w];
        way.text.set(t);
        way.bits
            .set(NanBox::number(i.map_or(-1.0, f64::from)).to_bits());
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
