//! `+` of a short string and an integer (`'k' + i`, `i + 'px'`), the commonest
//! concatenation a program makes: the result is written straight into the
//! inline buffer of one new small rope, with no intermediate digit string and
//! one heap lookup for the string operand, and recent results are handed out
//! again ([`StrIntMemo`]). And `+` of two strings.

use super::Realm;
use crate::cell::Cell;
use crate::heap::Handle;
use crate::nanbox::NanBox;
use crate::rope::Rope;

/// Integers below this magnitude print as plain decimal digits (2^53).
const LIMIT: f64 = 9_007_199_254_740_992.0;

/// Entries in [`StrIntMemo`] (a power of two).
const MEMO_LEN: usize = 256;

/// [`StrIntMemo::credit`] to begin with, and after a pause.
const CREDIT_START: i32 = 64;

/// What a hit adds to [`StrIntMemo::credit`], and its ceiling.
const CREDIT_HIT: i32 = 8;
const CREDIT_MAX: i32 = 4096;

/// How many `'k' + i` evaluations a memo out of credit lets by before it is
/// tried again.
const PAUSE: u32 = 4096;

/// A direct-mapped memo of recent `'k' + i` / `i + 'k'` results
/// ([`Realm::add_str_int`]): a string is a value whose identity no program
/// can observe, so the same text may be handed out again instead of being
/// built anew — a loop that makes the same few keys (`o['k' + (i & 15)]`)
/// then allocates nothing. Each entry is the string operand's handle bits
/// (`0`: empty), the integer with the operand order in its low bit, and the
/// result's handle bits.
///
/// The memo roots nothing: it is valid only while the heap's handle epoch
/// (`Heap::epoch`, which every sweep, compaction and free ends) is the one
/// it was filled under, so every handle in it still names the value it
/// named; a stale memo is emptied on its next use. It is not used while an
/// incremental marking cycle runs (handing out a value the marker may not
/// have reached).
///
/// A program whose keys never repeat (`'k' + i` over a growing `i`) would
/// pay for the lookups and stores for nothing: each miss spends one unit of
/// `credit` and each hit earns some, and a memo out of credit is left alone
/// for a while ([`PAUSE`] evaluations) before it is tried again.
pub(crate) struct StrIntMemo {
    /// The heap epoch the entries were taken under.
    epoch: u32,
    /// Lookups left before a pause (see the type's docs).
    credit: i32,
    /// Evaluations let by in the current pause.
    paused: u32,
    /// `(string operand, integer << 1 | number first, result)`.
    ents: [[u64; 3]; MEMO_LEN],
}

/// [`StrIntMemo`]'s entry for a string operand's handle bits and an encoded
/// integer.
#[inline]
fn memo_index(s: u64, key: u64) -> usize {
    let h = s.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ key.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    (h >> (64 - MEMO_LEN.trailing_zeros())) as usize
}

/// Whether `+` of two short strings may hand out a recent equal result
/// ([`Realm::add_str_str`]): `KATAAN_STR_MEMO=0` turns it off; read once per
/// process.
fn str_memo_enabled() -> bool {
    #[cfg(feature = "std")]
    {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("KATAAN_STR_MEMO").is_none_or(|v| v != "0"))
    }
    #[cfg(not(feature = "std"))]
    {
        true
    }
}

/// Whether `+` of two short strings held in their rope nodes joins them in
/// one step (`Rope::try_concat_small`): `KATAAN_SMALL_JOIN=0` turns it off;
/// read once per process.
fn small_join_enabled() -> bool {
    #[cfg(feature = "std")]
    {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("KATAAN_SMALL_JOIN").is_none_or(|v| v != "0"))
    }
    #[cfg(not(feature = "std"))]
    {
        true
    }
}

impl Realm {
    /// `a + b` for two plain strings, or a string and an integer
    /// ([`add_str_str`](Self::add_str_str), [`add_str_int`](Self::add_str_int)):
    /// the result; `None`, having done nothing, for any other operands (the
    /// general `+` then runs).
    #[inline]
    pub(crate) fn add_fast(&mut self, a: NanBox, b: NanBox) -> Option<NanBox> {
        match (a.as_handle(), b.as_handle()) {
            (Some(x), Some(y)) => match self.str_memo_hit(x, y) {
                Some(v) => Some(v),
                None => self.add_str_str(a, b),
            },
            (Some(_), None) | (None, Some(_)) => self.add_str_int(a, b),
            (None, None) => None,
        }
    }

    /// [`add_str_str`](Self::add_str_str)'s memo hit for the operands'
    /// handle bits `x`, `y` (see there), in its caller's line: a few
    /// compares and no frame. `None` (the memo untouched) for anything else.
    #[inline(always)]
    fn str_memo_hit(&mut self, x: u64, y: u64) -> Option<NanBox> {
        if self.jit_marking {
            return None;
        }
        let epoch = self.heap.epoch();
        let m = self.str_str_memo.as_deref_mut()?;
        if m.credit <= 0 || m.epoch != epoch {
            return None;
        }
        let [ms, mk, mr] = m.ents[memo_index(x, y)];
        if ms != x || mk != y || mr == 0 {
            return None;
        }
        m.credit = (m.credit + CREDIT_HIT).min(CREDIT_MAX);
        Some(NanBox::from_bits(mr))
    }

    /// `a + b` for two plain string cells (no auxiliary object, which may
    /// carry a function marker): their concatenation, in one step. `None`,
    /// having done nothing, for any other operands or a result past the
    /// maximum string length (the general `+` then runs, and throws).
    #[inline(never)]
    pub(crate) fn add_str_str(&mut self, a: NanBox, b: NanBox) -> Option<NanBox> {
        let (x, y) = (a.as_handle()?, b.as_handle()?);
        // Two short texts joined into one node: a recent join of the very
        // same two cells is handed out again (`str_str_memo`, the
        // `StrIntMemo` scheme keyed by both operands' handle bits), so a
        // loop assembling the same few short strings piece by piece
        // (`name += String.fromCharCode(c)`, each piece an interned unit
        // string and each prefix then itself a memoized result) allocates
        // nothing. Not while a string cell may carry an auxiliary object
        // (`str_special`, which empties the memo for good), nor during
        // incremental marking. Only small joins of two string cells are
        // ever entered, and within the epoch both cells are still those, so
        // a hit needs neither of them read.
        let slot = memo_index(x, y);
        let epoch = self.heap.epoch();
        let mut probed = false;
        if !self.jit_marking
            && let Some(m) = self.str_str_memo.as_deref_mut()
            && m.credit > 0
            && m.epoch == epoch
        {
            let [ms, mk, mr] = m.ents[slot];
            if ms == x && mk == y && mr != 0 {
                m.credit = (m.credit + CREDIT_HIT).min(CREDIT_MAX);
                return Some(NanBox::from_bits(mr));
            }
            probed = true;
        }
        let ra = self.heap.get_hot(Handle::from_raw(x))?.as_str()?;
        let rb = self.heap.get_hot(Handle::from_raw(y))?.as_str()?;
        if self.str_special && (self.aux_props.contains_key(&x) || self.aux_props.contains_key(&y))
        {
            return None;
        }
        if !small_join_enabled() || !ra.small_join_fits(rb) {
            let joined = ra.try_concat(rb)?;
            return Some(NanBox::handle(self.heap.alloc(Cell::Str(joined)).to_raw()));
        }
        let memo = !self.str_special && !self.jit_marking && str_memo_enabled();
        if memo && let Some(m) = self.str_str_memo.as_deref_mut() {
            if m.credit > 0 {
                // (A miss of the probe above, or a memo of an older epoch.)
                if probed || m.epoch == epoch {
                    m.credit -= 1;
                }
            } else {
                m.paused += 1;
                if m.paused >= PAUSE {
                    m.paused = 0;
                    m.credit = CREDIT_START;
                }
                return self.add_small_join(a, b);
            }
        }
        let v = self.add_small_join(a, b)?;
        if memo {
            let m = match self.str_str_memo.as_deref_mut() {
                Some(m) => m,
                None => self.str_str_memo.insert(Self::new_str_memo(epoch)),
            };
            if m.epoch != epoch {
                m.ents = [[0; 3]; MEMO_LEN];
                m.epoch = epoch;
            }
            m.ents[slot] = [x, y, v.to_bits()];
        }
        Some(v)
    }

    /// The small join of the string cells `a` and `b` (which
    /// `Rope::small_join_fits`), in a new cell.
    #[inline]
    fn add_small_join(&mut self, a: NanBox, b: NanBox) -> Option<NanBox> {
        let ra = self
            .heap
            .get_hot(Handle::from_raw(a.as_handle()?))?
            .as_str()?;
        let rb = self
            .heap
            .get_hot(Handle::from_raw(b.as_handle()?))?
            .as_str()?;
        let joined = ra.try_concat_small(rb)?;
        Some(NanBox::handle(self.heap.alloc(Cell::Str(joined)).to_raw()))
    }

    /// An empty memo for [`Realm::add_str_str`].
    #[cold]
    #[inline(never)]
    fn new_str_memo(epoch: u32) -> alloc::boxed::Box<StrIntMemo> {
        alloc::boxed::Box::new(StrIntMemo {
            epoch,
            credit: CREDIT_START,
            paused: 0,
            ents: [[0; 3]; MEMO_LEN],
        })
    }

    /// `a + b` when one operand is a string cell whose `typeof` is `"string"`
    /// and the other an integer Number of magnitude below 2^53: that new
    /// string (built in one small rope when it fits one, and then possibly a
    /// memoized earlier result, [`StrIntMemo`]). `None`, having done nothing,
    /// for any other operands or a result past the maximum string length
    /// (the general `+` then runs).
    #[inline]
    pub(crate) fn add_str_int(&mut self, a: NanBox, b: NanBox) -> Option<NanBox> {
        let (s, n, number_first) = match b.as_number() {
            Some(n) => (a, n, false),
            None => (b, a.as_number()?, true),
        };
        let sh = s.as_handle()?;
        if !(n > -LIMIT && n < LIMIT) {
            return None;
        }
        let i = n as i64;
        // (`-0` passes as `0`, which is also how it prints.)
        if i as f64 != n {
            return None;
        }
        let sbits = s.to_bits();
        let key = ((i as u64) << 1) | u64::from(number_first);
        let slot = memo_index(sbits, key);
        // Whether to look the result up (and record it).
        let mut memo = false;
        if !self.jit_marking {
            let epoch = self.heap.epoch();
            if self.str_int_memo.is_none() {
                self.str_int_memo_new();
            }
            let m = self.str_int_memo.as_deref_mut()?;
            if m.credit > 0 {
                memo = true;
                if m.epoch == epoch {
                    let [ms, mk, mr] = m.ents[slot];
                    if ms == sbits && mk == key {
                        m.credit = (m.credit + CREDIT_HIT).min(CREDIT_MAX);
                        let hit = NanBox::from_bits(mr);
                        // (A cell given an auxiliary object since is no
                        // plain string any more.)
                        if !(self.str_special
                            && hit
                                .as_handle()
                                .is_none_or(|r| self.aux_props.contains_key(&r)))
                        {
                            return Some(hit);
                        }
                    }
                }
                m.credit -= 1;
            } else {
                m.paused += 1;
                if m.paused >= PAUSE {
                    m.paused = 0;
                    m.credit = CREDIT_START;
                }
            }
        }
        let r = self.heap.get(Handle::from_raw(sh))?.as_str()?;
        let Some(rope) = r.small_with_int(i, number_first) else {
            // A longer result (`s += i` building a long string): the
            // digits, then one concatenation.
            let digits = Rope::from_int(i);
            let joined = if number_first {
                digits.try_concat(r)?
            } else {
                r.try_concat(&digits)?
            };
            if self.str_special && self.aux_props.contains_key(&sh) {
                return None;
            }
            return Some(NanBox::handle(self.heap.alloc(Cell::Str(joined)).to_raw()));
        };
        // A string cell carrying a function marker (or any other auxiliary
        // object) is left to the general path, which decides its `typeof`.
        if self.str_special && self.aux_props.contains_key(&sh) {
            return None;
        }
        let v = NanBox::handle(self.heap.alloc(Cell::Str(rope)).to_raw());
        if memo {
            let epoch = self.heap.epoch();
            if let Some(m) = self.str_int_memo.as_deref_mut() {
                if m.epoch != epoch {
                    m.ents = [[0; 3]; MEMO_LEN];
                    m.epoch = epoch;
                }
                m.ents[slot] = [sbits, key, v.to_bits()];
            }
        }
        Some(v)
    }

    /// The `'k' + i` memo, made (empty) on first use.
    #[cold]
    #[inline(never)]
    fn str_int_memo_new(&mut self) {
        self.str_int_memo = Some(alloc::boxed::Box::new(StrIntMemo {
            epoch: self.heap.epoch(),
            credit: CREDIT_START,
            paused: 0,
            ents: [[0; 3]; MEMO_LEN],
        }));
    }
}
