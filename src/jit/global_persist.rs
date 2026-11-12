//! The generic tier's **persistent global binding cache** (`ROADMAP.md` §2.1):
//! the global read addresses a compiled body validated
//! (`jit_helper_global_ref`), kept across its activations.
//!
//! A body caches a validated global binding address in its frame for as long
//! as the global binding epoch it was validated under is current (the body's
//! own cache, `compile_generic`, and a promoted loop's, `numeric`). The frame
//! is new each call, so every activation used to validate its globals afresh
//! — a helper call per global name per call, which for a small hot function
//! (`fib` reading `fib`, a step function reading its arrays and `Math`) cost
//! more than its body.
//!
//! The addresses are not the activation's: a `LoadGlobal`'s validation
//! depends only on the host's global scope and that scope's bindings (their
//! slot storage, `const`ness, initialization), and the epoch is bumped
//! whenever any of those may change — including whenever the host's global
//! scope is swapped for another (entering or leaving a realm, a module's
//! evaluation). So an address validated under epoch `E` is right in any
//! activation of the same body while `E` is current. Each body therefore owns
//! a small array ([`Persist`]) holding the epoch its entries were validated
//! under and the entries themselves: a frame whose cache finds its noted
//! epoch stale first takes the array's entries when the array's epoch is the
//! current one, and a validation writes what it found back.
//!
//! A write's validation persists the same way when its code mirrors nothing
//! onto the global object ([`writes_persist`]): `jit_helper_global_ref`
//! validates a write's address (a plain, initialized, non-`const` binding
//! with no like-named own property of the global object to mirror onto)
//! under the same epoch rule, every change of those facts bumping it. A write
//! whose code mirrors (`global_mirror`) needs its op's mirror target restored
//! by each validation, and keeps validating per activation (through its
//! per-op memo, `StoreRefMemo`).
//!
//! The array's invariant: every non-empty entry of a group was validated
//! under (or after) the group's noted epoch. A group's entries are emptied
//! whenever its noted epoch changes, then filled.

use super::call_seq::r;
use super::{Label, X64Assembler};

/// Whether a global write's validated address persists like a read's (see
/// the module docs); `KATAAN_JIT_GWPERSIST=0` turns it off (read once per
/// process).
pub(super) fn writes_persist() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("KATAAN_JIT_GWPERSIST").is_none_or(|v| v != "0"))
}

/// Whether the global binding cache's reset takes the persistent entries
/// before emptying the frame's slots, so that it empties only the others
/// (`X64Assembler::gp_restore_first`); `KATAAN_JIT_GPFIRST=0` turns it off
/// (read once per process).
pub(super) fn restore_first() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("KATAAN_JIT_GPFIRST").is_none_or(|v| v != "0"))
}

/// The array's length (`u64` words).
pub(super) const LEN: usize = 64;

/// An empty entry, and the "no epoch" mark (the global binding epoch never
/// reaches it): `RECV_CACHE_EMPTY` sign-extended.
pub(super) const EMPTY: u64 = u64::MAX;

/// The words of the body's own global binding cache (`compile_generic`): its
/// epoch, then one per slot.
pub(super) const BODY_WORDS: usize = 1 + 8;

/// The counters past the cache, each starting at zero: per region (the
/// first [`COUNTERS`] of a body), how often a speculative region's integer
/// version gave up (`LEN + region`), and how many times in a row an
/// integer version's entry found a loop-carried value that is not an int32
/// (`LEN + COUNTERS + region`): see `numeric::int32`; and how often (lately:
/// halved at each of its back-edge polls) a speculative region's fast
/// version left for its plain copy (`LEN + 2 * COUNTERS + region`): see
/// `numeric::Spec`; and which of a region's integer version's overflow
/// bails have been taken (`LEN + 3 * COUNTERS + region`, a bit per site):
/// see `numeric::int32`'s demotion.
pub(super) const COUNTERS: usize = 16;

/// The word past the counters: the native-call entry's countdown to its
/// next recompilation poll (`tierup::ENTRY_POLL`), in a body that polls
/// there. It starts at [`ENTRY_FIRST`].
pub(super) const ENTRY_POLL: usize = LEN + 4 * COUNTERS;

/// The native-call entry's first countdown (see [`ENTRY_POLL`]).
pub(super) const ENTRY_FIRST: u64 = 64;

/// Byte offset of region `ri`'s bail counter.
pub(super) fn counter_off(ri: usize) -> i32 {
    off(LEN + ri)
}

/// Byte offset of region `ri`'s entry-failure counter.
pub(super) fn entry_off(ri: usize) -> i32 {
    off(LEN + COUNTERS + ri)
}

/// Byte offset of region `ri`'s speculative-leave counter.
pub(super) fn leave_off(ri: usize) -> i32 {
    off(LEN + 2 * COUNTERS + ri)
}

/// Byte offset of region `ri`'s overflow-bail mask.
pub(super) fn demote_off(ri: usize) -> i32 {
    off(LEN + 3 * COUNTERS + ri)
}

/// A body's persistent cache array (owned by its `JitProto`; the code holds
/// its address).
pub(super) struct Persist {
    words: alloc::boxed::Box<[core::cell::UnsafeCell<u64>]>,
}

impl Persist {
    /// A fresh array, every entry and epoch empty, with `extra` zeroed words
    /// past [`ENTRY_POLL`] for other memos ([`extra_off`]).
    pub(super) fn with_extra(extra: usize) -> Self {
        Self {
            words: (0..=ENTRY_POLL + extra)
                .map(|w| {
                    core::cell::UnsafeCell::new(match w {
                        _ if w < LEN => EMPTY,
                        ENTRY_POLL => ENTRY_FIRST,
                        _ => 0,
                    })
                })
                .collect(),
        }
    }

    /// Region `ri`'s speculative-leave count ([`leave_off`]).
    pub(super) fn leaves(&self, ri: usize) -> u64 {
        if ri >= COUNTERS {
            return 0;
        }
        self.word(LEN + 2 * COUNTERS + ri)
    }

    /// Region `ri`'s overflow-bail mask ([`demote_off`]).
    pub(super) fn demotes(&self, ri: usize) -> u64 {
        if ri >= COUNTERS {
            return 0;
        }
        self.word(LEN + 3 * COUNTERS + ri)
    }

    /// The address of the native-call entry's poll countdown
    /// ([`ENTRY_POLL`]).
    pub(super) fn entry_poll_addr(&self) -> u64 {
        self.addr() + (ENTRY_POLL * 8) as u64
    }

    /// Sets the native-call entry's poll countdown ([`ENTRY_POLL`]).
    pub(super) fn set_entry_poll(&self, v: u64) {
        if let Some(w) = self.words.get(ENTRY_POLL) {
            // SAFETY: the body's code writes the word only while it runs on
            // this thread, never during this call.
            #[allow(unsafe_code)]
            unsafe {
                *w.get() = v;
            }
        }
    }

    fn word(&self, i: usize) -> u64 {
        self.words.get(i).map_or(0, |w| {
            // SAFETY: the compiled code writes the word only while it runs,
            // on this thread, which is not running it now (a back-edge poll
            // or a call out of it reads it here).
            #[allow(unsafe_code)]
            unsafe {
                *w.get()
            }
        })
    }

    /// The array's address (stable: the boxed slice never moves).
    pub(super) fn addr(&self) -> u64 {
        core::cell::UnsafeCell::raw_get(self.words.as_ptr()) as usize as u64
    }
}

/// Byte offset of the extra word `k` ([`Persist::with_extra`]).
pub(super) fn extra_off(k: usize) -> i32 {
    off(ENTRY_POLL + 1 + k)
}

/// Byte offset of word `w`.
pub(super) fn off(w: usize) -> i32 {
    i32::try_from(w * 8).unwrap_or(i32::MAX)
}

impl X64Assembler {
    /// With `rax` the current global binding epoch: when the persistent group
    /// noted at word `noted` was validated under it, copies its entries
    /// (`(word, frame displacement)`) into the frame and jumps to `hit`; else
    /// falls through. Clobbers `rcx`, `r11`.
    pub(super) fn gp_restore(
        &mut self,
        persist: u64,
        noted: usize,
        entries: &[(usize, i32)],
        hit: Label,
    ) {
        let miss = self.new_label();
        self.g_movabs(r::R11, persist);
        self.g_cmp_mem(r::RAX, r::R11, off(noted));
        self.g_jcc(super::call_seq::cc::NE, miss);
        for &(w, d) in entries {
            self.g_load(r::RCX, r::R11, off(w));
            self.g_store(r::RBP, d, r::RCX);
        }
        self.jmp(hit);
        self.bind(miss);
    }

    /// The global binding cache's reset when the epoch changed, with `rax`
    /// the current epoch: when the persistent group noted at word `noted`
    /// was validated under it, copies its entries (`(word, frame
    /// displacement)`) into the frame, empties the frame's other
    /// entry-cleared slots, notes the epoch at `[rbp + frame_epoch]` and
    /// jumps to `hit`; else falls through (to the full reset). Clobbers
    /// `rcx`, `r11`.
    pub(super) fn gp_restore_first(
        &mut self,
        persist: u64,
        noted: usize,
        entries: &[(usize, i32)],
        frame_epoch: i32,
        hit: Label,
    ) {
        let miss = self.new_label();
        self.g_movabs(r::R11, persist);
        self.g_cmp_mem(r::RAX, r::R11, off(noted));
        self.g_jcc(super::call_seq::cc::NE, miss);
        for &(w, d) in entries {
            self.g_load(r::RCX, r::R11, off(w));
            self.g_store(r::RBP, d, r::RCX);
        }
        let mut keep: alloc::vec::Vec<i32> = entries.iter().map(|&(_, d)| d).collect();
        keep.push(frame_epoch);
        self.clear_entry_slots_except(&keep);
        self.g_store(r::RBP, frame_epoch, r::RAX);
        self.jmp(hit);
        self.bind(miss);
    }

    /// Records entry word `word` = `rax` (an address just validated) in the
    /// persistent group noted at `noted`, under the epoch in frame slot
    /// `[rbp + frame_epoch]` (the one the frame's entry is held under): a
    /// group noted under another epoch is emptied (its words `group`) and
    /// noted under this one first. Preserves `rax`; clobbers `rcx`, `r11`.
    pub(super) fn gp_save_one(
        &mut self,
        persist: u64,
        noted: usize,
        group: &[usize],
        word: usize,
        frame_epoch: i32,
    ) {
        let same = self.new_label();
        self.g_load(r::RCX, r::RBP, frame_epoch);
        self.g_movabs(r::R11, persist);
        self.g_cmp_mem(r::RCX, r::R11, off(noted));
        self.g_jcc(super::call_seq::cc::E, same);
        for &w in group {
            self.g_store_imm32(r::R11, off(w), -1);
        }
        self.g_store(r::R11, off(noted), r::RCX);
        self.bind(same);
        self.g_store(r::R11, off(word), r::RAX);
    }

    /// Records a whole validated group: its entries (`(word, frame
    /// displacement)`) from the frame, then its epoch from frame slot
    /// `[rbp + frame_epoch]`. Clobbers `rcx`, `r11`.
    pub(super) fn gp_save_all(
        &mut self,
        persist: u64,
        noted: usize,
        entries: &[(usize, i32)],
        frame_epoch: i32,
    ) {
        self.g_movabs(r::R11, persist);
        for &(w, d) in entries {
            self.g_load(r::RCX, r::RBP, d);
            self.g_store(r::R11, off(w), r::RCX);
        }
        self.g_load(r::RCX, r::RBP, frame_epoch);
        self.g_store(r::R11, off(noted), r::RCX);
    }
}
