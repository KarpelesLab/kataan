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
//! Only reads persist: a write's validation also depends on the host's
//! running scope (`global_scope_if_current`), which an activation does not
//! fix.
//!
//! The array's invariant: every non-empty entry of a group was validated
//! under (or after) the group's noted epoch. A group's entries are emptied
//! whenever its noted epoch changes, then filled.

use super::call_seq::r;
use super::{Label, X64Assembler};

/// The array's length (`u64` words).
pub(super) const LEN: usize = 64;

/// An empty entry, and the "no epoch" mark (the global binding epoch never
/// reaches it): `RECV_CACHE_EMPTY` sign-extended.
pub(super) const EMPTY: u64 = u64::MAX;

/// The words of the body's own global binding cache (`compile_generic`): its
/// epoch, then one per slot.
pub(super) const BODY_WORDS: usize = 1 + 8;

/// A body's persistent cache array (owned by its `JitProto`; the code holds
/// its address).
pub(super) struct Persist {
    words: alloc::boxed::Box<[core::cell::UnsafeCell<u64>]>,
}

impl Persist {
    /// A fresh array, every entry and epoch empty.
    pub(super) fn new() -> Self {
        Self {
            words: (0..LEN)
                .map(|_| core::cell::UnsafeCell::new(EMPTY))
                .collect(),
        }
    }

    /// The array's address (stable: the boxed slice never moves).
    pub(super) fn addr(&self) -> u64 {
        core::cell::UnsafeCell::raw_get(self.words.as_ptr()) as usize as u64
    }
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
