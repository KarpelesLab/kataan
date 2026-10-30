//! The heap arena **pinned in machine registers** for a generic body.
//!
//! Every inline heap access of the generic tier decodes a handle against the
//! heap arena: the slot array's base address and its length, both read from
//! the realm (`ctx -> realm -> heap`). Unpinned, each receiver walk reloads
//! them -- three dependent loads and a move, a fifth of the walk. Neither can
//! change without Rust code running: the arena grows (and moves) only in
//! `Heap::alloc`, and the compiled code's own inline allocations take a slot
//! from the free list, never growing it; and Rust code runs only behind a
//! `call`. So a body that walks keeps them in two callee-saved registers that
//! nothing else in the compiled code names, **`r12` = the arena base and `r13`
//! = its length**, valid at every instruction of the body:
//!
//! * both entries (the frame-ABI prologue, the native-call entry) save the
//!   caller's `r12`/`r13` in two frame slots and load the arena
//!   ([`X64Assembler::pin_enter`]); every other way into the body's code -- an
//!   OSR or catch entry, a slow path's resume, a loop header, an inlined
//!   callee -- is a jump from code where they already hold it;
//! * every `call` the assembler emits ([`X64Assembler::call_rax`],
//!   [`X64Assembler::g_call_mem`]) reloads them after it returns: the callee
//!   may have grown the arena. (The only other `call`s -- an inlined call's
//!   frame-materialization stub, reached through `inline_frame_hook`, and
//!   the handler-segment wrapper's -- run no Rust code, or run in a frame of
//!   their own, which does not pin.)
//! * every [`X64Assembler::frame_epilogue`] restores the caller's registers
//!   (`r12`/`r13` are callee-saved in the System V ABI; a compiled callee
//!   entered through the call sequence saves them the same way).
//!
//! The registers hold raw addresses, never `NanBox` values: nothing the
//! collector scans (it roots the register window, not machine registers). An
//! assembler that is not pinned (every other tier, a body without heap ops,
//! the handler-segment wrapper) emits exactly the unpinned sequences.

use super::X64Assembler;
use super::call_seq::r;

/// `r12` / `r13`'s register numbers.
pub(crate) const R12: u8 = 12;
pub(crate) const R13: u8 = 13;

/// Where a pinned body finds the arena and keeps its caller's registers.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PinArena {
    /// `ctx`-relative offset of the realm pointer.
    pub(crate) realm: i32,
    /// Realm-relative offsets of the arena's base pointer and length.
    pub(crate) heap_ptr: i32,
    pub(crate) heap_len: i32,
    /// `rbp`-relative frame slots saving the caller's `r12` (and, 8 bytes
    /// above, `r13`).
    pub(crate) save: i32,
}

/// Whether a generic body pins the arena: when it has an op whose fast
/// path decodes a heap handle. (Pinning costs four instructions an
/// activation and three a `call`; a body that only does arithmetic and
/// calls, such as a recursive `fib`, would pay them for nothing. Either
/// choice is sound: the unpinned sequences load the arena at each walk.)
pub(crate) fn worth_pinning(ops: &[super::GenericOp]) -> bool {
    use super::GenericOp as G;
    ops.iter().any(|o| {
        matches!(
            o,
            G::GetProp { .. }
                | G::GetMethod { .. }
                | G::SetProp { .. }
                | G::GetElem { .. }
                | G::SetElem { .. }
                | G::ArrayLen { .. }
                | G::InitProp { .. }
                | G::CellGet { .. }
                | G::CellSet { .. }
                | G::ArrGet { .. }
                | G::ArrSet { .. }
                | G::CallMethod { .. }
                | G::CallWithMethod { .. }
                | G::CallKey { .. }
                | G::InlineGuard { .. }
        )
    })
}

impl X64Assembler {
    /// Pins the arena for the body being emitted (see the module docs).
    pub(crate) fn set_pin(&mut self, pin: Option<PinArena>) {
        self.pin = pin;
    }

    /// The current pinning (to suspend it around code that runs in another
    /// frame, such as the handler-segment wrapper).
    pub(crate) fn pin(&self) -> Option<PinArena> {
        self.pin
    }

    /// Whether the arena is pinned (`r12`/`r13`) rather than in `r10`/`rdx`.
    pub(crate) fn pinned(&self) -> bool {
        self.pin.is_some()
    }

    /// After a pinned body's frame prologue (`r15` = `ctx`): saves the
    /// caller's `r12`/`r13` and loads the arena. Clobbers nothing else.
    pub(crate) fn pin_enter(&mut self) {
        if let Some(p) = self.pin {
            self.g_store(r::RBP, p.save, R12);
            self.g_store(r::RBP, p.save + 8, R13);
            self.pin_reload();
        }
    }

    /// Loads the arena into `r12`/`r13` (after a `call`: the callee may have
    /// grown it): three loads, no scratch register, the flags untouched.
    pub(crate) fn pin_reload(&mut self) {
        if let Some(p) = self.pin {
            self.g_load(R12, r::R15, p.realm);
            self.g_load(R13, R12, p.heap_len);
            self.g_load(R12, R12, p.heap_ptr);
        }
    }

    /// Before a pinned body's epilogue: restores the caller's `r12`/`r13`.
    pub(crate) fn pin_leave(&mut self) {
        if let Some(p) = self.pin {
            self.g_load(R12, r::RBP, p.save);
            self.g_load(R13, r::RBP, p.save + 8);
        }
    }

    /// Makes the arena available to a walk: nothing when pinned, else the
    /// base in `r10` and the length in `rdx` (clobbering `rax` and `r11`).
    pub(crate) fn p_arena(&mut self, realm_off: i32, l: &super::props::PropsLayout) {
        if !self.pinned() {
            self.p_load_arena(realm_off, l);
            self.mov_r10_rax();
        }
    }

    /// `cmp idx, <arena length>` (`idx` is `rax` or `rcx`).
    pub(crate) fn p_cmp_arena_len(&mut self, idx: u8) {
        debug_assert!(idx == r::RAX || idx == r::RCX);
        if self.pinned() {
            // cmp idx, r13
            self.code.extend_from_slice(&[0x4c, 0x39, 0xe8 | idx]);
        } else {
            // cmp idx, rdx
            self.code.extend_from_slice(&[0x48, 0x39, 0xd0 | idx]);
        }
    }

    /// `idx = <arena base> + idx * 72` (the usual slot stride; `idx` is `rax`
    /// or `rcx`), as two `lea`s.
    pub(crate) fn p_slot_lea72(&mut self, idx: u8) {
        debug_assert!(idx == r::RAX || idx == r::RCX);
        let i = idx & 7;
        // lea idx, [idx + idx*8]
        self.code
            .extend_from_slice(&[0x48, 0x8d, (i << 3) | 0x04, 0xc0 | (i << 3) | i]);
        // lea idx, [r12 + idx*8] (pinned) / [r10 + idx*8]
        let base = if self.pinned() { R12 & 7 } else { r::R10 & 7 };
        self.code
            .extend_from_slice(&[0x49, 0x8d, (i << 3) | 0x04, 0xc0 | (i << 3) | base]);
    }

    /// `add idx, <arena base>` (`idx` is `rax` or `rcx`).
    pub(crate) fn p_add_arena_base(&mut self, idx: u8) {
        debug_assert!(idx == r::RAX || idx == r::RCX);
        // add idx, r12 (pinned) / r10
        let base = if self.pinned() { R12 & 7 } else { r::R10 & 7 };
        self.code
            .extend_from_slice(&[0x4c, 0x01, 0xc0 | (base << 3) | idx]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pinned() -> X64Assembler {
        let mut a = X64Assembler::new();
        a.set_pin(Some(PinArena {
            realm: 0x10,
            heap_ptr: 0x20,
            heap_len: 0x28,
            save: -48,
        }));
        a
    }

    /// `mov r12, [r15+0x10]; mov r13, [r12+0x28]; mov r12, [r12+0x20]`.
    const RELOAD: [u8; 23] = [
        0x4d, 0x8b, 0xa7, 0x10, 0, 0, 0, // mov r12, [r15+0x10]
        0x4d, 0x8b, 0xac, 0x24, 0x28, 0, 0, 0, // mov r13, [r12+0x28]
        0x4d, 0x8b, 0xa4, 0x24, 0x20, 0, 0, 0, // mov r12, [r12+0x20]
    ];

    #[test]
    fn walk_encodings() {
        let mut a = X64Assembler::new();
        a.p_cmp_arena_len(r::RCX);
        a.p_slot_lea72(r::RCX);
        a.p_add_arena_base(r::RAX);
        // Unpinned: the exact bytes the walks always emitted.
        assert_eq!(
            a.code(),
            &[
                0x48, 0x39, 0xd1, // cmp rcx, rdx
                0x48, 0x8d, 0x0c, 0xc9, // lea rcx, [rcx + rcx*8]
                0x49, 0x8d, 0x0c, 0xca, // lea rcx, [r10 + rcx*8]
                0x4c, 0x01, 0xd0, // add rax, r10
            ]
        );
        let mut a = pinned();
        a.p_cmp_arena_len(r::RAX);
        a.p_slot_lea72(r::RAX);
        a.p_add_arena_base(r::RCX);
        assert_eq!(
            a.code(),
            &[
                0x4c, 0x39, 0xe8, // cmp rax, r13
                0x48, 0x8d, 0x04, 0xc0, // lea rax, [rax + rax*8]
                0x49, 0x8d, 0x04, 0xc4, // lea rax, [r12 + rax*8]
                0x4c, 0x01, 0xe1, // add rcx, r12
            ]
        );
    }

    #[test]
    fn entry_calls_and_exits_keep_the_arena() {
        let mut a = pinned();
        a.pin_enter();
        assert_eq!(
            &a.code()[..16],
            &[
                0x4c, 0x89, 0xa5, 0xd0, 0xff, 0xff, 0xff, // mov [rbp-48], r12
                0x4c, 0x89, 0xad, 0xd8, 0xff, 0xff, 0xff, // mov [rbp-40], r13
                0x4d, 0x8b,
            ]
        );
        assert_eq!(&a.code()[14..], &RELOAD);
        // Every call reloads after it.
        let mut a = pinned();
        a.call_rax();
        assert_eq!(&a.code()[..2], &[0xff, 0xd0]);
        assert_eq!(&a.code()[2..], &RELOAD);
        let mut a = pinned();
        a.g_call_mem(r::R11, 8);
        assert_eq!(&a.code()[a.code().len() - RELOAD.len()..], &RELOAD);
        // The epilogue restores the caller's.
        let mut a = pinned();
        a.frame_epilogue();
        assert_eq!(
            &a.code()[..14],
            &[
                0x4c, 0x8b, 0xa5, 0xd0, 0xff, 0xff, 0xff, // mov r12, [rbp-48]
                0x4c, 0x8b, 0xad, 0xd8, 0xff, 0xff, 0xff, // mov r13, [rbp-40]
            ]
        );
        // Unpinned: none of it.
        let mut a = X64Assembler::new();
        a.pin_enter();
        a.call_rax();
        assert_eq!(a.code(), &[0xff, 0xd0]);
    }
}
