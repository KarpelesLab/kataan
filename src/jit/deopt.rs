//! **Deoptimization exits** for the generic tier: compiled code that leaves
//! in the middle of a body and lets the interpreter finish the activation
//! from a bytecode pc.
//!
//! # Why a body can leave anywhere
//!
//! A generic body runs on the interpreter's own register file and numbers
//! its ops as the bytecode does (`compile_generic`), so at the start of most
//! ops the interpreter's whole frame state is already in memory: the
//! registers (in the window `rbx` points at), the handler stack (the
//! activation's segment of `jit_tier::TierRuntime::handlers`, in the
//! function's own pcs) and the pc itself. Leaving there costs no value
//! translation. The exceptions are what this module's eligibility test
//! ([`points`]) rules out:
//!
//! * a promoted loop region (`numeric`): its registers live in SSE
//!   registers, and the memory file is stale until a flush (its fast
//!   versions already have an exit of their own, the plain copy);
//! * the consumer of a fused test (a `JumpIfFalse` right after a guard, a
//!   `Not` or a compare whose result the emitter keeps in the flags only);
//! * an op of a fresh object's add chain, whose straight runs the emitter
//!   makes at once;
//! * an op of the body's own bookkeeping that has no pc of the function's
//!   (a later way's guard, the jump past an inlined site's fallback);
//! * any op of a generator or an `async` body (a suspension the
//!   interpreter would make there has no compiled resumption to hand back).
//!
//! # Inlined callees need no frame of their own
//!
//! An inlined callee (`inline`, up to three levels) has no interpreter
//! frame, and making one would mean inverting the splice: renamed
//! registers, parameters read from the caller's registers, a `return`
//! turned into a move and a jump. None of that is needed. The spliced
//! bytecode is itself a program the interpreter runs correctly on the
//! window: its inlined bodies are ordinary ops on the sub-window, an
//! `InlineGuard` the interpreter runs is a miss (`false`, so the guarded
//! call op runs instead) and an `InlineLeave` does nothing. So an exit at
//! pc `X` inside `k` levels of inlined bodies runs, in the interpreter, the
//! spliced ops from `X` up to the innermost level's `InlineLeave` (the slice
//! of the program before it, which ends the run as falling off its end
//! would), then that level's leave, then the next level's ops up to its
//! leave, and so on, then the outermost site's tail (a `super(...)`'s
//! binding of `this` and the jump past the fallback call) — after which the
//! pc is the function's own again and the rest of the activation runs on
//! the function's own bytecode.
//!
//! What the native code keeps for an inlined activation besides its
//! registers is its native-call header and depth count (pushed lazily,
//! when Rust code is about to run in it: `call_seq::emit_inline_guard`).
//! The exit is a `call`, so the inline frame hook materializes every
//! pending level's header first, exactly as any call from there would;
//! the interpreter's segments then run with them in place (so `f.caller`,
//! `f.arguments` and the depth limit see what they would have), and the
//! exit pops each level's header where its `InlineLeave` would have
//! (`jit_tier::jit_helper_deopt`). A throw out of a segment ends every
//! pending level (its header popped, as the body's throw exit does) and is
//! offered to the activation's own handlers.
//!
//! # Where the rest of the activation runs
//!
//! An activation the interpreter entered (a call or an on-stack
//! replacement, `jit_tier::run_tiering`) hands its pc and handler stack
//! back to that caller, which goes on interpreting the same frame; an
//! on-stack replacement's interpreter loop simply continues there (so the
//! next hot back-edge may enter the compiled body again). An activation a
//! native call sequence entered has no interpreter loop below it: the
//! exit's helper runs the rest itself and returns its outcome as the
//! body's return value, so the native caller sees an ordinary return.
//!
//! # Using it
//!
//! A body compiled with a [`DeoptInfo`] (only under the knob below, for
//! now) can branch to an exit at any eligible op start: `compile_generic`
//! collects `(label, op)` pairs in `deopt_stubs` and emits, out of line,
//! one stub per pair that calls the helper and returns what it returns
//! (the body's fall-off path when the interpreter fell off the end). An
//! exit never rejoins compiled code, so whatever the code after it assumes
//! ("no Rust code has run since …") is untouched by having one: that is
//! what lets a slow path leave instead of rejoining, and the code after it
//! drop a re-check. (A slow path that leaves this way must take it before
//! its op has done anything: the interpreter runs the op from its start.)
//!
//! A site class that relies on exits needs a policy for exits that keep
//! happening (an exit costs the rest of the activation in the interpreter
//! until its next hot back-edge): count them per site and recompile
//! without the assumption, as speculative loops do with `no_spec`.
//!
//! # Measured value (2026-11-06)
//!
//! The first site class tried, a receiver-cache way taken as still filled
//! (`recv_cache::SitePlan::known`) outside speculative loops — a site after
//! one on the same object whose slow paths, and those of every op between,
//! would exit instead of rejoining — was measured with an unsound
//! upper-bound build that marked every such site known: `richards_like`
//! ran 3.0% fewer instructions (88 → 86 ms best-of-7 at 4x), and the other
//! benchmarks did not move; restricted to chains through property reads,
//! writes, moves and constants (the slow paths cheap to convert), 0.4%.
//! The JIT's checks after a slow path are mostly one compare of a value
//! that only Rust code changes (the receiver cache's keys, which every
//! call empties), so dropping them buys little; what an exit really
//! unlocks is keeping values in machine registers across ops and slow
//! paths (a register allocator whose exits write them back).
//!
//! `KATAAN_JIT_DEOPT_STRESS=1` (a test knob) gives every body an exit at
//! every eligible op start, taken every few such starts (a short
//! pseudo-random period), so the interpreter and the compiled code take
//! turns all through a run: the differential corpus under it is the
//! mechanism's test.

use alloc::vec::Vec;
use core::cell::Cell;

/// The bit of an exit's `at` word that says it is taken after the fallback
/// op before `at`'s pc ran (`sink::SinkPlan::exit_after`), not at the start
/// of the op at its pc.
pub(crate) const AFTER_SHIFT: u32 = 40;

use super::GenericOp;

/// Whether `KATAAN_JIT_DEOPT_STRESS` asks for an exit at every eligible op
/// start (read once per process).
pub(crate) fn stress() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("KATAAN_JIT_DEOPT_STRESS").is_some_and(|v| v != "0"))
}

/// What a body's exits tell their helper (one per body, at a stable address
/// the exits bake in; owned with the code).
pub struct DeoptInfo {
    /// The function's id in its function table (`u32::MAX` until the
    /// compiled body is recorded as the function's: `JitProto::note_function`).
    pub func: Cell<u32>,
    /// The body's own bytecode (the spliced one when it inlined calls), which
    /// lives as long as the code.
    pub body: *const crate::nbvm::Op,
    /// Its length.
    pub len: usize,
    /// Per body pc, the function's own pc (`u32::MAX` inside an inlined
    /// body); empty when the two numberings are the same.
    pub orig: Vec<u32>,
    /// Per own pc, its first body pc (where the interpreter's op starts:
    /// an inlined computed-key call's read comes before its guard, which
    /// has the call's own pc); empty with `orig`.
    pub first: Vec<u32>,
    /// The body pushes handlers of its own (its activation has a handler
    /// segment of its own; otherwise the segment in force is a caller's).
    pub handlers: bool,
    /// Under `KATAAN_JIT_DEOPT_STRESS`: the eligible op starts left until
    /// the next exit.
    pub countdown: Cell<u64>,
    /// The exits taken so far.
    pub taken: Cell<u64>,
    /// Whether the body leaves every few eligible op starts
    /// (`KATAAN_JIT_DEOPT_STRESS`); else only where its sunk objects need
    /// it (`sink`).
    pub stress: bool,
    /// The bytecode `body` points at when the body's own (the emitter's)
    /// was rewritten by `super::sink` (the interpreter runs the original).
    pub owned: Option<alloc::boxed::Box<[crate::nbvm::Op]>>,
    /// The body's sunk objects, which every exit materializes first.
    pub sink: Option<alloc::boxed::Box<super::sink::SinkExits>>,
}

impl DeoptInfo {
    /// A body's record: `body` its bytecode, `(orig, first)` its pc maps
    /// (body pc to own pc and back; empty: the function's own numbering),
    /// `handlers` whether it has `try` regions.
    pub(crate) fn new(
        body: &[crate::nbvm::Op],
        (orig, first): (Vec<u32>, Vec<u32>),
        handlers: bool,
    ) -> Self {
        Self {
            func: Cell::new(u32::MAX),
            body: body.as_ptr(),
            len: body.len(),
            orig,
            first,
            handlers,
            countdown: Cell::new(1),
            taken: Cell::new(0),
            stress: stress(),
            owned: None,
            sink: None,
        }
    }

    /// [`Self::new`] for a body with sunk objects (`sink`): the
    /// interpreter's bytecode is `orig`, the body's own before the rewrite.
    pub(crate) fn with_sink(
        orig: alloc::boxed::Box<[crate::nbvm::Op]>,
        maps: (Vec<u32>, Vec<u32>),
        handlers: bool,
        sink: alloc::boxed::Box<super::sink::SinkExits>,
    ) -> Self {
        let mut d = Self::new(&orig, maps, handlers);
        // (The box's contents do not move with it.)
        d.owned = Some(orig);
        d.sink = Some(sink);
        d
    }

    /// The body's bytecode.
    ///
    /// # Safety
    /// The body the record was made for must still be alive (it is while
    /// its code can run: both are owned by the same `JitProto`).
    #[allow(unsafe_code)]
    pub(crate) unsafe fn body(&self) -> &[crate::nbvm::Op] {
        // SAFETY: the caller's contract; `body`/`len` came from a slice.
        unsafe { core::slice::from_raw_parts(self.body, self.len) }
    }

    /// The function's own pc where body pc `pc` starts that op (`None`
    /// inside an inlined body, or past the start of the own op `pc` is
    /// part of).
    pub(crate) fn own_start(&self, pc: usize) -> Option<usize> {
        let own = self.own_pc(pc)?;
        match self.first.get(own) {
            Some(&f) if f as usize != pc => None,
            _ => Some(own),
        }
    }

    /// The function's own pc of body pc `pc` (`None` inside an inlined body).
    pub(crate) fn own_pc(&self, pc: usize) -> Option<usize> {
        if self.orig.is_empty() {
            return Some(pc);
        }
        match self.orig.get(pc) {
            Some(&p) if p != u32::MAX => Some(p as usize),
            _ => None,
        }
    }

    /// Records an exit; under [`stress`] also sets the next one's distance
    /// (1 to 13 eligible op starts, varying from exit to exit).
    pub(crate) fn note_exit(&self) {
        let n = self.taken.get().wrapping_add(1);
        self.taken.set(n);
        // (A multiplicative hash of the count: a period that does not
        // resonate with a loop's length.)
        let h = n.wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 60;
        self.countdown.set(1 + h % 13);
    }
}

/// Per op, how many inlined activations are active while it runs: `0` in
/// the function's own code, `k` inside `k` levels of inlined bodies (a
/// guard is its caller's, a leave its callee's).
pub(crate) fn levels(ops: &[GenericOp]) -> Vec<u8> {
    let mut depth = 0u8;
    ops.iter()
        .map(|op| match *op {
            GenericOp::InlineGuard { .. } => {
                let d = depth;
                depth = depth.saturating_add(1);
                d
            }
            GenericOp::InlineLeave => {
                let d = depth;
                depth = depth.saturating_sub(1);
                d
            }
            _ => depth,
        })
        .collect()
}

/// Per op, whether an exit may be taken at its start (see the module docs):
/// `in_region` tells the ops of promoted loops, `fresh` the ops of fresh
/// objects' add and literal chains, `own` the function's own pc a body pc
/// starts (`None` inside an inlined body, or inside one own op's expansion:
/// [`DeoptInfo::own_start`]), `levels` the inlining depth of each op
/// ([`levels`]). (A generator's or an `async` body gets no
/// [`DeoptInfo`], so no exits: the caller's test.)
pub(crate) fn points(
    ops: &[GenericOp],
    levels: &[u8],
    in_region: impl Fn(usize) -> bool,
    fresh: impl Fn(usize) -> bool,
    own: impl Fn(usize) -> Option<usize>,
) -> Vec<bool> {
    (0..ops.len())
        .map(|pc| {
            let level = levels.get(pc).copied().unwrap_or(0);
            !in_region(pc)
                && !fresh(pc)
                && !matches!(
                    ops[pc],
                    GenericOp::JumpIfFalse { .. } | GenericOp::Jump { .. } | GenericOp::InlineLeave
                )
                // In the function's own code, only at an op of its own.
                && (level > 0 || own(pc).is_some())
        })
        .collect()
}

/// The pc of the `InlineLeave` that ends the innermost inlined body
/// enclosing body pc `from` (`None` if there is none).
pub(crate) fn leave_of(body: &[crate::nbvm::Op], from: usize) -> Option<usize> {
    use crate::nbvm::Op;
    let mut depth = 0usize;
    for (pc, op) in body.iter().enumerate().skip(from) {
        match op {
            Op::InlineGuard { .. } => depth += 1,
            Op::InlineLeave if depth == 0 => return Some(pc),
            Op::InlineLeave => depth -= 1,
            _ => {}
        }
    }
    None
}

/// The end of an inlined site's tail, which starts at `from` (just past its
/// way's `InlineLeave`): the pc its jump past the fallback call goes to,
/// the op after the site (`inline::expand`: a `super(…)`'s binding of
/// `this`, then that jump).
pub(crate) fn site_end(body: &[crate::nbvm::Op], from: usize) -> Option<usize> {
    use crate::nbvm::Op;
    for op in body.get(from..)?.iter().take(2) {
        match op {
            Op::Jump { target } => return Some(*target),
            Op::Move { .. } => {}
            _ => return None,
        }
    }
    None
}
