//! Numeric code quality for the generic tier (`ROADMAP.md` §2.1): a static
//! register **type inference** over the bytecode, inline type-specialized
//! arithmetic that uses it, and **loop register promotion** (keeping a numeric
//! loop's registers in SSE registers across iterations).
//!
//! # Type inference
//!
//! [`infer`] runs a forward dataflow analysis over a function's bytecode and
//! records, for every pc, what each register is known to hold on entry to that
//! op: nothing yet ([`T_BOT`]), a Number ([`T_NUM`]), a boolean ([`T_BOOL`]) or
//! anything ([`T_ANY`]). The facts are about *JavaScript semantics*, not about
//! the native code: `x * 3` is a Number whatever `x` is (a BigInt operand mixed
//! with a Number throws), `a | 0` likewise, `i < n` is a boolean, a numeric
//! constant is a Number. Because the interpreter and the compiled body run the
//! same ops on the same register file, the facts hold for an activation that
//! started in the interpreter and entered the native code mid-function (on-stack
//! replacement at a loop header): it reached that pc along real control flow
//! from the function's start, where every register is assumed [`T_ANY`]. A
//! `catch` entry is assumed [`T_ANY`] throughout. An op the analysis does not
//! model makes every register it names [`T_ANY`].
//!
//! An operand proven a Number needs no type guard, so arithmetic over proven
//! operands compiles to straight SSE code with no slow path at all.
//!
//! # Loop register promotion
//!
//! An innermost loop whose body is made only of ops this module emits (numeric
//! ops, moves, constants and branches) is a *region*: up to 14 of the
//! registers it names live in `xmm2..=xmm15` (raw `NanBox` bits; a non-number
//! simply rides along) from the loop header to every exit. The register file in
//! memory is brought up to date ("flushed") before every call and on every
//! exit, and the SSE registers are reloaded after every call (they are
//! caller-saved), so the interpreter, the runtime helpers and the collector
//! always see the true values; calls only happen on slow paths and in the
//! back-edge poll, so a loop that stays on its fast paths runs with no memory
//! traffic for its promoted registers.
//!
//! A call-free **loop nest** of such generic loops is one region too
//! ([`plan_nests`]): the outer loop's registers stay in their SSE registers
//! across the inner loops (a register live into an inner header holds its
//! register over that whole loop), the inner loops' entries and exits are
//! plain branches, and their back-edges poll like the outer one. An entry at
//! an inner header (on-stack replacement) loads the registers live there;
//! in a speculative nest so does the plain copy's inner back-edge, so a
//! leave for the copy inside an inner loop costs the rest of that inner
//! iteration, not of the whole outer one.
//!
//! In a speculative region's fast version, the NaN check a guarded op makes
//! (an operand not proven a Number shows as a NaN result) is **merged** into
//! a later op's check when what it computes only flows on by arithmetic to
//! it ([`plan_defer`]). A NaN there is told apart out of line: when every
//! operand the merged ops did not prove is a Number, it is a Number's NaN
//! and the fast version goes on; else it leaves for the plain copy at the
//! start of that side-effect-free stretch, which runs it again exactly.
//!
//! A value the native code produces is bit-identical to the interpreter's
//! `NanBox`. For `+ - * /` and `sqrt` over Numbers that takes no NaN fix-up:
//! the canonical NaN is the hardware's default NaN, which an invalid
//! operation makes and an operand NaN (only ever the canonical one) carries
//! through unchanged. A sign-bit operation (negation, `Math.abs`) does
//! rewrite a NaN result to the canonical one.

use super::{GenericOp, Label, X64Assembler};
use crate::nanbox::NanBox;

mod int32;
use crate::nbvm::Op;
use alloc::vec;
use alloc::vec::Vec;

/// Nothing is known to reach this point yet (the analysis' initial state).
pub(super) const T_BOT: u8 = 0;
/// A Number (any `f64`, NaN included).
pub(super) const T_NUM: u8 = 1;
/// A boolean (`true`/`false`).
pub(super) const T_BOOL: u8 = 2;
/// Anything.
pub(super) const T_ANY: u8 = 3;

/// The largest `ops × registers` state table [`infer`] builds; a bigger function
/// gets no facts (every register [`T_ANY`]), which is always sound.
const MAX_STATE: usize = 1 << 22;

/// `is_number(bits) == (bits & QNAN) != QNAN`.
const QNAN: u64 = 0x7ffc_0000_0000_0000;
/// The x86 "integer indefinite" a `cvttsd2si` of NaN / ±∞ / out-of-range yields.
const INDEFINITE: u64 = 0x8000_0000_0000_0000;
/// The `f64` sign bit.
const SIGN: u64 = 0x8000_0000_0000_0000;
/// The back-edge poll countdown frame slot (`[rbp-24]`, see `compile_generic`).
const COUNTDOWN: i8 = -24;
/// The first SSE register promotion uses (`xmm0`/`xmm1` are scratch).
const FIRST_XMM: u8 = 2;
/// How many registers a region may promote (`xmm2..=xmm15`).
const MAX_PROMOTED: usize = 14;

/// x86 condition codes (the low nibble of `Jcc`/`SETcc`).
const CC_B: u8 = 0x2;
const CC_AE: u8 = 0x3;
const CC_E: u8 = 0x4;
const CC_NE: u8 = 0x5;
const CC_BE: u8 = 0x6;
const CC_A: u8 = 0x7;
const CC_P: u8 = 0xa;

/// GPR numbers used by the encoders.
const RAX: u8 = 0;
const RCX: u8 = 1;
const RDX: u8 = 2;

/// The inferred register types: the in-state of every pc.
pub(super) struct Types {
    n_regs: usize,
    /// `st[pc * n_regs + r]`; empty when the analysis was skipped.
    st: Vec<u8>,
}

impl Types {
    /// What register `r` holds on entry to op `pc`.
    pub(super) fn get(&self, pc: usize, r: u16) -> u8 {
        if self.st.is_empty() {
            return T_ANY;
        }
        self.st
            .get(pc * self.n_regs + r as usize)
            .copied()
            .unwrap_or(T_ANY)
    }
}

fn join(a: u8, b: u8) -> u8 {
    if a == b || b == T_BOT {
        a
    } else if a == T_BOT {
        b
    } else {
        T_ANY
    }
}

/// Every register `op` names (read or written), from the verifier's own operand
/// enumeration.
fn op_regs(op: &Op) -> Vec<u16> {
    let out = core::cell::RefCell::new(Vec::new());
    crate::bytecode::op_for_each_reg(op, &|r| out.borrow_mut().push(r));
    out.into_inner()
}

/// Applies `op`'s effect on register types to `s`.
fn transfer(op: &Op, s: &mut [u8]) {
    use crate::nbvm::{VB_BIT_AND, VB_LOOSE_EQ, VB_LOOSE_NEQ, VB_SHR, VB_USHR};
    let t = |s: &[u8], r: u16| s.get(r as usize).copied().unwrap_or(T_ANY);
    // `num_if(c)`: a Number when `c`, else anything.
    let num_if = |c: bool| if c { T_NUM } else { T_ANY };
    let (dst, v) = match op {
        Op::LoadConst { dst, value } => {
            let v = if value.is_number() {
                T_NUM
            } else if value.as_boolean().is_some() {
                T_BOOL
            } else {
                T_ANY
            };
            (*dst, v)
        }
        Op::Move { dst, src } => (*dst, t(s, *src)),
        // ToNumeric(x) <op> ToNumeric(y): a Number unless both are BigInts (a
        // BigInt mixed with a Number throws).
        Op::Sub { dst, a, b }
        | Op::Mul { dst, a, b }
        | Op::Div { dst, a, b }
        | Op::Mod { dst, a, b } => (*dst, num_if(t(s, *a) == T_NUM || t(s, *b) == T_NUM)),
        Op::ValueBin { dst, op, a, b } => {
            let v = match *op {
                // `>>>` has no BigInt form: a Number or a throw.
                VB_USHR => T_NUM,
                VB_LOOSE_EQ | VB_LOOSE_NEQ => T_BOOL,
                o if (VB_BIT_AND..=VB_SHR).contains(&o) || o == 0 => {
                    num_if(t(s, *a) == T_NUM || t(s, *b) == T_NUM)
                }
                _ => T_ANY,
            };
            (*dst, v)
        }
        // `+` is a Number only over two Numbers (a string operand concatenates).
        Op::Add { dst, a, b } | Op::AddValue { dst, a, b } => {
            (*dst, num_if(t(s, *a) == T_NUM && t(s, *b) == T_NUM))
        }
        Op::Lt { dst, .. }
        | Op::StrictEq { dst, .. }
        | Op::Not { dst, .. }
        | Op::PlainPrims { dst, .. } => (*dst, T_BOOL),
        Op::Compare { dst, op, .. } => (*dst, if *op <= 3 { T_BOOL } else { T_ANY }),
        Op::ToNumeric { dst, src } | Op::Step { dst, src, .. } => {
            (*dst, num_if(t(s, *src) == T_NUM))
        }
        Op::Neg { dst, a } | Op::BitNot { dst, a } => (*dst, num_if(t(s, *a) == T_NUM)),
        // A step writes only booleans to `done` (or leaves it as it was).
        Op::IterNext {
            dst, done, iter, ..
        } => {
            let d = if t(s, *done) == T_BOOL { T_BOOL } else { T_ANY };
            for r in [*dst, *iter] {
                if let Some(x) = s.get_mut(r as usize) {
                    *x = T_ANY;
                }
            }
            (*done, d)
        }
        // Ops that write their `dst` only.
        Op::GetProp { dst, .. }
        | Op::GetMethod { dst, .. }
        | Op::GetKey { dst, .. }
        | Op::GetElem { dst, .. }
        | Op::ArrayLen { dst, .. }
        | Op::CallNative { dst, .. }
        | Op::Call { dst, .. }
        | Op::CallMethod { dst, .. }
        | Op::CallWithMethod { dst, .. }
        | Op::LoadGlobal { dst, .. }
        | Op::TypeOf { dst, .. } => (*dst, T_ANY),
        // Ops that write no register (a `PushHandler`'s register is bound at its
        // catch entry, which the analysis assumes holds anything).
        // An inlined call's guard: a boolean, and the callee's captures.
        Op::InlineGuard {
            dst, caps, n_caps, ..
        } => {
            for i in 0..*n_caps {
                if let Some(x) = s.get_mut(caps.wrapping_add(i) as usize) {
                    *x = T_ANY;
                }
            }
            (*dst, T_BOOL)
        }
        Op::SetProp { .. }
        | Op::SetPropStrict { .. }
        | Op::SetKey { .. }
        | Op::SetKeyStrict { .. }
        | Op::SetElem { .. }
        | Op::StoreGlobal { .. }
        | Op::InitProp { .. }
        | Op::InitGlobal { .. }
        | Op::CheckTdz { .. }
        | Op::RequireObjectCoercible { .. }
        | Op::PushHandler { .. }
        | Op::PopHandler
        | Op::InlineLeave
        | Op::Jump { .. }
        | Op::JumpIfFalse { .. }
        | Op::Return { .. }
        | Op::Throw { .. } => return,
        _ => {
            for r in op_regs(op) {
                if let Some(x) = s.get_mut(r as usize) {
                    *x = T_ANY;
                }
            }
            return;
        }
    };
    if let Some(x) = s.get_mut(dst as usize) {
        *x = v;
    }
}

/// The successor pcs of op `pc` (a catch entry is seeded separately).
fn successors(op: &Op, pc: usize, out: &mut Vec<usize>) {
    out.clear();
    match op {
        Op::Jump { target } => out.push(*target),
        Op::JumpIfFalse { target, .. } => {
            out.push(pc + 1);
            out.push(*target);
        }
        Op::Return { .. }
        | Op::Throw { .. }
        | Op::ThrowTypeError { .. }
        | Op::ThrowReferenceError { .. } => {}
        _ => out.push(pc + 1),
    }
}

/// The integrality facts ([`infer_integral`]): a register is [`I_INT`] at a
/// pc when it surely holds a Number whose value has no fractional part (an
/// integer, `-0`, NaN or ±∞), [`T_ANY`] otherwise.
pub(super) const I_INT: u8 = 1;

/// Calls `f` with every register `op` may write (as [`transfer`] models
/// them: an op it does not list may write any register it names).
fn for_each_write(op: &Op, f: &mut dyn FnMut(u16)) {
    match op {
        Op::LoadConst { dst, .. }
        | Op::Move { dst, .. }
        | Op::Sub { dst, .. }
        | Op::Mul { dst, .. }
        | Op::Div { dst, .. }
        | Op::Mod { dst, .. }
        | Op::ValueBin { dst, .. }
        | Op::Add { dst, .. }
        | Op::AddValue { dst, .. }
        | Op::Lt { dst, .. }
        | Op::StrictEq { dst, .. }
        | Op::Not { dst, .. }
        | Op::PlainPrims { dst, .. }
        | Op::Compare { dst, .. }
        | Op::ToNumeric { dst, .. }
        | Op::Step { dst, .. }
        | Op::Neg { dst, .. }
        | Op::BitNot { dst, .. }
        | Op::GetProp { dst, .. }
        | Op::GetMethod { dst, .. }
        | Op::GetKey { dst, .. }
        | Op::GetElem { dst, .. }
        | Op::ArrayLen { dst, .. }
        | Op::CallNative { dst, .. }
        | Op::Call { dst, .. }
        | Op::CallMethod { dst, .. }
        | Op::CallWithMethod { dst, .. }
        | Op::LoadGlobal { dst, .. }
        | Op::TypeOf { dst, .. } => f(*dst),
        Op::IterNext {
            dst, done, iter, ..
        } => {
            f(*dst);
            f(*done);
            f(*iter);
        }
        Op::InlineGuard {
            dst, caps, n_caps, ..
        } => {
            f(*dst);
            for i in 0..*n_caps {
                f(caps.wrapping_add(i));
            }
        }
        Op::SetProp { .. }
        | Op::SetPropStrict { .. }
        | Op::SetKey { .. }
        | Op::SetKeyStrict { .. }
        | Op::SetElem { .. }
        | Op::StoreGlobal { .. }
        | Op::InitProp { .. }
        | Op::InitGlobal { .. }
        | Op::CheckTdz { .. }
        | Op::RequireObjectCoercible { .. }
        | Op::PushHandler { .. }
        | Op::PopHandler
        | Op::InlineLeave
        | Op::Jump { .. }
        | Op::JumpIfFalse { .. }
        | Op::Return { .. }
        | Op::Throw { .. } => {}
        _ => {
            for r in op_regs(op) {
                f(r);
            }
        }
    }
}

/// Infers the register types of `ops` (see the module docs).
pub(super) fn infer(ops: &[Op], n_regs: usize) -> Types {
    infer_with(ops, n_regs, &transfer)
}

/// The integrality facts of `ops` ([`I_INT`]), for the registers that may
/// reach an element read's index (the only consumer): those indices, and
/// the operands of the ops below that write a tracked register.
pub(super) struct Ints {
    /// Per register: its column, or `u32::MAX` (untracked: [`T_ANY`]).
    col: Vec<u32>,
    facts: Types,
}

impl Ints {
    /// The fact of register `r` on entry to op `pc`.
    pub(super) fn get(&self, pc: usize, r: u16) -> u8 {
        match self.col.get(r as usize) {
            Some(&c) if c != u32::MAX => self.facts.get(pc, c as u16),
            _ => T_ANY,
        }
    }
}

/// Infers [`Ints`]: an integral constant, a copy, `+` / `-` / `*` / `%` /
/// unary `-` / `++` / `--` over integral Numbers (whose result is an
/// integer, or NaN / ±∞ past the exact range), and a bitwise op with an
/// integral operand (a Number, so no BigInt result: mixing one in throws)
/// are integral; anything else written is not.
pub(super) fn infer_integral(ops: &[Op], n_regs: usize) -> Ints {
    use crate::nbvm::{VB_BIT_AND, VB_USHR};
    // The int-producing ops' operands: (dst, operands).
    let feeds = |op: &Op| -> Option<(u16, [Option<u16>; 2])> {
        match *op {
            Op::Move { dst, src }
            | Op::ToNumeric { dst, src }
            | Op::Step { dst, src, .. }
            | Op::Neg { dst, a: src } => Some((dst, [Some(src), None])),
            Op::Add { dst, a, b }
            | Op::AddValue { dst, a, b }
            | Op::Sub { dst, a, b }
            | Op::Mul { dst, a, b }
            | Op::Mod { dst, a, b } => Some((dst, [Some(a), Some(b)])),
            Op::ValueBin { dst, op, a, b } if (VB_BIT_AND..=VB_USHR).contains(&op) => {
                Some((dst, [Some(a), Some(b)]))
            }
            _ => None,
        }
    };
    let mut col = vec![u32::MAX; n_regs];
    let mut k = 0u32;
    let track = |r: u16, col: &mut Vec<u32>, k: &mut u32| -> bool {
        match col.get_mut(r as usize) {
            Some(c) if *c == u32::MAX => {
                *c = *k;
                *k += 1;
                true
            }
            _ => false,
        }
    };
    for op in ops {
        if let Op::GetKey { key, .. } = op {
            track(*key, &mut col, &mut k);
        }
    }
    let mut changed = k > 0;
    while changed {
        changed = false;
        for op in ops {
            if let Some((d, srcs)) = feeds(op)
                && col.get(d as usize).is_some_and(|&c| c != u32::MAX)
            {
                for r in srcs.into_iter().flatten() {
                    changed |= track(r, &mut col, &mut k);
                }
            }
        }
    }
    if k == 0 {
        return Ints {
            col,
            facts: Types {
                n_regs: 0,
                st: Vec::new(),
            },
        };
    }
    let c = |r: u16| {
        col.get(r as usize)
            .copied()
            .filter(|&c| c != u32::MAX)
            .map(|c| c as usize)
    };
    let step = |op: &Op, s: &mut [u8]| {
        let t = |s: &[u8], r: u16| c(r).map_or(T_ANY, |i| s[i]);
        let int_if = |b: bool| if b { I_INT } else { T_ANY };
        let v = match op {
            Op::LoadConst { value, .. } => Some(int_if(
                value
                    .as_number()
                    .is_some_and(|n| n.is_nan() || n == n.trunc()),
            )),
            Op::Move { src, .. } => Some(t(s, *src)),
            Op::ValueBin { op: vop, a, b, .. } if (VB_BIT_AND..=VB_USHR).contains(vop) => Some(
                int_if(*vop == VB_USHR || t(s, *a) == I_INT || t(s, *b) == I_INT),
            ),
            _ => match feeds(op) {
                Some((_, [Some(a), Some(b)])) => Some(int_if(t(s, a) == I_INT && t(s, b) == I_INT)),
                Some((_, [Some(a), None])) => Some(int_if(t(s, a) == I_INT)),
                _ => None,
            },
        };
        let dst = match op {
            Op::LoadConst { dst, .. } => Some(*dst),
            _ => feeds(op).map(|f| f.0),
        };
        match (dst, v) {
            (Some(d), Some(v)) => {
                if let Some(i) = c(d) {
                    s[i] = v;
                }
            }
            _ => for_each_write(op, &mut |r| {
                if let Some(i) = c(r) {
                    s[i] = T_ANY;
                }
            }),
        }
    };
    let facts = infer_with(ops, k as usize, &step);
    Ints { col, facts }
}

/// The forward analysis [`infer`] and [`infer_integral`] run, over the
/// `transfer` function of their lattice ([`T_BOT`] below every fact,
/// [`T_ANY`] above).
fn infer_with(ops: &[Op], n_regs: usize, transfer: &dyn Fn(&Op, &mut [u8])) -> Types {
    let n = ops.len();
    if n == 0 || n_regs == 0 || (n + 1).saturating_mul(n_regs) > MAX_STATE {
        return Types {
            n_regs,
            st: Vec::new(),
        };
    }
    let mut st = vec![T_BOT; (n + 1) * n_regs];
    let mut queued = vec![false; n + 1];
    let mut work: Vec<usize> = Vec::new();
    let seed = |st: &mut [u8], pc: usize, work: &mut Vec<usize>, queued: &mut [bool]| {
        if pc < n {
            st[pc * n_regs..(pc + 1) * n_regs].fill(T_ANY);
            if !queued[pc] {
                queued[pc] = true;
                work.push(pc);
            }
        }
    };
    seed(&mut st, 0, &mut work, &mut queued);
    for op in ops {
        if let Op::PushHandler { target, .. } = op {
            seed(&mut st, *target, &mut work, &mut queued);
        }
    }
    let mut cur = vec![T_BOT; n_regs];
    let mut succ = Vec::with_capacity(2);
    while let Some(pc) = work.pop() {
        queued[pc] = false;
        cur.copy_from_slice(&st[pc * n_regs..(pc + 1) * n_regs]);
        transfer(&ops[pc], &mut cur);
        successors(&ops[pc], pc, &mut succ);
        for &s in &succ {
            if s >= n {
                continue;
            }
            let row = &mut st[s * n_regs..(s + 1) * n_regs];
            let mut changed = false;
            for (x, &c) in row.iter_mut().zip(cur.iter()) {
                let j = join(*x, c);
                if j != *x {
                    *x = j;
                    changed = true;
                }
            }
            if changed && !queued[s] {
                queued[s] = true;
                work.push(s);
            }
        }
    }
    Types { n_regs, st }
}

/// One promoted register of a region: register `reg` lives in `xmm` at pcs
/// `start..=end` (the whole region for a loop-carried register, else its live
/// interval within the body).
struct Slot {
    reg: u16,
    xmm: u8,
    start: usize,
    end: usize,
    /// Live on entry to the header: loaded there.
    carried: bool,
    /// Written by some op of the region (only such a slot is ever stored back).
    dirty: bool,
}

/// A promoted loop: bytecode pcs `head..=end` (`end` the unconditional back-edge
/// `Jump` to `head`) and its promoted registers.
struct Region {
    head: usize,
    end: usize,
    slots: Vec<Slot>,
    /// Bound just past the header's loads: where in-region branches to the
    /// header go.
    inner: Option<Label>,
    /// The region's integer version ([`int32`]), if planned.
    int: Option<int32::IntPlan>,
    /// Whether it contains ops the generic emitter emits
    /// ([`generic_region_op`]).
    generic: bool,
    /// The registers live on entry to the header (planned for a `generic`
    /// region only).
    live_head: Vec<u16>,
    /// Its speculative double version, if planned ([`plan_spec`]).
    spec: Option<Spec>,
    /// The property reads the speculative version replaces by a copy of the
    /// register already holding the value ([`plan_reuse`]): `(pc, src)`.
    reuse: Vec<(usize, u16)>,
    /// A loop nest's inner back-edges, as `(header, back-edge)` ([`plan_nests`]).
    loops: Vec<(usize, usize)>,
    /// Per inner header of a generic nest: the registers live on entry to
    /// it (its entry from the plain copy, [`NumGen::emit_inner_entries`]).
    live_inner: Vec<(usize, Vec<u16>)>,
    /// The speculative version's merged NaN checks ([`plan_defer`]).
    defer: DeferPlan,
    /// The speculative version's **hoisted typed-array views**
    /// ([`plan_views`]): `(register, first of its frame slots, where its
    /// entry check finds the value)`.
    views: Vec<(u16, u32, ViewSrc)>,
    /// The region's **guard diamonds** ([`guard_diamonds`]), as `(test,
    /// fallback, join)`: in the speculative version's fast version the
    /// guard's test (`test`, a `JumpIfFalse`) emits nothing and the
    /// fallback arm `fallback..join` is not there at all.
    cold: Vec<(usize, usize, usize)>,
    /// For a body that branches inside itself: the registers live on entry
    /// to each of its ops (`live_words` words per pc from `head`), which
    /// [`Region::defined`] reads; empty otherwise.
    live: Vec<u64>,
    live_words: usize,
}

impl Region {
    /// Whether `pc` lies in a guard diamond's fallback arm ([`Region::cold`]).
    fn is_cold(&self, pc: usize) -> bool {
        self.cold.iter().any(|&(_, f, e)| f <= pc && pc < e)
    }

    /// Whether register `r` is live on entry to region op `pc` (`true`
    /// when not recorded).
    fn live_at(&self, pc: usize, r: u16) -> bool {
        if self.live_words == 0 || pc < self.head {
            return true;
        }
        let w = (pc - self.head) * self.live_words + r as usize / 64;
        self.live.get(w).is_none_or(|w| w & (1u64 << (r % 64)) != 0)
    }

    /// Whether slot `s`'s SSE register holds register `s.reg`'s value at op
    /// `pc` (whose own write is still to come) on every path there, so that
    /// storing it writes the register file a true value. A loop-carried
    /// slot is loaded at the region's entry. A temporary's interval runs
    /// from its first mention (a write, else the register would be live
    /// at the header) to its last, in pc order, which in a body that only
    /// branches forward (the back-edge aside) every path takes; but past a
    /// join a temporary one arm wrote and the other has not yet (`t = c ?
    /// x : y`, left in the second arm before its write) holds whatever
    /// that SSE register last held, maybe a heap handle since freed. Such a
    /// register is dead there (each path writes it before reading it), so
    /// it is stored only where it is live: on entry to `pc`, or to `to`,
    /// where a leave at `pc` resumes (a merged-check stretch's start,
    /// whose live registers no op of the stretch writes).
    fn defined(&self, s: &Slot, pc: usize, to: usize) -> bool {
        s.carried
            || (s.start < pc && (self.live_at(pc, s.reg) || (to != pc && self.live_at(to, s.reg))))
    }
}

/// Region `r`'s **guard diamonds**: a call whose arguments the compiler
/// guarded (`Op::PlainPrims`: every argument a primitive other than a
/// String, so evaluating them runs no code, and a `Math` function may be
/// called inline) has two arms ???
///
/// ```text
/// g     PlainPrims t, args
/// g+1   JumpIfFalse t -> F
///       the fast arm (the call with its arguments in place)
/// F-1   Jump E
/// F     the fallback arm (the method read, the arguments, a real call)
/// E     (the join)
/// ```
///
/// ??? as `(g + 1, F, E)`, where nothing but the test branches into
/// `F..E` and no entry lies there. A speculative version's fast version
/// keeps only the fast arm: its `PlainPrims` leaves for the plain copy
/// when the test would fail (`jit::compile_generic`), so the test needs no
/// code and the fallback arm, never reached, need not be there ??? nor does
/// its call keep the loop from having a fast version.
fn guard_diamonds(
    r: &Region,
    ops: &[GenericOp],
    entries: &[usize],
    catches: &[usize],
) -> Vec<(usize, usize, usize)> {
    let mut out = Vec::new();
    for g in r.head..r.end {
        let (
            Some(&GenericOp::PlainPrims { dst, .. }),
            Some(&GenericOp::JumpIfFalse { cond, target: f }),
        ) = (ops.get(g), ops.get(g + 1))
        else {
            continue;
        };
        if cond != dst || f <= g + 2 || f > r.end {
            continue;
        }
        let Some(&GenericOp::Jump { target: e }) = ops.get(f - 1) else {
            continue;
        };
        if e <= f || e > r.end {
            continue;
        }
        // Only the test reaches the fallback arm (a branch inside it stays
        // inside it, its own business), and no entry lies there.
        let reached_only_by_test = ops.iter().enumerate().all(|(pc, op)| match *op {
            GenericOp::Jump { target: t } | GenericOp::JumpIfFalse { target: t, .. }
                if (f..e).contains(&t) =>
            {
                pc == g + 1 || ((f..e).contains(&pc) && t > pc)
            }
            GenericOp::PushHandler { target: t, .. } => !(f..e).contains(&t),
            _ => true,
        });
        // And nothing in it branches out of it but to the join.
        let stays = ops[f..e].iter().all(|op| match *op {
            GenericOp::Jump { target: t } | GenericOp::JumpIfFalse { target: t, .. } => {
                t > f && t <= e
            }
            GenericOp::PushHandler { .. } => false,
            _ => true,
        });
        if reached_only_by_test
            && stays
            && !entries.iter().chain(catches).any(|x| (f..e).contains(x))
        {
            out.push((g + 1, f, e));
        }
    }
    out
}

/// A speculative version's **merged NaN checks** ([`plan_defer`]): a guarded
/// arithmetic op whose result (and whose unproven operands) only flow, by
/// arithmetic, into another op's NaN check before anything observes them
/// leaves its own check out — a NaN-boxed operand makes a NaN, which every
/// SSE operation carries on, so the later check sees it. That check then
/// stands for both ([`Cover`]): a NaN there is a NaN of Numbers when every
/// operand it and the merged ops did not prove is a Number (the fast
/// version then goes on, as their own checks would have: a NaN of Numbers
/// is the canonical NaN); else something upstream was not a Number, and it leaves
/// for the plain copy at the start of the side-effect-free stretch it lies
/// in, which runs it all again exactly, as does every other leave from
/// inside the stretch.
#[derive(Default)]
struct DeferPlan {
    /// The region pcs whose NaN check is left out ([`DEFER_SKIP`]) or
    /// stands for others ([`DEFER_COVER`]), ascending.
    kind: Vec<(usize, u8)>,
    /// `(first, last, start)`: a leave of op `first..=last` goes to op
    /// `start` of the plain copy.
    spans: Vec<(usize, usize, usize)>,
    /// `(pc, register, xmm)`: before arithmetic op `pc`, store the register
    /// (live on entry to a stretch, its SSE register handed on there).
    saves: Vec<(usize, u16, u8)>,
    /// The checks that stand for others ([`DEFER_COVER`]), ascending.
    covers: Vec<Cover>,
}

/// A check that stands for merged ones ([`DeferPlan`]): what its stub
/// ([`StubKind::DeferNan`]) looks at when the op's result is a NaN.
struct Cover {
    pc: usize,
    /// The merged ops' operands not proven Numbers, each still where it was
    /// when its op read it: when they all are Numbers (as are the op's
    /// own), the NaN is a Number's.
    roots: Vec<u16>,
}

/// [`DeferPlan`]: the op's NaN check is left out.
const DEFER_SKIP: u8 = 1;
/// [`DeferPlan`]: the op's check stands for others ([`Cover`]).
const DEFER_COVER: u8 = 2;

/// A region's **speculative** double version: arithmetic whose operands are
/// not Numbers does not run its slow path in the region but leaves it — all
/// promoted registers flushed — for the op's own pc in a plain copy of the
/// loop body (emitted by the generic emitter, `compile_generic`), which runs
/// the op exactly as the interpreter would and loops back through the
/// region's entry. Every arithmetic result in the region is therefore a
/// Number, so the ops that consume it need no type guard. Loop-carried
/// registers the region assumes are Numbers are checked once at its entry
/// (else the copy runs the iteration).
///
/// Every other slow path that would run Rust code leaves for the copy the
/// same way (the numeric ones by [`NumGen::stub`], the generic emitter's
/// through [`NumGen::spec_bail`], a global binding's validation after it has
/// validated), as do a property add and a compiled-in layout's mismatch: the
/// fast version makes no call from the header to the back-edge, which lets
/// the receiver cache carry what one site proved to the next sites of the
/// iteration (`recv_cache::plan`).
struct Spec {
    /// The in-state register types of each region pc (`(pc - head) * n_regs
    /// + r`), under the speculation.
    types: Vec<u8>,
    /// The registers checked to be Numbers at the region's entry.
    checks: Vec<u16>,
    /// The **field facts** the fast version relies on, as `(mask address,
    /// slots)`: a property read whose layout is compiled in (which the fast
    /// version proves, or leaves) from a slot that layout records as
    /// Number-only (`Shape::may_be_non_number`) yields a Number, with no
    /// type check. A record changes only in Rust code (`jit::fieldrep`),
    /// which the fast version never runs: every entry into it (the
    /// region's, a back-edge poll's after its call, an inner header's)
    /// checks that the slots are still Number-only, else the plain copy
    /// runs (`spec_globals_ready`).
    fields: Vec<(u64, u64)>,
}

/// The helper entry points and frame facts the emitter calls into.
pub(super) struct NumEnv {
    pub(super) n_regs: usize,
    pub(super) bytecode: *const Op,
    pub(super) bytecode_len: usize,
    pub(super) add: u64,
    pub(super) arith: u64,
    pub(super) value_bin: u64,
    pub(super) lt: u64,
    pub(super) strict_eq: u64,
    pub(super) truthy: u64,
    pub(super) exec_op: u64,
    pub(super) backedge: u64,
    pub(super) global_ref: u64,
    pub(super) math1: u64,
    /// The body's persistent global binding cache (`super::global_persist`).
    pub(super) persist: u64,
    /// `Ctx`-relative offset of the realm pointer (the inline heap reads').
    pub(super) realm: i32,
    /// The call sequence's layout (the back-edge poll's inline test).
    pub(super) layout: super::CallLayout,
    /// Per pc: for a `GetProp` whose layout is compiled in, and which that
    /// layout had as a Number-only slot when compiled, the address of the
    /// layout's may-be-non-Number mask and the slot (see [`Spec::fields`]).
    pub(super) fields: Vec<Option<(u64, u32)>>,
}

/// A region truthiness test's receiver-cache way: its key and slot-address
/// frame slots (`rbp`-relative), whether to compare the key first, and the
/// layout the way proves (`recv_cache::shape_ways`), which a fill then
/// requires. An ordinary shaped, non-exotic object condition fills the way
/// (with its walk's slot address), so a later property op on the same
/// object skips the walk.
#[derive(Clone, Copy, Debug)]
pub(super) struct TruthyWay {
    pub(super) key: i32,
    pub(super) cell: i32,
    pub(super) check: bool,
    pub(super) shape: Option<u64>,
    /// The way's layout (`shape`) lacks the `[[IsHTMLDDA]]` brand.
    pub(super) plain: bool,
}

/// Out-of-line slow paths, emitted after the body ([`NumGen::finish`]).
enum StubKind {
    /// `rax = helper(ctx, [a], [b], extra?)`, sentinel-checked when `throws`,
    /// stored to `dst`.
    Helper {
        helper: u64,
        a: u16,
        b: u16,
        extra: Option<u64>,
        dst: u16,
        throws: bool,
    },
    /// The interpreter runs bytecode op `pc`.
    Exec { pc: u32 },
    /// `ToBoolean([cond])` through the helper, then branch: falsy → bytecode
    /// pc `target`, truthy → `resume`.
    Truthy { cond: u16, target: usize },
    /// `xmm0 = canonical NaN` (no call).
    CanonNan,
    /// `xmm x` is NaN-patterned: to `slow` unless it is a Number (a NaN),
    /// else resume (no call).
    NanCheck { x: u8, slow: Label },
    /// An arithmetic result in `xmm x` is a NaN (see `float_bin`): to
    /// `slow` if one of `regs` is not a Number, else resume (no call).
    DeferNan {
        x: u8,
        regs: [Option<u16>; 2],
        slow: Label,
        /// First `xmm x = xmm0` (the value the op's result replaced), and
        /// on resuming `xmm x = canonical NaN` (the op's result).
        restore: bool,
        /// The check stands for others ([`Cover`]): its merged ops'
        /// operands are checked too (`slow` leaves for the stretch's
        /// start).
        cover: bool,
    },
    /// Leave the region: flush, then jump to bytecode pc `target`.
    Exit { target: usize },
    /// A global op at `pc` found no validated binding address: validate it
    /// (`jit_helper_global_ref`) and retry at `retry`, or run the op through the
    /// interpreter if it does not qualify.
    GlobalRef { pc: usize, retry: Label },
    /// The integer version gives up at bytecode pc `pc` (see [`int32`]).
    Bail { pc: u32, store: bool },
    /// The speculative double version leaves for op `pc` in the plain copy
    /// (see [`Spec`]).
    SpecBail { pc: u32 },
}

struct Stub {
    entry: Label,
    resume: Label,
    region: Option<usize>,
    /// The pc of the op the stub belongs to (which promoted registers are live).
    pc: usize,
    kind: StubKind,
    /// A fused comparison's branch, taken after the slow op instead of resuming:
    /// `(cond, target)` — truthy `cond` continues at pc `pc + 2`, falsy at
    /// `target`.
    branch: Option<(u16, usize)>,
    /// Emitted by a region's integer version (its flush / reload convert the
    /// int registers).
    int: bool,
    /// Emitted in a speculative region's plain copy (its branches stay in
    /// the copy: [`NumGen::branch_label`]).
    copy: bool,
}

/// The per-compilation state of the numeric emitter.
pub(super) struct NumGen {
    types: Types,
    /// The integrality facts ([`infer_integral`]).
    ints: Ints,
    regions: Vec<Region>,
    /// Per pc: the region it belongs to (`u32::MAX` for none).
    region_of: Vec<u32>,
    /// Per pc: a branch target or entry point (so the op's label can be reached
    /// other than by falling through from the previous op).
    is_target: Vec<bool>,
    /// Per register: how many ops name it.
    mentions: Vec<u32>,
    stubs: Vec<Stub>,
    /// Per pc: a `JumpIfFalse` folded into the comparison before it (emits
    /// nothing).
    skip: Vec<bool>,
    /// The pc being emitted (which promoted registers are live).
    cur_pc: usize,
    /// Per pc: the frame slot holding an in-region global op's validated
    /// binding address (`u32::MAX` for none); slot `k` is at `gbase - 8(k+1)`.
    gslot: Vec<u32>,
    /// Per region: its global-op frame slots.
    region_gslots: Vec<Vec<u32>>,
    n_gslots: usize,
    gbase: i32,
    env: NumEnv,
    /// While emitting a region's integer version (or one of its stubs): that
    /// region.
    imode: Option<usize>,
    canonical_nan: u64,
    tag_true: u64,
    tag_false: u64,
    /// While the generic emitter emits a speculative region's plain copy:
    /// no op is in a region (the copy runs on the register file).
    copy: bool,
    /// Per pc: its label in its region's plain copy ([`Spec`]).
    copy_labels: Vec<Option<Label>>,
    /// Per pc: the receiver-cache way of a region truthiness test on an
    /// object ([`Self::truthy_cacheable`]), as `recv_cache::plan` placed it.
    truthy_ways: Vec<Option<TruthyWay>>,
    /// Where a back-edge poll that asks the body to leave for the
    /// interpreter jumps (with the promoted registers flushed).
    pub(super) leave_exit: Option<Label>,
    /// The entries at a loop nest's inner headers (on-stack replacement
    /// there), as `(label, pc)` ([`Self::entry_label`]).
    nest_entries: Vec<(Label, usize)>,
    /// Per inner header of a speculative nest: the entry into its fast
    /// version there ([`Self::emit_inner_entries`]), which the plain copy's
    /// inner back-edge (and an on-stack replacement there) takes, as
    /// `(header, label)`.
    inner_entries: Vec<(usize, Label)>,
    /// The hoisted views' frame slots ([`Region::views`]), after the
    /// regions' epoch slots.
    n_view_slots: usize,
}

/// The ops a region may contain: each is emitted here, with calls only on slow
/// paths.
fn region_op_ok(op: &GenericOp, bc: &Op) -> bool {
    match op {
        GenericOp::Const { .. }
        | GenericOp::Move { .. }
        | GenericOp::Nop
        | GenericOp::Add { .. }
        | GenericOp::NumAdd { .. }
        | GenericOp::Lt { .. }
        | GenericOp::Compare { .. }
        | GenericOp::StrictEq { .. }
        | GenericOp::ToNumeric { .. }
        | GenericOp::Step { .. }
        | GenericOp::Jump { .. }
        | GenericOp::JumpIfFalse { .. }
        | GenericOp::CheckTdz { .. }
        | GenericOp::Ret { .. } => true,
        GenericOp::Arith { .. } => true,
        GenericOp::ValueBin { op, .. } => *op != 0,
        GenericOp::Interp { .. } => {
            matches!(
                bc,
                Op::Neg { .. }
                    | Op::BitNot { .. }
                    | Op::LoadGlobal { .. }
                    | Op::StoreGlobal { .. }
                    | Op::RequireObjectCoercible { .. }
            ) || generic_region_op(op, bc)
        }
        _ => generic_region_op(op, bc),
    }
}

/// The region ops the generic emitter emits (`compile_generic`): a property or
/// element read, a property write, an array's `.length`, and a call (the
/// native call sequence, or a `Math` function the generic tier computes
/// inline). They read their operands from the register file, so the
/// registers they read are never promoted; a promoted register they write is
/// reloaded after them ([`NumGen::after_generic`]). Their calls (slow paths,
/// getters, setters, misses, the callee itself) store the promoted registers
/// first and reload them after ([`NumGen::call_spill`]), so a loop that reads
/// and writes object fields, or calls a function, keeps its arithmetic in SSE
/// registers. (A callee reaches the caller's registers only through the
/// register file: as the collector's roots, which a relocation may update.)
fn generic_region_op(op: &GenericOp, bc: &Op) -> bool {
    match op {
        GenericOp::GetProp { .. }
        | GenericOp::GetMethod { .. }
        | GenericOp::SetProp { .. }
        | GenericOp::GetElem { .. }
        | GenericOp::SetElem { .. }
        | GenericOp::ArrayLen { .. }
        | GenericOp::CallMethod { .. }
        | GenericOp::CallValue { .. }
        | GenericOp::CallWithMethod { .. }
        | GenericOp::CallKey { .. } => true,
        // An inlined call's guard and end (`super::inline`): no call.
        GenericOp::InlineGuard { .. } | GenericOp::InlineLeave => true,
        // A guard's type test (inline, its slow path a pure check).
        GenericOp::PlainPrims { .. } => true,
        // `!x` (a boolean or Number inline) and an array iteration's step in
        // by-index mode (`for (… of array)`, inline while the array lasts).
        GenericOp::Not { .. } => true,
        GenericOp::Interp { .. } => {
            matches!(
                bc,
                Op::IterNext { iter, next, .. } if usize::from(*next) == usize::from(*iter) + 1
            ) || allocating_region_op(op, bc)
        }
        _ => allocating_region_op(op, bc),
    }
}

/// The generic emitter's ops that may allocate or run code on their fast
/// path -- a literal, a closure, a binding cell, a `new` -- and a captured
/// binding's read or write: each is emitted as anywhere else (its calls
/// bracketed by the promoted registers' stores and reloads like every
/// generic region op's), so a loop that makes objects or closures, or
/// counts in a captured variable, still keeps its arithmetic in SSE
/// registers. Its every call can run Rust code ([`region_call`]), so such a
/// loop has no speculative version.
fn allocating_region_op(op: &GenericOp, bc: &Op) -> bool {
    match op {
        GenericOp::CellGet { .. } | GenericOp::CellSet { .. } | GenericOp::Construct { .. } => true,
        GenericOp::Interp { .. } => matches!(
            bc,
            Op::NewObjectLit { .. }
                | Op::NewObject { .. }
                | Op::NewArray { .. }
                | Op::NewArrayFrom { .. }
                | Op::NewBox { .. }
                | Op::MakeClosure { .. }
        ),
        _ => false,
    }
}

/// Whether region op `op` may run arbitrary code (a call that is not an
/// inline `Math` function): a speculative version needs a call-free body.
fn region_call(op: &GenericOp, bc: &Op) -> bool {
    match op {
        GenericOp::CallMethod { .. } => super::inline_math(bc).is_none(),
        // (A `CallWithMethod` of `Math.imul` computed inline counts too:
        // the `GetMethod` before it reads the built-in through a helper.)
        GenericOp::CallValue { .. }
        | GenericOp::CallWithMethod { .. }
        | GenericOp::CallKey { .. } => true,
        // A captured binding's read: its fast path is loads, and its slow
        // path leaves a fast version for the plain copy (`spec_bail`).
        GenericOp::CellGet { .. } => false,
        _ => allocating_region_op(op, bc),
    }
}

impl NumGen {
    /// Plans the numeric emission of `ops` (lowered from `bytecode`):
    /// infers the types and picks the promoted loop regions. `entries` are the
    /// body's OSR entry pcs and `catches` its catch entries.
    pub(super) fn new(
        ops: &[GenericOp],
        bytecode: &[Op],
        env: NumEnv,
        entries: &[usize],
        catches: &[usize],
    ) -> Self {
        let n = ops.len();
        let types = infer(bytecode, env.n_regs);
        let ints = infer_integral(bytecode, env.n_regs);
        let mut is_target = vec![false; n + 1];
        let mut mentions = vec![0u32; env.n_regs];
        for op in bytecode {
            match op {
                Op::Jump { target }
                | Op::JumpIfFalse { target, .. }
                | Op::PushHandler { target, .. } => {
                    if let Some(t) = is_target.get_mut(*target) {
                        *t = true;
                    }
                }
                _ => {}
            }
            for r in op_regs(op) {
                if let Some(m) = mentions.get_mut(r as usize) {
                    *m += 1;
                }
            }
        }
        for &e in entries.iter().chain(catches) {
            if let Some(t) = is_target.get_mut(e) {
                *t = true;
            }
        }
        let mut regions = plan_regions(ops, bytecode, catches, env.n_regs);
        let mut skip = vec![false; n + 1];
        for r in &mut regions {
            r.cold = guard_diamonds(r, ops, entries, catches);
            r.spec = plan_spec(r, ops, bytecode, &types, env.n_regs, &env.fields);
            match &r.spec {
                // The fast version has no guard test and no fallback arm.
                Some(_) => {
                    for &(test, fallback, join) in &r.cold {
                        skip[test] = true;
                        for s in &mut skip[fallback..join] {
                            *s = true;
                        }
                    }
                }
                None => r.cold.clear(),
            }
        }
        if regions.iter().any(|r| r.spec.is_some())
            && let Some((words, live)) = liveness(bytecode, env.n_regs, catches)
        {
            let live_at = |pc: usize, r: u16| {
                (r as usize) < env.n_regs
                    && live
                        .get(pc * words + r as usize / 64)
                        .is_some_and(|w| w & (1u64 << (r % 64)) != 0)
            };
            for r in &mut regions {
                r.defer = plan_defer(r, ops, bytecode, env.n_regs, &is_target, &live_at);
            }
        }
        let mut region_of = vec![u32::MAX; n];
        let mut gslot = vec![u32::MAX; n];
        let mut region_gslots = vec![Vec::new(); regions.len()];
        let mut n_gslots = 0u32;
        // The integer versions' promoted globals' binding addresses.
        for r in &mut regions {
            if let Some(p) = r.int.as_mut() {
                for g in &mut p.globals {
                    g.slot = n_gslots;
                    n_gslots += 1;
                }
            }
        }
        for (i, r) in regions.iter().enumerate() {
            for pc in r.head..=r.end {
                region_of[pc] = i as u32;
                if matches!(bytecode[pc], Op::LoadGlobal { .. } | Op::StoreGlobal { .. }) {
                    gslot[pc] = n_gslots;
                    region_gslots[i].push(n_gslots);
                    n_gslots += 1;
                }
            }
        }
        // The hoisted views' slots, past the epoch slots (one per region).
        // Only once typed arrays have been indexed: for an array the
        // hoisted path's test is one more compare per access.
        let mut n_view_slots = 0usize;
        if crate::realm::typed_elems_seen() {
            let first = n_gslots as usize + regions.len();
            for r in &mut regions {
                if r.spec.is_some() {
                    for (v, src) in plan_views(r, ops, bytecode, env.n_regs) {
                        r.views.push((v, (first + n_view_slots) as u32, src));
                        // (A register the region may write: and the value
                        // the facts are about, `ViewSlots::held`.)
                        n_view_slots += if src == ViewSrc::Fixed { 3 } else { 4 };
                    }
                }
            }
        }
        Self {
            types,
            ints,
            regions,
            region_of,
            is_target,
            mentions,
            stubs: Vec::new(),
            skip,
            cur_pc: 0,
            gslot,
            region_gslots,
            n_gslots: n_gslots as usize,
            gbase: 0,
            env,
            imode: None,
            canonical_nan: NanBox::number(f64::NAN).to_bits(),
            tag_true: NanBox::boolean(true).to_bits(),
            tag_false: NanBox::boolean(false).to_bits(),
            copy: false,
            copy_labels: vec![None; n],
            truthy_ways: Vec::new(),
            leave_exit: None,
            nest_entries: Vec::new(),
            inner_entries: Vec::new(),
            n_view_slots,
        }
    }

    /// Allocates the labels of the speculative regions' plain copies.
    pub(super) fn alloc_copy_labels(&mut self, a: &mut X64Assembler) {
        for r in &self.regions {
            if r.spec.is_some() {
                for pc in r.head..=r.end {
                    self.copy_labels[pc] = Some(a.new_label());
                }
                for &(h, _) in &r.loops {
                    if !self.inner_entries.iter().any(|e| e.0 == h) {
                        self.inner_entries.push((h, a.new_label()));
                    }
                }
            }
        }
    }

    /// The entry into speculative nest `ri`'s fast version at its inner
    /// header `h`, if it has one.
    fn inner_entry(&self, h: usize) -> Option<Label> {
        self.inner_entries.iter().find(|e| e.0 == h).map(|e| e.1)
    }

    /// The order the generic emitter emits the body in, as `(pc, copy)`: every
    /// op once, and after a speculative region its plain copy.
    pub(super) fn schedule(&self, n: usize) -> Vec<(usize, bool)> {
        let mut out = Vec::with_capacity(n);
        for pc in 0..n {
            out.push((pc, false));
            if let Some(ri) = self.region_at(pc)
                && self.regions[ri].end == pc
                && self.regions[ri].spec.is_some()
            {
                out.extend((self.regions[ri].head..=pc).map(|q| (q, true)));
            }
        }
        out
    }

    /// Where the body's entry at loop header `pc` (`0`, or an on-stack
    /// replacement) goes: an inner header of a loop nest is entered through
    /// a stub ([`Self::finish`]) that loads the promoted registers live
    /// there, or — in a speculative nest — runs the plain copy from there
    /// (which re-enters the fast version at the nest's own header).
    pub(super) fn entry_label(
        &mut self,
        a: &mut X64Assembler,
        labels: &[Label],
        pc: usize,
    ) -> Option<Label> {
        match self.region_at(pc) {
            Some(ri) if self.regions[ri].head != pc => {
                let l = a.new_label();
                self.nest_entries.push((l, pc));
                Some(l)
            }
            _ => labels.get(pc).copied(),
        }
    }

    /// Sets where a back-edge poll that asks the body to leave jumps.
    pub(super) fn set_leave_exit(&mut self, l: Label) {
        self.leave_exit = Some(l);
    }

    /// Enters (or leaves) the emission of a speculative region's plain copy.
    pub(super) fn set_copy(&mut self, on: bool) {
        self.copy = on;
    }

    /// Op `pc`'s label in its region's plain copy.
    pub(super) fn copy_label(&self, pc: usize) -> Option<Label> {
        self.copy_labels.get(pc).copied().flatten()
    }

    /// How many 8-byte native frame slots the emitted code needs: the
    /// global binding addresses, then one epoch slot per region, then the
    /// hoisted views' ([`Region::views`]).
    pub(super) fn frame_slots(&self) -> usize {
        self.n_gslots + self.regions.len() + self.n_view_slots
    }

    /// For a typed-array element op `pc` (`GetElem` / `SetElem` /
    /// `ArrayLen`) in a speculative region's fast version, on receiver
    /// register `r`: its hoisted view's slots, when the region hoists `r`
    /// ([`plan_views`]). Its entry check filled them (see
    /// [`super::typed::ViewSlots`]).
    pub(super) fn hoisted_view(&self, pc: usize, r: u16) -> Option<super::typed::ViewSlots> {
        let ri = self.region_at(pc)?;
        self.regions[ri].spec.as_ref()?;
        self.hoisted_view_slots(ri, r)
    }

    /// Speculative region `ri`'s hoisted views' entry checks (see
    /// [`super::typed::ViewSlots`]), at a point where nothing promoted is
    /// live and the register file is up to date. Clobbers `rax`, `rcx`,
    /// `rdx`, `r8`..`r11`.
    fn hoist_views(&self, a: &mut X64Assembler, ri: usize) {
        if self.regions[ri].views.is_empty() {
            return;
        }
        let l = super::compute_jit_layout();
        let t = super::typed::typed_layout();
        for &(r, _, src) in &self.regions[ri].views {
            let Some(v) = self.hoisted_view_slots(ri, r) else {
                continue;
            };
            // A global binding's value through its op's validated address
            // (`spec_globals_ready` just proved them all filled).
            let from = match src {
                ViewSrc::Global(pc) => match self.gslot.get(pc) {
                    Some(&k) if k != u32::MAX => {
                        super::typed::HoistFrom::Binding(self.gslot_disp(k))
                    }
                    _ => super::typed::HoistFrom::Reg(disp(r)),
                },
                _ => super::typed::HoistFrom::Reg(disp(r)),
            };
            a.t_hoist(l, t, self.env.realm, from, v);
        }
    }

    fn hoisted_view_slots(&self, ri: usize, r: u16) -> Option<super::typed::ViewSlots> {
        let &(_, k, src) = self.regions[ri].views.iter().find(|v| v.0 == r)?;
        Some(super::typed::ViewSlots {
            kind: self.gslot_disp(k),
            len: self.gslot_disp(k + 1),
            base: self.gslot_disp(k + 2),
            held: (src != ViewSrc::Fixed).then(|| self.gslot_disp(k + 3)),
        })
    }

    /// The frame slots the prologue must empty: the epoch slots of the
    /// regions with global ops (an empty slot is no epoch, so the first
    /// entry validates afresh).
    pub(super) fn entry_clear_slots(&self) -> Vec<i32> {
        (0..self.regions.len())
            .filter(|&ri| !self.region_gslots[ri].is_empty())
            .map(|ri| self.gslot_disp((self.n_gslots + ri) as u32))
            .collect()
    }

    /// Whether op `pc` lies in a promoted loop region (whose code this
    /// emitter owns).
    pub(super) fn in_region(&self, pc: usize) -> bool {
        !self.copy && self.region_of.get(pc).is_some_and(|&r| r != u32::MAX)
    }

    /// Where those frame slots start: slot `k` is at `[rbp + base - 8(k+1)]`.
    /// Whether region op `pc` is a forward `JumpIfFalse` this emitter tests
    /// with [`Self::branch_if_false`]'s general path (its condition neither
    /// a proven Number nor a boolean): the receiver cache may give it a way,
    /// which an object condition then fills (see [`TruthyWay`]).
    pub(super) fn truthy_cacheable(&self, pc: usize, ops: &[GenericOp]) -> bool {
        matches!(ops.get(pc), Some(&GenericOp::JumpIfFalse { cond, target })
            if target > pc && self.in_region(pc) && self.ty(pc, cond) != T_NUM && self.ty(pc, cond) != T_BOOL)
    }

    /// The receiver-cache ways of the region truthiness tests (per pc).
    pub(super) fn set_truthy_ways(&mut self, ways: Vec<Option<TruthyWay>>) {
        self.truthy_ways = ways;
    }

    pub(super) fn set_frame_base(&mut self, base: i32) {
        self.gbase = base;
    }

    fn gslot_disp(&self, k: u32) -> i32 {
        self.gbase - 8 * (k as i32 + 1)
    }

    /// Drops the region's validated global binding addresses (after a call:
    /// Rust code may have moved or retired them, or changed what they mean).
    fn zero_globals(&self, a: &mut X64Assembler, reg: Option<usize>) {
        let Some(ri) = reg else { return };
        for &k in &self.region_gslots[ri] {
            a.mov_frame_zero(self.gslot_disp(k));
        }
    }

    /// Keeps region `ri`'s validated global binding addresses across a
    /// region entry or a back-edge poll while the global binding epoch they
    /// were validated under is current, else drops them (and notes the
    /// current epoch). Only Rust code changes the epoch, and the addresses
    /// stay valid exactly as long as it does not (`jit_helper_global_ref`'s
    /// contract, which the body's own global binding cache relies on too).
    /// Every slot was filled at or after the epoch the region noted, as the
    /// epoch only grows and every call in the region drops them all: a
    /// noted epoch that is still current proves every filled slot current.
    ///
    /// With `validate` (a speculative region, from a point where no
    /// promoted register is live yet: the entry before its loads, or a
    /// back-edge poll's before the reload), a changed epoch also validates
    /// every slot afresh (`jit_helper_global_ref`, which leaves a binding
    /// that does not qualify empty) — so the fast version, whose own
    /// validation leaves for the plain copy, finds them filled. Clobbers
    /// `rax`, `r11` (and, validating, every caller-saved register).
    fn refresh_globals(&self, a: &mut X64Assembler, ri: usize, validate: bool) {
        if self.region_gslots[ri].is_empty() {
            return;
        }
        let keep = a.new_label();
        let noted = self.gslot_disp((self.n_gslots + ri) as u32);
        a.movabs_r11(crate::env::global_binding_epoch_addr() as i64);
        a.raw(&[0x49, 0x8b, 0x03]); // mov rax, [r11]
        a.cmp_rax_mem(noted);
        a.jcc(CC_E, keep);
        a.g_store(super::call_seq::r::RBP, noted, RAX);
        self.zero_globals(a, Some(ri));
        if validate {
            // The addresses another activation validated under this very
            // epoch, when the region's global ops are all reads
            // (`super::global_persist`).
            let persisted = self.persisted_gslots(ri);
            if let Some((pn, entries)) = &persisted {
                a.gp_restore(self.env.persist, *pn, entries, keep);
            }
            let r = &self.regions[ri];
            for pc in r.head..=r.end {
                let k = self.gslot.get(pc).copied().unwrap_or(u32::MAX);
                if k == u32::MAX {
                    continue;
                }
                let op_ptr = self.env.bytecode.wrapping_add(pc) as i64;
                a.mov_rdi_r15();
                a.movabs_argreg(1, op_ptr);
                a.movabs_rax(self.env.global_ref as i64);
                a.call_rax();
                a.store_rax(self.gslot_disp(k));
            }
            // (Validating ran Rust code: its epoch is the one noted only if
            // it did not change it.)
            a.movabs_r11(crate::env::global_binding_epoch_addr() as i64);
            a.raw(&[0x49, 0x8b, 0x03]); // mov rax, [r11]
            a.cmp_rax_mem(noted);
            match &persisted {
                Some((pn, entries)) => {
                    let failed = a.new_label();
                    a.jcc(CC_NE, failed);
                    a.gp_save_all(self.env.persist, *pn, entries, noted);
                    a.jmp(keep);
                    a.bind(failed);
                }
                None => a.jcc(CC_E, keep),
            }
            self.zero_globals(a, Some(ri));
            a.store_empty(noted);
        }
        a.bind(keep);
    }

    /// Region `ri`'s persistent global binding entries, when its global ops
    /// are all reads and the body's array has room: the word its epoch is
    /// noted at, and per global op its `(word, frame displacement)`. (The
    /// array's words past the body's own cache: one epoch per region, then
    /// one per global slot.)
    fn persisted_gslots(&self, ri: usize) -> Option<(usize, Vec<(usize, i32)>)> {
        use super::global_persist::{BODY_WORDS, LEN};
        if self.env.persist == 0 || BODY_WORDS + self.regions.len() + self.n_gslots > LEN {
            return None;
        }
        let r = &self.regions[ri];
        let mut entries = Vec::new();
        for pc in r.head..=r.end {
            let k = self.gslot.get(pc).copied().unwrap_or(u32::MAX);
            if k == u32::MAX {
                continue;
            }
            if !matches!(self.bc(pc), Some(Op::LoadGlobal { .. })) {
                return None;
            }
            entries.push((
                BODY_WORDS + self.regions.len() + k as usize,
                self.gslot_disp(k),
            ));
        }
        Some((BODY_WORDS + ri, entries))
    }

    /// A global op `pc` of speculative region `ri`'s fast version, its
    /// binding address in frame slot `sd`. The region validated every one
    /// of its global ops' addresses before entering the fast version
    /// ([`spec_globals_ready`](Self::spec_globals_ready)), and from there to
    /// the back-edge no Rust code runs — nothing drops or retires them — so
    /// the address is used unchecked. For the same reason a read of a binding
    /// the iteration already read or wrote (with that value still in its
    /// register) is that value: only the region's own writes can change it.
    fn spec_global(&mut self, a: &mut X64Assembler, ri: usize, pc: usize, sd: i32) -> Option<bool> {
        let reg = Some(ri);
        match *self.bc(pc)? {
            Op::LoadGlobal { dst, .. } => {
                if (dst as usize) >= self.env.n_regs {
                    return None;
                }
                if let Some(src) = self.global_known(ri, pc) {
                    self.copy_reg(a, reg, dst, src);
                    return Some(true);
                }
                a.load_rax(sd);
                match self.xmm_of(reg, dst) {
                    Some(x) => a.sse_mem_rax(0x10, x), // movsd xmm, [rax]
                    None => {
                        a.raw(&[0x48, 0x8b, 0x00]); // mov rax, [rax]
                        self.store_gpr(a, reg, RAX, dst);
                    }
                }
            }
            Op::StoreGlobal { src, .. } => {
                if (src as usize) >= self.env.n_regs {
                    return None;
                }
                let mirror = super::global_mirror::mirror_cell(self.bc(pc)?);
                a.load_rcx(sd);
                match self.xmm_of(reg, src) {
                    Some(x) => {
                        a.sse_mem_rcx(0x11, x); // movsd [rcx], xmm
                        // (A promoted register holds any value's bits, a
                        // heap handle's among them.)
                        if let Some(cell) = mirror {
                            a.movq_gpr_x(RAX, x);
                            let bail = self.spec_bail(a, pc)?;
                            a.gm_store_gpr(cell, RCX, RAX, bail);
                        }
                    }
                    None => {
                        self.load_gpr(a, reg, RAX, src);
                        a.raw(&[0x48, 0x89, 0x01]); // mov [rcx], rax
                        // (A heap handle's mirror needs the write barrier:
                        // the plain copy's op does it.)
                        if let Some(cell) = mirror {
                            let bail = self.spec_bail(a, pc)?;
                            a.gm_store_gpr(cell, RCX, RAX, bail);
                        }
                    }
                }
            }
            _ => return None,
        }
        Some(true)
    }

    /// For the `LoadGlobal` at `pc` of speculative region `ri`: the register
    /// still holding its binding's value, when an earlier global op of the
    /// iteration on the same name read it into one, or wrote it from one,
    /// that no op since has written. (Only back to the last join before
    /// `pc`: past it the value may come from another edge.)
    fn global_known(&self, ri: usize, pc: usize) -> Option<u16> {
        let Op::LoadGlobal { name, .. } = self.bc(pc)? else {
            return None;
        };
        let head = self.regions[ri].head;
        let (mut d, mut u) = (Vec::new(), Vec::new());
        for q in (head..pc).rev() {
            // (Not across a join: an arm's end, a loop nest's inner header
            // or exit.)
            if self.is_target.get(q + 1).copied().unwrap_or(true) {
                return None;
            }
            let bq = self.bc(q)?;
            let held = match bq {
                Op::LoadGlobal { name: n, dst, .. } if n == name => Some(*dst),
                Op::StoreGlobal { name: n, src, .. } if n == name => Some(*src),
                _ => None,
            };
            if let Some(r) = held {
                // Unwritten since (an op between writes no register but
                // the ones `def_use` names: the region's ops are all
                // modelled ones).
                let clean = (q + 1..pc).all(|x| {
                    self.bc(x).is_some_and(|o| {
                        def_use(o, &mut d, &mut u);
                        !d.contains(&r) && written_reg(o) != Some(r)
                    })
                });
                // And the value is still where `pc` would read it: a
                // promoted register's SSE register only holds it within
                // its live interval (past it the register file is stale).
                let slots = &self.regions[ri].slots;
                let held = !slots.iter().any(|s| s.reg == r)
                    || slots
                        .iter()
                        .any(|s| s.reg == r && s.start <= q && pc <= s.end);
                return (clean && held && self.gslot.get(q).is_some_and(|&k| k != u32::MAX))
                    .then_some(r);
            }
        }
        None
    }

    /// For a `LoadGlobal` at `pc` in a speculative region's fast version that
    /// reads a value a register still holds ([`global_known`](Self::global_known)):
    /// that register (the op is a copy of it).
    pub(super) fn global_copy(&self, pc: usize) -> Option<u16> {
        let ri = self.region_at(pc)?;
        if self.regions[ri].spec.is_none() || self.gslot.get(pc).is_none_or(|&k| k == u32::MAX) {
            return None;
        }
        self.global_known(ri, pc)
    }

    /// `x++` / `x--` on a promoted Number `x` — the compiler's `ToNumeric t1
    /// ← x; Step t2 ← t1; Move x ← t2`, with `t1` and `t2` named by nothing
    /// else and no entry into the sequence — as one `addsd`/`subsd` on
    /// `x`'s SSE register (the two temporaries' values go nowhere), the
    /// `Step` and the `Move` then emitting nothing. Regions without an
    /// integer version only (whose single emission the skip marks apply
    /// to). `true` if emitted.
    fn fused_step(
        &mut self,
        a: &mut X64Assembler,
        reg: Option<usize>,
        pc: usize,
        ops: &[GenericOp],
    ) -> bool {
        let Some(ri) = reg else { return false };
        if self.imode.is_some() || self.regions[ri].int.is_some() || pc + 2 > self.regions[ri].end {
            return false;
        }
        let (
            GenericOp::ToNumeric {
                dst: t1, src: x, ..
            },
            GenericOp::Step {
                dst: t2,
                src: s1,
                dec,
                ..
            },
            GenericOp::Move { dst: x2, src: s2 },
        ) = (ops[pc], ops[pc + 1], ops[pc + 2])
        else {
            return false;
        };
        let once = |r: u16| self.mentions.get(r as usize).copied() == Some(2);
        let entered = |q: usize| self.is_target.get(q).copied().unwrap_or(true);
        if s1 != t1 || s2 != t2 || x2 != x || t1 == x || t2 == x || t1 == t2 {
            return false;
        }
        if !once(t1) || !once(t2) || entered(pc + 1) || entered(pc + 2) {
            return false;
        }
        if self.ty(pc, x) != T_NUM {
            return false;
        }
        let Some(xx) = self.xmm_of(reg, x) else {
            return false;
        };
        a.movabs_rax(1.0f64.to_bits() as i64);
        a.movq_x_gpr(1, RAX);
        a.sse_rr(0xf2, if dec { 0x5c } else { 0x58 }, xx, 1);
        self.skip[pc + 1] = true;
        self.skip[pc + 2] = true;
        true
    }

    /// The entry check of speculative region `ri`'s fast version: every one
    /// of its global ops has a validated binding address (else, with
    /// nothing promoted yet, the plain copy runs the iteration) and its
    /// field facts ([`Spec::fields`]) still hold. Clobbers `r11` when it has
    /// any, else nothing.
    fn spec_globals_ready(&self, a: &mut X64Assembler, ri: usize, pc: usize) -> Option<()> {
        let copy = self.copy_label(pc)?;
        for &k in &self.region_gslots[ri] {
            a.cmp_frame_imm8(self.gslot_disp(k), 0);
            a.jcc(CC_E, copy);
        }
        // The field facts ([`Spec::fields`]): still Number-only slots.
        if let Some(sp) = &self.regions[ri].spec {
            for &(addr, mask) in &sp.fields {
                a.p_fields_still_numbers(addr, mask, copy);
            }
        }
        Some(())
    }

    /// Whether op `pc` lies in the fast version of a speculative region:
    /// none of its paths runs Rust code and then rejoins it (every one
    /// leaves for the region's plain copy instead), so from the region's
    /// header to its back-edge no call is made.
    pub(super) fn in_spec(&self, pc: usize) -> bool {
        self.region_at(pc)
            .is_some_and(|ri| self.regions[ri].spec.is_some())
    }

    /// For a generic op `pc` in the fast version of a speculative region
    /// ([`in_spec`](Self::in_spec)): a label that leaves for the op in the
    /// region's plain copy (the promoted registers written back), which its
    /// slow paths take instead of their own code. Taken before the op has
    /// done anything, it is the op from scratch.
    pub(super) fn spec_bail(&mut self, a: &mut X64Assembler, pc: usize) -> Option<Label> {
        if !self.in_spec(pc) || self.copy_label(pc).is_none() {
            return None;
        }
        let entry = a.new_label();
        self.stubs.push(Stub {
            entry,
            resume: entry,
            region: self.region_at(pc),
            pc,
            kind: StubKind::SpecBail {
                pc: self.bail_pc(pc) as u32,
            },
            branch: None,
            int: false,
            copy: self.copy,
        });
        Some(entry)
    }

    /// Region `ri`'s [`Cover`] at `pc`: the operands its stub checks
    /// besides the op's own.
    fn cover(&self, ri: Option<usize>, pc: usize) -> Option<Vec<u16>> {
        let c = &self.regions[ri?].defer.covers;
        let i = c.binary_search_by_key(&pc, |c| c.pc).ok()?;
        Some(c[i].roots.clone())
    }

    /// Region op `pc`'s [`DeferPlan`] kind (`0`: its own check), in the
    /// fast version.
    fn defer_kind(&self, pc: usize) -> u8 {
        let Some(ri) = self.region_at(pc) else {
            return 0;
        };
        let k = &self.regions[ri].defer.kind;
        k.binary_search_by_key(&pc, |e| e.0).map_or(0, |i| k[i].1)
    }

    /// Where a leave of the fast version at op `pc` goes in the plain copy:
    /// the start of a merged-check stretch it lies in ([`DeferPlan`]), else
    /// the op itself.
    fn bail_pc(&self, pc: usize) -> usize {
        let Some(ri) = self.region_at(pc) else {
            return pc;
        };
        self.regions[ri]
            .defer
            .spans
            .iter()
            .find(|&&(f, l, _)| f <= pc && pc <= l)
            .map_or(pc, |x| x.2)
    }

    fn region_at(&self, pc: usize) -> Option<usize> {
        if self.copy {
            return None;
        }
        match self.region_of.get(pc) {
            Some(&r) if r != u32::MAX => Some(r as usize),
            _ => None,
        }
    }

    /// The SSE register promoted register `r` lives in within region `reg`.
    fn xmm_of(&self, reg: Option<usize>, r: u16) -> Option<u8> {
        let pc = self.cur_pc;
        self.regions[reg?]
            .slots
            .iter()
            .find(|s| s.reg == r && s.start <= pc && pc <= s.end)
            .map(|s| s.xmm)
    }

    fn ty(&self, pc: usize, r: u16) -> u8 {
        if let Some(ri) = self.region_at(pc)
            && let Some(sp) = &self.regions[ri].spec
        {
            let n = self.env.n_regs;
            return sp
                .types
                .get((pc - self.regions[ri].head) * n + r as usize)
                .copied()
                .unwrap_or(T_ANY);
        }
        self.types.get(pc, r)
    }

    /// Emits op `pc` if this module handles it (returns `false`, having emitted
    /// nothing, otherwise). The op's label is already bound.
    #[allow(clippy::too_many_lines)]
    pub(super) fn emit(
        &mut self,
        a: &mut X64Assembler,
        labels: &[Label],
        throw_exit: Label,
        pc: usize,
        ops: &[GenericOp],
    ) -> Option<bool> {
        let reg = self.region_at(pc);
        self.cur_pc = pc;
        if !self.copy && self.skip.get(pc).copied().unwrap_or(false) {
            return Some(true);
        }
        if self.imode.is_some() {
            if !self.emit_int_op(a, labels, throw_exit, pc, ops)? {
                self.ishadow_uses(a, pc)?;
                if !self.emit_dbl(a, labels, throw_exit, pc, ops, reg)? {
                    return None;
                }
            }
            return Some(true);
        }
        if let Some(ri) = reg
            && self.regions[ri].head == pc
        {
            // Region entry (fall-through, a branch from outside, OSR): try the
            // integer version first, else load the loop-carried registers,
            // then the in-region header.
            if self.regions[ri].int.is_some() {
                let entry = a.new_label();
                let dbl_entry = a.new_label();
                a.jmp(entry);
                a.bind(dbl_entry);
                if let Some(p) = self.regions[ri].int.as_mut() {
                    p.entry = Some(entry);
                    p.dbl_entry = Some(dbl_entry);
                }
            }
            let validate = self.regions[ri].spec.is_some();
            if validate {
                self.refresh_globals(a, ri, true);
                self.spec_globals_ready(a, ri, pc)?;
                self.hoist_views(a, ri);
            }
            for s in &self.regions[ri].slots {
                if s.carried {
                    a.movsd_x_mem(s.xmm, disp(s.reg));
                }
            }
            // The speculation's entry checks (nothing is dirty yet: a failed
            // one runs the iteration in the plain copy).
            if let Some(sp) = &self.regions[ri].spec {
                let copy = self.copy_label(pc)?;
                for &r in &sp.checks {
                    self.num_guard(a, reg, r, copy);
                }
            }
            if !validate {
                self.refresh_globals(a, ri, false);
            }
            let inner = a.new_label();
            a.bind(inner);
            self.regions[ri].inner = Some(inner);
        }
        let done = self.emit_dbl(a, labels, throw_exit, pc, ops, reg)?;
        if done
            && let Some(ri) = reg
            && self.regions[ri].end == pc
            && self.regions[ri].int.is_some()
        {
            self.emit_int_region(a, labels, throw_exit, ri, ops)?;
        }
        Some(done)
    }

    /// The double emitter of op `pc` (see [`emit`](Self::emit)).
    #[allow(clippy::too_many_lines)]
    fn emit_dbl(
        &mut self,
        a: &mut X64Assembler,
        labels: &[Label],
        throw_exit: Label,
        pc: usize,
        ops: &[GenericOp],
        reg: Option<usize>,
    ) -> Option<bool> {
        let op = ops[pc];
        let bc = self.bc(pc)?;
        match op {
            GenericOp::Const { dst, imm } if reg.is_some() => {
                a.movabs_rax(imm as i64);
                self.store_gpr(a, reg, RAX, dst);
            }
            GenericOp::Move { dst, src } if reg.is_some() => {
                match (self.xmm_of(reg, src), self.xmm_of(reg, dst)) {
                    (Some(xs), Some(xd)) if xs == xd => {}
                    (Some(xs), Some(xd)) => a.movapd_xx(xd, xs),
                    (Some(xs), None) => a.movsd_mem_x(disp(dst), xs),
                    (None, Some(xd)) => a.movsd_x_mem(xd, disp(src)),
                    (None, None) => {
                        a.load_rax(disp(src));
                        a.store_rax(disp(dst));
                    }
                }
            }
            GenericOp::Nop if reg.is_some() => {}
            GenericOp::Ret { src } if reg.is_some() => {
                self.flush(a, reg);
                a.load_rax(disp(src));
                a.frame_epilogue();
            }
            // A TDZ check already made, in the speculative version
            // (`plan_reuse`).
            GenericOp::CheckTdz { .. }
                if reg.is_some_and(|ri| self.regions[ri].spec.is_some())
                    && self.reuse_planned(pc).is_some() => {}
            GenericOp::CheckTdz { src, pc: opc } if reg.is_some() => {
                let stub = a.new_label();
                let resume = a.new_label();
                self.load_gpr(a, reg, RCX, src);
                a.movabs_r11(NanBox::tdz().to_bits() as i64);
                a.cmp_rcx_r11();
                a.jcc(CC_E, stub);
                a.bind(resume);
                self.stub(stub, resume, reg, StubKind::Exec { pc: opc });
            }
            GenericOp::Add { dst, a: x, b: y } => {
                let helper = self.env.add;
                self.float_bin(a, pc, reg, 0x58, dst, x, y, |_| StubKind::Helper {
                    helper,
                    a: x,
                    b: y,
                    extra: None,
                    dst,
                    throws: true,
                });
            }
            GenericOp::NumAdd {
                dst,
                a: x,
                b: y,
                pc: opc,
            } => {
                self.float_bin(a, pc, reg, 0x58, dst, x, y, |_| StubKind::Exec { pc: opc });
            }
            GenericOp::Arith {
                dst,
                a: x,
                b: y,
                op: gop,
            } => {
                let helper = self.env.arith;
                let slow = move |_: &Self| StubKind::Helper {
                    helper,
                    a: x,
                    b: y,
                    extra: Some(u64::from(gop)),
                    dst,
                    throws: true,
                };
                match gop {
                    crate::nbvm::GA_SUB => self.float_bin(a, pc, reg, 0x5c, dst, x, y, slow),
                    crate::nbvm::GA_MUL => self.float_bin(a, pc, reg, 0x59, dst, x, y, slow),
                    crate::nbvm::GA_DIV => self.float_bin(a, pc, reg, 0x5e, dst, x, y, slow),
                    _ => self.int_mod(a, pc, reg, dst, x, y, slow),
                }
            }
            GenericOp::ValueBin {
                dst,
                a: x,
                b: y,
                op: vop,
            } => {
                use crate::nbvm::{VB_LOOSE_EQ, VB_LOOSE_NEQ};
                let helper = self.env.value_bin;
                let slow = StubKind::Helper {
                    helper,
                    a: x,
                    b: y,
                    extra: Some(u64::from(vop)),
                    dst,
                    throws: true,
                };
                match vop {
                    0 => return Some(false),
                    VB_LOOSE_EQ | VB_LOOSE_NEQ => {
                        let fused = self.fusable(pc, ops, dst);
                        self.compare(
                            a,
                            labels,
                            pc,
                            reg,
                            CmpKind::Eq(vop == VB_LOOSE_NEQ),
                            dst,
                            x,
                            y,
                            slow,
                            fused,
                        )?;
                    }
                    _ => self.bitwise(a, pc, reg, vop, dst, x, y, slow),
                }
            }
            GenericOp::Lt { dst, a: x, b: y } => {
                let slow = StubKind::Helper {
                    helper: self.env.lt,
                    a: x,
                    b: y,
                    extra: None,
                    dst,
                    throws: true,
                };
                let fused = self.fusable(pc, ops, dst);
                self.compare(a, labels, pc, reg, CmpKind::Rel(0), dst, x, y, slow, fused)?;
            }
            GenericOp::Compare {
                dst,
                a: x,
                b: y,
                op: cop,
                pc: opc,
            } => {
                let fused = self.fusable(pc, ops, dst);
                self.compare(
                    a,
                    labels,
                    pc,
                    reg,
                    CmpKind::Rel(cop),
                    dst,
                    x,
                    y,
                    StubKind::Exec { pc: opc },
                    fused,
                )?;
            }
            GenericOp::StrictEq { dst, a: x, b: y } => {
                let slow = StubKind::Helper {
                    helper: self.env.strict_eq,
                    a: x,
                    b: y,
                    extra: None,
                    dst,
                    throws: false,
                };
                let fused = self.fusable(pc, ops, dst);
                self.compare(
                    a,
                    labels,
                    pc,
                    reg,
                    CmpKind::Eq(false),
                    dst,
                    x,
                    y,
                    slow,
                    fused,
                )?;
            }
            GenericOp::ToNumeric { dst, src, pc: opc } => {
                if self.fused_step(a, reg, pc, ops) {
                    return Some(true);
                }
                let resume = a.new_label();
                if self.ty(pc, src) != T_NUM {
                    let stub = a.new_label();
                    self.num_guard(a, reg, src, stub);
                    self.stub(stub, resume, reg, StubKind::Exec { pc: opc });
                }
                self.copy_reg(a, reg, dst, src);
                a.bind(resume);
            }
            GenericOp::Step {
                dst,
                src,
                dec,
                pc: opc,
            } => {
                let resume = a.new_label();
                if self.ty(pc, src) != T_NUM {
                    let stub = a.new_label();
                    self.num_guard(a, reg, src, stub);
                    self.stub(stub, resume, reg, StubKind::Exec { pc: opc });
                }
                // `x ± 1` of a canonical NaN is that NaN, so no fix-up is needed.
                self.load_xmm0(a, reg, src);
                a.movabs_rax(1.0f64.to_bits() as i64);
                a.movq_x_gpr(1, RAX);
                a.sse_rr(0xf2, if dec { 0x5c } else { 0x58 }, 0, 1);
                self.store_xmm0(a, reg, dst);
                a.bind(resume);
            }
            GenericOp::Interp { .. }
                if reg.is_some()
                    && matches!(bc, Op::LoadGlobal { .. } | Op::StoreGlobal { .. }) =>
            {
                let k = *self.gslot.get(pc)?;
                if k == u32::MAX {
                    return None;
                }
                let sd = self.gslot_disp(k);
                if let Some(ri) = reg
                    && self.regions[ri].spec.is_some()
                {
                    return self.spec_global(a, ri, pc, sd);
                }
                let retry = a.new_label();
                let stub = a.new_label();
                let resume = a.new_label();
                a.bind(retry);
                match *bc {
                    Op::LoadGlobal { dst, .. } => {
                        if (dst as usize) >= self.env.n_regs {
                            return None;
                        }
                        a.load_rax(sd);
                        a.test_rax_rax();
                        a.jcc(CC_E, stub);
                        match self.xmm_of(reg, dst) {
                            // movsd xmm, [rax]
                            Some(x) => a.sse_mem_rax(0x10, x),
                            None => {
                                a.raw(&[0x48, 0x8b, 0x00]); // mov rax, [rax]
                                self.store_gpr(a, reg, RAX, dst);
                            }
                        }
                    }
                    Op::StoreGlobal { src, .. } => {
                        if (src as usize) >= self.env.n_regs {
                            return None;
                        }
                        let mirror = super::global_mirror::mirror_cell(bc);
                        a.load_rcx(sd);
                        a.raw(&[0x48, 0x85, 0xc9]); // test rcx, rcx
                        a.jcc(CC_E, stub);
                        match self.xmm_of(reg, src) {
                            Some(x) => {
                                a.sse_mem_rcx(0x11, x); // movsd [rcx], xmm
                                // (A promoted register holds any value's
                                // bits, a heap handle's among them.)
                                if let Some(cell) = mirror {
                                    a.movq_gpr_x(RAX, x);
                                    let h = a.new_label();
                                    a.gm_store_gpr(cell, RCX, RAX, h);
                                    self.stub(h, resume, reg, StubKind::Exec { pc: pc as u32 });
                                }
                            }
                            None => {
                                self.load_gpr(a, reg, RAX, src);
                                a.raw(&[0x48, 0x89, 0x01]); // mov [rcx], rax
                                // A heap handle's mirror needs the write
                                // barrier: the interpreter runs the op.
                                if let Some(cell) = mirror {
                                    let h = a.new_label();
                                    a.gm_store_gpr(cell, RCX, RAX, h);
                                    self.stub(h, resume, reg, StubKind::Exec { pc: pc as u32 });
                                }
                            }
                        }
                    }
                    _ => return None,
                }
                a.bind(resume);
                self.stub(stub, resume, reg, StubKind::GlobalRef { pc, retry });
            }
            GenericOp::Interp { .. }
                if reg.is_none()
                    && matches!(
                        bc,
                        Op::LoadGlobal { .. } | Op::RequireObjectCoercible { .. }
                    ) =>
            {
                let op_ptr = self.env.bytecode.wrapping_add(pc) as i64;
                let slow = a.new_label();
                let done = a.new_label();
                match *bc {
                    Op::LoadGlobal { dst, .. } => {
                        if (dst as usize) >= self.env.n_regs {
                            return None;
                        }
                        // A plain binding: read it in place (no one-op
                        // interpreter frame).
                        a.mov_rdi_r15();
                        a.movabs_argreg(1, op_ptr);
                        a.movabs_rax(self.env.global_ref as i64);
                        a.call_rax();
                        a.test_rax_rax();
                        a.jcc(CC_E, slow);
                        a.raw(&[0x48, 0x8b, 0x00]); // mov rax, [rax]
                        a.store_rax(disp(dst));
                    }
                    Op::RequireObjectCoercible { src, .. } => {
                        if (src as usize) >= self.env.n_regs {
                            return None;
                        }
                        // Only `undefined` / `null` throw.
                        a.load_rax(disp(src));
                        a.movabs_r11(NanBox::undefined().to_bits() as i64);
                        a.cmp_rax_r11();
                        a.jcc(CC_E, slow);
                        a.movabs_r11(NanBox::null().to_bits() as i64);
                        a.cmp_rax_r11();
                        a.jcc(CC_E, slow);
                    }
                    _ => return None,
                }
                a.jmp(done);
                a.bind(slow);
                a.mov_rdi_r15();
                a.mov_rsi_rbx();
                a.movabs_argreg(2, self.env.n_regs as i64);
                a.movabs_argreg(3, op_ptr);
                a.movabs_rax(self.env.exec_op as i64);
                a.call_rax();
                a.movabs_r11(NanBox::jit_throw_bits() as i64);
                a.cmp_rax_r11();
                a.je(throw_exit);
                a.bind(done);
            }
            GenericOp::Interp { pc: opc } if matches!(bc, Op::Neg { .. } | Op::BitNot { .. }) => {
                let (dst, src, neg) = match *bc {
                    Op::Neg { dst, a } => (dst, a, true),
                    Op::BitNot { dst, a } => (dst, a, false),
                    _ => return Some(false),
                };
                if (src as usize) >= self.env.n_regs || (dst as usize) >= self.env.n_regs {
                    return None;
                }
                let resume = a.new_label();
                let stub = a.new_label();
                if self.ty(pc, src) != T_NUM {
                    self.num_guard(a, reg, src, stub);
                }
                if neg {
                    self.load_gpr(a, reg, RAX, src);
                    a.movabs_r11(SIGN as i64);
                    a.raw(&[0x4c, 0x31, 0xd8]); // xor rax, r11
                    a.movq_x_gpr(0, RAX);
                    self.canon_xmm0(a, reg);
                } else {
                    self.to_int32(a, reg, RAX, src, stub);
                    a.not_eax();
                    a.to_int32_rax();
                    a.cvtsi2sd_x_gpr(0, RAX);
                }
                self.store_xmm0(a, reg, dst);
                a.bind(resume);
                self.stub(stub, resume, reg, StubKind::Exec { pc: opc });
            }
            GenericOp::JumpIfFalse { cond, target } => {
                let back = target <= pc;
                if back && reg.is_none() {
                    return Some(false);
                }
                let t = self.ty(pc, cond);
                if reg.is_none() && t != T_NUM && t != T_BOOL {
                    return Some(false);
                }
                if back {
                    // An in-region backward conditional branch is never a region's
                    // (the planner admits only unconditional back-edges).
                    return None;
                }
                self.branch_if_false(a, labels, pc, reg, cond, target);
            }
            GenericOp::Jump { target } if reg.is_some() => {
                let ri = reg?;
                if target <= pc {
                    // The region's back-edge (or, in a loop nest, an inner
                    // loop's): poll, then the in-region header.
                    let nested = pc != self.regions[ri].end || target != self.regions[ri].head;
                    if nested && !self.regions[ri].loops.contains(&(target, pc)) {
                        return None;
                    }
                    let skip = a.new_label();
                    a.dec_frame8(COUNTDOWN);
                    a.jne(skip);
                    super::call_seq::emit_quiet_poll(
                        a,
                        &self.env.layout,
                        i32::from(COUNTDOWN),
                        skip,
                    );
                    self.flush(a, reg);
                    a.mov_rdi_r15();
                    a.mov_rsi_rbx();
                    a.movabs_argreg(2, self.env.n_regs as i64);
                    a.movabs_argreg(3, self.env.bytecode as i64);
                    a.movabs_argreg(4, self.env.bytecode_len as i64);
                    a.movabs_argreg(5, target as i64);
                    a.movabs_rax(self.env.backedge as i64);
                    a.call_rax();
                    a.movabs_r11(NanBox::jit_throw_bits() as i64);
                    a.cmp_rax_r11();
                    a.je(throw_exit);
                    // Leaving at the header: the flush above wrote back
                    // everything the interpreter can read there.
                    a.movabs_r11(NanBox::jit_exit_bits() as i64);
                    a.cmp_rax_r11();
                    a.je(self.leave_exit?);
                    a.store_rax_frame8(COUNTDOWN);
                    let validate = self.regions[ri].spec.is_some();
                    if validate {
                        self.refresh_globals(a, ri, true);
                        self.spec_globals_ready(a, ri, target)?;
                        self.hoist_views(a, ri);
                    }
                    self.reload(a, reg);
                    if !validate {
                        self.refresh_globals(a, ri, false);
                    }
                    a.bind(skip);
                    if nested {
                        a.jmp(*labels.get(target)?);
                    } else {
                        a.jmp(self.regions[ri].inner?);
                    }
                } else {
                    self.jump(a, labels, reg, target);
                }
            }
            GenericOp::Interp { pc: opc }
                if reg.is_some() && matches!(bc, Op::RequireObjectCoercible { .. }) =>
            {
                let Op::RequireObjectCoercible { src, .. } = *bc else {
                    return None;
                };
                if (src as usize) >= self.env.n_regs {
                    return None;
                }
                if self.roc_folded(pc) {
                    // The `Math.f(x)` call after it checks instead.
                    return Some(true);
                }
                // Only `undefined` / `null` throw (the interpreter's op).
                let stub = a.new_label();
                let resume = a.new_label();
                self.load_gpr(a, reg, RAX, src);
                a.movabs_r11(NanBox::undefined().to_bits() as i64);
                a.cmp_rax_r11();
                a.jcc(CC_E, stub);
                a.movabs_r11(NanBox::null().to_bits() as i64);
                a.cmp_rax_r11();
                a.jcc(CC_E, stub);
                a.bind(resume);
                self.stub(stub, resume, reg, StubKind::Exec { pc: opc });
            }
            // A captured binding's read whose value a register already
            // holds, in the speculative version (`plan_reuse`).
            GenericOp::CellGet { dst, .. }
                if reg.is_some_and(|ri| self.regions[ri].spec.is_some())
                    && let Some(src) = self.reuse_planned(pc) =>
            {
                if (dst as usize) >= self.env.n_regs || (src as usize) >= self.env.n_regs {
                    return None;
                }
                self.copy_reg(a, reg, dst, src);
            }
            // A property read whose value a register already holds, in the
            // speculative version (`plan_reuse`).
            GenericOp::GetProp { dst, .. }
                if reg.is_some_and(|ri| self.regions[ri].spec.is_some())
                    && let Some(src) = self.reuse_planned(pc) =>
            {
                if (dst as usize) >= self.env.n_regs || (src as usize) >= self.env.n_regs {
                    return None;
                }
                self.copy_reg(a, reg, dst, src);
            }
            // The generic emitter's (see `generic_region_op`).
            _ if reg.is_some() && generic_region_op(&op, bc) => return Some(false),
            _ => {
                if reg.is_some() {
                    // The planner admitted an op this emitter does not handle.
                    return None;
                }
                return Some(false);
            }
        }
        Some(true)
    }

    /// The promoted registers live at region op `pc` that a call in the
    /// generic emitter's code for it must store before it and reload after
    /// it, as `(xmm, register-file displacement, store first)` (empty outside
    /// a region). A temporary op `pc` first defines holds nothing yet: it is
    /// reloaded but never stored.
    pub(super) fn call_spill(&self, pc: usize) -> Vec<(u8, i32, bool)> {
        let Some(ri) = self.region_at(pc) else {
            return Vec::new();
        };
        let rg = &self.regions[ri];
        rg.slots
            .iter()
            .filter(|s| s.start <= pc && pc <= s.end)
            .map(|s| (s.xmm, disp(s.reg), s.dirty && rg.defined(s, pc, pc)))
            .collect()
    }

    /// The SSE register a region op `pc`'s result register `dst` is promoted
    /// to: the generic emitter's fast path may leave the result there only
    /// (the register is dirty, so a flush stores it), and must then skip the
    /// reload [`after_generic`](Self::after_generic) emits for its slow
    /// paths.
    pub(super) fn result_xmm(&self, pc: usize, dst: u16) -> Option<u8> {
        let ri = self.region_at(pc)?;
        self.regions[ri]
            .slots
            .iter()
            .find(|s| s.reg == dst && s.start <= pc && pc <= s.end)
            .map(|s| s.xmm)
    }

    /// For a generic region op `pc` that may read operand `r` from an SSE
    /// register (a `SetProp`'s value, an inline `Math.f(x)`'s argument: see
    /// `plan_regions`): the register, when `r` is promoted there and the op
    /// is in a speculative fast version with `r` a Number — every path of
    /// the op but its inline one then leaves for the plain copy, through a
    /// flush that stores `r`. Otherwise `r` (if promoted) is first stored
    /// to the register file, which every path of the op then reads.
    /// (`inline`: whether the op's code has its inline path at all.)
    pub(super) fn operand_xmm(
        &self,
        a: &mut X64Assembler,
        pc: usize,
        r: u16,
        inline: bool,
    ) -> Option<u8> {
        let x = self.result_xmm(pc, r)?;
        if inline && self.in_spec(pc) && self.copy_label(pc).is_some() && self.is_number_at(pc, r) {
            return Some(x);
        }
        a.movsd_mem_x(disp(r), x);
        None
    }

    /// Whether register `r` provably holds a Number on entry to op `pc` (in
    /// a speculative region, under its speculation).
    pub(super) fn is_number_at(&self, pc: usize, r: u16) -> bool {
        self.ty(pc, r) == T_NUM
    }

    /// Whether the `RequireObjectCoercible` at `pc` of a speculative region's
    /// fast version emits nothing, its check folded into the inline
    /// `Math.f(x)` call on the same register right after it (no entry in
    /// between): that call's fast path proves its receiver an object (a
    /// heap handle walked to a plain object holding the built-in), and its
    /// every other path leaves for the plain copy *at this op* — which then
    /// runs the check, and throws, exactly as the interpreter would
    /// (nothing has happened since: the check has no other effect).
    pub(super) fn roc_folded(&self, pc: usize) -> bool {
        let (
            Some(&Op::RequireObjectCoercible { src, .. }),
            Some(call @ Op::CallMethod { recv, .. }),
        ) = (self.bc(pc), self.bc(pc + 1))
        else {
            return false;
        };
        src == *recv
            && super::inline_math(call).is_some()
            && self.in_spec(pc)
            && self.in_spec(pc + 1)
            && self.region_at(pc) == self.region_at(pc + 1)
            && !self.is_target.get(pc + 1).copied().unwrap_or(true)
            && self.copy_label(pc).is_some()
    }

    /// The register whose value the speculative version of region op `pc`
    /// (a `GetProp`) copies instead of reading the property ([`plan_reuse`]),
    /// in either version: the plain copy reads it, and the receiver cache
    /// must not count on that op's way access ([`recv_cache::plan`]).
    pub(super) fn reuse_planned(&self, pc: usize) -> Option<u16> {
        let ri = *self.region_of.get(pc)?;
        let r = self.regions.get(ri as usize)?;
        r.spec.as_ref()?;
        r.reuse.iter().find(|x| x.0 == pc).map(|x| x.1)
    }

    /// For an array iteration's step `pc` (`IterNext`) followed by the
    /// loop test `t = !done; if (!t) goto exit`: where the step's
    /// present-element path continues, the test skipped — that path leaves
    /// `done` as it found it, `false` (the step tests it is not `true`, and
    /// it is provably a boolean), so `t` is `true` and the branch falls
    /// through. Only when nothing else names `t` (dead past the branch),
    /// nothing branches to the test, and the loop body (`pc + 3`) lies in
    /// the same region (or both in none): the label in the version being
    /// emitted.
    pub(super) fn iter_step_body(&self, labels: &[Label], pc: usize) -> Option<Label> {
        let (
            Some(&Op::IterNext { done, .. }),
            Some(&Op::Not { dst: t, a }),
            Some(&Op::JumpIfFalse { cond, .. }),
        ) = (self.bc(pc), self.bc(pc + 1), self.bc(pc + 2))
        else {
            return None;
        };
        if a != done
            || cond != t
            || t == done
            || self.mentions.get(t as usize).copied() != Some(2)
            || self.is_target.get(pc + 1).copied().unwrap_or(true)
            || self.is_target.get(pc + 2).copied().unwrap_or(true)
            || !self.is_bool_at(pc, done)
            || pc + 3 >= self.region_of.len()
            || self.region_of[pc] != self.region_of[pc + 3]
        {
            return None;
        }
        // (No promoted register the step writes, which `after_generic`
        // would reload after it.)
        let (mut d, mut u) = (Vec::new(), Vec::new());
        def_use(self.bc(pc)?, &mut d, &mut u);
        if d.iter().any(|&r| self.result_xmm(pc, r).is_some()) {
            return None;
        }
        if self.copy {
            self.copy_label(pc + 3)
        } else {
            labels.get(pc + 3).copied()
        }
    }

    /// Whether register `r` provably holds a boolean on entry to op `pc` (in
    /// a speculative region, under its speculation).
    pub(super) fn is_bool_at(&self, pc: usize, r: u16) -> bool {
        self.ty(pc, r) == T_BOOL
    }

    /// `xmm = rax` (the bits).
    pub(super) fn set_xmm_from_rax(a: &mut X64Assembler, x: u8) {
        a.movq_x_gpr(x, RAX);
    }

    /// `xmm0 = (f64) (r8 + 1)`.
    pub(super) fn inc_r8_to_xmm0(a: &mut X64Assembler) {
        a.raw(&[0x49, 0xff, 0xc0]); // inc r8
        a.raw(&[0x0f, 0x57, 0xc0]); // xorps xmm0, xmm0 (no false dependency)
        a.raw(&[0xf2, 0x49, 0x0f, 0x2a, 0xc0]); // cvtsi2sd xmm0, r8
    }

    /// `rax` = the exact non-negative-or-not integer value of the Number in
    /// `xmm x` (an element read's promoted index), else to `miss` (a
    /// fraction, NaN, out of `i64`'s range). A non-Number (`check`: `x` is
    /// not proven a Number) needs no test of its own: every other `NanBox`
    /// reads as a NaN, which the exactness test rejects. Clobbers `xmm1`.
    pub(super) fn index_from_xmm(a: &mut X64Assembler, x: u8, check: bool, miss: Label) {
        let _ = check;
        a.cvttsd2si_gpr_x(RAX, x);
        a.cvtsi2sd_x_gpr(1, RAX);
        a.sse_rr(0x66, 0x2e, x, 1); // ucomisd x, xmm1
        a.jcc(CC_NE, miss);
        a.jcc(CC_P, miss);
    }

    /// `rax` = the truncation of the Number in `xmm x` (an element read's
    /// promoted index proven integral, `is_integral_at`).
    pub(super) fn trunc_from_xmm(a: &mut X64Assembler, x: u8) {
        a.cvttsd2si_gpr_x(RAX, x);
    }

    /// Whether register `r` surely holds an integral Number on entry to op
    /// `pc` ([`I_INT`]).
    pub(super) fn is_integral_at(&self, pc: usize, r: u16) -> bool {
        self.ints.get(pc, r) == I_INT
    }

    /// `gpr = xmm x` (the bits; `gpr` one of `rax`/`rcx`/`rdx`).
    pub(super) fn gpr_from_xmm(a: &mut X64Assembler, gpr: u8, x: u8) {
        a.movq_gpr_x(gpr, x);
    }

    /// `xmm = (f64) rax`.
    pub(super) fn set_xmm_from_int_rax(a: &mut X64Assembler, x: u8) {
        a.cvtsi2sd_x_gpr(x, RAX);
    }

    /// `xmm0 = xmm x`.
    pub(super) fn xmm0_from_xmm(a: &mut X64Assembler, x: u8) {
        a.movapd_xx(0, x);
    }

    /// `xmm d = sqrt(xmm s)` (of a Number: a NaN result is the canonical NaN).
    pub(super) fn sqrt_xmm(a: &mut X64Assembler, d: u8, s: u8) {
        a.sse_rr(0xf2, 0x51, d, s); // sqrtsd d, s
    }

    /// `xmm = xmm0`.
    pub(super) fn set_xmm_from_xmm0(a: &mut X64Assembler, x: u8) {
        a.movapd_xx(x, 0);
    }

    /// After the generic emitter's code for region op `pc`: a promoted
    /// register it wrote (in the register file) is reloaded into its SSE
    /// register.
    pub(super) fn after_generic(&self, a: &mut X64Assembler, pc: usize) {
        let Some(ri) = self.region_at(pc) else { return };
        let Some(bc) = self.bc(pc) else { return };
        let (mut defs, mut uses) = (Vec::new(), Vec::new());
        def_use(bc, &mut defs, &mut uses);
        for s in &self.regions[ri].slots {
            if defs.contains(&s.reg) && s.start <= pc && pc <= s.end {
                a.movsd_x_mem(s.xmm, disp(s.reg));
            }
        }
    }

    /// Before the general `CallMethod` code of op `pc`: when it is a
    /// one-argument call of a `Math` function name, ask `jit_helper_math1`
    /// (the built-in still installed, a Number argument) for the result and
    /// jump to `done` with it stored; on a miss fall through to the general
    /// call. Emits nothing for any other call.
    pub(super) fn math_call(&self, a: &mut X64Assembler, pc: usize, done: Label) {
        let Some(Op::CallMethod {
            dst,
            recv,
            key,
            args,
        }) = self.bc(pc)
        else {
            return;
        };
        let n = self.env.n_regs;
        if args.len() != 1
            || (*dst as usize) >= n
            || (*recv as usize) >= n
            || (args[0] as usize) >= n
            || !matches!(
                key.as_str(),
                "sqrt"
                    | "abs"
                    | "floor"
                    | "ceil"
                    | "trunc"
                    | "round"
                    | "sign"
                    | "exp"
                    | "log"
                    | "sin"
                    | "cos"
                    | "tan"
                    | "atan"
                    | "cbrt"
            )
        {
            return;
        }
        let miss = a.new_label();
        a.mov_rdi_r15();
        a.load_argreg(1, disp(*recv));
        a.load_argreg(2, disp(args[0]));
        a.movabs_argreg(3, self.env.bytecode.wrapping_add(pc) as i64);
        a.movabs_rax(self.env.math1 as i64);
        a.call_rax();
        a.movabs_r11(NanBox::jit_throw_bits() as i64);
        a.cmp_rax_r11();
        a.jcc(CC_E, miss);
        a.store_rax(disp(*dst));
        a.jmp(done);
        a.bind(miss);
    }

    fn bc(&self, pc: usize) -> Option<&Op> {
        if pc >= self.env.bytecode_len {
            return None;
        }
        // SAFETY: `bytecode` points at `bytecode_len` ops that outlive the
        // compilation (`compile_generic`'s `bytecode` argument).
        #[allow(unsafe_code)]
        Some(unsafe { &*self.env.bytecode.add(pc) })
    }

    /// Whether the comparison at `pc` writing `dst` can branch directly into the
    /// `JumpIfFalse` that follows it: `Some((target, store))`, `store` false when
    /// nothing else can read `dst` (the boolean need not be materialized).
    fn fusable(&self, pc: usize, ops: &[GenericOp], dst: u16) -> Option<(usize, bool)> {
        if self.copy {
            return None;
        }
        match ops.get(pc + 1) {
            Some(&GenericOp::JumpIfFalse { cond, target }) if cond == dst && target > pc + 1 => {
                let others = self.mentions.get(dst as usize).copied().unwrap_or(u32::MAX);
                let store = others > 2 || self.is_target.get(pc + 1).copied().unwrap_or(true);
                // Fusing skips op `pc + 1`'s own code on the fast path, so that op
                // must belong to the same region as `pc`.
                (self.region_at(pc + 1) == self.region_at(pc)).then_some((target, store))
            }
            _ => None,
        }
    }

    fn stub(&mut self, entry: Label, resume: Label, region: Option<usize>, kind: StubKind) {
        // The integer version runs no slow path itself: it hands the op to the
        // double version.
        let kind = match kind {
            StubKind::Helper { .. } | StubKind::Exec { .. } | StubKind::Truthy { .. }
                if self.imode.is_some() =>
            {
                StubKind::Bail {
                    pc: self.cur_pc as u32,
                    store: true,
                }
            }
            // In a speculative region every slow path that would run Rust
            // code leaves for the plain copy (the fast version makes no call:
            // see `in_spec`).
            StubKind::Helper { .. } | StubKind::Exec { .. } | StubKind::Truthy { .. }
                if region.is_some_and(|ri| self.regions[ri].spec.is_some()) =>
            {
                StubKind::SpecBail {
                    pc: self.bail_pc(self.cur_pc) as u32,
                }
            }
            k => k,
        };
        self.stubs.push(Stub {
            entry,
            resume,
            region,
            pc: self.cur_pc,
            kind,
            branch: None,
            int: self.imode.is_some(),
            copy: self.copy,
        });
    }

    /// Stores the promoted registers live at the current pc that the region
    /// writes back to the register file.
    fn flush(&self, a: &mut X64Assembler, reg: Option<usize>) {
        let Some(ri) = reg else { return };
        if self.imode == Some(ri) {
            self.iflush(a, ri);
            return;
        }
        let pc = self.cur_pc;
        let to = self.bail_pc(pc);
        let rg = &self.regions[ri];
        for s in &rg.slots {
            // A temporary op `pc` first defines holds no value yet (its first
            // mention is a write: a read would make it live at the header).
            if s.dirty && s.start <= pc && pc <= s.end && rg.defined(s, pc, to) {
                a.movsd_mem_x(disp(s.reg), s.xmm);
            }
        }
    }

    /// Reloads the promoted registers live at the current pc from the register
    /// file (after a call: the SSE registers are caller-saved).
    fn reload(&self, a: &mut X64Assembler, reg: Option<usize>) {
        let Some(ri) = reg else { return };
        if self.imode == Some(ri) {
            self.ireload(a, ri);
            return;
        }
        let pc = self.cur_pc;
        // A temporary op `pc` defines last: it may share its SSE register
        // with an operand that ends at `pc` (`plan_regions`).
        for first in [true, false] {
            for s in &self.regions[ri].slots {
                if s.start <= pc && pc <= s.end && (s.start != pc || s.carried) == first {
                    a.movsd_x_mem(s.xmm, disp(s.reg));
                }
            }
        }
    }

    /// `gpr = bits of register r`.
    fn load_gpr(&self, a: &mut X64Assembler, reg: Option<usize>, gpr: u8, r: u16) {
        match self.xmm_of(reg, r) {
            Some(x) => a.movq_gpr_x(gpr, x),
            None => a.mov_gpr_mem(gpr, disp(r)),
        }
    }

    /// `register r = bits in gpr`.
    fn store_gpr(&self, a: &mut X64Assembler, reg: Option<usize>, gpr: u8, r: u16) {
        match self.xmm_of(reg, r) {
            Some(x) => a.movq_x_gpr(x, gpr),
            None => a.mov_mem_gpr(disp(r), gpr),
        }
    }

    /// `xmm0 = register r`.
    fn load_xmm0(&self, a: &mut X64Assembler, reg: Option<usize>, r: u16) {
        match self.xmm_of(reg, r) {
            Some(x) => a.movapd_xx(0, x),
            None => a.movsd_x_mem(0, disp(r)),
        }
    }

    /// The SSE register holding register `r` (promoted), or `scratch` loaded
    /// with it.
    fn xmm_src(&self, a: &mut X64Assembler, reg: Option<usize>, r: u16, scratch: u8) -> u8 {
        match self.xmm_of(reg, r) {
            Some(x) => x,
            None => {
                a.movsd_x_mem(scratch, disp(r));
                scratch
            }
        }
    }

    /// `register r = xmm0`.
    fn store_xmm0(&self, a: &mut X64Assembler, reg: Option<usize>, r: u16) {
        match self.xmm_of(reg, r) {
            Some(x) => a.movapd_xx(x, 0),
            None => a.movsd_mem_x(disp(r), 0),
        }
    }

    /// `register dst = register src`.
    fn copy_reg(&self, a: &mut X64Assembler, reg: Option<usize>, dst: u16, src: u16) {
        if dst == src {
            return;
        }
        match (self.xmm_of(reg, src), self.xmm_of(reg, dst)) {
            (Some(xs), Some(xd)) => a.movapd_xx(xd, xs),
            (Some(xs), None) => a.movsd_mem_x(disp(dst), xs),
            (None, Some(xd)) => a.movsd_x_mem(xd, disp(src)),
            (None, None) => {
                a.load_rax(disp(src));
                a.store_rax(disp(dst));
            }
        }
    }

    /// Jumps to `stub` unless registers `x` and `y` (those not proven) are
    /// Numbers. A promoted one is tested by `ucomisd` alone: only a
    /// NaN-patterned value (a NaN, or any non-Number) takes the exact test,
    /// out of line ([`StubKind::NanCheck`]). Clobbers `rcx`/`r11`.
    fn guard_numbers(
        &mut self,
        a: &mut X64Assembler,
        pc: usize,
        reg: Option<usize>,
        x: u16,
        y: u16,
        stub: Label,
    ) {
        let mut mask = false;
        for (i, r) in [x, y].into_iter().enumerate() {
            if self.ty(pc, r) == T_NUM || (i == 1 && y == x) {
                continue;
            }
            match self.xmm_of(reg, r) {
                Some(xr) if self.imode.is_none() => {
                    let check = a.new_label();
                    let back = a.new_label();
                    a.sse_rr(0x66, 0x2e, xr, xr); // ucomisd xr, xr
                    a.jcc(CC_P, check);
                    a.bind(back);
                    self.stub(check, back, reg, StubKind::NanCheck { x: xr, slow: stub });
                }
                _ => {
                    if !mask {
                        a.movabs_r11(QNAN as i64);
                        mask = true;
                    }
                    self.load_gpr(a, reg, RCX, r);
                    a.and_rcx_r11();
                    a.cmp_rcx_r11();
                    a.jcc(CC_E, stub);
                }
            }
        }
    }

    /// Jumps to `stub` unless register `r` is a Number. Clobbers `rcx`/`r11`.
    fn num_guard(&self, a: &mut X64Assembler, reg: Option<usize>, r: u16, stub: Label) {
        self.load_gpr(a, reg, RCX, r);
        a.movabs_r11(QNAN as i64);
        a.and_rcx_r11();
        a.cmp_rcx_r11();
        a.jcc(CC_E, stub);
    }

    /// Canonicalizes a NaN in `xmm0` (out of line).
    fn canon_xmm0(&mut self, a: &mut X64Assembler, reg: Option<usize>) {
        let stub = a.new_label();
        let resume = a.new_label();
        a.sse_rr(0x66, 0x2e, 0, 0); // ucomisd xmm0, xmm0
        a.jcc(CC_P, stub);
        a.bind(resume);
        self.stub(stub, resume, reg, StubKind::CanonNan);
    }

    /// `gpr = ToInt32(register r)` (in the low 32 bits) for a Number `r`, or a
    /// jump to `stub` for NaN / ±∞ / |x| ≥ 2^63 (the helper's case). Clobbers
    /// `r11` and `xmm1`.
    fn to_int32(&self, a: &mut X64Assembler, reg: Option<usize>, gpr: u8, r: u16, stub: Label) {
        let x = self.xmm_src(a, reg, r, 1);
        a.cvttsd2si_gpr_x(gpr, x);
        a.movabs_r11(INDEFINITE as i64);
        a.cmp_gpr_r11(gpr);
        a.jcc(CC_E, stub);
    }

    /// `dst = x <sse op> y` over two Numbers (`opc`: `0x58` add, `0x5c` sub,
    /// `0x59` mul, `0x5e` div), guarded unless proven.
    #[allow(clippy::too_many_arguments)]
    fn float_bin(
        &mut self,
        a: &mut X64Assembler,
        pc: usize,
        reg: Option<usize>,
        opc: u8,
        dst: u16,
        x: u16,
        y: u16,
        slow: impl FnOnce(&Self) -> StubKind,
    ) {
        let resume = a.new_label();
        if let Some(ri) = reg {
            for &(at, r, xmm) in &self.regions[ri].defer.saves {
                if at == pc {
                    a.movsd_mem_x(disp(r), xmm);
                }
            }
        }
        let guarded = self.ty(pc, x) != T_NUM || self.ty(pc, y) != T_NUM;
        let stub = a.new_label();
        // An operand not proven a Number is checked *after* the arithmetic,
        // through the result: a NaN-boxed value (anything but a Number) is
        // a NaN to the hardware, so the result of any operation on it is a
        // NaN, and only then (out of line, `StubKind::DeferNan`) are the
        // operands looked at — one that is not a Number takes the slow
        // path, which runs the op from scratch (the operands are intact:
        // the result is computed where no operand lives); else the NaN is a
        // Number's, the canonical one.
        let defer = guarded && self.imode.is_none();
        // A check merged into a later one ([`DeferPlan`]): none here.
        let dk = if defer { self.defer_kind(pc) } else { 0 };
        let cover = dk == DEFER_COVER;
        if guarded && !defer {
            self.guard_numbers(a, pc, reg, x, y, stub);
        }
        let unproven = |s: &Self, r: u16| (s.ty(pc, r) != T_NUM).then_some(r);
        let (ux, uy) = (unproven(self, x), unproven(self, y).filter(|&r| r != x));
        match self.xmm_of(reg, dst) {
            Some(xd) if defer => {
                let (xx, xy) = (self.xmm_of(reg, x), self.xmm_of(reg, y));
                // Computed in its own register. When that holds a value a
                // slow path still needs (an operand's, or the register's
                // own before this op), the value is first saved in `xmm0`,
                // which the check's stub puts back (the result is then a
                // NaN anyway); a non-commuting op into its second operand's
                // register goes through `xmm0` instead.
                let fresh = reg.is_some_and(|ri| {
                    self.regions[ri]
                        .slots
                        .iter()
                        .any(|s| s.reg == dst && s.start == pc && !s.carried && s.xmm == xd)
                }) && xx != Some(xd)
                    && xy != Some(xd);
                let commutes = opc == 0x58 || opc == 0x59;
                let shared_y = xy == Some(xd) && xx != Some(xd);
                let st = if shared_y && !commutes { 0 } else { xd };
                let restore = !fresh && st == xd && dk != DEFER_SKIP;
                if restore {
                    a.movapd_xx(0, xd);
                }
                if shared_y && commutes {
                    let xs = self.xmm_src(a, reg, x, 1);
                    a.sse_rr(0xf2, opc, xd, xs);
                } else {
                    match xx {
                        Some(s) if s == st => {}
                        Some(s) => a.movapd_xx(st, s),
                        None => a.movsd_x_mem(st, disp(x)),
                    }
                    let ys = self.xmm_src(a, reg, y, 1);
                    a.sse_rr(0xf2, opc, st, ys);
                }
                if dk != DEFER_SKIP {
                    let check = a.new_label();
                    let back = a.new_label();
                    a.sse_rr(0x66, 0x2e, st, st); // ucomisd st, st
                    a.jcc(CC_P, check);
                    a.bind(back);
                    self.stub(
                        check,
                        back,
                        reg,
                        StubKind::DeferNan {
                            x: st,
                            regs: [ux, uy],
                            slow: stub,
                            restore,
                            cover,
                        },
                    );
                }
                if st != xd {
                    a.movapd_xx(xd, st);
                }
            }
            // A promoted result in a region: computed in its own register
            // (`x` copied in first, or `y` when `+` / `*` can commute).
            Some(xd) if self.imode.is_none() => {
                let (xx, xy) = (self.xmm_of(reg, x), self.xmm_of(reg, y));
                if xy == Some(xd) && xx != Some(xd) {
                    if opc == 0x58 || opc == 0x59 {
                        let xs = self.xmm_src(a, reg, x, 1);
                        a.sse_rr(0xf2, opc, xd, xs);
                    } else {
                        self.load_xmm0(a, reg, x);
                        a.sse_rr(0xf2, opc, 0, xd);
                        a.movapd_xx(xd, 0);
                    }
                } else {
                    match xx {
                        Some(s) if s == xd => {}
                        Some(s) => a.movapd_xx(xd, s),
                        None => a.movsd_x_mem(xd, disp(x)),
                    }
                    let ys = self.xmm_src(a, reg, y, 1);
                    a.sse_rr(0xf2, opc, xd, ys);
                }
                // (A NaN of Numbers is the canonical NaN.)
            }
            _ => {
                self.load_xmm0(a, reg, x);
                let xy = self.xmm_src(a, reg, y, 1);
                a.sse_rr(0xf2, opc, 0, xy);
                if dk == DEFER_SKIP {
                    // (Checked later, [`DeferPlan`].)
                } else if defer {
                    let check = a.new_label();
                    let back = a.new_label();
                    a.sse_rr(0x66, 0x2e, 0, 0); // ucomisd xmm0, xmm0
                    a.jcc(CC_P, check);
                    a.bind(back);
                    self.stub(
                        check,
                        back,
                        reg,
                        StubKind::DeferNan {
                            x: 0,
                            regs: [ux, uy],
                            slow: stub,
                            restore: false,
                            cover,
                        },
                    );
                }
                // (A NaN of Numbers is the canonical NaN.)
                self.store_xmm0(a, reg, dst);
            }
        }
        a.bind(resume);
        if guarded && dk != DEFER_SKIP {
            let k = slow(self);
            self.stub(stub, resume, reg, k);
        }
    }

    /// The integer constant register `r` surely holds on entry to op `pc`: the
    /// `LoadConst` just before writes it, and nothing else reaches `pc`.
    fn const_int_at(&self, pc: usize, r: u16) -> Option<i64> {
        if pc == 0 || self.is_target.get(pc).copied().unwrap_or(true) {
            return None;
        }
        let Some(Op::LoadConst { dst, value }) = self.bc(pc - 1) else {
            return None;
        };
        let v = value.as_number().filter(|_| *dst == r)?;
        // (`-0.0` is not an integer constant here: `x % -0` is NaN.)
        (v == v.trunc() && v.abs() < 9.0e15 && !(v == 0.0 && v.is_sign_negative()))
            .then_some(v as i64)
    }

    /// `dst = x % y`: inline when both are integral Numbers that convert
    /// exactly to i64, with `y` neither `0` nor `-1` (no `idiv` overflow): the
    /// integer remainder is then fmod's exact result (the sign of a zero
    /// result follows `x`, `-0` included); else the helper (`fmod`).
    ///
    /// The exactness test also rejects any operand that is not a Number:
    /// every other `NanBox` reads as a NaN `f64`, whose round trip through
    /// an integer compares unordered. A divisor the `LoadConst` just before
    /// sets (`const_int_at`) is compiled in.
    #[allow(clippy::too_many_arguments)]
    fn int_mod(
        &mut self,
        a: &mut X64Assembler,
        pc: usize,
        reg: Option<usize>,
        dst: u16,
        x: u16,
        y: u16,
        slow: impl FnOnce(&Self) -> StubKind,
    ) {
        let resume = a.new_label();
        let stub = a.new_label();
        match self.const_int_at(pc, y).filter(|&k| k != 0 && k != -1) {
            // rcx = the divisor, compiled in.
            Some(k) => a.g_movabs(RCX, k as u64),
            None => {
                // rcx = int(y), exact (an out-of-range y converts to
                // i64::MIN, which only -2^63 itself converts back to), neither
                // 0 nor -1.
                let xy = self.xmm_src(a, reg, y, 1);
                a.cvttsd2si_gpr_x(RCX, xy);
                a.raw(&[0x48, 0x85, 0xc9]); // test rcx, rcx
                a.jcc(CC_E, stub);
                a.raw(&[0x48, 0x83, 0xf9, 0xff]); // cmp rcx, -1
                a.jcc(CC_E, stub);
                a.cvtsi2sd_x_gpr(0, RCX);
                a.sse_rr(0x66, 0x2e, 0, xy); // ucomisd xmm0, y
                a.jcc(CC_NE, stub);
                a.jcc(CC_P, stub);
            }
        }
        // rax = int(x), exact.
        let xx = self.xmm_src(a, reg, x, 1);
        a.cvttsd2si_gpr_x(RAX, xx);
        a.cvtsi2sd_x_gpr(0, RAX);
        a.sse_rr(0x66, 0x2e, 0, xx); // ucomisd xmm0, x
        a.jcc(CC_NE, stub);
        a.jcc(CC_P, stub);
        // `y` is not -1, so the 64-bit `idiv` cannot overflow.
        a.cqo();
        a.idiv_rcx();
        // A zero remainder takes the dividend's sign (x's raw sign bit: -0 too).
        let nonzero = a.new_label();
        let have = a.new_label();
        a.raw(&[0x48, 0x85, 0xd2]); // test rdx, rdx
        a.jcc(CC_NE, nonzero);
        self.load_gpr(a, reg, RAX, x);
        a.movabs_r11(SIGN as i64);
        a.raw(&[0x4c, 0x21, 0xd8]); // and rax, r11 → ±0.0 bits
        a.movq_x_gpr(0, RAX);
        a.jmp(have);
        a.bind(nonzero);
        a.cvtsi2sd_x_gpr(0, RDX);
        a.bind(have);
        self.store_xmm0(a, reg, dst);
        a.bind(resume);
        let k = slow(self);
        self.stub(stub, resume, reg, k);
    }

    /// `dst = x <bitwise op> y` (`&`, `|`, `^`, `<<`, `>>`, `>>>`).
    #[allow(clippy::too_many_arguments)]
    fn bitwise(
        &mut self,
        a: &mut X64Assembler,
        pc: usize,
        reg: Option<usize>,
        vop: u8,
        dst: u16,
        x: u16,
        y: u16,
        slow: StubKind,
    ) {
        use crate::nbvm::{VB_BIT_AND, VB_BIT_OR, VB_BIT_XOR, VB_SHL, VB_SHR, VB_USHR};
        let resume = a.new_label();
        let stub = a.new_label();
        for r in [x, y] {
            if self.ty(pc, r) != T_NUM {
                self.num_guard(a, reg, r, stub);
            }
        }
        self.to_int32(a, reg, RAX, x, stub);
        self.to_int32(a, reg, RCX, y, stub);
        match vop {
            VB_BIT_AND => a.raw(&[0x21, 0xc8]), // and eax, ecx
            VB_BIT_OR => a.raw(&[0x09, 0xc8]),  // or eax, ecx
            VB_BIT_XOR => a.raw(&[0x31, 0xc8]), // xor eax, ecx
            VB_SHL => a.raw(&[0xd3, 0xe0]),     // shl eax, cl
            VB_SHR => a.raw(&[0xd3, 0xf8]),     // sar eax, cl
            VB_USHR => a.raw(&[0xd3, 0xe8]),    // shr eax, cl
            _ => {}
        }
        if vop == VB_USHR {
            a.raw(&[0x89, 0xc0]); // mov eax, eax (zero-extend: a uint32)
        } else {
            a.to_int32_rax(); // movsxd rax, eax
        }
        // An integer result is never NaN or -0.
        a.cvtsi2sd_x_gpr(0, RAX);
        self.store_xmm0(a, reg, dst);
        a.bind(resume);
        self.stub(stub, resume, reg, slow);
    }

    /// A comparison: `<`/`>`/`<=`/`>=` (`Rel`) or (loose/strict, same on two
    /// Numbers) equality (`Eq(negated)`), materialized as a boolean and/or fused
    /// into the following `JumpIfFalse` (`fused` = `(target, store)`).
    #[allow(clippy::too_many_arguments)]
    fn compare(
        &mut self,
        a: &mut X64Assembler,
        labels: &[Label],
        pc: usize,
        reg: Option<usize>,
        kind: CmpKind,
        dst: u16,
        x: u16,
        y: u16,
        slow: StubKind,
        fused: Option<(usize, bool)>,
    ) -> Option<()> {
        let stub = a.new_label();
        let guarded = self.ty(pc, x) != T_NUM || self.ty(pc, y) != T_NUM;
        if guarded {
            self.guard_numbers(a, pc, reg, x, y, stub);
        }
        // `ucomisd p, q` then `a`/`ae` = p > q / p >= q (false when unordered).
        let (p, q, cc) = match kind {
            CmpKind::Rel(0) => (y, x, CC_A),
            CmpKind::Rel(1) => (x, y, CC_A),
            CmpKind::Rel(2) => (y, x, CC_AE),
            CmpKind::Rel(_) => (x, y, CC_AE),
            CmpKind::Eq(_) => (x, y, CC_E),
        };
        let xp = self.xmm_src(a, reg, p, 0);
        let xq = self.xmm_src(a, reg, q, 1);
        a.sse_rr(0x66, 0x2e, xp, xq); // ucomisd p, q
        let store = fused.is_none_or(|(_, s)| s);
        if store {
            // rax = the 0/1 result (ordered equality needs ZF && !PF).
            match kind {
                CmpKind::Eq(neg) => {
                    a.setcc_al(CC_E);
                    a.raw(&[0x0f, 0x9b, 0xc1]); // setnp cl
                    a.raw(&[0x20, 0xc8]); // and al, cl
                    a.raw(&[0x0f, 0xb6, 0xc0]); // movzx eax, al
                    if neg {
                        a.xor_rax_imm(1);
                    }
                }
                CmpKind::Rel(_) => {
                    a.setcc_al(cc);
                    a.raw(&[0x0f, 0xb6, 0xc0]); // movzx eax, al
                }
            }
            a.raw(&[0x48, 0x89, 0xc2]); // mov rdx, rax (the 0/1, for the branch)
            a.movabs_r11(self.tag_false as i64);
            a.raw(&[0x4c, 0x01, 0xd8]); // add rax, r11
            self.store_gpr(a, reg, RAX, dst);
        }
        match fused {
            None => {
                let resume = a.new_label();
                a.bind(resume);
                if guarded {
                    self.stub(stub, resume, reg, slow);
                }
            }
            Some((target, _)) => {
                // Falsy → `target`; truthy → past the `JumpIfFalse`.
                let past = *labels.get(pc + 2)?;
                if store {
                    a.raw(&[0x48, 0x85, 0xd2]); // test rdx, rdx
                    self.jcc_to(a, labels, reg, CC_E, target);
                } else {
                    match kind {
                        CmpKind::Eq(false) => {
                            self.jcc_to(a, labels, reg, CC_NE, target);
                            self.jcc_to(a, labels, reg, CC_P, target);
                        }
                        CmpKind::Eq(true) => {
                            // `!=` is false exactly when ordered-equal.
                            let not_eq = a.new_label();
                            a.jcc(CC_NE, not_eq);
                            a.jcc(CC_P, not_eq);
                            self.jump(a, labels, reg, target);
                            a.bind(not_eq);
                        }
                        CmpKind::Rel(_) => {
                            let not_cc = if cc == CC_A { CC_BE } else { CC_B };
                            self.jcc_to(a, labels, reg, not_cc, target);
                        }
                    }
                }
                let jif_reachable = self.is_target.get(pc + 1).copied().unwrap_or(true);
                if jif_reachable {
                    a.jmp(past);
                }
                if guarded {
                    // The slow path materializes the boolean, then branches on
                    // it: through the `JumpIfFalse` emitted normally at pc + 1
                    // when that op is reachable otherwise, else by itself.
                    self.stub(stub, *labels.get(pc + 1)?, reg, slow);
                    if !jif_reachable && let Some(st) = self.stubs.last_mut() {
                        st.branch = Some((dst, target));
                    }
                }
                if !jif_reachable {
                    // Nothing else reaches the `JumpIfFalse`: the fast path falls
                    // straight through to pc + 2.
                    if let Some(k) = self.skip.get_mut(pc + 1) {
                        *k = true;
                    }
                }
            }
        }
        Some(())
    }

    /// `if !ToBoolean(register cond) goto target` for a branch from `pc`.
    fn branch_if_false(
        &mut self,
        a: &mut X64Assembler,
        labels: &[Label],
        pc: usize,
        reg: Option<usize>,
        cond: u16,
        target: usize,
    ) {
        match self.ty(pc, cond) {
            T_BOOL => {
                self.load_gpr(a, reg, RAX, cond);
                a.movabs_r11(self.tag_true as i64);
                a.cmp_rax_r11();
                self.jcc_to(a, labels, reg, CC_NE, target);
            }
            T_NUM => {
                let x = self.xmm_src(a, reg, cond, 0);
                a.sse_rr(0x66, 0x57, 1, 1); // xorpd xmm1, xmm1
                a.sse_rr(0x66, 0x2e, x, 1); // ucomisd x, xmm1 (ZF: zero or NaN)
                self.jcc_to(a, labels, reg, CC_E, target);
            }
            _ => {
                // Number / boolean inline, anything else through the helper.
                let num = a.new_label();
                let stub = a.new_label();
                let resume = a.new_label();
                self.load_gpr(a, reg, RAX, cond);
                if let Some(w) = self.truthy_ways.get(pc).copied().flatten() {
                    // An object first (the usual condition here): through
                    // its way, filling it (anything else goes on below).
                    let other = a.new_label();
                    a.p_truthy_way(
                        self.env.realm,
                        (w.key, w.cell),
                        w.check,
                        (w.shape, w.plain),
                        resume,
                        other,
                    );
                    a.bind(other);
                    self.load_gpr(a, reg, RAX, cond);
                }
                a.movabs_r11(self.tag_true as i64);
                a.cmp_rax_r11();
                a.jcc(CC_E, resume);
                a.movabs_r11(self.tag_false as i64);
                a.cmp_rax_r11();
                self.jcc_to(a, labels, reg, CC_E, target);
                a.mov_rcx_rax();
                a.movabs_r11(QNAN as i64);
                a.and_rcx_r11();
                a.cmp_rcx_r11();
                a.jcc(CC_NE, num);
                // `null` / `undefined`, an ordinary object or a function,
                // inline; anything else through the helper.
                let falsy = a.new_label();
                a.p_truthy_heap(self.env.realm, resume, falsy);
                a.jmp(stub);
                a.bind(falsy);
                self.jump(a, labels, reg, target);
                a.bind(num);
                a.movq_x_gpr(0, RAX);
                a.sse_rr(0x66, 0x57, 1, 1); // xorpd xmm1, xmm1
                a.sse_rr(0x66, 0x2e, 0, 1); // ucomisd xmm0, xmm1
                self.jcc_to(a, labels, reg, CC_E, target);
                a.bind(resume);
                self.stub(stub, resume, reg, StubKind::Truthy { cond, target });
            }
        }
    }

    /// Where a branch from `from` (in region `reg`) to bytecode pc `target`
    /// lands without leaving the region, or `None` if it leaves it.
    fn in_region_target(
        &self,
        labels: &[Label],
        reg: Option<usize>,
        target: usize,
    ) -> Option<Label> {
        match reg {
            None => self.branch_label(labels, self.cur_pc, target),
            Some(ri) => {
                let rg = &self.regions[ri];
                if target == rg.head && self.imode == Some(ri) {
                    rg.int.as_ref().and_then(int32::IntPlan::inner)
                } else if target == rg.head {
                    rg.inner
                } else if target > rg.head && target <= rg.end {
                    labels.get(target).copied()
                } else {
                    None
                }
            }
        }
    }

    /// Where a branch from op `from` to `target` goes: in a speculative
    /// loop nest's plain copy, a target in the nest past its header is the
    /// copy's own op (the copy runs on the register file, the fast version
    /// on its SSE registers; only the outer back-edge goes back through the
    /// region's entry, which loads and checks them), else `target`'s label.
    pub(super) fn branch_label(
        &self,
        labels: &[Label],
        from: usize,
        target: usize,
    ) -> Option<Label> {
        if self.copy
            && let Some(&ri) = self.region_of.get(from)
            && ri != u32::MAX
            && self.region_of.get(target) == Some(&ri)
            && self.regions[ri as usize].head != target
        {
            // A speculative nest's inner back-edge goes back into the fast
            // version at that header (`emit_inner_entries`): after a leave
            // for the copy, the copy runs the rest of that inner iteration
            // only, not of the whole outer one.
            if self.regions[ri as usize].loops.contains(&(target, from))
                && let Some(l) = self.inner_entry(target)
            {
                return Some(l);
            }
            return self.copy_label(target);
        }
        labels.get(target).copied()
    }

    /// `goto target` (flushing first if that leaves region `reg`).
    fn jump(&mut self, a: &mut X64Assembler, labels: &[Label], reg: Option<usize>, target: usize) {
        match self.in_region_target(labels, reg, target) {
            Some(l) => a.jmp(l),
            None => {
                self.flush(a, reg);
                if let Some(&l) = labels.get(target) {
                    a.jmp(l);
                }
            }
        }
    }

    /// `if cc goto target`, flushing out of line if that leaves region `reg`.
    fn jcc_to(
        &mut self,
        a: &mut X64Assembler,
        labels: &[Label],
        reg: Option<usize>,
        cc: u8,
        target: usize,
    ) {
        match self.in_region_target(labels, reg, target) {
            Some(l) => a.jcc(cc, l),
            None => {
                let stub = a.new_label();
                let resume = a.new_label();
                a.jcc(cc, stub);
                a.bind(resume);
                self.stub(stub, resume, reg, StubKind::Exit { target });
            }
        }
    }

    /// Emits the out-of-line slow paths (after the body).
    pub(super) fn finish(&mut self, a: &mut X64Assembler, labels: &[Label], throw_exit: Label) {
        let stubs = core::mem::take(&mut self.stubs);
        for s in &stubs {
            a.bind(s.entry);
            self.cur_pc = s.pc;
            self.copy = s.copy;
            self.imode = if s.int { s.region } else { None };
            match s.kind {
                StubKind::Bail { pc, store } => {
                    if let Some(ri) = s.region {
                        self.emit_bail(a, labels, ri, pc as usize, store);
                    }
                }
                StubKind::SpecBail { pc } => {
                    self.flush(a, s.region);
                    // Planned with every region pc's copy label.
                    if let Some(l) = self.copy_label(pc as usize) {
                        a.jmp(l);
                    }
                }
                StubKind::CanonNan => {
                    a.movabs_rax(self.canonical_nan as i64);
                    a.movq_x_gpr(0, RAX);
                    a.jmp(s.resume);
                }
                StubKind::NanCheck { x, slow } => {
                    a.movq_gpr_x(RCX, x);
                    a.movabs_r11(QNAN as i64);
                    a.and_rcx_r11();
                    a.cmp_rcx_r11();
                    a.jcc(CC_E, slow);
                    a.jmp(s.resume);
                }
                StubKind::DeferNan {
                    x,
                    regs,
                    slow,
                    restore,
                    cover,
                } => {
                    if restore {
                        a.movapd_xx(x, 0);
                    }
                    let roots = if cover {
                        let Some(roots) = self.cover(s.region, s.pc) else {
                            a.jmp(slow);
                            continue;
                        };
                        roots
                    } else {
                        Vec::new()
                    };
                    let mut first = true;
                    for r in regs.into_iter().flatten().chain(roots) {
                        self.load_gpr(a, s.region, RCX, r);
                        if core::mem::take(&mut first) {
                            a.movabs_r11(QNAN as i64);
                        }
                        a.and_rcx_r11();
                        a.cmp_rcx_r11();
                        a.jcc(CC_E, slow);
                    }
                    // A Number's NaN: the canonical one (every merged op's
                    // too), which the op's restored register is then given.
                    if restore {
                        a.movabs_rax(self.canonical_nan as i64);
                        a.movq_x_gpr(x, RAX);
                    }
                    a.jmp(s.resume);
                }
                StubKind::Exit { target } => {
                    self.flush(a, s.region);
                    if let Some(&l) = labels.get(target) {
                        a.jmp(l);
                    }
                }
                StubKind::Helper {
                    helper,
                    a: x,
                    b: y,
                    extra,
                    dst,
                    throws,
                } => {
                    self.flush(a, s.region);
                    a.mov_rdi_r15();
                    a.load_argreg(1, disp(x));
                    a.load_argreg(2, disp(y));
                    if let Some(e) = extra {
                        a.movabs_argreg(3, e as i64);
                    }
                    a.movabs_rax(helper as i64);
                    a.call_rax();
                    if throws {
                        a.movabs_r11(NanBox::jit_throw_bits() as i64);
                        a.cmp_rax_r11();
                        a.je(throw_exit);
                    }
                    a.store_rax(disp(dst));
                    self.reload(a, s.region);
                    self.zero_globals(a, s.region);
                    self.after_slow(a, labels, s);
                }
                StubKind::GlobalRef { pc, retry } => {
                    let k = self.gslot.get(pc).copied().unwrap_or(u32::MAX);
                    let op_ptr = self.env.bytecode.wrapping_add(pc) as u64;
                    self.flush(a, s.region);
                    a.mov_rdi_r15();
                    a.movabs_argreg(1, op_ptr as i64);
                    a.movabs_rax(self.env.global_ref as i64);
                    a.call_rax();
                    if k != u32::MAX {
                        a.store_rax(self.gslot_disp(k));
                    }
                    // A speculative region's fast version makes no call that
                    // rejoins it: the copy runs the op (the next iteration
                    // finds the address validated).
                    if !s.int
                        && let Some(ri) = s.region
                        && self.regions[ri].spec.is_some()
                        && let Some(l) = self.copy_label(self.bail_pc(pc))
                    {
                        a.jmp(l);
                        continue;
                    }
                    self.reload(a, s.region);
                    a.test_rax_rax();
                    a.jcc(CC_NE, retry);
                    if s.int
                        && let Some(ri) = s.region
                    {
                        // The double version runs the op.
                        self.emit_bail(a, labels, ri, pc, true);
                        continue;
                    }
                    // Not a plain global binding: the interpreter's op.
                    a.mov_rdi_r15();
                    a.mov_rsi_rbx();
                    a.movabs_argreg(2, self.env.n_regs as i64);
                    a.movabs_argreg(3, op_ptr as i64);
                    a.movabs_rax(self.env.exec_op as i64);
                    a.call_rax();
                    a.movabs_r11(NanBox::jit_throw_bits() as i64);
                    a.cmp_rax_r11();
                    a.je(throw_exit);
                    self.reload(a, s.region);
                    self.zero_globals(a, s.region);
                    a.jmp(s.resume);
                }
                StubKind::Exec { pc } => {
                    self.flush(a, s.region);
                    let op_ptr = self.env.bytecode.wrapping_add(pc as usize);
                    a.mov_rdi_r15();
                    a.mov_rsi_rbx();
                    a.movabs_argreg(2, self.env.n_regs as i64);
                    a.movabs_argreg(3, op_ptr as i64);
                    a.movabs_rax(self.env.exec_op as i64);
                    a.call_rax();
                    a.movabs_r11(NanBox::jit_throw_bits() as i64);
                    a.cmp_rax_r11();
                    a.je(throw_exit);
                    self.reload(a, s.region);
                    self.zero_globals(a, s.region);
                    self.after_slow(a, labels, s);
                }
                StubKind::Truthy { cond, target } => {
                    self.flush(a, s.region);
                    a.mov_rdi_r15();
                    a.load_argreg(1, disp(cond));
                    a.movabs_rax(self.env.truthy as i64);
                    a.call_rax();
                    self.reload(a, s.region);
                    self.zero_globals(a, s.region);
                    a.test_rax_rax();
                    // Truthy → resume; falsy → the target (already flushed).
                    a.jcc(CC_NE, s.resume);
                    match self.in_region_target(labels, s.region, target) {
                        Some(l) => a.jmp(l),
                        None => {
                            if let Some(&l) = labels.get(target) {
                                a.jmp(l);
                            }
                        }
                    }
                }
            }
        }
        self.imode = None;
        self.copy = false;
        for (l, pc) in core::mem::take(&mut self.nest_entries) {
            a.bind(l);
            let Some(ri) = self.region_at(pc) else {
                continue;
            };
            self.cur_pc = pc;
            if self.regions[ri].spec.is_some() {
                if let Some(c) = self.inner_entry(pc).or_else(|| self.copy_label(pc)) {
                    a.jmp(c);
                }
                continue;
            }
            self.refresh_globals(a, ri, false);
            for s in &self.regions[ri].slots {
                if s.start <= pc && pc <= s.end {
                    a.movsd_x_mem(s.xmm, disp(s.reg));
                }
            }
            if let Some(&t) = labels.get(pc) {
                a.jmp(t);
            }
        }
        self.emit_inner_entries(a, labels);
    }

    /// The entries into the speculative nests' fast versions at their inner
    /// headers, taken from the plain copy (its inner back-edges) and by an
    /// on-stack replacement there, with the register file up to date: like
    /// the nest's own entry, the global bindings validated (else the copy
    /// runs on), then the promoted registers live at the header loaded and
    /// each one the fast version assumes there to be a Number checked (a
    /// failed check runs the inner iteration in the copy, whose back-edge
    /// tries again). A register live at the header whose speculative type
    /// there is neither its plain one nor a Number makes the header have no
    /// entry (the copy runs on to the outer back-edge, as without one).
    fn emit_inner_entries(&mut self, a: &mut X64Assembler, labels: &[Label]) {
        for (h, l) in core::mem::take(&mut self.inner_entries) {
            a.bind(l);
            let (Some(ri), Some(copy), Some(&fast)) =
                (self.region_at(h), self.copy_label(h), labels.get(h))
            else {
                continue;
            };
            self.cur_pc = h;
            let Some(sp) = self.regions[ri].spec.as_ref() else {
                a.jmp(copy);
                continue;
            };
            let n = self.env.n_regs;
            let row = (h - self.regions[ri].head) * n;
            let mut checks = Vec::new();
            let mut ok = true;
            let live: &[u16] = self.regions[ri]
                .live_inner
                .iter()
                .find(|x| x.0 == h)
                .map_or(&[], |x| x.1.as_slice());
            for &r in live {
                let spec = sp.types.get(row + r as usize).copied().unwrap_or(T_ANY);
                let plain = self.types.get(h, r);
                if spec == plain || spec == T_ANY {
                    continue;
                }
                if spec == T_NUM {
                    checks.push(r);
                } else {
                    ok = false;
                }
            }
            if !ok {
                a.jmp(copy);
                continue;
            }
            self.refresh_globals(a, ri, true);
            if self.spec_globals_ready(a, ri, h).is_none() {
                a.jmp(copy);
                continue;
            }
            self.hoist_views(a, ri);
            for s in &self.regions[ri].slots {
                if s.start <= h && h <= s.end {
                    a.movsd_x_mem(s.xmm, disp(s.reg));
                }
            }
            for r in checks {
                self.num_guard(a, Some(ri), r, copy);
            }
            a.jmp(fast);
        }
    }
}

impl NumGen {
    /// The tail of a slow-path stub: resume, or a fused comparison's branch on
    /// the boolean the slow op stored (in memory, which is up to date).
    fn after_slow(&self, a: &mut X64Assembler, labels: &[Label], s: &Stub) {
        let Some((cond, target)) = s.branch else {
            a.jmp(s.resume);
            return;
        };
        a.load_rax(disp(cond));
        a.movabs_r11(self.tag_true as i64);
        a.cmp_rax_r11();
        if let Some(&past) = labels.get(s.pc + 2) {
            a.jcc(CC_E, past);
        }
        match self.in_region_target(labels, s.region, target) {
            Some(l) => a.jmp(l),
            None => {
                if let Some(&l) = labels.get(target) {
                    a.jmp(l);
                }
            }
        }
    }
}

#[derive(Clone, Copy)]
enum CmpKind {
    /// `Compare`'s op: 0 `<`, 1 `>`, 2 `<=`, 3 `>=`.
    Rel(u8),
    /// Equality over Numbers; `true` for the negated form.
    Eq(bool),
}

/// Register `r`'s displacement in the register file.
fn disp(r: u16) -> i32 {
    i32::from(r) * 8
}

/// The registers `op` writes (`defs`) and reads (`uses`), for liveness: exact
/// for the ops regions are made of, and conservatively "reads everything it
/// names, writes nothing" for every other op.
pub(super) fn def_use(op: &Op, defs: &mut Vec<u16>, uses: &mut Vec<u16>) {
    defs.clear();
    uses.clear();
    match op {
        Op::LoadConst { dst, .. } => defs.push(*dst),
        Op::Move { dst, src }
        | Op::ToNumeric { dst, src }
        | Op::Step { dst, src, .. }
        | Op::Neg { dst, a: src }
        | Op::BitNot { dst, a: src }
        | Op::Not { dst, a: src } => {
            defs.push(*dst);
            uses.push(*src);
        }
        Op::Add { dst, a, b }
        | Op::AddValue { dst, a, b }
        | Op::Sub { dst, a, b }
        | Op::Mul { dst, a, b }
        | Op::Div { dst, a, b }
        | Op::Mod { dst, a, b }
        | Op::Lt { dst, a, b }
        | Op::StrictEq { dst, a, b }
        | Op::ValueBin { dst, a, b, .. }
        | Op::Compare { dst, a, b, .. } => {
            defs.push(*dst);
            uses.push(*a);
            uses.push(*b);
        }
        Op::JumpIfFalse { cond, .. } => uses.push(*cond),
        Op::PlainPrims { dst, regs } => {
            uses.extend(regs.iter().copied());
            if !uses.contains(dst) {
                defs.push(*dst);
            }
        }
        Op::LoadGlobal { dst, .. }
        | Op::NewBox { dst }
        | Op::NewObject { dst }
        | Op::NewArray { dst, .. }
        | Op::NewString { dst, .. } => defs.push(*dst),
        // (A source register may also be the destination.)
        Op::NewObjectLit { dst, lit } => {
            uses.extend(lit.srcs.iter().copied());
            if !uses.contains(dst) {
                defs.push(*dst);
            }
        }
        Op::NewArrayFrom { dst, elems: srcs }
        | Op::MakeClosure {
            dst,
            captures: srcs,
            ..
        } => {
            uses.extend(srcs.iter().copied());
            if !uses.contains(dst) {
                defs.push(*dst);
            }
        }
        Op::Construct { dst, ctor, args } => {
            uses.push(*ctor);
            uses.extend(args.iter().copied());
            if !uses.contains(dst) {
                defs.push(*dst);
            }
        }
        Op::CellGet { dst, cell } => {
            defs.push(*dst);
            uses.push(*cell);
        }
        Op::CellSet { cell, src } => {
            uses.push(*cell);
            uses.push(*src);
        }
        Op::Return { src } | Op::CheckTdz { src } | Op::StoreGlobal { src, .. } => uses.push(*src),
        Op::RequireObjectCoercible { src, .. } => uses.push(*src),
        Op::GetProp { dst, obj, .. }
        | Op::GetMethod { dst, obj, .. }
        | Op::ArrayLen { dst, arr: obj } => {
            defs.push(*dst);
            uses.push(*obj);
        }
        Op::GetKey { dst, obj, key } => {
            defs.push(*dst);
            uses.push(*obj);
            uses.push(*key);
        }
        Op::SetProp { obj, src, .. } | Op::SetPropStrict { obj, src, .. } => {
            uses.push(*obj);
            uses.push(*src);
        }
        Op::CallMethod {
            dst, recv, args, ..
        } => {
            uses.push(*recv);
            uses.extend(args.iter().copied());
            // An argument register may also be the destination.
            if !uses.contains(dst) {
                defs.push(*dst);
            }
        }
        Op::CallWithMethod {
            dst,
            callee,
            recv,
            args,
            ..
        } => {
            uses.push(*callee);
            uses.push(*recv);
            uses.extend(args.iter().copied());
            if !uses.contains(dst) {
                defs.push(*dst);
            }
        }
        Op::Jump { .. } | Op::InlineLeave => {}
        // The guard writes its result and (on a hit) the inlined body's
        // captures, and reads the call's callee and arguments.
        Op::InlineGuard {
            dst,
            callee,
            args,
            caps,
            n_caps,
            ..
        } => {
            defs.push(*dst);
            defs.extend((0..*n_caps).map(|i| caps.wrapping_add(i)));
            uses.push(*callee);
            uses.extend(args.iter().copied());
        }
        // A step writes its value, `done` and the cursor, but not on every
        // path: each is a use as well (live across it), and a def (written).
        Op::IterNext {
            dst,
            done,
            iter,
            next,
        } => {
            defs.extend([*dst, *done, *iter]);
            uses.extend([*dst, *done, *iter, *next]);
        }
        _ => uses.extend(op_regs(op)),
    }
}

/// Backward register liveness: the live-in set of every pc, as bitsets of
/// `words` words. A `catch` entry's live-in is live out of every op (any op
/// may throw to it). `None` if the table would be too large.
fn liveness(ops: &[Op], n_regs: usize, catches: &[usize]) -> Option<(usize, Vec<u64>)> {
    let n = ops.len();
    let words = n_regs.div_ceil(64);
    if n.saturating_mul(words).saturating_mul(64) > MAX_STATE * 8 {
        return None;
    }
    let mut live = vec![0u64; n * words];
    let mut defs = Vec::new();
    let mut uses = Vec::new();
    let mut succ = Vec::with_capacity(2);
    let mut out = vec![0u64; words];
    let mut catch_live = vec![0u64; words];
    loop {
        let mut changed = false;
        catch_live.fill(0);
        for &c in catches {
            if c < n {
                for w in 0..words {
                    catch_live[w] |= live[c * words + w];
                }
            }
        }
        for pc in (0..n).rev() {
            out.copy_from_slice(&catch_live);
            successors(&ops[pc], pc, &mut succ);
            for &t in &succ {
                if t < n {
                    for w in 0..words {
                        out[w] |= live[t * words + w];
                    }
                }
            }
            def_use(&ops[pc], &mut defs, &mut uses);
            for &d in &defs {
                if (d as usize) < n_regs {
                    out[d as usize / 64] &= !(1u64 << (d % 64));
                }
            }
            for &u in &uses {
                if (u as usize) < n_regs {
                    out[u as usize / 64] |= 1u64 << (u % 64);
                }
            }
            let row = &mut live[pc * words..(pc + 1) * words];
            if row != out.as_slice() {
                row.copy_from_slice(&out);
                changed = true;
            }
        }
        if !changed {
            return Some((words, live));
        }
    }
}

/// Picks the promoted loop regions of `ops` and allocates their SSE registers
/// (see the module docs).
fn plan_regions(
    ops: &[GenericOp],
    bytecode: &[Op],
    catches: &[usize],
    n_regs: usize,
) -> Vec<Region> {
    let n = ops.len();
    let branch_target = |op: &GenericOp| match *op {
        GenericOp::Jump { target } | GenericOp::JumpIfFalse { target, .. } => Some(target),
        GenericOp::PushHandler { target, .. } => Some(target),
        _ => None,
    };
    // The candidate loops.
    let mut cands: Vec<(usize, usize)> = Vec::new();
    for (end, op) in ops.iter().enumerate() {
        let GenericOp::Jump { target: head } = *op else {
            continue;
        };
        if head > end || cands.iter().any(|&(h, e)| h <= end && head <= e) {
            continue;
        }
        let body = head..=end;
        // Innermost and simple: every op emitted by this module, no other
        // backward branch inside, no catch entry inside.
        let ok = body.clone().all(|pc| {
            region_op_ok(&ops[pc], &bytecode[pc])
                && (pc == end || branch_target(&ops[pc]).is_none_or(|t| t > pc))
        }) && !catches.iter().any(|c| body.contains(c));
        // No branch from outside the region into its body (past the header).
        if ok
            && !ops.iter().enumerate().any(|(pc, o)| {
                !body.contains(&pc) && branch_target(o).is_some_and(|t| t > head && t <= end)
            })
        {
            cands.push((head, end));
        }
    }
    if cands.is_empty() {
        return Vec::new();
    }
    // Loop nests around them (each with the inner back-edges it contains).
    let cands = plan_nests(ops, bytecode, catches, cands);
    let Some((words, live)) = liveness(bytecode, n_regs, catches) else {
        return Vec::new();
    };
    let is_live = |pc: usize, r: u16| {
        pc < n
            && (r as usize) < n_regs
            && live[pc * words + r as usize / 64] & (1u64 << (r % 64)) != 0
    };
    let mut out = Vec::new();
    for (head, end, loops) in cands {
        // Per register named in the body: first / last mention, uses, written.
        let mut info: Vec<(u16, usize, usize, u32, bool)> = Vec::new();
        let mut exits: Vec<usize> = Vec::new();
        for pc in head..=end {
            for r in op_regs(&bytecode[pc]) {
                if (r as usize) >= n_regs {
                    continue;
                }
                match info.iter_mut().find(|x| x.0 == r) {
                    Some(x) => {
                        x.2 = pc;
                        x.3 += 1;
                    }
                    None => info.push((r, pc, pc, 1, false)),
                }
            }
            if let Some(d) = written_reg(&bytecode[pc])
                && let Some(x) = info.iter_mut().find(|x| x.0 == d)
            {
                x.4 = true;
            }
            // A generic op's fast path may leave its result in the SSE
            // register only (`NumGen::result_xmm`).
            if generic_region_op(&ops[pc], &bytecode[pc]) {
                let (mut d, mut u) = (Vec::new(), Vec::new());
                def_use(&bytecode[pc], &mut d, &mut u);
                for r in d {
                    if let Some(x) = info.iter_mut().find(|x| x.0 == r) {
                        x.4 = true;
                    }
                }
            }
            if let Some(t) = branch_target(&ops[pc])
                && (t < head || t > end)
            {
                exits.push(t);
            }
        }
        if info.is_empty() {
            continue;
        }
        // A reused value's register lives on to the read that copies it.
        let reuse = plan_reuse(ops, bytecode, head, end);
        for &(pc, src) in &reuse {
            if let Some(x) = info.iter_mut().find(|x| x.0 == src) {
                x.2 = x.2.max(pc);
                x.3 += 1;
            }
        }
        // The registers the generic emitter's ops read stay in the register
        // file (see `generic_region_op`).
        let mut generic = false;
        let mut pinned: Vec<u16> = Vec::new();
        {
            let (mut d, mut u) = (Vec::new(), Vec::new());
            for pc in head..=end {
                if generic_region_op(&ops[pc], &bytecode[pc]) {
                    generic = true;
                    def_use(&bytecode[pc], &mut d, &mut u);
                    // An element read's index may stay promoted: the
                    // emitter reads it from its SSE register
                    // (`NumGen::index_from_xmm`).
                    // So may a property store's value, an element store's
                    // index and value, and an inline `Math.f(x)`'s argument
                    // (`NumGen::operand_xmm`).
                    let free = match (&ops[pc], &bytecode[pc]) {
                        (_, Op::GetKey { obj, key, .. }) if key != obj => [Some(*key), None],
                        (
                            GenericOp::SetProp { .. },
                            Op::SetProp { obj, src, .. } | Op::SetPropStrict { obj, src, .. },
                        ) if src != obj => [Some(*src), None],
                        (
                            GenericOp::SetElem { .. },
                            Op::SetKey { obj, key, src } | Op::SetKeyStrict { obj, key, src },
                        ) => [
                            Some(*key).filter(|k| k != obj),
                            Some(*src).filter(|v| v != obj),
                        ],
                        (GenericOp::CallMethod { .. }, Op::CallMethod { recv, args, .. })
                            if args.len() == 1
                                && args[0] != *recv
                                && super::inline_math(&bytecode[pc])
                                    .is_some_and(|(_, f)| f.float_like()) =>
                        {
                            [Some(args[0]), None]
                        }
                        _ => [None, None],
                    };
                    pinned.extend(u.iter().copied().filter(|&r| !free.contains(&Some(r))));
                }
            }
        }
        info.retain(|x| !pinned.contains(&x.0));
        // In a nest, a register live into an inner loop's header holds its
        // value around that loop's back-edge too: its interval covers the
        // whole inner loop (to a fixpoint, for loops nested deeper).
        let mut grown = !loops.is_empty();
        while grown {
            grown = false;
            for x in &mut info {
                for &(h, e) in &loops {
                    if is_live(h, x.0) && (x.1 > h || x.2 < e) {
                        x.1 = x.1.min(h);
                        x.2 = x.2.max(e);
                        grown = true;
                    }
                }
            }
        }
        // Loop-carried: live into the header, at an exit, or at a catch entry.
        let carried_r = |r: u16| {
            is_live(head, r)
                || exits.iter().any(|&t| is_live(t, r))
                || catches.iter().any(|&c| is_live(c, r))
        };
        let mut carried: Vec<&(u16, usize, usize, u32, bool)> =
            info.iter().filter(|x| carried_r(x.0)).collect();
        carried.sort_by(|p, q| q.3.cmp(&p.3).then(p.0.cmp(&q.0)));
        // Body temporaries, by the start of their live intervals.
        let mut temps: Vec<&(u16, usize, usize, u32, bool)> =
            info.iter().filter(|x| !carried_r(x.0)).collect();
        temps.sort_by(|p, q| p.1.cmp(&q.1).then(p.0.cmp(&q.0)));
        // One linear scan: the loop-carried registers first (their interval
        // the whole region), then the temporaries. When the registers run
        // out in a generic region, the interval that loses one is the one
        // whose register buys the least (`spill_weight`): an operand that
        // is only read, once an iteration of a long interval (an outer
        // value an inner loop reads), costs one load from the register file
        // where it stays, while a temporary of the inner loop's arithmetic
        // left in memory puts a store and a reload on its dependency chain.
        let weights = spill_weights(bytecode, head, end, &loops, &reuse);
        let weight = |r: u16| weights.iter().find(|x| x.0 == r).map_or(0, |x| x.1);
        let mut slots: Vec<Slot> = Vec::new();
        let mut free: Vec<u8> = (FIRST_XMM..FIRST_XMM + MAX_PROMOTED as u8).rev().collect();
        // (interval end, xmm, register, interval start, carried)
        let mut active: Vec<(usize, u8, u16, usize, bool)> = Vec::new();
        let ordered = carried
            .iter()
            .map(|x| (*x, true))
            .chain(temps.iter().map(|x| (*x, false)));
        for (x, is_carried) in ordered {
            let (start, stop) = if is_carried { (head, end) } else { (x.1, x.2) };
            active.retain(|&(e, xmm, ..)| {
                if e < start {
                    free.push(xmm);
                    false
                } else {
                    true
                }
            });
            // A `Move` that ends its source's interval hands the source's
            // SSE register to its destination (the move then emits nothing).
            // Generic regions only (no integer version shadows the slots).
            let mut reuse = None;
            if generic
                && !is_carried
                && let Op::Move { dst, src } = bytecode[x.1]
                && dst == x.0
                && let Some(i) = active
                    .iter()
                    .position(|&(e, _, r, ..)| r == src && e == x.1)
            {
                reuse = Some(active.remove(i).1);
            }
            // Likewise an arithmetic op (`float_bin`'s) whose first operand
            // (or, commuting, second) ends there: the result is computed in
            // place, with no copy. Every slow path of the op leaves before
            // the result is written (a stub that resumes reloads the result
            // last, `NumGen::reload`), so the operand's value is intact
            // wherever one stores it.
            if generic
                && !is_carried
                && reuse.is_none()
                && let Some((p, q, commutes)) = float_bin_operands(&ops[x.1], x.0)
            {
                let ends = |r: u16| {
                    active
                        .iter()
                        .position(|&(e, _, ar, ..)| ar == r && e == x.1 && r != x.0)
                };
                if let Some(i) = ends(p).or_else(|| if commutes { ends(q) } else { None }) {
                    reuse = Some(active.remove(i).1);
                }
            }
            let xmm = match (reuse, free.pop()) {
                (Some(m), f) => {
                    free.extend(f);
                    m
                }
                (None, Some(m)) => m,
                // Out of registers: in a generic region the active interval
                // whose register buys the least gives it up, unless that is
                // this one (an interval only ever holds one register, from
                // its start to its end, so the loser stays in memory).
                (None, None) if generic => {
                    let w = |r: u16, a: usize, b: usize| {
                        weight(r).saturating_mul(1024) / (b.saturating_sub(a) as u64 + 1)
                    };
                    let mine = w(x.0, start, stop);
                    let victim = active
                        .iter()
                        .enumerate()
                        .map(|(i, &(e, _, r, s0, _))| (w(r, s0, e), core::cmp::Reverse(e), i))
                        .min();
                    match victim {
                        Some((vw, core::cmp::Reverse(ve), i))
                            if vw < mine || (vw == mine && ve > stop) =>
                        {
                            let (_, m, vr, vs, _) = active.remove(i);
                            slots.retain(|s| !(s.reg == vr && s.start == vs));
                            m
                        }
                        _ => continue,
                    }
                }
                (None, None) => continue,
            };
            active.push((stop, xmm, x.0, start, is_carried));
            slots.push(Slot {
                reg: x.0,
                xmm,
                start,
                end: stop,
                carried: is_carried,
                dirty: x.4,
            });
        }
        let dbl_xmm: Vec<u8> = slots.iter().map(|s| s.xmm).collect();
        // The integer version covers numeric ops only.
        let int = if generic {
            None
        } else {
            int32::plan(bytecode, head, end, &info, &carried_r, n_regs, &dbl_xmm)
        };
        let live_head = if generic {
            (0..n_regs as u16).filter(|&r| is_live(head, r)).collect()
        } else {
            Vec::new()
        };
        let live_inner = if generic {
            loops
                .iter()
                .map(|&(h, _)| (h, (0..n_regs as u16).filter(|&r| is_live(h, r)).collect()))
                .collect()
        } else {
            Vec::new()
        };
        // A body that branches inside itself records its liveness
        // (`Region::defined`).
        let branchy =
            (head..end).any(|pc| branch_target(&ops[pc]).is_some_and(|t| t > head && t <= end));
        let live_rows = if branchy && generic {
            live[head * words..(end + 1) * words].to_vec()
        } else {
            Vec::new()
        };
        out.push(Region {
            live: live_rows,
            live_words: if branchy && generic { words } else { 0 },
            head,
            end,
            slots,
            inner: None,
            int,
            generic,
            live_head,
            spec: None,
            reuse,
            loops,
            live_inner,
            defer: DeferPlan::default(),
            views: Vec::new(),
            cold: Vec::new(),
        });
    }
    out
}

/// What keeping each register of region `head..=end` in an SSE register is
/// worth ([`plan_regions`]' spill choice): its mentions in the region (and
/// the reads [`plan_reuse`] turns into copies of it), each weighted by how
/// often it runs: 8x per inner loop of a nest (`loops`) around it.
fn spill_weights(
    bytecode: &[Op],
    head: usize,
    end: usize,
    loops: &[(usize, usize)],
    reuse: &[(usize, u16)],
) -> Vec<(u16, u64)> {
    let mut w: Vec<(u16, u64)> = Vec::new();
    let add = |w: &mut Vec<(u16, u64)>, r: u16, pc: usize| {
        let depth = loops.iter().filter(|&&(h, e)| h <= pc && pc <= e).count();
        let k = 8u64.saturating_pow(depth.min(6) as u32);
        match w.iter_mut().find(|x| x.0 == r) {
            Some(x) => x.1 = x.1.saturating_add(k),
            None => w.push((r, k)),
        }
    };
    for (pc, op) in bytecode.iter().enumerate().take(end + 1).skip(head) {
        for r in op_regs(op) {
            add(&mut w, r, pc);
        }
    }
    for &(pc, src) in reuse {
        add(&mut w, src, pc);
    }
    w
}

/// Plans speculative region `r`'s merged NaN checks ([`DeferPlan`]).
///
/// The fast version's body is cut into **stretches**: runs of ops with no
/// effect a second run could see (none stores anything but registers, and
/// none writes a register live on entry to the stretch), no branch, no
/// join inside. A guarded arithmetic op ([`NumGen::float_bin`]'s check:
/// one operand not proven a Number) is
/// *deferred*, its check left out, when what it makes is then only carried
/// by arithmetic to the check of a later op of the stretch — as an operand
/// it did not prove, which the speculation then counts as a Number, or as
/// its result. A value carrying such a pending NaN is *tainted* by the
/// deferred ops it came through (a move or a property read the fast
/// version replaces by a copy passes the taint on; `+ - * /` and an inline
/// one-argument `Math` function combine it; a check of a guarded op clears
/// it, from every register, once that check passes). Anything else reading
/// a tainted value (a store, a branch, a comparison, an element index...)
/// makes the deferral it came from keep its check, as does reaching the
/// end of the stretch unchecked. A check that clears a taint *covers*
/// those deferrals ([`Cover`]). A NaN there is a Number's when every
/// operand the covered ops did not prove is a Number, so those must still
/// hold the values the ops read at the check: one written in between
/// makes its deferral keep its check, and one whose promoted interval
/// ends before the check is stored to the register file by the deferred
/// op (where the check's stub reads it). A NaN with some operand not a
/// Number leaves for the stretch's start in the plain copy, as does every
/// leave of the fast version from the first deferred op to the last
/// covering check — so every register live on entry to the stretch must
/// still hold its value there (its SSE register not handed on while the
/// stretch may still leave, unless the register file has it).
fn plan_defer(
    r: &Region,
    ops: &[GenericOp],
    bytecode: &[Op],
    n_regs: usize,
    is_target: &[bool],
    live: &dyn Fn(usize, u16) -> bool,
) -> DeferPlan {
    let Some(sp) = r.spec.as_ref() else {
        return DeferPlan::default();
    };
    let (head, end) = (r.head, r.end);
    if end >= ops.len() || end >= bytecode.len() {
        return DeferPlan::default();
    }
    let ty = |pc: usize, x: u16| {
        sp.types
            .get((pc - head) * n_regs + x as usize)
            .copied()
            .unwrap_or(T_ANY)
    };
    let slot_at = |pc: usize, x: u16| {
        r.slots
            .iter()
            .position(|s| s.reg == x && s.start <= pc && pc <= s.end)
    };
    let is_float = |pc: usize| {
        written_reg(&bytecode[pc]).is_some_and(|dst| float_bin_operands(&ops[pc], dst).is_some())
    };
    let mut kept = vec![false; end - head + 1];
    let (mut d, mut u) = (Vec::new(), Vec::new());
    for _ in 0..32 {
        let mut st = DeferState::new(n_regs);
        let mut plan = DeferPlan::default();
        let mut changed = false;
        st.start = head;
        for pc in head..=end {
            let (op, bc) = (&ops[pc], &bytecode[pc]);
            def_use(bc, &mut d, &mut u);
            // A join, or a write of a register live on entry to the stretch,
            // starts a new one.
            if pc > st.start
                && (is_target.get(pc).copied().unwrap_or(true)
                    || d.iter().any(|&x| live(st.start, x)))
            {
                changed |= st.close(r, &mut kept, &mut plan, head, live, &is_float);
                st.start = pc;
            }
            let float =
                written_reg(bc).and_then(|dst| float_bin_operands(op, dst).map(|o| (dst, o)));
            let math = match (op, bc) {
                (
                    GenericOp::CallMethod { .. },
                    Op::CallMethod {
                        dst, recv, args, ..
                    },
                ) if args.len() == 1
                    && super::inline_math(bc).is_some_and(|(_, f)| f.float_like()) =>
                {
                    Some((*dst, *recv, args[0]))
                }
                _ => None,
            };
            let mut boundary = false;
            if let Some((dst, (x, y, _))) = float {
                let guarded = ty(pc, x) != T_NUM || ty(pc, y) != T_NUM;
                let t = st.taint[x as usize] | st.taint[y as usize];
                if guarded && !kept[pc - head] && st.cands.len() < 64 {
                    let bit = 1u64 << st.cands.len();
                    st.cands.push(pc);
                    st.first.get_or_insert(pc);
                    let mut roots = [None; 2];
                    for (i, o) in [x, y].into_iter().enumerate() {
                        if ty(pc, o) != T_NUM {
                            st.taint[o as usize] |= bit;
                            if i == 0 || o != x {
                                roots[i] = Some(o);
                            }
                        }
                    }
                    st.roots.push(roots);
                    st.set(dst, t | bit, pc);
                } else if guarded {
                    if t != 0 {
                        changed |= st.cover(pc, t, &slot_at, &mut kept, head);
                        st.last = Some(pc);
                    }
                    st.set(dst, 0, pc);
                } else {
                    st.set(dst, t, pc);
                }
            } else if let Some((dst, recv, arg)) = math {
                changed |= st.observe(recv, &mut kept, head);
                if ty(pc, arg) != T_NUM {
                    changed |= st.observe(arg, &mut kept, head);
                }
                let t = st.taint[arg as usize];
                st.set(dst, t, pc);
            } else {
                match (op, bc) {
                    (GenericOp::Move { .. }, &Op::Move { dst, src }) => st.copy(dst, src, pc),
                    (GenericOp::Const { .. } | GenericOp::Nop, _) => {
                        for &x in &d {
                            st.set(x, 0, pc);
                        }
                    }
                    (GenericOp::GetProp { .. }, &Op::GetProp { dst, obj, .. }) => {
                        match r.reuse.iter().find(|x| x.0 == pc) {
                            Some(&(_, src)) => st.copy(dst, src, pc),
                            None => {
                                changed |= st.observe(obj, &mut kept, head);
                                st.set(dst, 0, pc);
                            }
                        }
                    }
                    (GenericOp::GetElem { .. } | GenericOp::ArrayLen { .. }, _)
                    | (
                        GenericOp::Interp { .. },
                        Op::LoadGlobal { .. } | Op::RequireObjectCoercible { .. },
                    ) => {
                        for &x in &u {
                            changed |= st.observe(x, &mut kept, head);
                        }
                        for &x in &d {
                            st.set(x, 0, pc);
                        }
                    }
                    _ => boundary = true,
                }
            }
            if boundary {
                for &x in &u {
                    changed |= st.observe(x, &mut kept, head);
                }
            }
            changed |= st.overwrite(&d, &mut kept, head);
            st.wrote.extend(d.iter().map(|&x| (pc, x)));
            // An op that ends the stretch (or writes what it reads, live on
            // entry to the stretch it starts) closes it.
            if boundary || d.iter().any(|&x| live(st.start, x)) || pc == end {
                changed |= st.close(r, &mut kept, &mut plan, head, live, &is_float);
                st.start = pc + 1;
            }
        }
        if !changed {
            plan.kind.sort_unstable();
            plan.covers.sort_unstable_by_key(|c| c.pc);
            return plan;
        }
    }
    DeferPlan::default()
}

/// [`plan_defer`]'s walk through one stretch.
struct DeferState {
    /// Per register: the deferred ops (bits of `cands`) whose pending NaN
    /// its value may carry.
    taint: Vec<u64>,
    /// Per register: the op that made its value (through copies).
    maker: Vec<usize>,
    /// The stretch's deferred ops.
    cands: Vec<usize>,
    /// Per deferred op: its operands not proven Numbers.
    roots: Vec<[Option<u16>; 2]>,
    /// The checks that clear a taint ([`DeferPlan::covers`]).
    covers: Vec<Cover>,
    /// `(pc, register)`: the stretch's register writes.
    wrote: Vec<(usize, u16)>,
    /// `(pc, register, slot)`: a [`Cover`]'s operand whose promoted
    /// interval ends before the check, stored by deferred op `pc`.
    root_saves: Vec<(usize, u16, usize)>,
    /// The cleared deferrals (bits of `cands`).
    verified: u64,
    start: usize,
    first: Option<usize>,
    last: Option<usize>,
    /// A read this walk already made keep a check (the stretch's other
    /// deferrals are judged on the next walk).
    revised: bool,
}

impl DeferState {
    fn new(n_regs: usize) -> Self {
        Self {
            taint: vec![0; n_regs],
            maker: vec![usize::MAX; n_regs],
            cands: Vec::new(),
            roots: Vec::new(),
            covers: Vec::new(),
            root_saves: Vec::new(),
            wrote: Vec::new(),
            verified: 0,
            start: 0,
            first: None,
            last: None,
            revised: false,
        }
    }

    fn set(&mut self, x: u16, t: u64, pc: usize) {
        if let Some(v) = self.taint.get_mut(x as usize) {
            *v = t;
            self.maker[x as usize] = pc;
        }
    }

    /// Op `pc` copies register `src` into `dst`.
    fn copy(&mut self, dst: u16, src: u16, pc: usize) {
        let (t, m) = match self.taint.get(src as usize) {
            Some(&t) => (t, self.maker[src as usize]),
            None => (0, pc),
        };
        self.set(dst, t, pc);
        if let Some(v) = self.maker.get_mut(dst as usize) {
            *v = m;
        }
    }

    /// Guarded op `pc`, whose operands carry the deferrals `t`, checks for
    /// them ([`Cover`]): each one's unproven operands must still be where
    /// they were (unwritten, `overwrite`, and in the same promoted interval
    /// or none, `slot_at`) for its stub to tell a Number's NaN from a type
    /// miss; a deferral whose are not keeps its own check. `true` if that
    /// changed the plan.
    fn cover(
        &mut self,
        pc: usize,
        t: u64,
        slot_at: &dyn Fn(usize, u16) -> Option<usize>,
        kept: &mut [bool],
        head: usize,
    ) -> bool {
        let mut changed = false;
        let mut roots = Vec::new();
        for (i, (&c, rs)) in self.cands.iter().zip(&self.roots).enumerate() {
            if t & (1u64 << i) == 0 {
                continue;
            }
            for r in rs.iter().flatten().copied() {
                let (then, now) = (slot_at(c, r), slot_at(pc, r));
                if then != now {
                    // Its interval ended (the register file then holds an
                    // older value): stored there by the merged op.
                    match then {
                        Some(k) if now.is_none() => {
                            if !self.root_saves.iter().any(|s| s.1 == r) {
                                self.root_saves.push((c, r, k));
                            }
                        }
                        _ => {
                            kept[c - head] = true;
                            changed = true;
                        }
                    }
                }
                if !roots.contains(&r) {
                    roots.push(r);
                }
            }
        }
        if changed {
            self.revised = true;
            return true;
        }
        self.covers.push(Cover { pc, roots });
        self.verify(t);
        false
    }

    /// Op writes registers `d`: a deferral still pending that one of them
    /// was an unproven operand of can no longer be told apart at a later
    /// check ([`cover`](Self::cover)), so it keeps its own. `true` if that
    /// changed the plan.
    fn overwrite(&mut self, d: &[u16], kept: &mut [bool], head: usize) -> bool {
        let mut changed = false;
        for (i, (&c, rs)) in self.cands.iter().zip(&self.roots).enumerate() {
            if self.verified & (1u64 << i) == 0
                && rs.iter().flatten().any(|r| d.contains(r))
                && !kept[c - head]
            {
                kept[c - head] = true;
                changed = true;
            }
        }
        self.revised |= changed;
        changed
    }

    /// A check passed: what it carried is cleared everywhere.
    fn verify(&mut self, t: u64) {
        self.verified |= t;
        for v in &mut self.taint {
            *v &= !t;
        }
    }

    /// Register `x` read by an op that is not arithmetic: what taints it
    /// keeps its own check — its maker if that is a deferral (which then
    /// clears the rest), else each deferral it came from. `true` if that
    /// changed the plan.
    fn observe(&mut self, x: u16, kept: &mut [bool], head: usize) -> bool {
        let t = self.taint.get(x as usize).copied().unwrap_or(0);
        if t == 0 {
            return false;
        }
        let maker = self.maker[x as usize];
        if self.cands.contains(&maker) {
            kept[maker - head] = true;
        } else {
            for (i, &c) in self.cands.iter().enumerate() {
                if t & (1u64 << i) != 0 {
                    kept[c - head] = true;
                }
            }
        }
        self.revised = true;
        true
    }

    /// The stretch ends: every deferral must have been cleared, and every
    /// register live on entry to it must still hold its value up to its
    /// last covering check; then its plan is recorded. `true` if that
    /// changed the plan (a deferral keeps its check).
    fn close(
        &mut self,
        r: &Region,
        kept: &mut [bool],
        plan: &mut DeferPlan,
        head: usize,
        live: &dyn Fn(usize, u16) -> bool,
        is_float: &dyn Fn(usize) -> bool,
    ) -> bool {
        let n = self.cands.len();
        let all = if n >= 64 { u64::MAX } else { (1u64 << n) - 1 };
        let mut changed = self.revised;
        if self.revised {
            // (Judged again on the next walk.)
        } else if all & !self.verified != 0 {
            for (i, &c) in self.cands.iter().enumerate() {
                if self.verified & (1u64 << i) == 0 {
                    kept[c - head] = true;
                }
            }
            changed = true;
        } else if let (Some(first), Some(last)) = (self.first, self.last) {
            // Where a leave from `first..=last` restarts: the first deferred
            // op itself when nothing from there on writes a register live
            // there (the ops before it ran checked), else the stretch's
            // start (whose own such writes split stretches). A register
            // live at the restart whose SSE register is handed on while
            // the stretch may still leave: stored to the register file by
            // the arithmetic op that ends its interval, just before that
            // op (only there: other ops are not this module's to emit).
            let restart = |at: usize| {
                if at != self.start
                    && self
                        .wrote
                        .iter()
                        .any(|&(pc, x)| at <= pc && pc <= last && live(at, x))
                {
                    return None;
                }
                let lost: Vec<&Slot> = r
                    .slots
                    .iter()
                    .filter(|s| {
                        s.start <= at && at <= s.end && s.end < last && s.dirty && live(at, s.reg)
                    })
                    .collect();
                (!lost.iter().any(|s| !is_float(s.end))).then_some((at, lost))
            };
            let Some((start, lost)) = restart(first).or_else(|| restart(self.start)) else {
                for &c in &self.cands {
                    kept[c - head] = true;
                }
                let n_regs = self.taint.len();
                *self = Self::new(n_regs);
                return true;
            };
            {
                plan.kind
                    .extend(self.cands.iter().map(|&c| (c, DEFER_SKIP)));
                plan.kind
                    .extend(self.covers.iter().map(|c| (c.pc, DEFER_COVER)));
                plan.covers.append(&mut self.covers);
                plan.spans.push((first, last, start));
                plan.saves
                    .extend(lost.iter().map(|s| (s.end, s.reg, s.xmm)));
                plan.saves.extend(
                    self.root_saves
                        .iter()
                        .map(|&(pc, x, k)| (pc, x, r.slots[k].xmm)),
                );
            }
        }
        let n_regs = self.taint.len();
        *self = Self::new(n_regs);
        changed
    }
}

/// Grows the innermost candidate loops `cands` into **loop nests**: an outer
/// loop `head..=end` (its back-edge the unconditional `Jump` at `end`) whose
/// every op is a region op and whose every other backward branch is the
/// back-edge of a loop nested in it that is itself a candidate (or a nest),
/// with no catch entry inside and no branch from outside into its body, is
/// promoted as one region: the outer loop's registers stay in their SSE
/// registers across the inner loops, whose entries and exits are then plain
/// in-region branches. Only nests of generic loops, all of them call-free
/// (an inner loop of numeric ops alone keeps its own region and its integer
/// version; a call would store and reload the outer registers around it).
/// The outermost qualifying loop wins. Returns the regions to plan as
/// `(head, end, inner back-edges)`.
#[allow(clippy::type_complexity)]
fn plan_nests(
    ops: &[GenericOp],
    bytecode: &[Op],
    catches: &[usize],
    cands: Vec<(usize, usize)>,
) -> Vec<(usize, usize, Vec<(usize, usize)>)> {
    // The back-edges of the body, innermost first.
    let mut backs: Vec<(usize, usize)> = ops
        .iter()
        .enumerate()
        .filter_map(|(e, op)| match *op {
            GenericOp::Jump { target: h } if h < e => Some((h, e)),
            _ => None,
        })
        .collect();
    let simple =
        |cands: &[(usize, usize)]| cands.iter().map(|&(h, e)| (h, e, Vec::new())).collect();
    if backs.len() < 2 {
        return simple(&cands);
    }
    backs.sort_by_key(|&(h, e)| e - h);
    // Every branch of the body, as `(from, to)`.
    let branches: Vec<(usize, usize)> = ops
        .iter()
        .enumerate()
        .filter_map(|(pc, op)| match *op {
            GenericOp::Jump { target } | GenericOp::JumpIfFalse { target, .. } => {
                Some((pc, target))
            }
            GenericOp::PushHandler { target, .. } => Some((pc, target)),
            _ => None,
        })
        .collect();
    let generic_loop = |h: usize, e: usize| {
        (h..=e).any(|pc| generic_region_op(&ops[pc], &bytecode[pc]))
            && !(h..=e).any(|pc| region_call(&ops[pc], &bytecode[pc]))
    };
    // Per back-edge (as in `backs`): the inner back-edges of the nest it heads.
    let mut nest: Vec<Option<Vec<(usize, usize)>>> = vec![None; backs.len()];
    for i in 0..backs.len() {
        let (head, end) = backs[i];
        let body = head..=end;
        if !body.clone().all(|pc| region_op_ok(&ops[pc], &bytecode[pc]))
            || catches.iter().any(|c| body.contains(c))
            || branches
                .iter()
                .any(|&(f, t)| !body.contains(&f) && t > head && t <= end)
            || !generic_loop(head, end)
        {
            continue;
        }
        // Every other backward branch: a nested loop's (a candidate or a
        // nest), properly inside.
        let mut inner: Vec<(usize, usize)> = Vec::new();
        let mut ok = true;
        for &(f, t) in &branches {
            if f == end || !body.contains(&f) || t > f {
                continue;
            }
            let edge = (t, f);
            let jump = matches!(ops[f], GenericOp::Jump { .. });
            let nested = cands.contains(&edge) && generic_loop(t, f)
                || backs[..i]
                    .iter()
                    .position(|&b| b == edge)
                    .and_then(|k| nest[k].as_ref())
                    .is_some();
            if !jump || t <= head || !nested {
                ok = false;
                break;
            }
            inner.push(edge);
        }
        if ok && !inner.is_empty() {
            nest[i] = Some(inner);
        }
    }
    // The outermost nests, then the candidates outside them.
    let mut out: Vec<(usize, usize, Vec<(usize, usize)>)> = Vec::new();
    for i in (0..backs.len()).rev() {
        let (h, e) = backs[i];
        if let Some(inner) = &nest[i]
            && !out.iter().any(|&(oh, oe, _)| oh <= e && h <= oe)
        {
            out.push((h, e, inner.clone()));
        }
    }
    if out.is_empty() {
        return simple(&cands);
    }
    for &(h, e) in &cands {
        if !out.iter().any(|&(oh, oe, _)| oh <= e && h <= oe) {
            out.push((h, e, Vec::new()));
        }
    }
    out.sort_by_key(|x| x.0);
    out
}

/// The operands of a region op the numeric emitter computes with
/// `float_bin` into `dst`: `(a, b, commutes)`.
fn float_bin_operands(op: &GenericOp, dst: u16) -> Option<(u16, u16, bool)> {
    match *op {
        GenericOp::Add { dst: d, a, b } | GenericOp::NumAdd { dst: d, a, b, .. } if d == dst => {
            Some((a, b, true))
        }
        GenericOp::Arith { dst: d, a, b, op } if d == dst => match op {
            crate::nbvm::GA_MUL => Some((a, b, true)),
            crate::nbvm::GA_SUB | crate::nbvm::GA_DIV => Some((a, b, false)),
            _ => None,
        },
        _ => None,
    }
}

/// The property reads of loop body `head..=end` whose value a register
/// already holds, for the loop's speculative version ([`Spec`]): `(pc,
/// src)` for a `GetProp` of `obj.key` after a `GetProp` of the same register's
/// `key` (into `src`) or a `SetProp` of it (of `src`), with neither register
/// written since, and since then only ops that write no property: register
/// ops, reads, branches out of the loop, and writes of other keys.
///
/// In that version every property op takes its inline path or leaves for the
/// plain copy before it has done anything (a getter, a setter, an add, a
/// miss), and no Rust code runs: the earlier op's value is the object's own
/// data property (no prototype walk is inline), and only those inline stores
/// change a property's value. Nothing else in the list can change an
/// object's own property or make it an accessor (a `SetElem` / global write
/// might, so either ends what is known), so the later read yields that very
/// value.
///
/// Likewise a captured binding's read (`CellGet`) after one of the same cell
/// register, with neither register written since: in that version nothing
/// writes a binding cell (a `CellSet` is a call, `region_call`). And a
/// `CheckTdz` of a register already checked (or a copy of one, unwritten
/// since) is listed as `(pc, src)` too: it has nothing to check.
fn plan_reuse(ops: &[GenericOp], bytecode: &[Op], head: usize, end: usize) -> Vec<(usize, u16)> {
    // (object register, key, register holding its value).
    let mut avail: Vec<(u16, &str, u16)> = Vec::new();
    // (cell register, register holding its value).
    let mut cells: Vec<(u16, u16)> = Vec::new();
    // Registers known not to hold the TDZ sentinel.
    let mut checked: Vec<u16> = Vec::new();
    let mut out = Vec::new();
    let (mut d, mut u) = (Vec::new(), Vec::new());
    // The body's in-body branch targets (an arm's end, a loop nest's inner
    // headers and exits) are joins: nothing is known there.
    let mut joins: Vec<usize> = Vec::new();
    for op in ops.get(head..=end).unwrap_or(&[]) {
        if let GenericOp::Jump { target } | GenericOp::JumpIfFalse { target, .. } = *op
            && target > head
            && target <= end
        {
            joins.push(target);
        }
    }
    for pc in head..end.min(ops.len()) {
        let (Some(op), Some(bc)) = (ops.get(pc), bytecode.get(pc)) else {
            break;
        };
        if joins.contains(&pc) {
            avail.clear();
            cells.clear();
            checked.clear();
        }
        match (op, bc) {
            (GenericOp::CellGet { .. }, Op::CellGet { dst, cell }) => {
                let (dst, cell) = (*dst, *cell);
                let hit = cells.iter().find(|e| e.0 == cell).map(|e| e.1);
                if let Some(src) = hit
                    && src != dst
                {
                    out.push((pc, src));
                }
                let known = hit.is_some_and(|src| checked.contains(&src));
                avail.retain(|e| e.0 != dst && e.2 != dst);
                cells.retain(|e| e.0 != dst && e.1 != dst);
                checked.retain(|&x| x != dst);
                if known {
                    checked.push(dst);
                }
                if hit.is_none() && cell != dst {
                    cells.push((cell, dst));
                }
                continue;
            }
            (GenericOp::CheckTdz { .. }, Op::CheckTdz { src }) => {
                if checked.contains(src) {
                    out.push((pc, *src));
                } else {
                    checked.push(*src);
                }
                continue;
            }
            (GenericOp::GetProp { .. }, Op::GetProp { dst, obj, key }) => {
                let (dst, obj) = (*dst, *obj);
                let hit = avail
                    .iter()
                    .find(|e| e.0 == obj && e.1 == key.as_str())
                    .map(|e| e.2);
                if let Some(src) = hit
                    && src != dst
                {
                    out.push((pc, src));
                }
                avail.retain(|e| e.0 != dst && e.2 != dst);
                cells.retain(|e| e.0 != dst && e.1 != dst);
                checked.retain(|&x| x != dst);
                if hit.is_none() && obj != dst {
                    avail.push((obj, key.as_str(), dst));
                }
            }
            (
                GenericOp::SetProp { .. },
                Op::SetProp { obj, key, src } | Op::SetPropStrict { obj, key, src },
            ) => {
                avail.retain(|e| e.1 != key.as_str());
                avail.push((*obj, key.as_str(), *src));
            }
            (
                GenericOp::Const { .. }
                | GenericOp::Move { .. }
                | GenericOp::Nop
                | GenericOp::Add { .. }
                | GenericOp::NumAdd { .. }
                | GenericOp::Arith { .. }
                | GenericOp::Lt { .. }
                | GenericOp::Compare { .. }
                | GenericOp::StrictEq { .. }
                | GenericOp::ValueBin { .. }
                | GenericOp::Not { .. }
                | GenericOp::ToNumeric { .. }
                | GenericOp::Step { .. }
                | GenericOp::CheckTdz { .. }
                | GenericOp::GetElem { .. }
                | GenericOp::ArrayLen { .. }
                | GenericOp::JumpIfFalse { .. },
                _,
            ) => {
                def_use(bc, &mut d, &mut u);
                avail.retain(|e| !d.contains(&e.0) && !d.contains(&e.2));
                cells.retain(|e| !d.contains(&e.0) && !d.contains(&e.1));
                checked.retain(|x| !d.contains(x));
            }
            (GenericOp::CallMethod { .. }, _) if super::inline_math(bc).is_some() => {
                def_use(bc, &mut d, &mut u);
                avail.retain(|e| !d.contains(&e.0) && !d.contains(&e.2));
                cells.retain(|e| !d.contains(&e.0) && !d.contains(&e.1));
                checked.retain(|x| !d.contains(x));
            }
            (
                GenericOp::Interp { .. },
                Op::LoadGlobal { .. }
                | Op::RequireObjectCoercible { .. }
                | Op::Neg { .. }
                | Op::BitNot { .. }
                | Op::IterNext { .. },
            ) => {
                def_use(bc, &mut d, &mut u);
                avail.retain(|e| !d.contains(&e.0) && !d.contains(&e.2));
                cells.retain(|e| !d.contains(&e.0) && !d.contains(&e.1));
                checked.retain(|x| !d.contains(x));
            }
            _ => {
                avail.clear();
                cells.clear();
                checked.clear();
            }
        }
    }
    out
}

/// The arithmetic a speculative region runs on Numbers only (see [`Spec`]):
/// each op's inline code computes a Number from Number operands, and any
/// other operand takes its slow path, which the speculation replaces by
/// leaving for the plain copy.
fn spec_num_op(op: &Op) -> bool {
    use crate::nbvm::{VB_BIT_AND, VB_USHR};
    match op {
        Op::Add { .. }
        | Op::AddValue { .. }
        | Op::Sub { .. }
        | Op::Mul { .. }
        | Op::Div { .. }
        | Op::Mod { .. }
        | Op::ToNumeric { .. }
        | Op::Step { .. }
        | Op::Neg { .. }
        | Op::BitNot { .. } => true,
        Op::ValueBin { op, .. } => (VB_BIT_AND..=VB_USHR).contains(op),
        _ => false,
    }
}

/// [`transfer`] under the speculation: a [`spec_num_op`]'s result is a
/// Number, and so is an `ArrayLen`'s (its fast path's array length: its
/// slow path, like every one of the fast version, leaves for the copy).
fn spec_transfer(op: &Op, s: &mut [u8]) {
    if let Op::ArrayLen { dst, .. } = op
        && let Some(x) = s.get_mut(*dst as usize)
    {
        *x = T_NUM;
        return;
    }
    if spec_num_op(op)
        && let Some(d) = written_reg(op)
        && let Some(x) = s.get_mut(d as usize)
    {
        *x = T_NUM;
        return;
    }
    transfer(op, s);
}

/// One op of a speculative region's straight-line body, `pc`, under the
/// speculation: [`spec_transfer`], plus what the fast version's continuing
/// past the op proves.
///
/// - A [`spec_num_op`] takes its inline path only on Number operands: every
///   other operand takes its slow path, which in the fast version leaves for
///   the plain copy ([`NumGen::stub`]). So past it each register it read
///   holds a Number (until written again): a later op reading one needs no
///   type guard.
/// - A property read the fast version replaces by a copy ([`plan_reuse`])
///   yields the copied register's value, so it has that register's type.
fn spec_step(
    op: &Op,
    pc: usize,
    reuse: &[(usize, u16)],
    fields: &[Option<(u64, u32)>],
    s: &mut [u8],
) {
    if let Op::GetProp { dst, .. } | Op::CellGet { dst, .. } = op
        && let Some(&(_, src)) = reuse.iter().find(|x| x.0 == pc)
        && let Some(&t) = s.get(src as usize)
        && let Some(x) = s.get_mut(*dst as usize)
    {
        *x = t;
        return;
    }
    // A Number-only slot of a compiled-in layout ([`Spec::fields`]).
    if let Op::GetProp { dst, .. } = op
        && fields.get(pc).is_some_and(Option::is_some)
        && let Some(x) = s.get_mut(*dst as usize)
    {
        *x = T_NUM;
        return;
    }
    spec_transfer(op, s);
    // `Math.f(x…)` computed inline: past it (in the fast version every other
    // case — another method, a non-Number argument — leaves for the copy)
    // each argument and the result are Numbers.
    if let Op::CallMethod { dst, args, .. } = op
        && super::inline_math(op).is_some()
    {
        for &r in args.iter().chain(core::iter::once(dst)) {
            if let Some(x) = s.get_mut(r as usize) {
                *x = T_NUM;
            }
        }
        return;
    }
    if spec_num_op(op) {
        let (mut d, mut u) = (Vec::new(), Vec::new());
        def_use(op, &mut d, &mut u);
        for r in u {
            if let Some(x) = s.get_mut(r as usize) {
                *x = T_NUM;
            }
        }
    }
}

/// Plans region `r`'s speculative double version ([`Spec`]): for a region of
/// generic ops (and no integer version) whose body makes no call. Its body
/// may branch inside itself (an `if`, `&&` / `||`, `?:`, a `continue` or an
/// early exit, a loop nest's inner loops): the facts then come from a
/// dataflow over the body's branches, a register's at a join being what
/// every edge into it agrees on ([`spec_nest_rows`]), and a leave for the
/// plain copy from inside an arm resumes at its op there, the copy's own
/// branches staying in the copy (`NumGen::branch_label`).
/// A loop-carried register an arithmetic op reads is assumed (and checked at
/// entry) to be a Number when the static types do not prove it, unless the
/// loop itself may leave a non-Number in it.
fn plan_spec(
    r: &Region,
    ops: &[GenericOp],
    bytecode: &[Op],
    types: &Types,
    n_regs: usize,
    fields: &[Option<(u64, u32)>],
) -> Option<Spec> {
    let (head, end) = (r.head, r.end);
    if !r.generic || r.int.is_some() || end >= ops.len() || end >= bytecode.len() {
        return None;
    }
    let len = end - head + 1;
    if len.saturating_mul(n_regs) > MAX_STATE {
        return None;
    }
    // Whether the body branches inside itself (an `if`, `&&`, `?:`, a
    // `continue`, a loop nest's inner loops).
    let mut branches = false;
    for (pc, op) in ops.iter().enumerate().take(end + 1).skip(head) {
        // (A guard diamond's fallback arm is not in the fast version.)
        if r.is_cold(pc) {
            continue;
        }
        if region_call(op, &bytecode[pc]) {
            return None;
        }
        let t = match *op {
            GenericOp::Jump { target } | GenericOp::JumpIfFalse { target, .. } => target,
            GenericOp::PushHandler { .. } => return None,
            _ => continue,
        };
        // A guard diamond's test, and its fast arm's jump to the join.
        if r.cold.iter().any(|&(test, fallback, join)| {
            (pc, t) == (test, fallback) || (pc + 1, t) == (fallback, join)
        }) {
            continue;
        }
        // (An in-body branch of the fast version stays in it, the plain
        // copy's in the copy, `NumGen::branch_label`.)
        if pc != end && (head..=end).contains(&t) {
            branches = true;
        }
    }
    let (mut d, mut u) = (Vec::new(), Vec::new());
    let mut written = vec![false; n_regs];
    let mut num_read = vec![false; n_regs];
    for op in &bytecode[head..=end] {
        def_use(op, &mut d, &mut u);
        for &x in &d {
            if let Some(w) = written.get_mut(x as usize) {
                *w = true;
            }
        }
        if spec_num_op(op) {
            for &x in &u {
                if let Some(w) = num_read.get_mut(x as usize) {
                    *w = true;
                }
            }
        }
    }
    let mut assumed: Vec<u16> = r
        .live_head
        .iter()
        .copied()
        .filter(|&x| num_read[x as usize] && types.get(head, x) != T_NUM)
        .collect();
    loop {
        let mut cur: Vec<u8> = (0..n_regs).map(|x| types.get(head, x as u16)).collect();
        for &x in &assumed {
            cur[x as usize] = T_NUM;
        }
        let mut rows = Vec::with_capacity(len * n_regs);
        if !branches {
            for (i, op) in bytecode[head..=end].iter().enumerate() {
                rows.extend_from_slice(&cur);
                // (The fast arm's jump skips the fallback arm: its join
                // is reached with the fast arm's state.)
                if !r.is_cold(head + i) {
                    spec_step(op, head + i, &r.reuse, fields, &mut cur);
                }
            }
        } else {
            cur = spec_nest_rows(r, ops, bytecode, n_regs, fields, cur, &mut rows);
        }
        // `cur` is the back-edge's state: an assumed register the loop writes
        // must come back a Number.
        let before = assumed.len();
        assumed.retain(|&x| !written[x as usize] || cur[x as usize] == T_NUM);
        if assumed.len() == before {
            let mut facts: Vec<(u64, u64)> = Vec::new();
            for (pc, op) in bytecode.iter().enumerate().take(end + 1).skip(head) {
                // (A read the fast version replaces by a copy reads no slot.)
                if r.reuse.iter().any(|x| x.0 == pc) {
                    continue;
                }
                if let Some(&Some((addr, slot))) = fields.get(pc)
                    && matches!(op, Op::GetProp { .. })
                {
                    match facts.iter_mut().find(|f| f.0 == addr) {
                        Some(f) => f.1 |= 1 << slot,
                        None => facts.push((addr, 1 << slot)),
                    }
                }
            }
            return Some(Spec {
                types: rows,
                checks: assumed,
                fields: facts,
            });
        }
    }
}

/// The most typed-array views a speculative region hoists (three or four
/// frame slots and an entry check each).
const MAX_VIEWS: usize = 4;

/// Where a hoisted view's entry check finds the register's value
/// ([`plan_views`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ViewSrc {
    /// No op of the region writes the register: its value at entry is its
    /// value throughout (so an access needs no compare).
    Fixed,
    /// Only reads of one global binding the region never writes (the
    /// `LoadGlobal` at this pc among them) write it: the binding's value,
    /// which every such read in the fast version yields.
    Global(usize),
    /// Reads of captured variables or properties write it: its value at
    /// entry, likely what they read again.
    Held,
}

/// The receivers speculative region `r`'s fast version keeps a **hoisted
/// view** of (`super::typed::ViewSlots`): the registers its element reads,
/// element writes and `.length` reads are on, found from the ops that name
/// them: each must be one known to leave the register alone (an element op
/// on it, or one whose only write is to another register) or, for a
/// register the region may write ([`ViewSrc`]), a read of a global
/// binding, a captured variable or a property into it — else the register
/// is not hoisted.
fn plan_views(
    r: &Region,
    ops: &[GenericOp],
    bytecode: &[Op],
    n_regs: usize,
) -> Vec<(u16, ViewSrc)> {
    let mut regs: Vec<u16> = Vec::new();
    for op in &ops[r.head..=r.end] {
        let recv = match *op {
            GenericOp::GetElem { obj, .. } | GenericOp::SetElem { obj, .. } => obj,
            GenericOp::ArrayLen { arr, .. } => arr,
            _ => continue,
        };
        if (recv as usize) < n_regs && !regs.contains(&recv) {
            regs.push(recv);
        }
    }
    let body = &bytecode[r.head..=r.end];
    let mut out = Vec::new();
    for v in regs {
        // The ops that may write `v`: each must be a read of a binding,
        // a captured variable or a property into it.
        let mut writers = Vec::new();
        let mut ok = true;
        for (i, op) in body.iter().enumerate() {
            let leaves = match op {
                Op::GetKey { dst, .. } | Op::ArrayLen { dst, .. } | Op::GetProp { dst, .. }
                    if *dst != v =>
                {
                    true
                }
                Op::SetKey { .. } | Op::SetKeyStrict { .. } | Op::JumpIfFalse { .. } => true,
                _ => written_reg(op).is_some_and(|d| d != v),
            };
            if leaves || !op_regs(op).contains(&v) {
                continue;
            }
            match op {
                Op::LoadGlobal { dst, .. } | Op::CellGet { dst, .. } if *dst == v => {
                    writers.push(r.head + i);
                }
                Op::GetProp { dst, obj, .. } if *dst == v && *obj != v => writers.push(r.head + i),
                _ => ok = false,
            }
        }
        if !ok {
            continue;
        }
        let src = match writers.first() {
            None => ViewSrc::Fixed,
            Some(&first) => {
                // One global binding the region never writes: its value
                // throughout the fast version (no Rust code runs there).
                let name = |pc: usize| match &bytecode[pc] {
                    Op::LoadGlobal { name, .. } => Some(name.as_str()),
                    _ => None,
                };
                match name(first) {
                    Some(n)
                        if writers.iter().all(|&w| name(w) == Some(n))
                            && !body.iter().any(
                                |op| matches!(op, Op::StoreGlobal { name: m, .. } if m == n),
                            ) =>
                    {
                        ViewSrc::Global(first)
                    }
                    _ => ViewSrc::Held,
                }
            }
        };
        out.push((v, src));
    }
    out.truncate(MAX_VIEWS);
    out
}

/// [`plan_spec`]'s in-state rows for region `r`, whose body branches inside
/// itself (into `rows`), its entry state `entry`: a forward dataflow over
/// the body to a fixpoint, each join (an `if`'s end, a loop nest's inner
/// header or exit) taking the [`join`] of its incoming edges' states, so a
/// fact holds there only if it holds on every edge. Returns the state the
/// back-edges carry to the header.
fn spec_nest_rows(
    r: &Region,
    ops: &[GenericOp],
    bytecode: &[Op],
    n_regs: usize,
    fields: &[Option<(u64, u32)>],
    entry: Vec<u8>,
    rows: &mut Vec<u8>,
) -> Vec<u8> {
    let (head, end) = (r.head, r.end);
    let len = end - head + 1;
    rows.clear();
    rows.resize(len * n_regs, T_BOT);
    rows[..n_regs].copy_from_slice(&entry);
    let mut back = vec![T_BOT; n_regs];
    let mut cur = vec![T_BOT; n_regs];
    let mut changed = true;
    while changed {
        changed = false;
        for i in 0..len {
            let pc = head + i;
            cur.copy_from_slice(&rows[i * n_regs..(i + 1) * n_regs]);
            if cur.iter().all(|&t| t == T_BOT) && n_regs > 0 {
                continue;
            }
            // (Not in the fast version: a guard diamond's fallback arm, and
            // its test's branch there.)
            if r.is_cold(pc) {
                continue;
            }
            spec_step(&bytecode[pc], pc, &r.reuse, fields, &mut cur);
            let (falls, target) = match ops[pc] {
                GenericOp::Jump { target } => (false, Some(target)),
                GenericOp::JumpIfFalse { .. } if r.cold.iter().any(|c| c.0 == pc) => (true, None),
                GenericOp::JumpIfFalse { target, .. } => (true, Some(target)),
                _ => (true, None),
            };
            let mut flow = |to: usize, rows: &mut Vec<u8>, changed: &mut bool| {
                if to == head {
                    for (b, &c) in back.iter_mut().zip(&cur) {
                        *b = join(*b, c);
                    }
                } else if to > head && to <= end {
                    let row = &mut rows[(to - head) * n_regs..(to - head + 1) * n_regs];
                    for (x, &c) in row.iter_mut().zip(&cur) {
                        let j = join(*x, c);
                        if j != *x {
                            *x = j;
                            *changed = true;
                        }
                    }
                }
            };
            if falls && pc < end {
                flow(pc + 1, rows, &mut changed);
            }
            if let Some(t) = target {
                flow(t, rows, &mut changed);
            }
        }
    }
    back
}

/// The register a region op writes (region ops write at most one, `dst`).
fn written_reg(op: &Op) -> Option<u16> {
    match op {
        Op::LoadConst { dst, .. }
        | Op::Move { dst, .. }
        | Op::Add { dst, .. }
        | Op::AddValue { dst, .. }
        | Op::Sub { dst, .. }
        | Op::Mul { dst, .. }
        | Op::Div { dst, .. }
        | Op::Mod { dst, .. }
        | Op::ValueBin { dst, .. }
        | Op::Lt { dst, .. }
        | Op::Compare { dst, .. }
        | Op::StrictEq { dst, .. }
        | Op::ToNumeric { dst, .. }
        | Op::Step { dst, .. }
        | Op::Neg { dst, .. }
        | Op::BitNot { dst, .. }
        | Op::Not { dst, .. }
        | Op::LoadGlobal { dst, .. } => Some(*dst),
        _ => None,
    }
}

// --- encoders (SSE / GPR forms with arbitrary registers) ---

impl X64Assembler {
    /// Before a `call`: stores the promoted registers the call brackets (see
    /// `X64Assembler::set_call_spill`).
    pub(super) fn spill_before_call(&mut self) {
        for i in 0..self.call_spill.len() {
            let (x, d, store) = self.call_spill[i];
            if store {
                self.movsd_mem_x(d, x);
            }
        }
    }

    /// After a `call`: reloads the promoted registers it brackets (the SSE
    /// registers are caller-saved, and the call may have changed the values,
    /// e.g. a collection relocating a handle).
    pub(super) fn reload_after_call(&mut self) {
        for i in 0..self.call_spill.len() {
            let (x, d, _) = self.call_spill[i];
            self.movsd_x_mem(x, d);
        }
    }

    /// Appends raw instruction bytes.
    pub(super) fn raw(&mut self, bytes: &[u8]) {
        self.code.extend_from_slice(bytes);
    }

    /// `[pfx] [REX] 0F opc /r` with `reg`/`rm` registers (mod = 11).
    fn sse_rr_w(&mut self, pfx: u8, w: bool, opc: u8, reg: u8, rm: u8) {
        self.code.push(pfx);
        let rex = 0x40 | (u8::from(w) << 3) | ((reg >> 3) << 2) | (rm >> 3);
        if rex != 0x40 {
            self.code.push(rex);
        }
        self.code
            .extend_from_slice(&[0x0f, opc, 0xc0 | ((reg & 7) << 3) | (rm & 7)]);
    }

    /// An SSE `xmm(reg), xmm(rm)` op.
    fn sse_rr(&mut self, pfx: u8, opc: u8, reg: u8, rm: u8) {
        self.sse_rr_w(pfx, false, opc, reg, rm);
    }

    /// `[pfx] [REX] 0F opc /r` with `[rbx + disp32]` (the register file).
    fn sse_rm(&mut self, pfx: u8, w: bool, opc: u8, reg: u8, disp: i32) {
        self.code.push(pfx);
        let rex = 0x40 | (u8::from(w) << 3) | ((reg >> 3) << 2);
        if rex != 0x40 {
            self.code.push(rex);
        }
        self.code
            .extend_from_slice(&[0x0f, opc, 0x80 | ((reg & 7) << 3) | 3]);
        self.code.extend_from_slice(&disp.to_le_bytes());
    }

    /// `F2 [REX.R] 0F opc` with `xmm(x)` and `[rax]` (`10` load, `11` store).
    fn sse_mem_rax(&mut self, opc: u8, x: u8) {
        self.code.push(0xf2);
        if x >= 8 {
            self.code.push(0x44);
        }
        self.code.extend_from_slice(&[0x0f, opc, (x & 7) << 3]);
    }

    /// Likewise with `[rcx]`.
    fn sse_mem_rcx(&mut self, opc: u8, x: u8) {
        self.code.push(0xf2);
        if x >= 8 {
            self.code.push(0x44);
        }
        self.code
            .extend_from_slice(&[0x0f, opc, ((x & 7) << 3) | 1]);
    }

    /// `movsd xmm(x), [rbx + disp]`.
    fn movsd_x_mem(&mut self, x: u8, disp: i32) {
        self.sse_rm(0xf2, false, 0x10, x, disp);
    }

    /// `movsd [rbx + disp], xmm(x)`.
    fn movsd_mem_x(&mut self, disp: i32, x: u8) {
        self.sse_rm(0xf2, false, 0x11, x, disp);
    }

    /// `movapd xmm(d), xmm(s)`.
    fn movapd_xx(&mut self, d: u8, s: u8) {
        if d != s {
            self.sse_rr(0x66, 0x28, d, s);
        }
    }

    /// `movq gpr, xmm(x)`.
    fn movq_gpr_x(&mut self, gpr: u8, x: u8) {
        self.sse_rr_w(0x66, true, 0x7e, x, gpr);
    }

    /// `movq xmm(x), gpr`.
    fn movq_x_gpr(&mut self, x: u8, gpr: u8) {
        self.sse_rr_w(0x66, true, 0x6e, x, gpr);
    }

    /// `cvttsd2si gpr, xmm(x)` (64-bit).
    fn cvttsd2si_gpr_x(&mut self, gpr: u8, x: u8) {
        self.sse_rr_w(0xf2, true, 0x2c, gpr, x);
    }

    /// `xorpd xmm(x), xmm(x); cvtsi2sd xmm(x), gpr` (64-bit source; the zeroing
    /// breaks the false dependency on `x`'s old value).
    fn cvtsi2sd_x_gpr(&mut self, x: u8, gpr: u8) {
        self.sse_rr(0x66, 0x57, x, x);
        self.sse_rr_w(0xf2, true, 0x2a, x, gpr);
    }

    /// `mov gpr, [rbx + disp]` (rax / rcx / rdx).
    fn mov_gpr_mem(&mut self, gpr: u8, disp: i32) {
        self.code
            .extend_from_slice(&[0x48, 0x8b, 0x80 | ((gpr & 7) << 3) | 3]);
        self.code.extend_from_slice(&disp.to_le_bytes());
    }

    /// `mov [rbx + disp], gpr` (rax / rcx / rdx).
    fn mov_mem_gpr(&mut self, disp: i32, gpr: u8) {
        self.code
            .extend_from_slice(&[0x48, 0x89, 0x80 | ((gpr & 7) << 3) | 3]);
        self.code.extend_from_slice(&disp.to_le_bytes());
    }

    /// `cmp gpr, r11` (rax / rcx / rdx).
    fn cmp_gpr_r11(&mut self, gpr: u8) {
        self.code.extend_from_slice(&[0x4c, 0x39, 0xd8 | (gpr & 7)]);
    }

    /// `cmp qword [rbp + disp32], imm8` (a native frame slot).
    fn cmp_frame_imm8(&mut self, disp: i32, imm: i8) {
        self.code.extend_from_slice(&[0x48, 0x83, 0xbd]);
        self.code.extend_from_slice(&disp.to_le_bytes());
        self.code.push(imm as u8);
    }

    /// `mov qword [rbp + disp32], 0` (a native frame slot).
    fn mov_frame_zero(&mut self, disp: i32) {
        self.code.extend_from_slice(&[0x48, 0xc7, 0x85]);
        self.code.extend_from_slice(&disp.to_le_bytes());
        self.code.extend_from_slice(&0i32.to_le_bytes());
    }

    /// `jcc rel32` to `label` (`cc` an x86 condition code).
    fn jcc(&mut self, cc: u8, label: Label) {
        self.code.extend_from_slice(&[0x0f, 0x80 | cc]);
        self.emit_rel32(label);
    }

    /// `setcc al`.
    fn setcc_al(&mut self, cc: u8) {
        self.code.extend_from_slice(&[0x0f, 0x90 | cc, 0xc0]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compiles `src` and returns the proto of function `name`.
    fn proto_of(src: &str, name: &str) -> crate::nbvm::FnProto {
        let program = crate::parser::Parser::parse_program(src).expect("parse");
        let funcs = crate::nbvm::compile_program(&program).expect("compile");
        funcs
            .into_iter()
            .find(|p| p.name == name)
            .unwrap_or_else(|| panic!("no function {name}"))
    }

    /// The pc of the first op of `proto` matching `pred`.
    fn pc_of(proto: &crate::nbvm::FnProto, pred: impl Fn(&Op) -> bool) -> usize {
        proto.ops.iter().position(pred).expect("op present")
    }

    #[test]
    fn inference_proves_numeric_loop_registers() {
        // `i` starts at 0 and only steps; `a * 3` is a Number whatever `a` is;
        // `x + 1` is unknown (`x` is a parameter: a string concatenates).
        let p = proto_of(
            "function f(x){ let a = x, s = 0; for (let i = 0; i < 10; i = i + 1) { s = (s + i * 3) | 0; a = a * 3; } return x + 1; }",
            "f",
        );
        let t = infer(&p.ops, p.n_regs);
        let mul = pc_of(&p, |o| matches!(o, Op::Mul { .. }));
        let Op::Mul { a, .. } = p.ops[mul] else {
            unreachable!()
        };
        assert_eq!(t.get(mul, a), T_NUM, "the loop counter is a Number");
        let or = pc_of(&p, |o| matches!(o, Op::ValueBin { .. }));
        let Op::ValueBin { a: lhs, .. } = p.ops[or] else {
            unreachable!()
        };
        assert_eq!(t.get(or, lhs), T_NUM, "s + i*3 is a Number");
        let cmp = pc_of(&p, |o| matches!(o, Op::Compare { .. } | Op::Lt { .. }));
        let jif = pc_of(&p, |o| matches!(o, Op::JumpIfFalse { .. }));
        let Op::JumpIfFalse { cond, .. } = p.ops[jif] else {
            unreachable!()
        };
        assert!(jif > cmp);
        assert_eq!(t.get(jif, cond), T_BOOL, "a comparison is a boolean");
        let add = p
            .ops
            .iter()
            .rposition(|o| matches!(o, Op::AddValue { .. }))
            .expect("x + 1");
        let Op::AddValue { a: xa, .. } = p.ops[add] else {
            unreachable!()
        };
        assert_eq!(t.get(add, xa), T_ANY, "a parameter is anything");
    }

    #[test]
    fn inference_is_conservative_across_unmodeled_ops_and_catches() {
        // `n` is a Number until an unmodeled op (`typeof`-free, a property get)
        // overwrites it; a catch entry knows nothing.
        let p = proto_of(
            "function f(o){ let n = 1; n = o.k; let r = n * 2; try { r = r + 1; } catch (e) { r = e; } return r; }",
            "f",
        );
        let t = infer(&p.ops, p.n_regs);
        let get = pc_of(&p, |o| matches!(o, Op::GetProp { .. }));
        let Op::GetProp { dst, .. } = p.ops[get] else {
            unreachable!()
        };
        assert_eq!(t.get(get + 1, dst), T_ANY);
        let Some(Op::PushHandler { target, .. }) =
            p.ops.iter().find(|o| matches!(o, Op::PushHandler { .. }))
        else {
            panic!("a try")
        };
        for r in 0..p.n_regs as u16 {
            assert_eq!(t.get(*target, r), T_ANY);
        }
    }

    #[test]
    fn regions_promote_carried_and_temporary_registers() {
        let p = proto_of(
            "function f(){ let a = 0, b = 1.5; for (let i = 0; i < 100; i = i + 1) { a = (a + i * 3) | 0; b = b * 0.5 + 1; } return a + b; }",
            "f",
        );
        let (ops, ..) = crate::jit::lower_nbvm_generic(&p).expect("generic");
        let regions = plan_regions(&ops, &p.ops, &[], p.n_regs);
        assert_eq!(regions.len(), 1, "one numeric loop");
        let r = &regions[0];
        assert!(matches!(ops[r.end], GenericOp::Jump { target } if target == r.head));
        let carried: Vec<u16> = r
            .slots
            .iter()
            .filter(|s| s.carried)
            .map(|s| s.reg)
            .collect();
        assert_eq!(carried.len(), 3, "a, b and i are loop-carried: {carried:?}");
        assert!(
            r.slots.iter().any(|s| !s.carried),
            "temporaries share registers"
        );
        // No two slots live at once share an SSE register (a move's source
        // may hand its register to the destination at the move).
        for x in &r.slots {
            for y in &r.slots {
                if !core::ptr::eq(x, y) && x.xmm == y.xmm {
                    let handoff = x.end == y.start
                        && matches!(p.ops[y.start], Op::Move { dst, src } if dst == y.reg && src == x.reg);
                    assert!(
                        x.end < y.start || y.end < x.start || handoff,
                        "overlapping intervals share xmm{}",
                        x.xmm
                    );
                }
            }
        }
        // A loop with a property access is a region of generic ops: the
        // object stays in the register file, the sum is promoted.
        let q = proto_of(
            "function g(o){ let s = 0; for (let i = 0; i < 100; i = i + 1) { s = s + o.x; } return s; }",
            "g",
        );
        let (qops, ..) = crate::jit::lower_nbvm_generic(&q).expect("generic");
        let regions = plan_regions(&qops, &q.ops, &[], q.n_regs);
        assert_eq!(regions.len(), 1);
        let r = &regions[0];
        assert!(r.generic && r.int.is_none());
        let get = pc_of(&q, |o| matches!(o, Op::GetProp { .. }));
        let Op::GetProp { dst, obj, .. } = q.ops[get] else {
            unreachable!()
        };
        assert!(r.slots.iter().all(|s| s.reg != obj), "the object is pinned");
        assert!(
            r.slots.iter().any(|s| s.reg == dst && s.dirty),
            "the field value is promoted (and dirty)"
        );
        // Its speculative version assumes the sum is a Number at entry.
        let t = infer(&q.ops, q.n_regs);
        let sp = plan_spec(r, &qops, &q.ops, &t, q.n_regs, &[]).expect("speculative");
        let add = pc_of(&q, |o| matches!(o, Op::AddValue { .. }));
        let Op::AddValue { a: sum, .. } = q.ops[add] else {
            unreachable!()
        };
        assert!(sp.checks.contains(&sum), "checks {:?}", sp.checks);
        // A branch inside the body: a speculative version too (its facts
        // from the dataflow over the body's branches).
        let b = proto_of(
            "function h(o){ let s = 0; for (let i = 0; i < 100; i = i + 1) { if (o.x > 1) s = s + o.x; } return s; }",
            "h",
        );
        let (bops, ..) = crate::jit::lower_nbvm_generic(&b).expect("generic");
        let regions = plan_regions(&bops, &b.ops, &[], b.n_regs);
        assert_eq!(regions.len(), 1);
        let t = infer(&b.ops, b.n_regs);
        assert!(plan_spec(&regions[0], &bops, &b.ops, &t, b.n_regs, &[]).is_some());
    }

    #[test]
    fn spills_favour_the_inner_loops_arithmetic() {
        // More values live across the inner loop than there are SSE
        // registers: the outer loop's read-only values (and its
        // accumulators) give theirs up, the inner loop's arithmetic keeps
        // them (each temporary of `dx*dx + dy*dy + ...` in a register).
        let p = proto_of(
            "function f(bs, dt){ for (let i = 0; i < bs.length; i = i + 1) { const a = bs[i]; \
             const ax = a.x, ay = a.y, az = a.z, am = a.m, aw = a.w, au = a.u; \
             let avx = a.vx, avy = a.vy, avz = a.vz; \
             for (let j = i + 1; j < bs.length; j = j + 1) { const b = bs[j]; \
             const dx = ax - b.x, dy = ay - b.y, dz = az - b.z; \
             const d2 = dx * dx + dy * dy + dz * dz + aw * au; \
             const mag = dt / (d2 * d2); \
             avx = avx - dx * b.m * mag; avy = avy - dy * b.m * mag; avz = avz - dz * b.m * mag; \
             b.vx = b.vx + dx * am * mag; } a.vx = avx; a.vy = avy; a.vz = avz; } }",
            "f",
        );
        let (ops, ..) = crate::jit::lower_nbvm_generic(&p).expect("generic");
        let regions = plan_regions(&ops, &p.ops, &[], p.n_regs);
        let r = regions
            .iter()
            .find(|r| !r.loops.is_empty())
            .expect("one nest");
        let (h, e) = r.loops[0];
        let promoted = |reg: u16| r.slots.iter().any(|s| s.reg == reg);
        // Every product of the inner body is promoted.
        let products: Vec<u16> = (h..=e)
            .filter_map(|pc| match p.ops[pc] {
                Op::Mul { dst, .. } => Some(dst),
                _ => None,
            })
            .collect();
        assert!(products.len() >= 8);
        for d in &products {
            assert!(promoted(*d), "inner product r{d} promoted");
        }
        // The registers still fit: no two live slots share an SSE register.
        for x in &r.slots {
            for y in &r.slots {
                if x.reg != y.reg && x.xmm == y.xmm {
                    assert!(
                        x.end < y.start || y.end < x.start || x.end == y.start || y.end == x.start
                    );
                }
            }
        }
    }

    #[test]
    fn nan_checks_merge_into_a_later_one() {
        // `d` and `e` (each from a property read) flow only into `f`, whose
        // operand check before the store stands for theirs; the unused
        // quotient keeps its own check, as does the op whose result the
        // store reads.
        let p = proto_of(
            "function f(xs, k){ let s = 0; for (let i = 0; i < xs.length; i = i + 1) { const o = xs[i]; \
             const d = o.a - 1; const e = o.b * 2; const u = o.c / 3; \
             const g = d * d + e * k; o.out = g + o.out; s = s + d; } return s; }",
            "f",
        );
        let plan = defer_plan(&p);
        let kind = |pc: usize| plan.kind.iter().find(|x| x.0 == pc).map_or(0, |x| x.1);
        let pc_mul2 = pc_of(&p, |o| matches!(o, Op::Mul { .. }));
        let pc_sub1 = pc_of(&p, |o| matches!(o, Op::Sub { .. }));
        let pc_div = pc_of(&p, |o| matches!(o, Op::Div { .. }));
        assert_eq!(kind(pc_sub1), DEFER_SKIP, "{:?}", plan.kind);
        assert_eq!(kind(pc_mul2), DEFER_SKIP, "{:?}", plan.kind);
        assert_eq!(kind(pc_div), 0, "the unused quotient keeps its check");
        // The store's operand: a covering check.
        let store = pc_of(&p, |o| matches!(o, Op::SetProp { .. }));
        assert!(
            plan.kind
                .iter()
                .any(|&(pc, k)| k == DEFER_COVER && pc < store)
        );
        // Every leave from the merged stretch restarts it.
        assert_eq!(plan.spans.len(), 1);
        let (first, last, start) = plan.spans[0];
        assert!(start <= pc_sub1 && first == pc_sub1 && last < store);
        // Its stub tells a Number's NaN by the merged ops' own unproven
        // operands: the reads of `o.a` and `o.b`.
        let read = |key: &str| {
            p.ops.iter().find_map(|o| match o {
                Op::GetProp { dst, key: k, .. } if &**k == key => Some(*dst),
                _ => None,
            })
        };
        assert_eq!(plan.covers.len(), 1);
        let roots = &plan.covers[0].roots;
        for key in ["a", "b"] {
            let r = read(key).expect("read");
            assert!(roots.contains(&r), "{key}: {roots:?}");
        }
    }

    /// The first speculative region of `p`, its merged NaN checks planned.
    fn defer_plan(p: &crate::nbvm::FnProto) -> DeferPlan {
        let (ops, ..) = crate::jit::lower_nbvm_generic(p).expect("generic");
        let mut regions = plan_regions(&ops, &p.ops, &[], p.n_regs);
        let types = infer(&p.ops, p.n_regs);
        let r = &mut regions[0];
        r.spec = plan_spec(r, &ops, &p.ops, &types, p.n_regs, &[]);
        assert!(r.spec.is_some());
        let mut is_target = vec![false; p.ops.len() + 1];
        for op in &p.ops {
            if let Op::Jump { target } | Op::JumpIfFalse { target, .. } = op {
                is_target[*target] = true;
            }
        }
        let (words, live) = liveness(&p.ops, p.n_regs, &[]).expect("liveness");
        let live_at =
            |pc: usize, x: u16| live[pc * words + x as usize / 64] & (1u64 << (x % 64)) != 0;
        plan_defer(r, &ops, &p.ops, p.n_regs, &is_target, &live_at)
    }

    #[test]
    fn a_merged_check_needs_its_operands_intact() {
        // `x` held the read the product did not prove, and is written
        // before the sum's check: that check could not tell a Number's NaN
        // from a non-Number there, so the product keeps its own.
        let p = proto_of(
            "function f(xs){ let t = 0; for (let i = 0; i < xs.length; i = i + 1) { const o = xs[i]; \
             let x = o.a; const y = x * 3; x = o.b; const z = y + x; o.out = z; t = t + z; } return t; }",
            "f",
        );
        let plan = defer_plan(&p);
        let mul = pc_of(&p, |o| matches!(o, Op::Mul { .. }));
        assert!(
            plan.kind.iter().all(|&(pc, _)| pc != mul),
            "{:?}",
            plan.kind
        );
    }

    #[test]
    fn loop_nests_promote_across_inner_loops() {
        // A call-free nest of generic loops: one region, its inner
        // back-edge recorded, the outer object and index live across the
        // inner loop held for all of it, and a speculative version.
        let p = proto_of(
            "function f(bs){ let s = 0; for (let i = 0; i < bs.length; i = i + 1) { const a = bs[i]; for (let j = i + 1; j < bs.length; j = j + 1) { s = s + a.x * bs[j].x; } } return s; }",
            "f",
        );
        let (ops, ..) = crate::jit::lower_nbvm_generic(&p).expect("generic");
        let mut regions = plan_regions(&ops, &p.ops, &[], p.n_regs);
        assert_eq!(regions.len(), 1, "one nest");
        let types = infer(&p.ops, p.n_regs);
        let r = &mut regions[0];
        assert_eq!(r.loops.len(), 1, "one inner loop");
        let (h, e) = r.loops[0];
        assert!(r.head < h && e < r.end);
        assert!(matches!(ops[e], GenericOp::Jump { target } if target == h));
        // A register live into the inner header covers the whole inner loop.
        let (words, live) = liveness(&p.ops, p.n_regs, &[]).expect("liveness");
        for s in &r.slots {
            let at_h = live[h * words + s.reg as usize / 64] & (1u64 << (s.reg % 64)) != 0;
            if at_h {
                assert!(
                    s.start <= h && s.end >= e,
                    "slot r{} spans the inner loop",
                    s.reg
                );
            }
        }
        r.spec = plan_spec(r, &ops, &p.ops, &types, p.n_regs, &[]);
        assert!(r.spec.is_some(), "the nest has a speculative version");
        // Its inner header's live registers (its entry from the plain
        // copy): the outer object and index among them.
        let at_h = &r
            .live_inner
            .iter()
            .find(|x| x.0 == h)
            .expect("inner header")
            .1;
        for x in at_h {
            assert!(live[h * words + *x as usize / 64] & (1u64 << (x % 64)) != 0);
        }
        assert!(at_h.len() >= 3, "{at_h:?}");
        // An inner loop with a call keeps its own region (no nest).
        let q = proto_of(
            "function g(bs, f){ let s = 0; for (let i = 0; i < bs.length; i = i + 1) { const a = bs[i]; for (let j = 0; j < bs.length; j = j + 1) { s = s + f(a.x); } } return s; }",
            "g",
        );
        let (qops, ..) = crate::jit::lower_nbvm_generic(&q).expect("generic");
        let qr = plan_regions(&qops, &q.ops, &[], q.n_regs);
        assert!(qr.iter().all(|r| r.loops.is_empty()));
    }

    #[test]
    fn branchy_loops_have_a_speculative_version() {
        // An `if` / `else` in the body: a speculative version, its liveness
        // recorded, and at the join only what both arms agree on: `x` is a
        // Number on one arm (`i * 0.5`) and an element on the other.
        let p = proto_of(
            "function f(a, strs){ let s = 0; for (let i = 0; i < a.length; i = i + 1) { let x; if (i & 1) x = i * 0.5; else x = strs[i]; s = s + x * 2; } return s; }",
            "f",
        );
        let (ops, ..) = crate::jit::lower_nbvm_generic(&p).expect("generic");
        let mut regions = plan_regions(&ops, &p.ops, &[], p.n_regs);
        assert_eq!(regions.len(), 1);
        let types = infer(&p.ops, p.n_regs);
        let r = &mut regions[0];
        assert!(r.loops.is_empty() && r.live_words > 0);
        r.spec = plan_spec(r, &ops, &p.ops, &types, p.n_regs, &[]);
        let sp = r.spec.as_ref().expect("a speculative version");
        let n = p.n_regs;
        let muls: Vec<usize> = (r.head..=r.end)
            .filter(|&pc| matches!(p.ops[pc], Op::Mul { .. }))
            .collect();
        assert_eq!(muls.len(), 2);
        let ty = |pc: usize, x: u16| sp.types[(pc - r.head) * n + x as usize];
        // `i * 0.5`, on the first arm: `i` is a Number there.
        let Op::Mul { a: i, .. } = p.ops[muls[0]] else {
            unreachable!()
        };
        assert_eq!(ty(muls[0], i), T_NUM);
        // `x * 2`, past the join: `x` is not.
        let Op::Mul { a: x, .. } = p.ops[muls[1]] else {
            unreachable!()
        };
        assert_ne!(ty(muls[1], x), T_NUM);
    }

    #[test]
    fn spec_facts_follow_guarded_operands() {
        // `o.x - o.y` passes only on Numbers: past it the two field values
        // are Numbers, and so is the reused read of `o.x` (a copy of the
        // first). A field read the arithmetic has not seen yet is not.
        let p = proto_of(
            "function f(o){ let s = 0; for (let i = 0; i < 100; i = i + 1) { const d = o.x - o.y; s = s + o.x * d + o.z; } return s; }",
            "f",
        );
        let (ops, ..) = crate::jit::lower_nbvm_generic(&p).expect("generic");
        let regions = plan_regions(&ops, &p.ops, &[], p.n_regs);
        assert_eq!(regions.len(), 1);
        let r = &regions[0];
        let t = infer(&p.ops, p.n_regs);
        let sp = plan_spec(r, &ops, &p.ops, &t, p.n_regs, &[]).expect("speculative");
        let ty = |pc: usize, reg: u16| sp.types[(pc - r.head) * p.n_regs + reg as usize];
        let sub = pc_of(&p, |o| matches!(o, Op::Sub { .. }));
        let Op::Sub { a: x, b: y, .. } = p.ops[sub] else {
            unreachable!()
        };
        assert_eq!(ty(sub, x), T_ANY, "o.x before the subtraction");
        assert_eq!(ty(sub + 1, x), T_NUM, "o.x past it");
        assert_eq!(ty(sub + 1, y), T_NUM, "o.y past it");
        let mul = pc_of(&p, |o| matches!(o, Op::Mul { .. }));
        let Op::Mul { a: x2, .. } = p.ops[mul] else {
            unreachable!()
        };
        assert!(
            r.reuse.iter().any(|&(_, src)| src == x),
            "the second o.x reuses the first: {:?}",
            r.reuse
        );
        assert_eq!(ty(mul, x2), T_NUM, "the reused read of o.x");
        let z = (r.head..=r.end)
            .rev()
            .find_map(|pc| match &p.ops[pc] {
                Op::GetProp { dst, key, .. } if key.as_str() == "z" => Some((pc, *dst)),
                _ => None,
            })
            .expect("o.z");
        assert_eq!(ty(z.0 + 1, z.1), T_ANY, "o.z is unchecked");
    }

    #[test]
    fn join_is_a_lattice() {
        assert_eq!(join(T_BOT, T_NUM), T_NUM);
        assert_eq!(join(T_NUM, T_NUM), T_NUM);
        assert_eq!(join(T_NUM, T_BOOL), T_ANY);
        assert_eq!(join(T_ANY, T_BOT), T_ANY);
    }

    #[test]
    fn sse_encodings() {
        let mut a = X64Assembler::new();
        a.movsd_x_mem(2, 8); // movsd xmm2, [rbx+8]
        assert_eq!(a.code(), &[0xf2, 0x0f, 0x10, 0x93, 8, 0, 0, 0]);
        let mut a = X64Assembler::new();
        a.movsd_mem_x(16, 9); // movsd [rbx+16], xmm9
        assert_eq!(a.code(), &[0xf2, 0x44, 0x0f, 0x11, 0x8b, 16, 0, 0, 0]);
        let mut a = X64Assembler::new();
        a.movq_gpr_x(RCX, 10); // movq rcx, xmm10
        assert_eq!(a.code(), &[0x66, 0x4c, 0x0f, 0x7e, 0xd1]);
        let mut a = X64Assembler::new();
        a.movq_x_gpr(0, RAX); // movq xmm0, rax
        assert_eq!(a.code(), &[0x66, 0x48, 0x0f, 0x6e, 0xc0]);
        let mut a = X64Assembler::new();
        a.cvttsd2si_gpr_x(RCX, 12); // cvttsd2si rcx, xmm12
        assert_eq!(a.code(), &[0xf2, 0x49, 0x0f, 0x2c, 0xcc]);
        let mut a = X64Assembler::new();
        a.sse_rr(0xf2, 0x59, 0, 15); // mulsd xmm0, xmm15
        assert_eq!(a.code(), &[0xf2, 0x41, 0x0f, 0x59, 0xc7]);
        let mut a = X64Assembler::new();
        a.sse_mem_rax(0x10, 9); // movsd xmm9, [rax]
        assert_eq!(a.code(), &[0xf2, 0x44, 0x0f, 0x10, 0x08]);
        let mut a = X64Assembler::new();
        a.sse_mem_rcx(0x11, 3); // movsd [rcx], xmm3
        assert_eq!(a.code(), &[0xf2, 0x0f, 0x11, 0x19]);
        let mut a = X64Assembler::new();
        a.mov_gpr_mem(RDX, 24); // mov rdx, [rbx+24]
        assert_eq!(a.code(), &[0x48, 0x8b, 0x93, 24, 0, 0, 0]);
    }
}
