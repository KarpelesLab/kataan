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
//! Every arithmetic result is NaN-canonicalized before it is stored, so a value
//! the native code produces is bit-identical to the interpreter's `NanBox`.

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

/// Infers the register types of `ops` (see the module docs).
pub(super) fn infer(ops: &[Op], n_regs: usize) -> Types {
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
    /// The pcs of arithmetic ops whose result needs no NaN canonicalization
    /// in its SSE register ([`raw_nan_results`]).
    raw_nan: Vec<usize>,
    /// The property reads the speculative version replaces by a copy of the
    /// register already holding the value ([`plan_reuse`]): `(pc, src)`.
    reuse: Vec<(usize, u16)>,
}

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
    /// `xmm x = canonical NaN` (no call).
    CanonNanX { x: u8 },
    /// `xmm x` is NaN-patterned: to `slow` unless it is a Number (a NaN),
    /// else resume (no call).
    NanCheck { x: u8, slow: Label },
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
}

/// The per-compilation state of the numeric emitter.
pub(super) struct NumGen {
    types: Types,
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
        GenericOp::Interp { .. } => matches!(
            bc,
            Op::IterNext { iter, next, .. } if usize::from(*next) == usize::from(*iter) + 1
        ),
        _ => false,
    }
}

/// Whether region op `op` may run arbitrary code (a call that is not an
/// inline `Math` function): a speculative version needs a call-free body.
fn region_call(op: &GenericOp, bc: &Op) -> bool {
    match op {
        GenericOp::CallMethod { .. } => super::inline_math(bc).is_none(),
        GenericOp::CallValue { .. }
        | GenericOp::CallWithMethod { .. }
        | GenericOp::CallKey { .. } => true,
        _ => false,
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
        for r in &mut regions {
            r.spec = plan_spec(r, ops, bytecode, &types, env.n_regs);
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
        Self {
            types,
            regions,
            region_of,
            is_target,
            mentions,
            stubs: Vec::new(),
            skip: vec![false; n + 1],
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
        }
    }

    /// Allocates the labels of the speculative regions' plain copies.
    pub(super) fn alloc_copy_labels(&mut self, a: &mut X64Assembler) {
        for r in &self.regions {
            if r.spec.is_some() {
                for pc in r.head..=r.end {
                    self.copy_labels[pc] = Some(a.new_label());
                }
            }
        }
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
    /// global binding addresses, then one epoch slot per region.
    pub(super) fn frame_slots(&self) -> usize {
        self.n_gslots + self.regions.len()
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
                a.load_rcx(sd);
                match self.xmm_of(reg, src) {
                    Some(x) => a.sse_mem_rcx(0x11, x), // movsd [rcx], xmm
                    None => {
                        self.load_gpr(a, reg, RAX, src);
                        a.raw(&[0x48, 0x89, 0x01]); // mov [rcx], rax
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
    /// that no op since has written. (The fast version runs straight from
    /// the header: its only branches leave it.)
    fn global_known(&self, ri: usize, pc: usize) -> Option<u16> {
        let Op::LoadGlobal { name, .. } = self.bc(pc)? else {
            return None;
        };
        let head = self.regions[ri].head;
        let (mut d, mut u) = (Vec::new(), Vec::new());
        for q in (head..pc).rev() {
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
    /// nothing promoted yet, the plain copy runs the iteration). Clobbers
    /// nothing.
    fn spec_globals_ready(&self, a: &mut X64Assembler, ri: usize) -> Option<()> {
        let copy = self.copy_label(self.regions[ri].head)?;
        for &k in &self.region_gslots[ri] {
            a.cmp_frame_imm8(self.gslot_disp(k), 0);
            a.jcc(CC_E, copy);
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
            kind: StubKind::SpecBail { pc: pc as u32 },
            branch: None,
            int: false,
        });
        Some(entry)
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
                self.spec_globals_ready(a, ri)?;
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
                        a.load_rcx(sd);
                        a.raw(&[0x48, 0x85, 0xc9]); // test rcx, rcx
                        a.jcc(CC_E, stub);
                        match self.xmm_of(reg, src) {
                            // movsd [rcx], xmm
                            Some(x) => a.sse_mem_rcx(0x11, x),
                            None => {
                                self.load_gpr(a, reg, RAX, src);
                                a.raw(&[0x48, 0x89, 0x01]); // mov [rcx], rax
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
                    Op::RequireObjectCoercible { src } => {
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
                    // The region's back-edge: poll, then the in-region header.
                    if pc != self.regions[ri].end || target != self.regions[ri].head {
                        return None;
                    }
                    let skip = a.new_label();
                    a.dec_frame8(COUNTDOWN);
                    a.jne(skip);
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
                        self.spec_globals_ready(a, ri)?;
                    }
                    self.reload(a, reg);
                    if !validate {
                        self.refresh_globals(a, ri, false);
                    }
                    a.bind(skip);
                    a.jmp(self.regions[ri].inner?);
                } else {
                    self.jump(a, labels, reg, target);
                }
            }
            GenericOp::Interp { pc: opc }
                if reg.is_some() && matches!(bc, Op::RequireObjectCoercible { .. }) =>
            {
                let Op::RequireObjectCoercible { src } = *bc else {
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
        self.regions[ri]
            .slots
            .iter()
            .filter(|s| s.start <= pc && pc <= s.end)
            .map(|s| (s.xmm, disp(s.reg), s.dirty && (s.carried || s.start < pc)))
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
        let (Some(&Op::RequireObjectCoercible { src }), Some(call @ Op::CallMethod { recv, .. })) =
            (self.bc(pc), self.bc(pc + 1))
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
        a.raw(&[0xf2, 0x49, 0x0f, 0x2a, 0xc0]); // cvtsi2sd xmm0, r8
    }

    /// `rax` = the exact non-negative-or-not integer value of the Number in
    /// `xmm x` (an element read's promoted index), else to `miss` (a
    /// fraction, NaN, out of `i64`'s range; with `check`, first a
    /// non-Number). Clobbers `xmm1` (and, checking, `r11`).
    pub(super) fn index_from_xmm(a: &mut X64Assembler, x: u8, check: bool, miss: Label) {
        if check {
            a.movq_gpr_x(RAX, x);
            a.movabs_r11(QNAN as i64);
            a.and_rax_r11();
            a.cmp_rax_r11();
            a.jcc(CC_E, miss);
        }
        a.cvttsd2si_gpr_x(RAX, x);
        a.cvtsi2sd_x_gpr(1, RAX);
        a.sse_rr(0x66, 0x2e, x, 1); // ucomisd x, xmm1
        a.jcc(CC_NE, miss);
        a.jcc(CC_P, miss);
    }

    /// `gpr = xmm x` (the bits; `gpr` one of `rax`/`rcx`/`rdx`).
    pub(super) fn gpr_from_xmm(a: &mut X64Assembler, gpr: u8, x: u8) {
        a.movq_gpr_x(gpr, x);
    }

    /// `xmm = (f64) rax`.
    pub(super) fn set_xmm_from_int_rax(a: &mut X64Assembler, x: u8) {
        a.cvtsi2sd_x_gpr(x, RAX);
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
                    pc: self.cur_pc as u32,
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
        for s in &self.regions[ri].slots {
            // A temporary op `pc` first defines holds no value yet (its first
            // mention is a write: a read would make it live at the header).
            if s.dirty && s.start <= pc && pc <= s.end && (s.carried || s.start < pc) {
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
        for s in &self.regions[ri].slots {
            if s.start <= pc && pc <= s.end {
                a.movsd_x_mem(s.xmm, disp(s.reg));
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
        let guarded = self.ty(pc, x) != T_NUM || self.ty(pc, y) != T_NUM;
        let stub = a.new_label();
        if guarded {
            self.guard_numbers(a, pc, reg, x, y, stub);
        }
        match self.xmm_of(reg, dst) {
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
                // A NaN made canonical (out of line), unless nothing can
                // observe its bits (`raw_nan_results`).
                if !reg.is_some_and(|ri| self.regions[ri].raw_nan.contains(&pc)) {
                    let fix = a.new_label();
                    let back = a.new_label();
                    a.sse_rr(0x66, 0x2e, xd, xd); // ucomisd xd, xd
                    a.jcc(CC_P, fix);
                    a.bind(back);
                    self.stub(fix, back, reg, StubKind::CanonNanX { x: xd });
                }
            }
            _ => {
                self.load_xmm0(a, reg, x);
                let xy = self.xmm_src(a, reg, y, 1);
                a.sse_rr(0xf2, opc, 0, xy);
                self.canon_xmm0(a, reg);
                self.store_xmm0(a, reg, dst);
            }
        }
        a.bind(resume);
        if guarded {
            let k = slow(self);
            self.stub(stub, resume, reg, k);
        }
    }

    /// `dst = x % y`: inline when both are integral Numbers that convert
    /// exactly to i64, with `y` neither `0` nor `-1` (no `idiv` overflow): the
    /// integer remainder is then fmod's exact result (the sign of a zero
    /// result follows `x`, `-0` included); else the helper (`fmod`).
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
        for r in [x, y] {
            if self.ty(pc, r) != T_NUM {
                self.num_guard(a, reg, r, stub);
            }
        }
        // rcx = int(y), exact (an out-of-range y converts to i64::MIN, which
        // only -2^63 itself converts back to), neither 0 nor -1.
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
        self.load_xmm0(a, reg, p);
        let xq = self.xmm_src(a, reg, q, 1);
        a.sse_rr(0x66, 0x2e, 0, xq); // ucomisd xmm0, q
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
                        w.shape,
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
            None => labels.get(target).copied(),
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
                StubKind::CanonNanX { x } => {
                    a.movabs_rax(self.canonical_nan as i64);
                    a.movq_x_gpr(x, RAX);
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
                        && let Some(l) = self.copy_label(pc)
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
        Op::LoadGlobal { dst, .. } | Op::NewBox { dst } => defs.push(*dst),
        Op::CellGet { dst, cell } => {
            defs.push(*dst);
            uses.push(*cell);
        }
        Op::CellSet { cell, src } => {
            uses.push(*cell);
            uses.push(*src);
        }
        Op::Return { src } | Op::CheckTdz { src } | Op::StoreGlobal { src, .. } => uses.push(*src),
        Op::RequireObjectCoercible { src } => uses.push(*src),
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
    let Some((words, live)) = liveness(bytecode, n_regs, catches) else {
        return Vec::new();
    };
    let is_live = |pc: usize, r: u16| {
        pc < n
            && (r as usize) < n_regs
            && live[pc * words + r as usize / 64] & (1u64 << (r % 64)) != 0
    };
    let mut out = Vec::new();
    for (head, end) in cands {
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
                    let free = match bytecode[pc] {
                        Op::GetKey { obj, key, .. } if key != obj => Some(key),
                        _ => None,
                    };
                    pinned.extend(u.iter().copied().filter(|&r| Some(r) != free));
                }
            }
        }
        info.retain(|x| !pinned.contains(&x.0));
        // Loop-carried: live into the header, at an exit, or at a catch entry.
        let carried_r = |r: u16| {
            is_live(head, r)
                || exits.iter().any(|&t| is_live(t, r))
                || catches.iter().any(|&c| is_live(c, r))
        };
        let mut carried: Vec<&(u16, usize, usize, u32, bool)> =
            info.iter().filter(|x| carried_r(x.0)).collect();
        carried.sort_by(|p, q| q.3.cmp(&p.3).then(p.0.cmp(&q.0)));
        let mut slots: Vec<Slot> = Vec::new();
        let mut free: Vec<u8> = (FIRST_XMM..FIRST_XMM + MAX_PROMOTED as u8).rev().collect();
        for x in carried {
            let Some(xmm) = free.pop() else { break };
            slots.push(Slot {
                reg: x.0,
                xmm,
                start: head,
                end,
                carried: true,
                dirty: x.4,
            });
        }
        // Body temporaries: a linear scan over their live intervals.
        let mut temps: Vec<&(u16, usize, usize, u32, bool)> =
            info.iter().filter(|x| !carried_r(x.0)).collect();
        temps.sort_by(|p, q| p.1.cmp(&q.1).then(p.0.cmp(&q.0)));
        let mut active: Vec<(usize, u8, u16)> = Vec::new();
        for x in temps {
            active.retain(|&(e, xmm, _)| {
                if e < x.1 {
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
                && let Op::Move { dst, src } = bytecode[x.1]
                && dst == x.0
                && let Some(i) = active.iter().position(|&(e, _, r)| r == src && e == x.1)
            {
                reuse = Some(active.remove(i).1);
            }
            let xmm = match reuse {
                Some(m) => m,
                None => {
                    let Some(m) = free.pop() else { continue };
                    m
                }
            };
            active.push((x.2, xmm, x.0));
            slots.push(Slot {
                reg: x.0,
                xmm,
                start: x.1,
                end: x.2,
                carried: false,
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
        out.push(Region {
            head,
            end,
            slots,
            inner: None,
            int,
            generic,
            live_head,
            spec: None,
            raw_nan: raw_nan_results(ops, bytecode, head, end, &carried_r),
            reuse,
        });
    }
    out
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
fn plan_reuse(ops: &[GenericOp], bytecode: &[Op], head: usize, end: usize) -> Vec<(usize, u16)> {
    // (object register, key, register holding its value).
    let mut avail: Vec<(u16, &str, u16)> = Vec::new();
    let mut out = Vec::new();
    let (mut d, mut u) = (Vec::new(), Vec::new());
    for pc in head..end.min(ops.len()) {
        let (Some(op), Some(bc)) = (ops.get(pc), bytecode.get(pc)) else {
            break;
        };
        match (op, bc) {
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
            }
            (GenericOp::CallMethod { .. }, _) if super::inline_math(bc).is_some() => {
                def_use(bc, &mut d, &mut u);
                avail.retain(|e| !d.contains(&e.0) && !d.contains(&e.2));
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
            }
            _ => avail.clear(),
        }
    }
    out
}

/// Whether `op` is an SSE arithmetic op whose inline result is computed from
/// two Numbers (`+`, `-`, `*`, `/`): a NaN result is whatever NaN the
/// hardware made, canonicalized before it is stored.
fn raw_nan_producer(op: &GenericOp) -> bool {
    match *op {
        GenericOp::Add { .. } | GenericOp::NumAdd { .. } => true,
        GenericOp::Arith { op, .. } => matches!(
            op,
            crate::nbvm::GA_SUB | crate::nbvm::GA_MUL | crate::nbvm::GA_DIV
        ),
        _ => false,
    }
}

/// Whether `op` only ever looks at a Number operand's value as an `f64`: it
/// treats every NaN alike and never stores the operand's bits anywhere (its
/// own result, or its slow path's, is a fresh value): the arithmetic of
/// [`raw_nan_producer`] (whose own result is canonicalized or raw by the same
/// rule) and the relational comparisons.
fn raw_nan_consumer(op: &GenericOp) -> bool {
    raw_nan_producer(op) || matches!(op, GenericOp::Lt { .. } | GenericOp::Compare { .. })
}

/// The pcs of region `head..=end`'s arithmetic whose NaN result may stay the
/// hardware's NaN in its SSE register: the canonical NaN exists so that a
/// NaN's *bits* are the interpreter's wherever they can be observed (stored
/// into an object, a typed array, compared by identity...), but such a
/// result never reaches anything that looks at its bits. Its register is
/// not loop-carried (dead at the header, the exits and the catch entries,
/// `carried`), and every op of the region that reads it is a
/// [`raw_nan_consumer`] or a `Move` into a register that satisfies the same
/// (a `const` binding's copy) — so a value of it reaches nothing else,
/// whichever way control goes: an op before the header's next execution
/// that reads such a register reads this value or a later write, and after
/// an exit (or a leave at the header) the register is dead. Wherever such a
/// value is written to the register file (a flush before a call) it still
/// reads as a Number: the hardware's NaNs (the default `0xfff8…` or a
/// quieted operand NaN, itself a Number) never have the boxed pattern.
fn raw_nan_results(
    ops: &[GenericOp],
    bytecode: &[Op],
    head: usize,
    end: usize,
    carried: &impl Fn(u16) -> bool,
) -> Vec<usize> {
    let (mut d, mut u) = (Vec::new(), Vec::new());
    let mut out = Vec::new();
    for pc in head..=end {
        if !raw_nan_producer(&ops[pc]) {
            continue;
        }
        let Some(r) = written_reg(&bytecode[pc]) else {
            continue;
        };
        // The registers the value may reach (through moves), each checked.
        let mut reach = alloc::vec![r];
        let mut i = 0;
        let mut ok = true;
        while ok && i < reach.len() && reach.len() <= 8 {
            let x = reach[i];
            i += 1;
            if carried(x) {
                ok = false;
                break;
            }
            for q in head..=end {
                if q == pc && x == r {
                    continue;
                }
                def_use(&bytecode[q], &mut d, &mut u);
                if !u.contains(&x) || raw_nan_consumer(&ops[q]) {
                    continue;
                }
                match bytecode[q] {
                    Op::Move { dst, src }
                        if src == x && matches!(ops[q], GenericOp::Move { .. }) =>
                    {
                        if !reach.contains(&dst) {
                            reach.push(dst);
                        }
                    }
                    _ => {
                        ok = false;
                        break;
                    }
                }
            }
        }
        if ok && i == reach.len() {
            out.push(pc);
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
fn spec_step(op: &Op, pc: usize, reuse: &[(usize, u16)], s: &mut [u8]) {
    if let Op::GetProp { dst, .. } = op
        && let Some(&(_, src)) = reuse.iter().find(|x| x.0 == pc)
        && let Some(&t) = s.get(src as usize)
        && let Some(x) = s.get_mut(*dst as usize)
    {
        *x = t;
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
/// generic ops (and no integer version) whose in-body branches all leave it
/// (the back-edge aside), so the plain copy needs no other in-body entry.
/// A loop-carried register an arithmetic op reads is assumed (and checked at
/// entry) to be a Number when the static types do not prove it, unless the
/// loop itself may leave a non-Number in it.
fn plan_spec(
    r: &Region,
    ops: &[GenericOp],
    bytecode: &[Op],
    types: &Types,
    n_regs: usize,
) -> Option<Spec> {
    let (head, end) = (r.head, r.end);
    if !r.generic || r.int.is_some() || end >= ops.len() || end >= bytecode.len() {
        return None;
    }
    let len = end - head + 1;
    if len.saturating_mul(n_regs) > MAX_STATE {
        return None;
    }
    for (pc, op) in ops.iter().enumerate().take(end + 1).skip(head) {
        if region_call(op, &bytecode[pc]) {
            return None;
        }
        let t = match *op {
            GenericOp::Jump { target } | GenericOp::JumpIfFalse { target, .. } => target,
            GenericOp::PushHandler { .. } => return None,
            _ => continue,
        };
        if pc != end && (head..=end).contains(&t) {
            return None;
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
        for (i, op) in bytecode[head..=end].iter().enumerate() {
            rows.extend_from_slice(&cur);
            spec_step(op, head + i, &r.reuse, &mut cur);
        }
        // `cur` is the back-edge's state: an assumed register the loop writes
        // must come back a Number.
        let before = assumed.len();
        assumed.retain(|&x| !written[x as usize] || cur[x as usize] == T_NUM);
        if assumed.len() == before {
            return Some(Spec {
                types: rows,
                checks: assumed,
            });
        }
    }
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
        let sp = plan_spec(r, &qops, &q.ops, &t, q.n_regs).expect("speculative");
        let add = pc_of(&q, |o| matches!(o, Op::AddValue { .. }));
        let Op::AddValue { a: sum, .. } = q.ops[add] else {
            unreachable!()
        };
        assert!(sp.checks.contains(&sum), "checks {:?}", sp.checks);
        // A branch inside the body: no speculative version.
        let b = proto_of(
            "function h(o){ let s = 0; for (let i = 0; i < 100; i = i + 1) { if (o.x > 1) s = s + o.x; } return s; }",
            "h",
        );
        let (bops, ..) = crate::jit::lower_nbvm_generic(&b).expect("generic");
        let regions = plan_regions(&bops, &b.ops, &[], b.n_regs);
        assert_eq!(regions.len(), 1);
        let t = infer(&b.ops, b.n_regs);
        assert!(plan_spec(&regions[0], &bops, &b.ops, &t, b.n_regs).is_none());
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
        let sp = plan_spec(r, &ops, &p.ops, &t, p.n_regs).expect("speculative");
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
    fn raw_nan_results_feed_only_arithmetic() {
        // `t = a * b` and `u = t - b` only feed arithmetic: raw. `w` is
        // stored into an object, `q` is loop-carried, `m` moves: canonical.
        let p = proto_of(
            "function f(o, a, b){ let q = 0; for (let i = 0; i < 100; i = i + 1) { const t = a * b; const u = t - b; const w = u / a; o.w = w; q = q + u; const m = t * 2; const n = m; o.n = n; } return q; }",
            "f",
        );
        let (ops, ..) = crate::jit::lower_nbvm_generic(&p).expect("generic");
        let regions = plan_regions(&ops, &p.ops, &[], p.n_regs);
        assert_eq!(regions.len(), 1);
        let r = &regions[0];
        let muls: Vec<usize> = (r.head..=r.end)
            .filter(|&pc| matches!(p.ops[pc], Op::Mul { .. }))
            .collect();
        let sub = pc_of(&p, |o| matches!(o, Op::Sub { .. }));
        let div = pc_of(&p, |o| matches!(o, Op::Div { .. }));
        assert_eq!(muls.len(), 2);
        assert!(
            r.raw_nan.contains(&muls[0]),
            "t feeds - and *: {:?}",
            r.raw_nan
        );
        assert!(r.raw_nan.contains(&sub), "u feeds / and +");
        assert!(!r.raw_nan.contains(&div), "w is stored");
        assert!(!r.raw_nan.contains(&muls[1]), "m is moved");
        let add = (r.head..=r.end)
            .find(|&pc| matches!(p.ops[pc], Op::AddValue { .. } | Op::Add { .. }) && pc > sub)
            .expect("q + u");
        assert!(!r.raw_nan.contains(&add), "q is loop-carried");
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
