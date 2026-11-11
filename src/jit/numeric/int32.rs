//! **Int32 speculation** for promoted loop regions (`ROADMAP.md` §2.1).
//!
//! A region whose registers mostly hold small integers (loop counters, `| 0`
//! arithmetic, bit manipulation) gets a second, *integer version* of its body,
//! emitted after the double version. In it, the *int registers* (planned by
//! [`plan`]) live as int32 values in general-purpose registers
//! ([`GPRS`]), in a speculative region's (a loop nest's, whose inner loops
//! keep many values live) also in frame slots ([`MEM_BASE`]), or, for a
//! constant temporary, as an immediate; every other
//! register keeps the location the double version gave it (an SSE register or
//! the register file).
//!
//! # Soundness
//!
//! An int register only ever holds an int32 that is *exactly* its JavaScript
//! Number value, never `-0`:
//!
//! - region entry checks every loop-carried int register by a bit-exact
//!   round trip (`cvttsd2si` / `cvtsi2sd` / compare the bits), which rejects
//!   `-0`, fractions, NaN, out-of-range values and non-Numbers alike, and enters
//!   the double version instead when any check fails; a `LoadGlobal` into an int
//!   register is checked the same way;
//! - `+`, `-`, `*`, `++`/`--` and unary `-` check for signed overflow, `*` for a
//!   zero product with a negative factor (`-0`), unary `-` for a zero operand,
//!   and `%` for a zero divisor and a zero remainder of a negative dividend
//!   (`-0`); the bitwise ops always produce an int32 (`>>>` never feeds an int
//!   register: its result is a uint32).
//!
//! A failed check **bails**: before op `p` writes anything, the live int
//! registers are converted (exactly) into the locations the double version
//! expects at `p`, and control jumps to the double version's code for `p`,
//! which then finishes the region (it stays double until the loop is entered
//! again). The double version is the plain emitter, so everything the integer
//! version does not do itself — mixed int/double operands, non-Number operands,
//! every slow path — is its code: the integer version runs the same emitters
//! for such ops, with each int operand first converted into its double
//! location (its *shadow*) and every slow path turned into a bail.
//!
//! The int registers are caller-saved GPRs, so around the only calls the
//! integer version makes (the back-edge poll and a global binding's
//! validation) they are converted into the register file and reloaded after;
//! the interpreter, the helpers and the collector see exact Numbers.

use super::{CC_AE, CC_E, CC_NE, NumGen, RAX, RCX, RDX, Region, StubKind, def_use, disp};
use super::{GenericOp, Label, X64Assembler};
use crate::nanbox::NanBox;
use crate::nbvm::Op;
use alloc::vec::Vec;

mod elems;
mod views;

/// The general-purpose registers int registers live in: caller-saved, and
/// never touched by the emitters' fast paths (`rax`/`rcx`/`rdx`/`r11` are
/// their scratch, `rbx`/`r15` are pinned).
const GPRS: [u8; 5] = [6, 7, 8, 9, 10];

/// How many times in a row an integer version's entry may find a
/// loop-carried value that is not an int32 and have it tried again after an
/// iteration elsewhere (`NumGen::entry_retry`).
const ENTRY_RETRIES: i32 = 8;

/// How many entry failures in a row (`ENTRY_RETRIES`) before the next one
/// asks for the failing loop-carried register to be demoted
/// (`NumGen::entry_fail`). One retry is enough for a hash's or a CRC's
/// uint32 start value, whose second entry, an iteration later, passes.
const ENTRY_DEMOTE: i32 = 1;

/// How many bits of a region's overflow-bail mask name one site each (the
/// last stands for every site from it on): an `or` with a sign-extended
/// 32-bit immediate sets them.
pub(crate) const DEMOTE_BITS: usize = 31;

/// How many int GPRs there are (their save slots, `NumGen::int_save`).
pub(super) const N_GPRS: usize = GPRS.len();

/// An [`ISlot::gpr`] from this value on names no GPR but frame slot
/// `gpr - MEM_BASE` past the int GPRs' save slots (`NumGen::home_disp`),
/// which holds the int register's int32 (a speculative region's integer
/// version, `spec_alloc`).
const MEM_BASE: u8 = 0x40;

/// How many such frame slots one integer version may use.
pub(super) const MAX_MEM: usize = 16;

/// x86 condition codes beyond the parent's.
const CC_O: u8 = 0x0;
const CC_S: u8 = 0x8;
const CC_L: u8 = 0xc;
const CC_GE: u8 = 0xd;
const CC_LE: u8 = 0xe;
const CC_G: u8 = 0xf;

const R11: u8 = 11;
const RSI: u8 = 6;
const RDI: u8 = 7;

/// One int register with a GPR: `reg` lives in `gpr` at pcs `start..=end`.
pub(super) struct ISlot {
    reg: u16,
    gpr: u8,
    start: usize,
    end: usize,
    carried: bool,
    dirty: bool,
}

/// The integer version of a region.
pub(super) struct IntPlan {
    /// Every int register (sorted), constants included.
    ints: Vec<u16>,
    /// Int registers that are constant temporaries: `(reg, value, start, end)`.
    consts: Vec<(u16, i32, usize, usize)>,
    slots: Vec<ISlot>,
    /// The pcs of the `+`/`-`/`*` ops that wrap instead of checking overflow.
    wraps: Vec<usize>,
    /// The pcs of the element reads ([`elems`]) and `>>>` ops whose int
    /// register holds the value's `ToInt32` bits rather than the value (a
    /// Uint32 element or a `>>>` result past `i32::MAX`): like a wrapped
    /// op's, read only by bitwise ops before anything can bail.
    wrap_bits: Vec<usize>,
    /// The pcs of the `>>>` ops by a constant count that is not a multiple
    /// of 32 (`i >>> 5`): their result is below 2^31, an exact int32 held
    /// like any other (an element index, say).
    exact_ushr: Vec<usize>,
    /// The promoted global bindings.
    pub(super) globals: Vec<PGlobal>,
    /// The integer version's entry (region entry jumps here first).
    pub(super) entry: Option<Label>,
    /// The double version's entry (its header loads).
    pub(super) dbl_entry: Option<Label>,
    /// Bound past the integer entry checks: in-version branches to the header.
    inner: Option<Label>,
    /// How many frame slots its int registers without a GPR live in
    /// ([`MEM_BASE`], `spec_alloc`).
    pub(super) n_mem: usize,
}

/// A global binding the integer version keeps in a register: `vreg` (a
/// pseudo-register past the register file) lives in an int GPR (an `ISlot`) or
/// in `xmm`. Region entry validates the binding's address (the interpreter's
/// fast-path conditions, `jit_helper_global_ref`, for both its load and its
/// store op) into frame slot `slot` and loads the value; stores to it are only
/// written back to the binding at exits, bails and before the back-edge poll
/// (the only call), after which the address is validated again. Nothing else
/// can observe the binding meanwhile: only calls run other code.
pub(super) struct PGlobal {
    vreg: u16,
    load_pc: Option<usize>,
    store_pc: Option<usize>,
    xmm: Option<u8>,
    dirty: bool,
    pub(super) slot: u32,
}

impl IntPlan {
    /// Whether `r` is one of its int registers.
    pub(super) fn has_int(&self, r: u16) -> bool {
        self.ints.binary_search(&r).is_ok()
    }

    /// Its int registers (`KATAAN_JIT_SPECALLOC=log`).
    pub(super) fn regs(&self) -> Vec<u16> {
        self.ints.clone()
    }

    /// Where the integer version's in-region branches to the header land.
    pub(super) fn inner(&self) -> Option<Label> {
        self.inner
    }
}

/// Where an int register lives.
#[derive(Clone, Copy)]
enum IK {
    Gpr(u8),
    Imm(i32),
    /// A frame slot (`[rbp + disp]`) holding the int32 (`MEM_BASE`).
    Mem(i32),
}

/// The exact int32 a constant is, if it is one (and not `-0`).
fn const_i32(v: NanBox) -> Option<i32> {
    let n = v.as_number()?;
    let i = n as i32;
    (f64::from(i).to_bits() == n.to_bits()).then_some(i)
}

/// Plans the integer version of region `head..=end` (`info`: per register named
/// in it, `(reg, first, last, mentions, written)`; `carried`: live into the
/// header or out of the region). `None` when no integer arithmetic would run.
///
/// With `elems` (a region whose only generic ops are element and `.length`
/// reads, [`elems`]), those reads' results may be int registers too (each
/// checked when read), as may `>>>` results that only bitwise ops read
/// ([`IntPlan::wrap_bits`]); the plan is then only made when every such
/// read's result and index are int registers (else the region keeps its
/// double versions).
///
/// `demote`: registers an earlier body's integer version of the region
/// found holding no int32 (an overflow bail's result, recorded in the
/// region's overflow-bail mask, `NumGen::ibail_ovf`): no int registers,
/// nor is any value moved into or computed from them alone, so only those
/// values are doubles and the rest of the loop stays integer.
#[allow(clippy::too_many_arguments)]
pub(super) fn plan(
    bytecode: &[Op],
    head: usize,
    end: usize,
    info: &[(u16, usize, usize, u32, bool)],
    carried: &dyn Fn(u16) -> bool,
    n_regs: usize,
    dbl_xmm: &[u8],
    elems: bool,
    spec: Option<&[(usize, usize, usize)]>,
    live: &dyn Fn(usize, u16) -> bool,
    gcarry: &[(alloc::string::String, u16, usize)],
    extra: &[u16],
    demote: &[u16],
    loops: &[(usize, usize)],
) -> Option<IntPlan> {
    // A speculative region's read of a binding its register carries
    // (`plan_carried_globals`) is a copy of that register: an int register
    // when that one is (`NumGen::carried_int_global`).
    let carrier = |name: &str| gcarry.iter().find(|g| g.0 == name).map(|g| g.1);
    // A speculative region's integer version (`spec`, with its guard
    // diamonds as `(test, fallback, join)`, whose fallback arms neither fast
    // version has): see `NumGen::emit_spec_int_op`.
    let (spec, cold) = (spec.is_some(), spec.unwrap_or(&[]));
    let arm_of = |pc: usize| cold.iter().find(|&&(_, f, e)| (f..e).contains(&pc));
    // There, a temporary's int intervals are its **live ranges** in the
    // fast versions: the runs of pcs where it is live or mentioned, or a
    // branch carries its value across (so a value only flows within one
    // run, which then holds one GPR), the fallback arms left out — a
    // mention there counts at its guard (an arm is only reached through
    // its test, so the register is live there, where the plain copy may
    // be entered). Two inlined calls' bodies reusing one register window
    // then hold their registers apart, as does a value dead in between.
    let spec_info: Vec<(u16, usize, usize, u32, bool)>;
    let info = if spec {
        let mut v = Vec::new();
        for &(r, first, last, m, w) in info {
            if carried(r) {
                v.push((r, first, last, m, w));
                continue;
            }
            // (One mentioned only in fallback arms is no int register.)
            if (head..=end)
                .all(|pc| arm_of(pc).is_some() || !super::op_regs(&bytecode[pc]).contains(&r))
            {
                continue;
            }
            let n = end - head + 1;
            let mut occ = alloc::vec![false; n];
            for pc in head..=end {
                if !super::op_regs(&bytecode[pc]).contains(&r) {
                    if arm_of(pc).is_none() && live(pc, r) {
                        occ[pc - head] = true;
                    }
                    continue;
                }
                match arm_of(pc) {
                    Some(&(t, _, _)) => occ[t - 1 - head] = true,
                    None => occ[pc - head] = true,
                }
            }
            for pc in head..=end {
                if arm_of(pc).is_some() {
                    continue;
                }
                let t = match bytecode[pc] {
                    Op::Jump { target } | Op::JumpIfFalse { target, .. } => target,
                    _ => continue,
                };
                if t > head && t <= end && arm_of(t).is_none() && live(t, r) {
                    for o in &mut occ[pc.min(t) - head..=pc.max(t) - head] {
                        *o = true;
                    }
                }
            }
            let mut i = 0;
            while i < n {
                if !occ[i] {
                    i += 1;
                    continue;
                }
                let s0 = i;
                while i + 1 < n && occ[i + 1] {
                    i += 1;
                }
                v.push((r, head + s0, head + i, m, w));
                i += 1;
            }
        }
        spec_info = v;
        &spec_info[..]
    } else {
        info
    };
    use crate::nbvm::{VB_LOOSE_EQ, VB_LOOSE_NEQ, VB_SHR, VB_USHR};
    let body = &bytecode[head..=end];
    // An element read's result is speculated an int32 only when bitwise
    // ops read it (whose operands the double versions convert to int32 at
    // every op: what the integer version saves), or another read as its
    // index, or a move into a register read only that way. (A Float64
    // element summed up would only ever bail; a loop that only compares
    // elements keeps its speculative version, whose views are hoisted.)
    // Whether some read of array register `o` feeds arithmetic, a
    // comparison or a bitwise op (its elements are numbers, not objects
    // being moved around, as a heap's swaps are).
    let numeric_reads = |o: u16| {
        let (mut d, mut u) = (Vec::new(), Vec::new());
        body.iter().any(|op| match op {
            Op::GetKey { obj, dst, .. } if *obj == o && *dst != o => body.iter().any(|q| {
                def_use(q, &mut d, &mut u);
                u.contains(dst)
                    && match q {
                        Op::Add { .. }
                        | Op::AddValue { .. }
                        | Op::Sub { .. }
                        | Op::Mul { .. }
                        | Op::Lt { .. } => true,
                        Op::Compare { op, .. } => *op <= 3,
                        Op::ValueBin { op, .. } => (1..=VB_USHR).contains(op),
                        _ => false,
                    }
            }),
            _ => false,
        })
    };
    let elem_reader_ok = |r: u16, obj: u16, inn: &dyn Fn(u16) -> bool, arith: bool| {
        let mut work = alloc::vec![r];
        let mut seen: Vec<u16> = Vec::new();
        let (mut d, mut u) = (Vec::new(), Vec::new());
        while let Some(x) = work.pop() {
            if seen.contains(&x) || seen.len() > 8 {
                return false;
            }
            seen.push(x);
            for (i, op) in body.iter().enumerate() {
                if arm_of(head + i).is_some() {
                    continue;
                }
                def_use(op, &mut d, &mut u);
                if !u.contains(&x) {
                    continue;
                }
                match op {
                    Op::ValueBin { op, .. } if (1..=VB_USHR).contains(op) => {}
                    Op::GetKey { key, obj, .. } if *key == x && *obj != x => {}
                    // An element store's value (`win[j + 1] = win[j]`, a
                    // sort's moves): a hoisted view's store takes an int
                    // register as is (`NumGen::ihoisted_view_set`). Only
                    // when the array read, or the one written, is
                    // otherwise read as numbers.
                    Op::SetKey { obj: o, key, src } | Op::SetKeyStrict { obj: o, key, src }
                        if arith
                            && *src == x
                            && *key != x
                            && *o != x
                            && [obj, *o].iter().any(|&q| {
                                // (Or the register it copies: `win[n++] = …`
                                // writes through a copy of `win`.)
                                numeric_reads(q)
                                    || body.iter().any(|m| {
                                        matches!(m, Op::Move { dst, src }
                                            if *dst == q && numeric_reads(*src))
                                    })
                            }) => {}
                    // (`ToNumeric` of an int register is a copy: a counter
                    // starting at an element, `for (k = p[i]; …; k++)`.)
                    Op::Move { dst, .. } | Op::ToNumeric { dst, .. } => work.push(*dst),
                    Op::Step { dst, src, .. } if arith && inn(*dst) && inn(*src) => {}
                    // In a speculative region (whose bails are counted, so a
                    // loop over doubles soon stops trying): checked int32
                    // arithmetic and comparisons over int registers, as when
                    // an int32 array is summed up or its elements compared.
                    Op::Add { dst, a, b }
                    | Op::AddValue { dst, a, b }
                    | Op::Sub { dst, a, b }
                    | Op::Mul { dst, a, b }
                        if arith && inn(*dst) && inn(*a) && inn(*b) => {}
                    // (Not `===`: elements compared by identity are most
                    // likely objects, and two such reads would each look
                    // like an int register if the other is.)
                    Op::Lt { a, b, .. } if arith && inn(*a) && inn(*b) => {}
                    Op::Compare { op, a, b, .. } if arith && *op <= 3 && inn(*a) && inn(*b) => {}
                    // (`Math.imul`, which takes its `ToInt32`.)
                    Op::CallMethod { .. } | Op::CallWithMethod { .. }
                        if spec && super::int_math_op(op).is_some() => {}
                    _ => return false,
                }
            }
        }
        true
    };
    let elem_def = |op: &Op| matches!(op, Op::GetKey { .. } | Op::ArrayLen { .. });
    // Promoted global bindings: each named global the loop reads or writes is
    // a pseudo-register `n_regs + k`, loop-carried, written back at exits and
    // before calls (see `PGlobal`).
    let mut gnames: Vec<&str> = Vec::new();
    let mut promote = true;
    for op in body {
        match op {
            Op::LoadGlobal { dst, name, .. } => {
                promote &= (*dst as usize) < n_regs && !name.starts_with('\u{1}');
                if !gnames.contains(&name.as_str()) {
                    gnames.push(name);
                }
            }
            Op::StoreGlobal {
                src,
                name,
                strict,
                resolved,
                ..
            } => {
                // (A global `var`'s write also goes to the global object,
                // which a write-back at an exit cannot do for a heap handle:
                // `global_mirror`.)
                promote &= (*src as usize) < n_regs
                    && !(*strict && resolved.is_some())
                    && !name.starts_with('\u{1}')
                    && super::super::global_mirror::mirror_cell(op).is_none();
                if !gnames.contains(&name.as_str()) {
                    gnames.push(name);
                }
            }
            _ => {}
        }
    }
    let free_xmm: Vec<u8> = (super::FIRST_XMM..super::FIRST_XMM + super::MAX_PROMOTED as u8)
        .filter(|x| !dbl_xmm.contains(x))
        .collect();
    if spec
        || !promote
        || gnames.len() > 4
        || gnames.len() > free_xmm.len()
        || n_regs + gnames.len() > 0xffff
    {
        gnames.clear();
    }
    let vreg_of = |name: &str| {
        gnames
            .iter()
            .position(|g| *g == name)
            .map(|k| (n_regs + k) as u16)
    };
    // The pseudo-registers join the register table: carried, written when the
    // loop stores the global.
    let mut info_v: Vec<(u16, usize, usize, u32, bool)> = info.to_vec();
    for (k, g) in gnames.iter().enumerate() {
        let mentions = body
            .iter()
            .filter(|op| matches!(op, Op::LoadGlobal { name, .. } | Op::StoreGlobal { name, .. } if name == g))
            .count() as u32;
        let written = body
            .iter()
            .any(|op| matches!(op, Op::StoreGlobal { name, .. } if name == g));
        info_v.push(((n_regs + k) as u16, head, end, mentions, written));
    }
    let info = &info_v[..];
    let carried = |r: u16| (r as usize) >= n_regs || carried(r);
    let mut defs: Vec<(u16, Vec<usize>)> = Vec::new();
    for x in info {
        if !defs.iter().any(|e| e.0 == x.0) {
            defs.push((x.0, Vec::new()));
        }
    }
    for (i, op) in body.iter().enumerate() {
        if spec {
            // (Every register any op of the fast versions writes: the
            // generic emitter's ops too, whose results are never int
            // registers.)
            if arm_of(head + i).is_some() {
                continue;
            }
            let (mut d, mut u) = (Vec::new(), Vec::new());
            def_use(op, &mut d, &mut u);
            d.sort_unstable();
            d.dedup();
            for r in d {
                if let Some(e) = defs.iter_mut().find(|e| e.0 == r) {
                    e.1.push(head + i);
                }
            }
            continue;
        }
        let d = match op {
            Op::StoreGlobal { name, .. } => vreg_of(name),
            Op::GetKey { dst, .. } | Op::ArrayLen { dst, .. } if elems => Some(*dst),
            _ => super::written_reg(op),
        };
        if let Some(d) = d
            && let Some(e) = defs.iter_mut().find(|e| e.0 == d)
        {
            e.1.push(head + i);
        }
    }
    let defs_of = |r: u16| -> &[usize] {
        defs.iter()
            .find(|e| e.0 == r)
            .map_or(&[][..], |e| e.1.as_slice())
    };
    // A hint (the entry checks are what make speculation sound): a carried
    // register initialized before the loop with a non-int32 number constant,
    // directly or through a move, will not pass them.
    let outside = |pc: usize| pc < head || pc > end;
    // (A uint32 constant, `h = 0x811c9dc5`, becomes an int32 once the loop
    // writes it, and an entry that finds it tries again after an iteration:
    // `NumGen::entry_retry`.)
    let dbl_const = |r: u16| {
        bytecode.iter().any(|op| {
            matches!(op, Op::LoadConst { dst, value } if *dst == r
                && value.as_number().is_some_and(|n| {
                    const_i32(*value).is_none() && !(n > 0.0 && n < 4_294_967_296.0 && f64::from(n as u32) == n)
                }))
        })
    };
    // (Likewise one set before the loop to an object, an array, a
    // function or a string: a loop-invariant receiver compared by `===`
    // would otherwise look like an int register, each of its readers being
    // one if it is.)
    let mk = |op: &Op, r: u16| match op {
        Op::NewObjectLit { dst, .. }
        | Op::NewObject { dst, .. }
        | Op::NewArray { dst, .. }
        | Op::NewArrayFrom { dst, .. }
        | Op::NewArrayCtor { dst, .. }
        | Op::NewString { dst, .. }
        | Op::NewRegExp { dst, .. }
        | Op::NewCollection { dst, .. }
        | Op::LoadFunc { dst, .. }
        | Op::MakeClosure { dst, .. }
        | Op::Construct { dst, .. } => *dst == r,
        _ => false,
    };
    let makes_nonnum = |r: u16| bytecode.iter().any(|op| mk(op, r));
    let starts_double = |r: u16| {
        bytecode.iter().enumerate().any(|(pc, op)| {
            outside(pc)
                && match op {
                    Op::LoadConst { dst, .. } => *dst == r && dbl_const(r),
                    Op::Move { dst, src } => {
                        *dst == r && (dbl_const(*src) || (spec && makes_nonnum(*src)))
                    }
                    // (A boolean: a compiled `for-in`'s `o === undefined`
                    // test, made before its loop and branched on in it,
                    // would fail every entry check, then be demoted.)
                    Op::StrictEq { dst, .. }
                    | Op::Lt { dst, .. }
                    | Op::Compare { dst, .. }
                    | Op::Not { dst, .. } => *dst == r,
                    Op::ValueBin { dst, op, .. } if *op == VB_LOOSE_EQ || *op == VB_LOOSE_NEQ => {
                        *dst == r
                    }
                    _ => spec && mk(op, r),
                }
        })
    };
    let mut cset: Vec<u16> = info
        .iter()
        .map(|x| x.0)
        .filter(|&r| !(carried(r) && starts_double(r)) && !demote.contains(&r))
        .collect();
    cset.sort_unstable();
    cset.dedup();
    let mut du_d = Vec::new();
    let mut du_u = Vec::new();
    loop {
        // Fixpoint: drop every register some def of which is not an exact int32
        // given the current set, and every speculative register no int op uses.
        loop {
            let c = cset.clone();
            let inn = |r: u16| c.binary_search(&r).is_ok();
            // (A `Math.imul` argument may also be a constant: its bits.)
            let imul_arg = |r: u16, pc: usize| {
                inn(r) || super::const_int32_bits(&|q| bytecode.get(q), head, end, r, pc).is_some()
            };
            let int_def = |pc: usize, op: &Op| match op {
                Op::LoadConst { value, .. } => const_i32(*value).is_some(),
                Op::Move { src, .. } => inn(*src),
                Op::Add { a, b, .. }
                | Op::AddValue { a, b, .. }
                | Op::Sub { a, b, .. }
                | Op::Mul { a, b, .. }
                | Op::Mod { a, b, .. } => inn(*a) && inn(*b),
                Op::ValueBin { op, .. } => {
                    (1..=VB_SHR).contains(op) || ((elems || spec) && *op == VB_USHR)
                }
                Op::ToNumeric { src, .. } | Op::Step { src, .. } | Op::Neg { a: src, .. } => {
                    inn(*src)
                }
                // (In a speculative region's, checked after the generic
                // emitter's read: `NumGen::int_post`.)
                Op::GetKey { dst, key, obj } => {
                    // (Arithmetic readers only for a read by an int index,
                    // whose read of a hoisted array is inline: a keyed
                    // read's generic code would cost the GPRs' saves.)
                    (elems || spec) && elem_reader_ok(*dst, *obj, &inn, spec && inn(*key))
                }
                Op::ArrayLen { .. } => elems || spec,
                // A promoted global reads / writes its pseudo-register.
                Op::LoadGlobal { name, .. } if spec => carrier(name).is_some_and(inn),
                Op::LoadGlobal { name, .. } => vreg_of(name).is_none_or(inn),
                Op::StoreGlobal { src, .. } => inn(*src),
                Op::BitNot { .. } => true,
                // `Math.imul` over int registers (`NumGen::emit_int_math`).
                Op::CallMethod { .. } | Op::CallWithMethod { .. } if spec => {
                    super::int_math_op(op).is_some_and(|(x, y)| imul_arg(x, pc) && imul_arg(y, pc))
                }
                _ => false,
            };
            let useful = |r: u16, du_d: &mut Vec<u16>, du_u: &mut Vec<u16>| {
                if (r as usize) >= n_regs {
                    // A promoted global: some load of it feeds an int register.
                    return body.iter().any(|op| {
                        matches!(op, Op::LoadGlobal { dst, name, .. } if vreg_of(name) == Some(r) && inn(*dst))
                    });
                }
                // A carried binding's register: the read it carries is one.
                if gcarry.iter().any(|g| {
                    g.1 == r && matches!(bytecode.get(g.2), Some(Op::LoadGlobal { dst, .. }) if inn(*dst))
                }) {
                    return true;
                }
                body.iter().any(|op| {
                    def_use(op, du_d, du_u);
                    if !du_u.contains(&r) {
                        return false;
                    }
                    match op {
                        Op::JumpIfFalse { .. } => true,
                        Op::ValueBin { op, a, b, .. }
                            if *op == VB_LOOSE_EQ || *op == VB_LOOSE_NEQ =>
                        {
                            inn(*a) && inn(*b)
                        }
                        Op::ValueBin { op, .. } => (1..=VB_USHR).contains(op),
                        Op::Lt { a, b, .. } | Op::StrictEq { a, b, .. } => inn(*a) && inn(*b),
                        Op::Compare { op, a, b, .. } => *op <= 3 && inn(*a) && inn(*b),
                        // An element read's index.
                        Op::GetKey { key, obj, .. } => (elems || spec) && *key == r && *obj != r,
                        // An element store's index (`NumGen::ihoisted_arr_set`).
                        // Or its value, by an int index: an element copied
                        // between views stays an int32 (no double round trip).
                        Op::SetKey { key, obj, src } | Op::SetKeyStrict { key, obj, src } => {
                            spec && *obj != r && (*key == r || (*src == r && inn(*key)))
                        }
                        Op::CallMethod { dst, .. } | Op::CallWithMethod { dst, .. }
                            if spec && super::int_math_op(op).is_some() =>
                        {
                            inn(*dst)
                        }
                        _ => super::written_reg(op).is_some_and(inn),
                    }
                })
            };
            let next: Vec<u16> = c
                .iter()
                .copied()
                .filter(|&r| {
                    let ds = defs_of(r);
                    if !ds.iter().all(|&pc| int_def(pc, &bytecode[pc])) {
                        return false;
                    }
                    let speculative = carried(r)
                        || extra.contains(&r)
                        || ds.iter().any(|&pc| {
                            matches!(bytecode[pc], Op::LoadGlobal { .. } | Op::LoadConst { .. })
                                || elem_def(&bytecode[pc])
                        });
                    !speculative || useful(r, &mut du_d, &mut du_u)
                })
                .collect();
            if super::specalloc_log() && next != cset {
                std::eprintln!(
                    "[int32] {head}..={end}: dropped {:?}",
                    cset.iter()
                        .filter(|r| !next.contains(r))
                        .collect::<Vec<_>>()
                );
            }
            if next == cset {
                break;
            }
            cset = next;
        }
        if cset.is_empty() {
            return None;
        }
        // Constant temporaries need no register.
        let mut consts: Vec<(u16, i32, usize, usize)> = Vec::new();
        for x in info {
            let ds = defs_of(x.0);
            if cset.binary_search(&x.0).is_ok()
                && !carried(x.0)
                && ds.len() == 1
                && let Op::LoadConst { value, .. } = bytecode[ds[0]]
                && let Some(v) = const_i32(value)
            {
                consts.push((x.0, v, x.1, x.2));
            }
        }
        if spec {
            // So is a live range (`spec_info`) whose one def is a move of a
            // constant (an inlined call's constant argument), to a fixpoint.
            loop {
                let before = consts.len();
                for x in info {
                    if !cset.binary_search(&x.0).is_ok()
                        || carried(x.0)
                        || consts.iter().any(|c| c.0 == x.0 && c.2 == x.1)
                    {
                        continue;
                    }
                    let ds: Vec<usize> = defs_of(x.0)
                        .iter()
                        .copied()
                        .filter(|&pc| x.1 <= pc && pc <= x.2)
                        .collect();
                    if let [d] = ds[..]
                        && let Op::Move { src, .. } = bytecode[d]
                        && let Some(c) = consts.iter().find(|c| c.0 == src && c.2 <= d && d <= c.3)
                    {
                        consts.push((x.0, c.1, x.1, x.2));
                    }
                }
                if consts.len() == before {
                    break;
                }
            }
        }
        // Allocate GPRs: carried registers pinned (most used first), then the
        // temporaries by a linear scan over their live intervals.
        let needs = |x: &&(u16, usize, usize, u32, bool)| {
            cset.binary_search(&x.0).is_ok() && !consts.iter().any(|c| c.0 == x.0 && c.2 == x.1)
        };
        // (A speculative region's integer version, or a nest's: by weight,
        // and the values that lose a GPR live in frame slots, `spec_alloc`.)
        let spec_slots = if spec || !loops.is_empty() {
            let in_arm = |pc: usize| arm_of(pc).is_some();
            let need = |x: &(u16, usize, usize, u32, bool)| needs(&x);
            match spec_alloc(bytecode, (head, end), info, &need, &carried, &in_arm, loops).and_then(
                |v| {
                    // (A promoted global lives in a GPR, `PGlobal`.)
                    let homeless: Vec<u16> =
                        v.0.iter()
                            .filter(|s| s.reg as usize >= n_regs && s.gpr >= MEM_BASE)
                            .map(|s| s.reg)
                            .collect();
                    if homeless.is_empty() {
                        Ok(v)
                    } else {
                        Err(homeless)
                    }
                },
            ) {
                Ok(v) => Some(v),
                Err(failed) => {
                    if super::specalloc_log() {
                        std::eprintln!("[int32] {head}..={end}: no home for {failed:?}");
                    }
                    cset.retain(|r| !failed.contains(r));
                    continue;
                }
            }
        } else {
            None
        };
        let (slots, n_mem) = match spec_slots {
            Some(v) => v,
            None => {
                let mut carried_v: Vec<&(u16, usize, usize, u32, bool)> =
                    info.iter().filter(needs).filter(|x| carried(x.0)).collect();
                carried_v.sort_by(|p, q| q.3.cmp(&p.3).then(p.0.cmp(&q.0)));
                let mut free: Vec<u8> = GPRS.iter().rev().copied().collect();
                let mut slots = Vec::new();
                let mut failed = Vec::new();
                for x in carried_v {
                    match free.pop() {
                        Some(gpr) => slots.push(ISlot {
                            reg: x.0,
                            gpr,
                            start: head,
                            end,
                            carried: true,
                            dirty: x.4,
                        }),
                        None => failed.push(x.0),
                    }
                }
                let mut temps: Vec<&(u16, usize, usize, u32, bool)> = info
                    .iter()
                    .filter(needs)
                    .filter(|x| !carried(x.0))
                    .collect();
                temps.sort_by(|p, q| p.1.cmp(&q.1).then(p.0.cmp(&q.0)));
                let mut active: Vec<(usize, u8)> = Vec::new();
                for x in temps {
                    // A temporary whose last read is the op that defines `x` hands its
                    // GPR on: every emitter reads its operands before writing its
                    // result (computing in a scratch register first when they share).
                    active.retain(|&(e, g)| {
                        if e <= x.1 {
                            free.push(g);
                            false
                        } else {
                            true
                        }
                    });
                    match free.pop() {
                        Some(gpr) => {
                            active.push((x.2, gpr));
                            slots.push(ISlot {
                                reg: x.0,
                                gpr,
                                start: x.1,
                                end: x.2,
                                carried: false,
                                dirty: x.4,
                            });
                        }
                        None => failed.push(x.0),
                    }
                }
                if !failed.is_empty() {
                    if super::specalloc_log() {
                        std::eprintln!("[int32] {head}..={end}: no GPR for {failed:?}");
                    }
                    cset.retain(|r| !failed.contains(r));
                    continue;
                }
                (slots, 0)
            }
        };
        // Worth it only if some integer arithmetic or comparison runs.
        let inn = |r: u16| cset.binary_search(&r).is_ok();
        let benefit = body.iter().any(|op| match op {
            Op::Add { dst, .. }
            | Op::AddValue { dst, .. }
            | Op::Sub { dst, .. }
            | Op::Mul { dst, .. }
            | Op::Mod { dst, .. }
            | Op::Step { dst, .. }
            | Op::ValueBin { dst, .. } => inn(*dst),
            Op::Lt { a, b, .. } | Op::Compare { a, b, .. } => inn(*a) && inn(*b),
            Op::CallMethod { dst, .. } | Op::CallWithMethod { dst, .. } => spec && inn(*dst),
            _ => false,
        }) || !gnames.is_empty();
        if !benefit {
            return None;
        }
        // A nest's integer version runs its double and generic ops a little
        // worse than its double version (no in-place SSE results, no run
        // proofs, int GPRs saved around generic code): only worth it when
        // its int work, weighted 8x per loop around it, is at least that.
        if spec && !loops.is_empty() {
            let (mut wi, mut wd) = (0u64, 0u64);
            for (i, op) in body.iter().enumerate() {
                let pc = head + i;
                if arm_of(pc).is_some() {
                    continue;
                }
                let d = loops.iter().filter(|&&(h, e)| h <= pc && pc <= e).count();
                let w = 8u64.pow(d.min(5) as u32);
                let int = match op {
                    Op::Add { dst, .. }
                    | Op::AddValue { dst, .. }
                    | Op::Sub { dst, .. }
                    | Op::Mul { dst, .. }
                    | Op::Mod { dst, .. }
                    | Op::Div { dst, .. }
                    | Op::Step { dst, .. }
                    | Op::Neg { dst, .. }
                    | Op::ValueBin { dst, .. }
                    | Op::BitNot { dst, .. } => Some(inn(*dst)),
                    Op::Lt { a, b, .. } | Op::Compare { a, b, .. } => Some(inn(*a) && inn(*b)),
                    Op::GetKey { key, .. }
                    | Op::SetKey { key, .. }
                    | Op::SetKeyStrict { key, .. } => Some(inn(*key)),
                    Op::Move { .. }
                    | Op::LoadConst { .. }
                    | Op::Jump { .. }
                    | Op::JumpIfFalse { .. }
                    | Op::ToNumeric { .. }
                    | Op::Not { .. }
                    | Op::StrictEq { .. }
                    | Op::CheckTdz { .. }
                    | Op::LoadGlobal { .. }
                    | Op::ArrayLen { .. } => None,
                    _ => Some(false),
                };
                match int {
                    Some(true) => wi += w,
                    Some(false) => wd += w,
                    None => {}
                }
            }
            if wi < wd {
                if super::specalloc_log() {
                    std::eprintln!("[int32] {head}..={end}: nest int work {wi} < other {wd}");
                }
                return None;
            }
        }
        let mut xmms = free_xmm.iter().copied();
        let globals: Vec<PGlobal> = gnames
            .iter()
            .enumerate()
            .map(|(k, g)| {
                let vreg = (n_regs + k) as u16;
                let pc_of = |store: bool| {
                    body.iter()
                        .position(|op| match op {
                            Op::LoadGlobal { name, .. } => !store && name == g,
                            Op::StoreGlobal { name, .. } => store && name == g,
                            _ => false,
                        })
                        .map(|i| head + i)
                };
                PGlobal {
                    vreg,
                    load_pc: pc_of(false),
                    store_pc: pc_of(true),
                    xmm: if inn(vreg) { None } else { xmms.next() },
                    dirty: pc_of(true).is_some(),
                    slot: u32::MAX,
                }
            })
            .collect();
        // Wrapping `+`/`-`/`*`: an expression tree of these over int32s whose
        // every intermediate is below 2^53 in magnitude is computed exactly by
        // the doubles JavaScript uses, so when it only feeds bitwise ops (which
        // take ToInt32 of it: the low 32 bits) the 32-bit wrapped computation
        // gives the same bits (they are a ring homomorphism mod 2^32). Each
        // wrapped op writes a temporary defined once and read only by all-int
        // bitwise ops or other wrapped ops, with nothing but constants, those,
        // in between (nothing else can bail or flush while a wrapped value is
        // live: it is not the true Number).
        // (A register's constant when every live range of it is that one.)
        let cval = |r: u16| {
            let c = consts.iter().find(|c| c.0 == r)?.1;
            info.iter()
                .filter(|x| x.0 == r)
                .all(|x| consts.iter().any(|k| k.0 == r && k.2 == x.1 && k.1 == c))
                .then_some(c)
        };
        // (In `elems` mode an element read's or a `>>>`'s value may be held
        // as its `ToInt32` bits the same way: `IntPlan::wrap_bits`.)
        let bits_op = |op: &Op| {
            (elems || spec)
                && match op {
                    Op::GetKey { dst, .. } => Some(*dst),
                    Op::ValueBin { op, dst, .. } if *op == VB_USHR => Some(*dst),
                    _ => None,
                }
                .is_some_and(|d| inn(d) && !carried(d) && defs_of(d).len() == 1)
        };
        let mut wraps: Vec<usize> = body
            .iter()
            .enumerate()
            .filter_map(|(i, op)| {
                if bits_op(op) {
                    return Some(head + i);
                }
                let (Op::Add { dst, a, b }
                | Op::AddValue { dst, a, b }
                | Op::Sub { dst, a, b }
                | Op::Mul { dst, a, b }) = op
                else {
                    return None;
                };
                (inn(*dst) && inn(*a) && inn(*b) && !carried(*dst) && defs_of(*dst).len() == 1)
                    .then_some(head + i)
            })
            .collect();
        loop {
            // log2 magnitude bounds of the wrapped temporaries, in pc order.
            let mut bound: Vec<(u16, u32)> = Vec::new();
            let bnd = |r: u16, bound: &[(u16, u32)]| {
                bound
                    .iter()
                    .find(|b| b.0 == r)
                    .map(|b| b.1)
                    .or_else(|| cval(r).map(|c| 32 - c.unsigned_abs().leading_zeros()))
                    .unwrap_or(31)
            };
            for &p in &wraps {
                let b = match bytecode[p] {
                    Op::Mul { a, b, .. } => bnd(a, &bound) + bnd(b, &bound),
                    Op::Add { a, b, .. } | Op::AddValue { a, b, .. } | Op::Sub { a, b, .. } => {
                        bnd(a, &bound).max(bnd(b, &bound)) + 1
                    }
                    // A 32-bit value (a uint32 at most).
                    Op::GetKey { .. } | Op::ValueBin { .. } => 32,
                    _ => u32::MAX / 2,
                };
                // Too wide to be exact: not wrapped, so later ops see a checked
                // int32 (bound 31) in its place.
                let d = match bytecode[p] {
                    Op::GetKey { dst, .. } => Some(dst),
                    _ => super::written_reg(&bytecode[p]),
                };
                if b <= 53
                    && let Some(d) = d
                {
                    bound.push((d, b));
                }
            }
            let next: Vec<usize> = wraps
                .iter()
                .copied()
                .filter(|&p| {
                    let d = match bytecode[p] {
                        Op::GetKey { dst, .. } => Some(dst),
                        _ => super::written_reg(&bytecode[p]),
                    };
                    let Some(d) = d else {
                        return false;
                    };
                    let Some(x) = info.iter().find(|x| x.0 == d && x.1 <= p && p <= x.2) else {
                        return false;
                    };
                    bound.iter().any(|b| b.0 == d)
                        && x.2 > p
                        && (p + 1..=x.2).all(|pc| match &bytecode[pc] {
                            Op::LoadConst { dst: c, .. } => *c != d,
                            Op::ValueBin { op, a, b, .. } => {
                                (1..=VB_USHR).contains(op) && inn(*a) && inn(*b)
                            }
                            // (An element read can bail: not between.)
                            _ => wraps.contains(&pc) && !matches!(bytecode[pc], Op::GetKey { .. }),
                        })
                })
                .collect();
            if next == wraps {
                break;
            }
            wraps = next;
        }
        let (wrap_bits, wraps): (Vec<usize>, Vec<usize>) =
            wraps.into_iter().partition(|&p| bits_op(&bytecode[p]));
        // A `>>>` by a constant count (mod 32) of at least one is below 2^31.
        let exact_ushr: Vec<usize> = body
            .iter()
            .enumerate()
            .filter_map(|(i, op)| match op {
                Op::ValueBin { op, dst, a, b }
                    if *op == VB_USHR
                        && inn(*dst)
                        && inn(*a)
                        && a != b
                        && cval(*b).is_some_and(|c| c & 31 != 0)
                        && !wrap_bits.contains(&(head + i)) =>
                {
                    Some(head + i)
                }
                _ => None,
            })
            .collect();
        if elems || spec {
            // A `>>>` result that is not held as bits (nor exact) is not
            // an int32.
            let lost: Vec<u16> = body
                .iter()
                .enumerate()
                .filter_map(|(i, op)| match op {
                    Op::ValueBin { op, dst, .. }
                        if *op == VB_USHR
                            && inn(*dst)
                            && !wrap_bits.contains(&(head + i))
                            && !exact_ushr.contains(&(head + i)) =>
                    {
                        Some(*dst)
                    }
                    _ => None,
                })
                .collect();
            if !lost.is_empty() {
                cset.retain(|r| !lost.contains(r));
                continue;
            }
        }
        // (A `===` the double emitter makes, operands not both int
        // registers, decides two distinct objects inline in an integer
        // version too: only Strings and BigInts bail,
        // `NumGen::emit_strict_eq_stub`.)
        if elems {
            // Every element / `.length` read's result, and every element
            // read's index, an int register: else the double versions.
            let all = body.iter().all(|op| match op {
                Op::GetKey { dst, key, .. } => inn(*dst) && inn(*key),
                Op::ArrayLen { dst, .. } => inn(*dst),
                _ => true,
            });
            if !all {
                return None;
            }
        }
        return Some(IntPlan {
            ints: cset,
            consts,
            slots,
            wraps,
            wrap_bits,
            exact_ushr,
            globals,
            entry: None,
            dbl_entry: None,
            inner: None,
            n_mem,
        });
    }
}

/// Whether op `op` of a speculative region's integer version is likely to
/// run the generic emitter's code, around which the int GPRs live there are
/// saved and reloaded (a property op, a call, an inlined call's guard, ...):
/// anything but the numeric, control and element ops the version does
/// inline. (A hint for `spec_alloc`.)
fn saves_gprs(op: &Op) -> bool {
    !matches!(
        op,
        Op::Move { .. }
            | Op::LoadConst { .. }
            | Op::Add { .. }
            | Op::AddValue { .. }
            | Op::Sub { .. }
            | Op::Mul { .. }
            | Op::Div { .. }
            | Op::Mod { .. }
            | Op::ValueBin { .. }
            | Op::Lt { .. }
            | Op::Compare { .. }
            | Op::StrictEq { .. }
            | Op::Not { .. }
            | Op::JumpIfFalse { .. }
            | Op::Jump { .. }
            | Op::ToNumeric { .. }
            | Op::Step { .. }
            | Op::Neg { .. }
            | Op::BitNot { .. }
            | Op::CheckTdz { .. }
            | Op::GetKey { .. }
            | Op::SetKey { .. }
            | Op::SetKeyStrict { .. }
            | Op::ArrayLen { .. }
            | Op::LoadGlobal { .. }
    )
}

/// The int register locations of a speculative region's integer version:
/// one linear scan over the live intervals (`info`'s that `needs` keeps;
/// a `carried` register's is the whole region), in which, when the GPRs
/// run out, the interval whose GPR buys the least gives it up — its uses
/// weighted 8x per loop of the nest around them (`loops`), divided by its
/// length: a value an inner loop only reads, or an outer loop's counter,
/// costs a load from its frame slot per use, while an inner loop's
/// temporary there would put a store and a load on its chain. An interval
/// without a GPR lives in a frame slot ([`MEM_BASE`]), as its int32, for
/// the whole interval (the frame slots survive the generic emitter's code
/// and its calls, so they need no save). `Err`: the registers no slot was
/// left for either.
#[allow(clippy::type_complexity)]
fn spec_alloc(
    bytecode: &[Op],
    (head, end): (usize, usize),
    info: &[(u16, usize, usize, u32, bool)],
    needs: &dyn Fn(&(u16, usize, usize, u32, bool)) -> bool,
    carried: &dyn Fn(u16) -> bool,
    in_arm: &dyn Fn(usize) -> bool,
    loops: &[(usize, usize)],
) -> Result<(Vec<ISlot>, usize), Vec<u16>> {
    // (register, start, end, carried, dirty, weight)
    let mut ivs: Vec<(u16, usize, usize, bool, bool, u64)> = Vec::new();
    for x in info.iter().filter(|x| needs(x)) {
        let c = carried(x.0);
        let (s, e) = if c { (head, end) } else { (x.1, x.2) };
        let w: u64 = (s..=e)
            .filter(|&pc| !in_arm(pc) && super::op_regs(&bytecode[pc]).contains(&x.0))
            .map(|pc| {
                let d = loops.iter().filter(|&&(h, le)| h <= pc && pc <= le).count();
                8u64.pow(d.min(5) as u32)
            })
            .sum();
        // (A register no op names is a promoted global's pseudo-register
        // (a numeric nest's, `PGlobal`), which must have a GPR: first.)
        let named = (head..=end).any(|pc| super::op_regs(&bytecode[pc]).contains(&x.0));
        let w = if named { w } else { u64::MAX >> 16 };
        ivs.push((x.0, s, e, c, x.4, w));
    }
    ivs.sort_by(|p, q| p.1.cmp(&q.1).then(q.3.cmp(&p.3)).then(p.0.cmp(&q.0)));
    // A GPR is saved and reloaded around the generic emitter's code of each
    // op it lives across (`NumGen::int_save`), a frame slot never: an
    // interval whose uses cost less than those saves (a counter of a loop of
    // property ops) goes to a frame slot at once.
    let depth_w = |pc: usize| {
        let d = loops.iter().filter(|&&(h, le)| h <= pc && pc <= le).count();
        8u64.pow(d.min(5) as u32)
    };
    // (In a nest: a lone loop's counter is carried through a frame slot's
    // store and load each iteration, while the saves around a run of
    // consecutive generic ops, made once, are off its chain.)
    let mem_first: Vec<bool> = ivs
        .iter()
        .map(|iv| {
            let saves: u64 = (iv.1 + 1..iv.2)
                .filter(|&pc| {
                    !in_arm(pc)
                        && saves_gprs(&bytecode[pc])
                        && (pc == iv.1 + 1 || !saves_gprs(&bytecode[pc - 1]))
                })
                .map(depth_w)
                .sum();
            !loops.is_empty() && iv.5 <= 2 * saves
        })
        .collect();
    let density = |i: usize| ivs[i].5.saturating_mul(1024) / (ivs[i].2 - ivs[i].1 + 1) as u64;
    let mut loc: Vec<Option<u8>> = alloc::vec![None; ivs.len()];
    let mut free: Vec<u8> = GPRS.iter().rev().copied().collect();
    // (interval end, gpr, interval)
    let mut active: Vec<(usize, u8, usize)> = Vec::new();
    for i in 0..ivs.len() {
        let start = ivs[i].1;
        // (A temporary whose last read is the op that defines this one
        // hands its GPR on, as in `plan`'s scan.)
        active.retain(|&(e, g, k)| {
            if !ivs[k].3 && e <= start {
                free.push(g);
                false
            } else {
                true
            }
        });
        if mem_first[i] {
            continue;
        }
        if let Some(g) = free.pop() {
            loc[i] = Some(g);
            active.push((ivs[i].2, g, i));
            continue;
        }
        let mine = density(i);
        let victim = active
            .iter()
            .enumerate()
            .map(|(j, &(e, _, k))| (density(k), core::cmp::Reverse(e), j))
            .min();
        if let Some((vd, core::cmp::Reverse(ve), j)) = victim
            && (vd < mine || (vd == mine && ve > ivs[i].2))
        {
            let (_, g, k) = active.remove(j);
            loc[k] = None;
            loc[i] = Some(g);
            active.push((ivs[i].2, g, i));
        }
    }
    // The frame slots: a second scan over the intervals left without a GPR.
    let mut slots = Vec::new();
    let mut failed = Vec::new();
    let mut mfree: Vec<u8> = (0..MAX_MEM as u8).rev().collect();
    let mut mact: Vec<(usize, u8)> = Vec::new();
    let mut n_mem = 0usize;
    for (i, iv) in ivs.iter().enumerate() {
        let gpr = match loc[i] {
            Some(g) => g,
            None => {
                mact.retain(|&(e, m)| {
                    if e < iv.1 {
                        mfree.push(m);
                        false
                    } else {
                        true
                    }
                });
                let Some(m) = mfree.pop() else {
                    failed.push(iv.0);
                    continue;
                };
                mact.push((iv.2, m));
                n_mem = n_mem.max(usize::from(m) + 1);
                MEM_BASE + m
            }
        };
        slots.push(ISlot {
            reg: iv.0,
            gpr,
            start: iv.1,
            end: iv.2,
            carried: iv.3,
            dirty: iv.4,
        });
    }
    if failed.is_empty() {
        Ok((slots, n_mem))
    } else {
        Err(failed)
    }
}

impl NumGen {
    fn iplan(&self) -> Option<&IntPlan> {
        self.regions.get(self.imode?)?.int.as_ref()
    }

    /// Whether register `r` is an int register of the integer version being
    /// emitted (or whose stub is).
    pub(crate) fn is_int(&self, r: u16) -> bool {
        self.iplan()
            .is_some_and(|p| p.ints.binary_search(&r).is_ok())
    }

    /// Where int register `r` lives at the current pc.
    fn ik(&self, r: u16) -> Option<IK> {
        let p = self.iplan()?;
        let pc = self.cur_pc;
        if let Some(c) = p.consts.iter().find(|c| c.0 == r && c.2 <= pc && pc <= c.3) {
            return Some(IK::Imm(c.1));
        }
        p.slots
            .iter()
            .find(|s| s.reg == r && s.start <= pc && pc <= s.end)
            .map(|s| self.slot_ik(s))
    }

    /// Where int slot `s` lives.
    fn slot_ik(&self, s: &ISlot) -> IK {
        if s.gpr >= MEM_BASE {
            IK::Mem(self.home_disp(usize::from(s.gpr - MEM_BASE)))
        } else {
            IK::Gpr(s.gpr)
        }
    }

    /// The frame slot of int register frame slot `k` (`MEM_BASE`).
    fn home_disp(&self, k: usize) -> i32 {
        self.int_save_disp(N_GPRS + k)
    }

    /// The GPR holding int register `r`, or `scratch` loaded with its constant
    /// or from its frame slot.
    fn ival(&self, a: &mut X64Assembler, r: u16, scratch: u8) -> Option<u8> {
        Some(match self.ik(r)? {
            IK::Gpr(g) => g,
            IK::Imm(v) => {
                a.mov32_imm(scratch, v);
                scratch
            }
            IK::Mem(d) => {
                a.ld32(scratch, d);
                scratch
            }
        })
    }

    /// `ik(r)`, a frame slot loaded into `scratch` first (so never `Mem`).
    fn ik_ld(&self, a: &mut X64Assembler, r: u16, scratch: u8) -> Option<IK> {
        Some(match self.ik(r)? {
            IK::Mem(d) => {
                a.ld32(scratch, d);
                IK::Gpr(scratch)
            }
            k => k,
        })
    }

    /// Where an emitter computes int register `dst`'s new value: its GPR,
    /// or (a frame slot) `rax`, stored by [`Self::iput`]. `None` for a
    /// constant.
    fn iout(&self, dst: u16) -> Option<u8> {
        match self.ik(dst)? {
            IK::Gpr(g) => Some(g),
            IK::Mem(_) => Some(RAX),
            IK::Imm(_) => None,
        }
    }

    /// Int register `dst` = the int32 in GPR `src`.
    fn iput(&self, a: &mut X64Assembler, dst: u16, src: u8) -> Option<()> {
        match self.ik(dst)? {
            IK::Gpr(g) => a.mov32_rr(g, src),
            IK::Mem(d) => a.st32(d, src),
            IK::Imm(_) => return None,
        }
        Some(())
    }

    /// `rax` = int register `r`, sign-extended to 64 bits (an index).
    fn index_rax(&self, a: &mut X64Assembler, r: u16) -> Option<()> {
        match self.ik(r)? {
            IK::Gpr(g) => a.movsxd(RAX, g),
            IK::Imm(v) => a.movabs_rax(i64::from(v)),
            IK::Mem(d) => a.movsxd_mem(RAX, d),
        }
        Some(())
    }

    /// `xmm(x) = (f64) int register r`.
    fn icvt(&self, a: &mut X64Assembler, x: u8, r: u16) -> Option<()> {
        match self.ik(r)? {
            IK::Gpr(g) => a.cvtsi2sd32(x, g),
            IK::Imm(v) => {
                a.movabs_rax(f64::from(v).to_bits() as i64);
                a.movq_x_gpr(x, RAX);
            }
            IK::Mem(d) => a.cvtsi2sd32_mem(x, d),
        }
        Some(())
    }

    /// `dst = (f64) int register src` into `dst`'s double-version location
    /// (`dst` no int register), `None` (nothing emitted) when no GPR or
    /// constant holds `src` here. Clobbers `xmm0` and `rax`.
    pub(super) fn int_copy_to(
        &self,
        a: &mut X64Assembler,
        reg: Option<usize>,
        dst: u16,
        src: u16,
    ) -> Option<()> {
        self.ik(src)?;
        match self.xmm_of(reg, dst) {
            Some(x) => self.icvt(a, x, src),
            None => {
                self.icvt(a, 0, src)?;
                a.movsd_mem_x(disp(dst), 0);
                Some(())
            }
        }
    }

    /// Converts int register `r` into its double-version location (its promoted
    /// SSE register, else the register file).
    fn ishadow(&self, a: &mut X64Assembler, reg: Option<usize>, r: u16) -> Option<()> {
        match self.xmm_of(reg, r) {
            Some(x) => self.icvt(a, x, r),
            None => {
                self.icvt(a, 0, r)?;
                a.movsd_mem_x(disp(r), 0);
                Some(())
            }
        }
    }

    /// The integer version's `flush`: dirty double registers, then every live
    /// written int register converted into the register file. Clobbers `rax`
    /// and `xmm0`.
    pub(super) fn iflush(&self, a: &mut X64Assembler, ri: usize) {
        let pc = self.cur_pc;
        let Some(p) = self.regions[ri].int.as_ref() else {
            return;
        };
        let rg = &self.regions[ri];
        for s in &rg.slots {
            if s.dirty
                && s.start <= pc
                && pc <= s.end
                && p.ints.binary_search(&s.reg).is_err()
                && rg.defined(s, pc, pc)
            {
                a.movsd_mem_x(disp(s.reg), s.xmm);
            }
        }
        for s in &p.slots {
            if s.dirty && s.start <= pc && pc <= s.end && !self.is_greg(s.reg) {
                match self.slot_ik(s) {
                    IK::Mem(d) => a.cvtsi2sd32_mem(0, d),
                    _ => a.cvtsi2sd32(0, s.gpr),
                }
                a.movsd_mem_x(disp(s.reg), 0);
            }
        }
        for &(r, v, start, end) in &p.consts {
            if start <= pc && pc <= end {
                a.movabs_rax(f64::from(v).to_bits() as i64);
                a.store_rax(disp(r));
            }
        }
        self.gstore(a, p);
        self.store_lazy_args(a);
    }

    /// Whether `r` is a promoted global's pseudo-register.
    fn is_greg(&self, r: u16) -> bool {
        r as usize >= self.env.n_regs
    }

    /// The GPR an int pseudo-register lives in (carried: the whole region).
    fn ggpr(p: &IntPlan, vreg: u16) -> Option<u8> {
        p.slots.iter().find(|s| s.reg == vreg).map(|s| s.gpr)
    }

    /// Writes the stored-to promoted globals back to their bindings. Clobbers
    /// `rcx` and `xmm0`.
    fn gstore(&self, a: &mut X64Assembler, p: &IntPlan) {
        for g in p.globals.iter().filter(|g| g.dirty) {
            // (No promoted global is a global `var`: see `plan`.)
            a.load_rcx(self.gslot_disp(g.slot));
            match (g.xmm, Self::ggpr(p, g.vreg)) {
                (Some(x), _) => a.sse_mem_rcx(0x11, x), // movsd [rcx], x
                (None, Some(gpr)) => {
                    a.cvtsi2sd32(0, gpr);
                    a.sse_mem_rcx(0x11, 0);
                }
                (None, None) => {}
            }
        }
    }

    /// Validates every promoted global's binding address into its frame slot
    /// (the load and the store op must both qualify, at one address); jumps to
    /// `fail` otherwise. A call per op: clobbers every caller-saved register.
    fn gvalidate(&self, a: &mut X64Assembler, p: &IntPlan, fail: Label) {
        for g in &p.globals {
            let sd = self.gslot_disp(g.slot);
            for (i, pc) in [g.load_pc, g.store_pc].into_iter().flatten().enumerate() {
                a.mov_rdi_r15();
                a.movabs_argreg(1, self.env.bytecode.wrapping_add(pc) as i64);
                a.movabs_rax(self.env.global_ref as i64);
                a.call_rax();
                a.test_rax_rax();
                a.jcc(CC_E, fail);
                if i == 0 {
                    a.store_rax(sd);
                } else {
                    a.load_rcx(sd);
                    a.raw(&[0x48, 0x39, 0xc8]); // cmp rax, rcx
                    a.jcc(CC_NE, fail);
                }
            }
        }
    }

    /// Loads the promoted globals from their (validated) bindings; jumps to
    /// `fail` when an int one does not hold an exact int32.
    fn gload(&self, a: &mut X64Assembler, p: &IntPlan, fail: Label) {
        for g in &p.globals {
            a.load_rax(self.gslot_disp(g.slot));
            match (g.xmm, Self::ggpr(p, g.vreg)) {
                (Some(x), _) => a.sse_mem_rax(0x10, x), // movsd x, [rax]
                (None, Some(gpr)) => {
                    a.raw(&[0xf2, 0x0f, 0x2c, 0x08]); // cvttsd2si ecx, [rax]
                    a.raw(&[0x48, 0x8b, 0x00]); // mov rax, [rax]
                    a.cvtsi2sd32(0, RCX);
                    a.movq_gpr_x(R11, 0);
                    a.cmp_rax_r11();
                    a.jcc(CC_NE, fail);
                    a.mov32_rr(gpr, RCX);
                }
                (None, None) => {}
            }
        }
    }

    /// Whether `op` reads or writes a promoted global.
    fn promoted_global_op(&self, op: &Op) -> bool {
        match op {
            Op::LoadGlobal { name, .. } | Op::StoreGlobal { name, .. } => {
                self.pglobal(name).is_some()
            }
            _ => false,
        }
    }

    /// A `LoadGlobal` / `StoreGlobal` of a promoted global: a move from / to
    /// its pseudo-register.
    fn iglobal(&self, a: &mut X64Assembler, pc: usize) -> Option<()> {
        let reg = self.imode;
        let (load, r, name) = match self.bc(pc)? {
            Op::LoadGlobal { dst, name, .. } => (true, *dst, name.as_str()),
            Op::StoreGlobal { src, name, .. } => (false, *src, name.as_str()),
            _ => return None,
        };
        let g = self.pglobal(name)?;
        let (vreg, gx) = (g.vreg, g.xmm);
        let p = self.iplan()?;
        let gg = Self::ggpr(p, vreg);
        match (load, gx, gg) {
            // Int global.
            (true, None, Some(gpr)) => {
                if self.is_int(r) {
                    let IK::Gpr(d) = self.ik(r)? else {
                        return None;
                    };
                    a.mov32_rr(d, gpr);
                } else {
                    a.cvtsi2sd32(0, gpr);
                    self.store_xmm0(a, reg, r);
                }
            }
            (false, None, Some(gpr)) => {
                // Planned: a store into an int global is of an int register.
                let s = self.ival(a, r, RAX)?;
                a.mov32_rr(gpr, s);
            }
            // Double global.
            (true, Some(x), _) => {
                if self.is_int(r) {
                    return None;
                }
                match self.xmm_of(reg, r) {
                    Some(xd) => a.movapd_xx(xd, x),
                    None => a.movsd_mem_x(disp(r), x),
                }
            }
            (false, Some(x), _) => {
                if self.is_int(r) {
                    self.icvt(a, x, r)?;
                } else {
                    let s = self.xmm_src(a, reg, r, 0);
                    a.movapd_xx(x, s);
                }
            }
            _ => return None,
        }
        Some(())
    }

    /// The promoted global `name`, if the integer version being emitted has it.
    fn pglobal(&self, name: &str) -> Option<&PGlobal> {
        let p = self.iplan()?;
        p.globals.iter().find(|g| {
            [g.load_pc, g.store_pc].into_iter().flatten().any(|pc| {
                matches!(self.bc(pc), Some(Op::LoadGlobal { name: n, .. } | Op::StoreGlobal { name: n, .. }) if n == name)
            })
        })
    }

    /// The integer version's `reload` (after a call, which `iflush` preceded).
    pub(super) fn ireload(&self, a: &mut X64Assembler, ri: usize) {
        let pc = self.cur_pc;
        let Some(p) = self.regions[ri].int.as_ref() else {
            return;
        };
        for s in &self.regions[ri].slots {
            if s.start <= pc && pc <= s.end && p.ints.binary_search(&s.reg).is_err() {
                a.movsd_x_mem(s.xmm, disp(s.reg));
            }
        }
        for s in &p.slots {
            // (A frame slot is not touched by the call.)
            if s.start <= pc && pc <= s.end && !self.is_greg(s.reg) && s.gpr < MEM_BASE {
                // Exact: the register file holds the int32 `iflush` (or the
                // region's entry) left there.
                a.cvttsd2si32_mem(s.gpr, disp(s.reg));
            }
        }
    }

    /// A bail at the current pc: its out-of-line entry.
    fn ibail(&mut self, a: &mut X64Assembler) -> Label {
        let entry = a.new_label();
        let resume = a.new_label();
        let pc = self.cur_pc as u32;
        self.stub(
            entry,
            resume,
            self.imode,
            StubKind::Bail { pc, store: true },
        );
        entry
    }

    /// A bail at the current pc of an op whose result is no int32 (an
    /// overflow, a `-0`, a NaN): as [`Self::ibail`], and the bail sets the
    /// op's bit in the region's overflow-bail mask, so a recompilation
    /// demotes its result (see [`plan`]'s `demote`).
    fn ibail_ovf(&mut self, a: &mut X64Assembler) -> Label {
        let entry = self.ibail(a);
        if let Some(r) = self.bc(self.cur_pc).and_then(super::written_reg) {
            self.demote_last(r);
        }
        entry
    }

    /// The stub just made (a bail of the current pc's op, whose result `r`
    /// was found no int32: an overflow, or an element read's value) sets
    /// the site's bit in the region's overflow-bail mask, so a
    /// recompilation demotes `r` (see [`plan`]'s `demote`).
    fn demote_last(&mut self, r: u16) {
        let (Some(ri), pc) = (self.imode, self.cur_pc) else {
            return;
        };
        if self.env.persist == 0 || ri >= super::super::global_persist::COUNTERS {
            return;
        }
        let bit = self.demote_bit(ri, pc, r);
        if let Some(s) = self.stubs.last_mut() {
            s.demote = Some(bit);
        }
    }

    /// The bit of region `ri`'s overflow-bail mask for site `(pc, r)`.
    fn demote_bit(&mut self, ri: usize, pc: usize, r: u16) -> u8 {
        if self.demote_sites.len() <= ri {
            self.demote_sites.resize(ri + 1, Vec::new());
        }
        let sites = &mut self.demote_sites[ri];
        let k = match sites.iter().position(|&q| q == (pc, r)) {
            Some(k) => k,
            None => {
                sites.push((pc, r));
                sites.len() - 1
            }
        };
        k.min(DEMOTE_BITS - 1) as u8
    }

    /// Where region `ri`'s integer version's entry check of loop-carried int
    /// register `r` goes when the value is no int32: to `efail` (counted,
    /// `entry_retry`), the last time before the version is turned off
    /// ([`ENTRY_RETRIES`] in a row) first setting `r`'s bit in the region's
    /// overflow-bail mask (its site the header), so a recompilation demotes
    /// `r` (a counter `for (let c = 2147483000; …; c += 3)` already past
    /// 2^31 when on-stack replacement enters the loop). The code goes into
    /// `fails`, emitted after the checks.
    fn entry_fail(
        &mut self,
        a: &mut X64Assembler,
        ri: usize,
        r: u16,
        efail: Label,
        fails: &mut Vec<(Label, u8)>,
    ) -> Label {
        if self.env.persist == 0 || ri >= super::super::global_persist::COUNTERS {
            return efail;
        }
        let head = self.regions[ri].head;
        let bit = self.demote_bit(ri, head, r);
        let l = a.new_label();
        fails.push((l, bit));
        l
    }

    /// The code of [`Self::entry_fail`]'s labels.
    fn emit_entry_fails(
        &self,
        a: &mut X64Assembler,
        ri: usize,
        efail: Label,
        fails: &[(Label, u8)],
    ) {
        let r11 = super::super::call_seq::r::R11;
        for &(l, bit) in fails {
            a.bind(l);
            a.g_movabs(r11, self.env.persist);
            a.g_mem(
                true,
                &[0x81],
                7,
                r11,
                super::super::global_persist::entry_off(ri),
            );
            a.raw(&ENTRY_DEMOTE.to_le_bytes()); // cmp qword [r11 + off], imm32
            a.jcc(CC_L, efail);
            self.note_demote(a, ri, bit);
            a.jmp(efail);
        }
    }

    /// In an overflow bail's stub: sets bit `bit` of region `ri`'s
    /// overflow-bail mask and, the first time, asks the next back-edge poll
    /// to have the body recompiled. Clobbers `r11` (and the flags).
    pub(super) fn note_demote(&self, a: &mut X64Assembler, ri: usize, bit: u8) {
        let r11 = super::super::call_seq::r::R11;
        let done = a.new_label();
        a.g_movabs(r11, self.env.persist);
        a.g_mem(
            true,
            &[0xf7],
            0,
            r11,
            super::super::global_persist::demote_off(ri),
        );
        a.raw(&(1i32 << bit).to_le_bytes()); // test qword [r11 + off], 1 << bit
        a.jcc(CC_NE, done);
        a.g_mem(
            true,
            &[0x81],
            1,
            r11,
            super::super::global_persist::demote_off(ri),
        );
        a.raw(&(1i32 << bit).to_le_bytes()); // or qword [r11 + off], 1 << bit
        // (The next back-edge poll calls, to have the body recompiled.)
        a.g_store_byte_imm(
            super::super::call_seq::r::R15,
            self.env.layout.poll.demote,
            1,
        );
        a.bind(done);
    }

    /// `(header, end, region, sites)` of the regions whose integer
    /// versions have overflow bails that set their mask's bits (`tierup`):
    /// site `k`, `(pc, register)`, sets bit `min(k, DEMOTE_BITS - 1)`.
    pub(crate) fn demote_heads(&self) -> Vec<super::DemoteRegion> {
        self.demote_sites
            .iter()
            .enumerate()
            .filter(|(_, s)| !s.is_empty())
            .map(|(ri, s)| (self.regions[ri].head, self.regions[ri].end, ri, s.clone()))
            .collect()
    }

    /// A bail's body (in `finish`): materialize the int registers live at `pc`
    /// into the double version's locations and (`store`) write the promoted
    /// globals back to their bindings, then continue in its code for `pc`.
    pub(super) fn emit_bail(
        &self,
        a: &mut X64Assembler,
        labels: &[Label],
        ri: usize,
        pc: usize,
        store: bool,
    ) {
        let Some(p) = self.regions[ri].int.as_ref() else {
            return;
        };
        let reg = Some(ri);
        if store {
            self.gstore(a, p);
        }
        for s in &p.slots {
            if s.start <= pc && pc <= s.end && !self.is_greg(s.reg) {
                let _ = self.ishadow(a, reg, s.reg);
            }
        }
        for &(r, _, start, end) in &p.consts {
            if start <= pc && pc <= end {
                let _ = self.ishadow(a, reg, r);
            }
        }
        let rg: &Region = &self.regions[ri];
        let target = if pc == rg.head {
            rg.inner
        } else {
            labels.get(pc).copied()
        };
        if let Some(t) = target {
            a.jmp(t);
        }
    }

    /// Emits region `ri`'s integer version (after its double version).
    pub(super) fn emit_int_region(
        &mut self,
        a: &mut X64Assembler,
        labels: &[Label],
        throw_exit: Label,
        ri: usize,
        ops: &[GenericOp],
    ) -> Option<()> {
        let (head, end) = (self.regions[ri].head, self.regions[ri].end);
        let p = self.regions[ri].int.as_ref()?;
        let (entry, dbl_entry) = (p.entry?, p.dbl_entry?);
        let mut ilabels = labels.to_vec();
        for l in ilabels.iter_mut().take(end + 1).skip(head) {
            *l = a.new_label();
        }
        // (A nest's: an entry at an inner header may go straight into this
        // version there, `plain_int_inner_entry`.)
        if !self.regions[ri].loops.is_empty() {
            for pc in head..=end {
                if let Some(l) = self.int_labels.get_mut(pc) {
                    *l = Some(ilabels[pc]);
                }
            }
        }
        self.cur_pc = head;
        a.bind(entry);
        // A value not an int32 runs an iteration of the double version, whose
        // back-edge tries again (`entry_retry`).
        let efail = self.entry_retry(a, ri, dbl_entry, dbl_entry);
        let p = self.regions[ri].int.as_ref()?;
        // Entry: the promoted globals' bindings must qualify (before anything is
        // loaded: these are calls), every carried int register must be an exact
        // int32 now.
        self.gvalidate(a, p, dbl_entry);
        let carried: Vec<(u16, u8)> = p
            .slots
            .iter()
            .filter(|s| s.carried && !self.is_greg(s.reg))
            .map(|s| (s.reg, s.gpr))
            .collect();
        let mut fails = Vec::new();
        for (r, gpr) in carried {
            a.load_rax(disp(r));
            a.exact_i32_check(RCX);
            let f = self.entry_fail(a, ri, r, efail, &mut fails);
            a.jcc(CC_NE, f);
            if gpr >= MEM_BASE {
                a.st32(self.home_disp(usize::from(gpr - MEM_BASE)), RCX);
            } else {
                a.mov32_rr(gpr, RCX);
            }
        }
        if !fails.is_empty() {
            let past = a.new_label();
            a.jmp(past);
            self.emit_entry_fails(a, ri, efail, &fails);
            a.bind(past);
        }
        let p = self.regions[ri].int.as_ref()?;
        for s in &self.regions[ri].slots {
            if s.carried && p.ints.binary_search(&s.reg).is_err() {
                a.movsd_x_mem(s.xmm, disp(s.reg));
            }
        }
        self.gload(a, p, efail);
        self.entry_ok(a, ri);
        self.refresh_globals(a, ri, false);
        let inner = ilabels[head];
        a.bind(inner);
        if let Some(p) = self.regions[ri].int.as_mut() {
            p.inner = Some(inner);
        }
        self.imode = Some(ri);
        let mut ok = Some(());
        for pc in head..=end {
            if pc != head {
                a.bind(ilabels[pc]);
            }
            if self.emit(a, &ilabels, throw_exit, pc, ops) != Some(true) {
                ok = None;
                break;
            }
        }
        self.imode = None;
        ok
    }

    /// Emits op `pc` of the integer version: `Some(true)` when done here, `None`
    /// on a planning mismatch, `Some(false)` to run the double emitter (with the
    /// op's int operands shadowed first).
    #[allow(clippy::too_many_lines)]
    pub(super) fn emit_int_op(
        &mut self,
        a: &mut X64Assembler,
        labels: &[Label],
        throw_exit: Label,
        pc: usize,
        ops: &[GenericOp],
    ) -> Option<bool> {
        use crate::nbvm::{GA_MOD, GA_MUL, GA_SUB, VB_LOOSE_EQ, VB_LOOSE_NEQ};
        let reg = self.imode;
        let bc = self.bc(pc)?;
        let ii = |s: &Self, r: u16| s.is_int(r);
        match ops[pc] {
            // (`elems`: only planned with int results.)
            GenericOp::GetElem { dst, obj, index } if ii(self, dst) && !self.spec_int_mode() => {
                self.iget_elem(a, dst, obj, index)?;
            }
            GenericOp::ArrayLen { dst, arr } if ii(self, dst) && !self.spec_int_mode() => {
                self.iarray_len(a, dst, arr)?;
            }
            GenericOp::Const { dst, .. } if ii(self, dst) => {
                let k = self.ik(dst)?;
                if !matches!(k, IK::Imm(_)) {
                    let Op::LoadConst { value, .. } = *bc else {
                        return None;
                    };
                    match k {
                        IK::Gpr(g) => a.mov32_imm(g, const_i32(value)?),
                        IK::Mem(d) => a.st32_imm(d, const_i32(value)?),
                        IK::Imm(_) => {}
                    }
                }
            }
            GenericOp::Move { dst, src } if ii(self, dst) => {
                // (A constant live range: the move of its constant, nothing.)
                if let IK::Imm(_) = self.ik(dst)? {
                    return Some(true);
                }
                let s = self.ival(a, src, RAX)?;
                self.iput(a, dst, s)?;
            }
            GenericOp::Move { dst, src } if ii(self, src) => {
                self.icvt(a, 0, src)?;
                self.store_xmm0(a, reg, dst);
            }
            GenericOp::ToNumeric { dst, src, .. } if ii(self, dst) => {
                self.iout(dst)?;
                let s = self.ival(a, src, RAX)?;
                self.iput(a, dst, s)?;
            }
            GenericOp::Step { dst, src, dec, .. } if ii(self, dst) => {
                let d = self.iout(dst)?;
                let s = self.ival(a, src, RAX)?;
                let t = if d != s { d } else { RAX };
                a.mov32_rr(t, s);
                a.add32_imm8(t, if dec { -1 } else { 1 });
                let bail = self.ibail_ovf(a);
                a.jcc(CC_O, bail);
                self.iput(a, dst, t)?;
            }
            GenericOp::Add { dst, a: x, b: y }
            | GenericOp::NumAdd {
                dst, a: x, b: y, ..
            } if ii(self, dst) => {
                self.iarith(a, 0x01, dst, x, y)?;
            }
            GenericOp::Arith {
                dst,
                a: x,
                b: y,
                op: gop,
            } if ii(self, dst) => match gop {
                GA_SUB => self.iarith(a, 0x29, dst, x, y)?,
                GA_MUL => self.imul(a, dst, x, y)?,
                GA_MOD => self.imod(a, dst, x, y)?,
                _ => return None,
            },
            GenericOp::ValueBin {
                dst,
                a: x,
                b: y,
                op: vop,
            } if (1..=crate::nbvm::VB_USHR).contains(&vop)
                && (ii(self, dst) || ii(self, x) || ii(self, y)) =>
            {
                if !self.iand_test(a, labels, pc, ops, vop, dst, x, y)? {
                    self.ibitwise(a, vop, dst, x, y)?;
                }
            }
            GenericOp::ValueBin {
                dst,
                a: x,
                b: y,
                op: vop,
            } if (vop == VB_LOOSE_EQ || vop == VB_LOOSE_NEQ) && ii(self, x) && ii(self, y) => {
                let cc = if vop == VB_LOOSE_EQ { CC_E } else { CC_NE };
                self.icompare(a, labels, pc, ops, cc, dst, x, y)?;
            }
            GenericOp::StrictEq { dst, a: x, b: y } if ii(self, x) && ii(self, y) => {
                self.icompare(a, labels, pc, ops, CC_E, dst, x, y)?;
            }
            GenericOp::Lt { dst, a: x, b: y } if ii(self, x) && ii(self, y) => {
                self.icompare(a, labels, pc, ops, CC_L, dst, x, y)?;
            }
            GenericOp::Compare {
                dst,
                a: x,
                b: y,
                op: cop,
                ..
            } if ii(self, x) && ii(self, y) => {
                let cc = match cop {
                    0 => CC_L,
                    1 => CC_G,
                    2 => CC_LE,
                    _ => CC_GE,
                };
                self.icompare(a, labels, pc, ops, cc, dst, x, y)?;
            }
            GenericOp::JumpIfFalse { cond, target } if ii(self, cond) && target > pc => {
                match self.ik_ld(a, cond, RAX)? {
                    IK::Gpr(g) => {
                        a.test32_rr(g, g);
                        self.jcc_to(a, labels, reg, CC_E, target);
                    }
                    IK::Mem(_) => return None,
                    IK::Imm(0) => self.jump(a, labels, reg, target),
                    IK::Imm(_) => {}
                }
            }
            // `if (!x)` (`dx || dy`) on an int register: to the target when
            // `x` is non-zero, the not's boolean never stored.
            GenericOp::Not { dst, a: x } if ii(self, x) && !ii(self, dst) => {
                let Some(target) = self.int_not_test(pc, ops, dst, x) else {
                    return Some(false);
                };
                match self.ik_ld(a, x, RAX)? {
                    IK::Gpr(g) => {
                        a.test32_rr(g, g);
                        self.jcc_to(a, labels, reg, CC_NE, target);
                    }
                    IK::Imm(0) => {}
                    IK::Imm(_) => self.jump(a, labels, reg, target),
                    IK::Mem(_) => return None,
                }
                self.skip_op(pc + 1);
            }
            GenericOp::CheckTdz { src, .. } if ii(self, src) => {}
            GenericOp::Jump { target } if target <= pc => {
                let ri = reg?;
                // (In a speculative nest's integer version, an inner loop's
                // back-edge too: to that header's label in this version.)
                let nested = pc != self.regions[ri].end || target != self.regions[ri].head;
                if nested && !self.regions[ri].loops.contains(&(target, pc)) {
                    return None;
                }
                let inner = if nested {
                    *labels.get(target)?
                } else {
                    self.regions[ri].int.as_ref()?.inner?
                };
                let skip = a.new_label();
                a.dec_frame8(super::COUNTDOWN);
                a.jne(skip);
                super::super::call_seq::emit_quiet_poll(
                    a,
                    &self.env.layout,
                    i32::from(super::COUNTDOWN),
                    skip,
                );
                self.iflush(a, ri);
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
                // Leaving at the header: the integer flush above wrote back
                // every promoted register as a Number.
                a.movabs_r11(NanBox::jit_exit_bits() as i64);
                a.cmp_rax_r11();
                a.je(self.leave_exit?);
                a.store_rax_frame8(super::COUNTDOWN);
                if self.regions[ri].spec_int {
                    // As the speculative version's back-edge: the global
                    // bindings validated again (else the plain copy runs on,
                    // the register file up to date), the views hoisted.
                    self.refresh_globals(a, ri, true);
                    self.spec_globals_ready(a, ri, target)?;
                    self.hoist_views(a, ri);
                    self.ireload(a, ri);
                    a.bind(skip);
                    a.jmp(inner);
                    return Some(true);
                }
                let p = self.regions[ri].int.as_ref()?;
                let has_globals = !p.globals.is_empty();
                let (vfail, lfail) = (a.new_label(), a.new_label());
                if has_globals {
                    // The poll may have moved or retired the bindings.
                    self.gvalidate(a, p, vfail);
                }
                self.ireload(a, ri);
                if has_globals {
                    self.gload(a, p, lfail);
                }
                self.zero_globals(a, reg);
                a.bind(skip);
                a.jmp(inner);
                if has_globals {
                    // A binding no longer qualifies: the double version (which
                    // reads the bindings op by op) finishes the loop. The
                    // bindings are already up to date.
                    let entry = a.new_label();
                    let resume = a.new_label();
                    a.bind(vfail);
                    self.ireload(a, ri);
                    a.bind(lfail);
                    a.jmp(entry);
                    self.stub(
                        entry,
                        resume,
                        reg,
                        StubKind::Bail {
                            pc: pc as u32,
                            store: false,
                        },
                    );
                }
            }
            GenericOp::Interp { .. } if self.promoted_global_op(bc) => {
                self.iglobal(a, pc)?;
            }
            GenericOp::Interp { .. } => match *bc {
                Op::Neg { dst, a: src } if ii(self, dst) => {
                    self.iout(dst)?;
                    let s = self.ival(a, src, RAX)?;
                    a.mov32_rr(RAX, s);
                    let bail = self.ibail_ovf(a);
                    a.test32_rr(RAX, RAX);
                    a.jcc(CC_E, bail); // -0
                    a.grp3_32(3, RAX); // neg eax
                    a.jcc(CC_O, bail); // -(-2^31)
                    self.iput(a, dst, RAX)?;
                }
                Op::BitNot { dst, a: src } if ii(self, dst) => {
                    self.iout(dst)?;
                    if ii(self, src) {
                        let s = self.ival(a, src, RAX)?;
                        a.mov32_rr(RAX, s);
                    } else {
                        let bail = self.ibail(a);
                        if self.ty(pc, src) != super::T_NUM {
                            self.num_guard(a, reg, src, bail);
                        }
                        self.to_int32(a, reg, RAX, src, bail);
                    }
                    a.grp3_32(2, RAX); // not eax
                    self.iput(a, dst, RAX)?;
                }
                Op::LoadGlobal { dst, .. } if ii(self, dst) => {
                    self.iout(dst)?;
                    if let Some(src) = self.carried_int_global(pc) {
                        let s = self.ival(a, src, RAX)?;
                        self.iput(a, dst, s)?;
                        return Some(true);
                    }
                    let k = *self.gslot.get(pc)?;
                    if k == u32::MAX {
                        return None;
                    }
                    let sd = self.gslot_disp(k);
                    let retry = a.new_label();
                    let stub = a.new_label();
                    let resume = a.new_label();
                    a.bind(retry);
                    a.load_rax(sd);
                    a.test_rax_rax();
                    a.jcc(CC_E, stub);
                    // The value converts straight from memory; the exactness
                    // check (off the dependency chain) compares the bits.
                    a.raw(&[0xf2, 0x0f, 0x2c, 0x08]); // cvttsd2si ecx, [rax]
                    a.raw(&[0x48, 0x8b, 0x00]); // mov rax, [rax]
                    a.cvtsi2sd32(0, RCX);
                    a.movq_gpr_x(R11, 0);
                    a.cmp_rax_r11();
                    let bail = self.ibail(a);
                    a.jcc(CC_NE, bail);
                    self.iput(a, dst, RCX)?;
                    a.bind(resume);
                    self.stub(stub, resume, reg, StubKind::GlobalRef { pc, retry });
                }
                Op::StoreGlobal { src, .. } if ii(self, src) && !self.spec_int_mode() => {
                    let k = *self.gslot.get(pc)?;
                    if k == u32::MAX {
                        return None;
                    }
                    let sd = self.gslot_disp(k);
                    let retry = a.new_label();
                    let stub = a.new_label();
                    let resume = a.new_label();
                    a.bind(retry);
                    a.load_rcx(sd);
                    a.raw(&[0x48, 0x85, 0xc9]); // test rcx, rcx
                    a.jcc(CC_E, stub);
                    self.icvt(a, 0, src)?;
                    a.raw(&[0xf2, 0x0f, 0x11, 0x01]); // movsd [rcx], xmm0
                    // (An int32, converted: a Number.)
                    if let Some(cell) = super::super::global_mirror::mirror_cell(self.bc(pc)?) {
                        a.gm_store_xmm(cell, 0);
                    }
                    a.bind(resume);
                    self.stub(stub, resume, reg, StubKind::GlobalRef { pc, retry });
                }
                _ => return Some(false),
            },
            _ => return Some(false),
        }
        Some(true)
    }

    /// `dst = x <add|sub> y` over int registers (`opc` the `op r/m32, r32` byte),
    /// bailing on overflow.
    fn iarith(&mut self, a: &mut X64Assembler, opc: u8, dst: u16, x: u16, y: u16) -> Option<()> {
        let d = self.iout(dst)?;
        let (kx, ky) = (self.ik(x)?, self.ik(y)?);
        let gpr = |k: IK| match k {
            IK::Gpr(g) => Some(g),
            _ => None,
        };
        let wrap = self.iplan().is_some_and(|p| p.wraps.contains(&self.cur_pc));
        // Computing in `d` itself is fine unless an operand lives there (a bail
        // must still see the operands).
        let t = if Some(d) != gpr(kx) && Some(d) != gpr(ky) {
            d
        } else {
            RAX
        };
        match kx {
            IK::Gpr(g) => a.mov32_rr(t, g),
            IK::Imm(v) => a.mov32_imm(t, v),
            IK::Mem(m) => a.ld32(t, m),
        }
        match ky {
            IK::Gpr(g) => a.alu32_rr(opc, t, g),
            // (`81 /0` add, `81 /5` sub.)
            IK::Imm(v) => a.alu32_imm(if opc == 0x01 { 0 } else { 5 }, t, v),
            // (`03` add, `2b` sub: `op r32, r/m32`.)
            IK::Mem(m) => a.g_mem(false, &[opc + 2], t, super::super::call_seq::r::RBP, m),
        }
        if !wrap {
            let bail = self.ibail_ovf(a);
            a.jcc(CC_O, bail);
        }
        self.iput(a, dst, t)
    }

    /// `dst = x * y` over int registers, bailing on overflow and on a zero
    /// product with a negative factor (`-0`).
    fn imul(&mut self, a: &mut X64Assembler, dst: u16, x: u16, y: u16) -> Option<()> {
        let d = self.iout(dst)?;
        let gx = self.ival(a, x, RDX)?;
        let gy = self.ival(a, y, RCX)?;
        if self.iplan().is_some_and(|p| p.wraps.contains(&self.cur_pc)) {
            // Wrapped (see `plan`): only the low 32 bits matter, `-0` included.
            let t = if d != gx && d != gy { d } else { RAX };
            a.mov32_rr(t, gx);
            a.imul32_rr(t, gy);
            return self.iput(a, dst, t);
        }
        a.mov32_rr(RAX, gx);
        a.imul32_rr(RAX, gy);
        let bail = self.ibail_ovf(a);
        a.jcc(CC_O, bail);
        let ok = a.new_label();
        a.test32_rr(RAX, RAX);
        a.jcc(CC_NE, ok);
        a.mov32_rr(R11, gx);
        a.alu32_rr(0x09, R11, gy); // or r11d, y
        a.jcc(CC_S, bail);
        a.bind(ok);
        self.iput(a, dst, RAX)
    }

    /// `dst = x % y` over int registers, bailing on a zero divisor (NaN) and on a
    /// zero remainder of a negative dividend (`-0`).
    fn imod(&mut self, a: &mut X64Assembler, dst: u16, x: u16, y: u16) -> Option<()> {
        self.iout(dst)?;
        if let IK::Imm(c) = self.ik(y)?
            && c != 0
            && c != i32::MIN
        {
            return self.imod_const(a, dst, x, c.unsigned_abs());
        }
        let gx = self.ival(a, x, RAX)?;
        a.movsxd(RAX, gx);
        let gy = self.ival(a, y, RCX)?;
        a.movsxd(RCX, gy);
        let bail = self.ibail_ovf(a);
        a.test32_rr(RCX, RCX);
        a.jcc(CC_E, bail);
        a.raw(&[0x49, 0x89, 0xc3]); // mov r11, rax (the dividend)
        // Sign-extended 64-bit operands: `-2^31 / -1` cannot overflow.
        a.cqo();
        a.idiv_rcx();
        let ok = a.new_label();
        a.test32_rr(RDX, RDX);
        a.jcc(CC_NE, ok);
        a.raw(&[0x4d, 0x85, 0xdb]); // test r11, r11
        a.jcc(CC_S, bail);
        a.bind(ok);
        self.iput(a, dst, RDX)
    }

    /// `dst = x % c` for a constant divisor `0 < c < 2^31` (the sign of a
    /// divisor does not matter: the remainder takes the dividend's), by a
    /// reciprocal multiplication instead of a division: for `0 <= n < 2^31`,
    /// `floor(n / c) = (n * m) >> (31 + l)` with `l = ceil(log2 c)` and
    /// `m = floor(2^(31+l) / c) + 1` (then `c <= 2^l` gives `2^(31+l) < m*c <=
    /// 2^(31+l) + 2^l`, the classic exactness condition; `m <= 2^32`, so the
    /// product stays below 2^63). Bails on `x = -2^31` (`|x|` out of range) and
    /// on a zero remainder of a negative dividend (`-0`).
    fn imod_const(&mut self, a: &mut X64Assembler, dst: u16, x: u16, c: u32) -> Option<()> {
        let l = 32 - (c - 1).leading_zeros(); // ceil(log2 c); 0 for c = 1
        let m: u64 = (1u64 << (31 + l)) / u64::from(c) + 1;
        let bail = self.ibail_ovf(a);
        let gx = self.ival(a, x, RAX)?;
        a.mov32_rr(RAX, gx);
        a.raw(&[0x3d]); // cmp eax, imm32
        a.raw(&i32::MIN.to_le_bytes());
        a.jcc(CC_E, bail);
        a.raw(&[0x89, 0xc2]); // mov edx, eax
        a.raw(&[0xc1, 0xfa, 31]); // sar edx, 31 (the sign mask)
        a.raw(&[0x31, 0xd0]); // xor eax, edx
        a.raw(&[0x29, 0xd0]); // sub eax, edx (|x|, zero-extended)
        a.raw(&[0x49, 0x89, 0xc3]); // mov r11, rax
        a.movabs_argreg(3, m as i64); // rcx = m
        a.raw(&[0x48, 0x0f, 0xaf, 0xc1]); // imul rax, rcx
        a.raw(&[0x48, 0xc1, 0xe8, (31 + l) as u8]); // shr rax, 31+l (the quotient)
        a.raw(&[0x69, 0xc0]); // imul eax, eax, imm32
        a.raw(&(c as i32).to_le_bytes());
        a.raw(&[0x44, 0x89, 0xd9]); // mov ecx, r11d
        a.raw(&[0x29, 0xc1]); // sub ecx, eax (|x| mod c)
        a.raw(&[0x31, 0xd1]); // xor ecx, edx
        a.raw(&[0x29, 0xd1]); // sub ecx, edx (the dividend's sign)
        let ok = a.new_label();
        a.test32_rr(RCX, RCX);
        a.jcc(CC_NE, ok);
        a.test32_rr(RDX, RDX);
        a.jcc(CC_NE, bail);
        a.bind(ok);
        self.iput(a, dst, RCX)
    }

    /// `if (x & y)` over int registers, `dst` only tested by the
    /// `JumpIfFalse` right after (nothing else names it, nothing else reaches
    /// the test: `fusable`): a `test` and the branch, `dst` never written (an
    /// int32 is falsy exactly when it is `0`). Whether it emitted the op so.
    #[allow(clippy::too_many_arguments)]
    fn iand_test(
        &mut self,
        a: &mut X64Assembler,
        labels: &[Label],
        pc: usize,
        ops: &[GenericOp],
        vop: u8,
        dst: u16,
        x: u16,
        y: u16,
    ) -> Option<bool> {
        if vop != crate::nbvm::VB_BIT_AND || !self.is_int(x) || !self.is_int(y) || !self.is_int(dst)
        {
            return Some(false);
        }
        let Some(f) = self.fusable(pc, ops, dst).filter(|f| !f.store) else {
            return Some(false);
        };
        if let (IK::Imm(_), IK::Imm(_)) = (self.ik(x)?, self.ik(y)?) {
            return Some(false);
        }
        match (self.ik_ld(a, x, RAX)?, self.ik_ld(a, y, RCX)?) {
            (IK::Gpr(g), IK::Imm(v)) | (IK::Imm(v), IK::Gpr(g)) => a.test32_imm(g, v),
            (IK::Gpr(g), IK::Gpr(h)) => a.test32_rr(g, h),
            _ => return None,
        }
        // (Through a `Not`: to the target when non-zero.)
        let cc = if f.neg { CC_NE } else { CC_E };
        self.jcc_to(a, labels, self.imode, cc, f.target);
        // (`fusable` without a store: nothing else reaches the test.)
        self.skip_fused(pc, f);
        Some(true)
    }

    /// `dst = x <bitwise op> y` with an int operand or result: double operands
    /// go through `ToInt32` (bailing on a non-Number or a huge value).
    fn ibitwise(&mut self, a: &mut X64Assembler, vop: u8, dst: u16, x: u16, y: u16) -> Option<()> {
        use crate::nbvm::{VB_BIT_AND, VB_BIT_OR, VB_BIT_XOR, VB_SHL, VB_SHR, VB_USHR};
        let reg = self.imode;
        let pc = self.cur_pc;
        let dbl_ops = !self.is_int(x) || !self.is_int(y);
        let bail = if dbl_ops { Some(self.ibail(a)) } else { None };
        for (r, g) in [(x, RAX), (y, RCX)] {
            if !self.is_int(r) {
                let bail = bail?;
                if self.ty(pc, r) != super::T_NUM {
                    self.num_guard(a, reg, r, bail);
                }
                self.to_int32(a, reg, g, r, bail);
            }
        }
        // (An `&`/`|`/`^` of an int register and a constant takes the
        // constant as an immediate.)
        let ext = match vop {
            VB_BIT_AND => Some(4),
            VB_BIT_OR => Some(1),
            VB_BIT_XOR => Some(6),
            _ => None,
        };
        if let Some(ext) = ext
            && !dbl_ops
        {
            let pair = match (self.ik(x)?, self.ik(y)?) {
                (IK::Imm(_), IK::Imm(_)) => None,
                (_, IK::Imm(v)) => Some((x, v)),
                (IK::Imm(v), _) => Some((y, v)),
                _ => None,
            };
            if let Some((r, v)) = pair {
                let g = self.ival(a, r, RAX)?;
                a.mov32_rr(RAX, g);
                a.alu32_imm(ext, RAX, v);
                return self.ibitwise_result(a, vop, dst);
            }
        }
        if self.is_int(y) {
            let gy = self.ival(a, y, RCX)?;
            a.mov32_rr(RCX, gy);
        }
        if self.is_int(x) {
            let gx = self.ival(a, x, RAX)?;
            a.mov32_rr(RAX, gx);
        }
        match vop {
            VB_BIT_AND => a.alu32_rr(0x21, RAX, RCX),
            VB_BIT_OR => a.alu32_rr(0x09, RAX, RCX),
            VB_BIT_XOR => a.alu32_rr(0x31, RAX, RCX),
            VB_SHL => a.shift32_cl(4, RAX),
            VB_SHR => a.shift32_cl(7, RAX),
            VB_USHR => a.shift32_cl(5, RAX),
            _ => return None,
        }
        self.ibitwise_result(a, vop, dst)
    }

    /// [`ibitwise`](Self::ibitwise)'s result, in `eax`, into `dst`.
    fn ibitwise_result(&mut self, a: &mut X64Assembler, vop: u8, dst: u16) -> Option<()> {
        use crate::nbvm::VB_USHR;
        let reg = self.imode;
        let pc = self.cur_pc;
        if self.is_int(dst) {
            self.iout(dst)?;
            // A `>>>` result only where it is held as its bits
            // (`IntPlan::wrap_bits`): else it is planned never (a uint32).
            // (Or where its count makes it exact, `IntPlan::exact_ushr`.)
            if vop == VB_USHR
                && !self.iplan()?.wrap_bits.contains(&pc)
                && !self.iplan()?.exact_ushr.contains(&pc)
            {
                return None;
            }
            self.iput(a, dst, RAX)?;
        } else {
            // A 32-bit op zero-extends: `>>>`'s uint32 is already in `rax`.
            if vop != VB_USHR {
                a.movsxd(RAX, RAX);
            }
            a.cvtsi2sd_x_gpr(0, RAX);
            self.store_xmm0(a, reg, dst);
        }
        Some(())
    }

    /// A comparison of two int registers (`cc` the signed condition for
    /// "true"), materialized as a boolean and/or fused into the following
    /// `JumpIfFalse`.
    #[allow(clippy::too_many_arguments)]
    fn icompare(
        &mut self,
        a: &mut X64Assembler,
        labels: &[Label],
        pc: usize,
        ops: &[GenericOp],
        cc: u8,
        dst: u16,
        x: u16,
        y: u16,
    ) -> Option<()> {
        let reg = self.imode;
        let fused = self.fusable(pc, ops, dst);
        let store = fused.is_none_or(|f| f.store);
        let gx = self.ival(a, x, RAX)?;
        match self.ik(y)? {
            IK::Imm(v) => a.cmp32_imm(gx, v),
            IK::Gpr(gy) => a.alu32_rr(0x39, gx, gy), // cmp x, y
            IK::Mem(d) => a.cmp32_mem(gx, d),
        }
        let mut branch_cc = cc ^ 1;
        if store {
            a.setcc_r8(cc, RDX); // dl = the 0/1
            a.raw(&[0x0f, 0xb6, 0xc2]); // movzx eax, dl
            a.movabs_r11(self.tag_false as i64);
            a.raw(&[0x4c, 0x01, 0xd8]); // add rax, r11
            self.store_gpr(a, reg, RAX, dst);
            a.raw(&[0x84, 0xd2]); // test dl, dl
            branch_cc = CC_E;
        }
        if let Some(f) = fused {
            // (Through a `Not`, never stored: to the target when true.)
            if f.neg {
                branch_cc = cc;
            }
            self.jcc_to(a, labels, reg, branch_cc, f.target);
            if !f.neg && self.is_target.get(pc + 1).copied().unwrap_or(true) {
                a.jmp(*labels.get(pc + 2)?);
            } else {
                // Nothing else reaches the test: fall through past it.
                self.skip_fused(pc, f);
            }
        }
        Some(())
    }

    /// Shadows the int registers op `pc` reads (before the double emitter runs
    /// it in the integer version).
    pub(super) fn ishadow_uses(&self, a: &mut X64Assembler, pc: usize) -> Option<()> {
        let bc = self.bc(pc)?;
        let mut d = Vec::new();
        let mut u = Vec::new();
        def_use(bc, &mut d, &mut u);
        u.sort_unstable();
        u.dedup();
        for r in u {
            if self.is_int(r) {
                self.ishadow(a, self.imode, r)?;
            }
        }
        Some(())
    }
}

/// A speculative region's **integer version** (`NumGen::emit_spec_int_op`).
///
/// A loop whose calls are all inlined (or are `Math` functions computed
/// inline) has a speculative double version ([`super::Spec`]) and a plain
/// copy; it may have an integer version as well, emitted as a pass of the
/// generic emitter (`NumGen::schedule`) between the two, so that the
/// generic emitter's ops (an inlined call's guard, a property read, ...)
/// run in it as in the speculative version, and the numeric ops as the
/// integer version runs them. It is the speculative version with int
/// registers:
///
/// - its entry checks are the speculative version's (global bindings,
///   field facts, guard misses, hoisted views, the registers assumed
///   Numbers) and the integer version's (each carried int register an
///   exact int32, else the speculative version runs), and its bail
///   counter is under [`super::INT_BAIL_LIMIT`];
/// - every bail (an int check failing, a slow path) leaves for the op in
///   the plain copy, the register file written back as the speculative
///   version's leaves do, and counts: the copy's back-edge enters the
///   region again, so the next iteration tries the integer version again,
///   until the count says it keeps failing;
/// - the generic emitter's code may use any caller-saved register, so the
///   int registers live at its op are saved in frame slots before it and
///   reloaded after it (a leave from inside it reloads them first).
///
/// A guard's miss, a property read's miss and every other slow path of the
/// generic emitter's leave for the plain copy as in the speculative version
/// (through the integer version's write-back).
impl NumGen {
    /// Whether the integer version being emitted is a speculative region's.
    pub(super) fn spec_int_mode(&self) -> bool {
        self.imode.is_some_and(|ri| self.regions[ri].spec_int)
    }

    /// For the `LoadGlobal` at `pc` of a speculative region's integer
    /// version: the int register carrying its binding around the back-edge
    /// (`plan_carried_globals`), when that one is a carried int register of
    /// the version. The version's entry loads it from the binding (exactly:
    /// an int32 or the double version runs), and the region's one store of
    /// the binding stores it, so wherever the read can be it equals the
    /// binding: the read is a copy of it.
    fn carried_int_global(&self, pc: usize) -> Option<u16> {
        if !self.spec_int_mode() {
            return None;
        }
        let ri = self.imode?;
        let Op::LoadGlobal { name, .. } = self.bc(pc)? else {
            return None;
        };
        let g = self.regions[ri].gcarry.iter().find(|g| &g.0 == name)?;
        if self.gslot.get(g.2).is_none_or(|&k| k == u32::MAX) {
            return None;
        }
        let p = self.iplan()?;
        p.slots
            .iter()
            .any(|s| s.reg == g.1 && s.carried)
            .then_some(g.1)
    }

    /// The int slots live at `pc`, as `(gpr, its save slot index)`.
    fn live_int_gprs(&self, pc: usize) -> Vec<(u8, usize)> {
        let Some(p) = self.iplan() else {
            return Vec::new();
        };
        let mut out: Vec<(u8, usize)> = p
            .slots
            .iter()
            .filter(|s| s.start <= pc && pc <= s.end)
            .filter_map(|s| GPRS.iter().position(|&g| g == s.gpr).map(|i| (s.gpr, i)))
            .collect();
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Before the generic emitter's code for op `pc` of a speculative
    /// region's integer version: the int GPRs live there into their save
    /// slots.
    fn int_save(&mut self, a: &mut X64Assembler, pc: usize) {
        for (g, i) in self.live_int_gprs(pc) {
            if self.int_clean.0 & (1 << i) == 0 {
                a.g_store(super::super::call_seq::r::RBP, self.int_save_disp(i), g);
                self.int_clean.0 |= 1 << i;
            }
        }
    }

    /// The int GPRs live at `pc` back from their save slots.
    pub(super) fn int_restore(&self, a: &mut X64Assembler, pc: usize) {
        for (g, i) in self.live_int_gprs(pc) {
            a.g_load(g, super::super::call_seq::r::RBP, self.int_save_disp(i));
        }
    }

    /// Where register `r` is for the generic emitter's code of region op
    /// `pc` (a compiled `for-in`'s key index, which its key read, its
    /// liveness check and its body's `o[k]` read: `super::super::forin`):
    /// in the integer version an int register's value is in its save slot
    /// (`int_save`, made before that code); else a promoted register is in
    /// its SSE register, any other in the register file. (Code that keeps
    /// `rsi` and `rdi`, `keeps_si_di`, saves nothing: then the GPR itself.)
    /// `None`: an int register held as a constant, which the code does not
    /// read.
    pub(crate) fn hidden_index(&self, pc: usize, r: u16) -> Option<super::super::forin::Idx> {
        use super::super::forin::Idx;
        if self.is_int(r) {
            let g = match self.ik(r)? {
                IK::Gpr(g) => g,
                IK::Mem(d) => return Some(Idx::Saved(d)),
                IK::Imm(_) => return None,
            };
            if !self.int_saved {
                return (g == RSI || g == RDI).then_some(Idx::Gpr(g));
            }
            let i = GPRS.iter().position(|&q| q == g)?;
            return Some(Idx::Saved(self.int_save_disp(i)));
        }
        Some(match self.result_xmm(pc, r) {
            Some(x) => Idx::Xmm(x),
            None => Idx::File,
        })
    }

    /// After the generic emitter's code for an op (any op): in a speculative
    /// region's integer version, the int registers it saved come back.
    pub(crate) fn after_generic_op(&mut self, a: &mut X64Assembler) {
        self.int_saved = false;
        self.int_math = None;
        // Save slots kept: those saved around this op's code, unless its
        // inline fast path (`int_skip`), which saves nothing, joins it;
        // never the GPR of the int register it writes.
        let saved = self.int_reload;
        if self.int_skip.is_some() {
            self.int_clean.0 = self.int_clean.1;
        }
        let pc = self.cur_pc;
        if let Some(op) = self.bc(pc) {
            let (mut d, mut u) = (Vec::new(), Vec::new());
            def_use(op, &mut d, &mut u);
            for r in d {
                if let Some(IK::Gpr(g)) = self.ik(r)
                    && let Some(i) = GPRS.iter().position(|&q| q == g)
                {
                    self.int_clean.0 &= !(1 << i);
                }
            }
        }
        self.int_clean_ok = saved && self.spec_int_mode();
        self.fcarry_after(a);
        if core::mem::take(&mut self.int_reload) {
            // (Those it reads last need not, nor its result.)
            let pc = self.cur_pc;
            let Some(p) = self.iplan() else {
                return;
            };
            let mut back: Vec<(u8, usize)> = p
                .slots
                .iter()
                .filter(|s| s.carried || (s.start < pc && s.end > pc))
                .filter_map(|s| GPRS.iter().position(|&g| g == s.gpr).map(|i| (s.gpr, i)))
                .collect();
            back.sort_unstable();
            back.dedup();
            for (g, i) in back {
                a.g_load(g, super::super::call_seq::r::RBP, self.int_save_disp(i));
            }
        }
        let skip = self.int_skip.take();
        self.after_generic_post(a);
        if let Some(l) = skip {
            a.bind(l);
        }
    }

    /// [`Self::after_generic_op`]'s check of an element or `.length` read's
    /// value.
    fn after_generic_post(&mut self, a: &mut X64Assembler) {
        if let Some(dst) = self.int_post.take() {
            // The value read, where the double version keeps it: an exact
            // int32 into its int register, else (the op done) the rest of
            // the iteration in the plain copy, from the next op.
            let pc = self.cur_pc;
            if !matches!(self.ik(dst), Some(IK::Gpr(_) | IK::Mem(_))) {
                return;
            }
            match self.xmm_of(self.imode, dst) {
                Some(x) => a.movq_gpr_x(RAX, x),
                None => a.load_rax(disp(dst)),
            }
            if self.iplan().is_some_and(|p| p.wrap_bits.contains(&pc)) {
                // Held as its `ToInt32` bits (`IntPlan::wrap_bits`): any
                // integral value below 2^63 in magnitude (the low 32 bits of
                // its exact truncation).
                a.movq_x_gpr(0, RAX);
                a.cvttsd2si_gpr_x(RCX, 0);
                a.cvtsi2sd_x_gpr(0, RCX);
                a.movq_gpr_x(R11, 0);
                a.cmp_rax_r11();
            } else {
                a.exact_i32_check(RCX);
            }
            let entry = a.new_label();
            let resume = a.new_label();
            a.jcc(CC_NE, entry);
            let _ = self.iput(a, dst, RCX);
            a.bind(resume);
            self.stub(
                entry,
                resume,
                self.imode,
                StubKind::PostBail {
                    pc: pc as u32 + 1,
                    dst,
                },
            );
            // (A Float64 element, say: the recompilation has `dst` a double.)
            self.demote_last(dst);
        }
    }

    /// A [`StubKind::PostBail`]: the int register `dst`'s value (its bits
    /// in `rax`) into the register file over the write-back at `pc`, which
    /// the plain copy then runs from. Counted as a bail.
    pub(super) fn emit_post_bail(&mut self, a: &mut X64Assembler, ri: usize, pc: usize, dst: u16) {
        a.movq_x_gpr(1, RAX);
        self.cur_pc = pc;
        self.iflush(a, ri);
        a.movsd_mem_x(disp(dst), 1);
        let r11 = super::super::call_seq::r::R11;
        a.g_movabs(r11, self.env.persist);
        a.g_inc_mem(r11, super::super::global_persist::counter_off(ri));
        if let Some(l) = self.copy_label(pc) {
            self.lazy_pend(a, pc, true);
            a.jmp(l);
        }
    }

    /// Whether op `pc`'s `Math.imul` is the integer version's
    /// ([`Self::emit_int_math`]).
    pub(crate) fn int_math_at(&self, pc: usize) -> bool {
        self.int_math == Some(pc)
    }

    /// In the generic emitter's code for op `pc`, once it has found the
    /// built-in: `Math.imul` of int registers into its int register
    /// (`true`), their values from their save slots (`int_save`), as the
    /// code before may have clobbered their GPRs. `false` anywhere else.
    pub(crate) fn emit_int_math(&self, a: &mut X64Assembler, pc: usize) -> Option<bool> {
        if self.int_math != Some(pc) {
            return Some(false);
        }
        let (x, y) = super::int_math_op(self.bc(pc)?)?;
        let (Some(Op::CallMethod { dst, .. }) | Some(Op::CallWithMethod { dst, .. })) = self.bc(pc)
        else {
            return None;
        };
        let (head, end) = {
            let r = &self.regions[self.imode?];
            (r.head, r.end)
        };
        for (r, g) in [(x, RAX), (y, RCX)] {
            if !self.is_int(r) {
                let bits = super::const_int32_bits(&|q| self.bc(q), head, end, r, pc)?;
                a.mov32_imm(g, bits);
                continue;
            }
            match self.ik(r)? {
                IK::Imm(v) => a.mov32_imm(g, v),
                IK::Gpr(s) if self.int_reload => {
                    let i = GPRS.iter().position(|&q| q == s)?;
                    a.g_load(g, super::super::call_seq::r::RBP, self.int_save_disp(i));
                }
                IK::Gpr(s) => a.mov32_rr(g, s),
                IK::Mem(d) => a.ld32(g, d),
            }
        }
        a.imul32_rr(RAX, RCX);
        self.iput(a, *dst, RAX)?;
        Some(true)
    }

    /// Int register `r` (live at the current pc) into the register file.
    pub(super) fn int_to_file(&self, a: &mut X64Assembler, r: u16) {
        if self.icvt(a, 0, r).is_some() {
            a.movsd_mem_x(disp(r), 0);
        }
    }

    /// At an integer version's entry: unless region `ri`'s entries have
    /// found a loop-carried value that is not an int32 [`ENTRY_RETRIES`]
    /// times in a row (then to `off`), on into the entry's checks; returns
    /// where a failed one goes — counted, then to `retry` (one iteration
    /// elsewhere, after which the region's integer version is tried
    /// again: a hash's `h = 0x811c9dc5` or a CRC's `c = 0xffffffff` is a
    /// uint32 before the first iteration and an int32 after it). Clobbers
    /// `r11`.
    fn entry_retry(&self, a: &mut X64Assembler, ri: usize, off: Label, retry: Label) -> Label {
        if self.env.persist == 0 || ri >= super::super::global_persist::COUNTERS {
            return off;
        }
        let efail = a.new_label();
        let r11 = super::super::call_seq::r::R11;
        let eoff = super::super::global_persist::entry_off(ri);
        let checks = a.new_label();
        a.g_movabs(r11, self.env.persist);
        a.g_mem(true, &[0x81], 7, r11, eoff);
        a.raw(&ENTRY_RETRIES.to_le_bytes());
        a.jcc(CC_AE, off);
        a.jmp(checks);
        a.bind(efail);
        a.g_movabs(r11, self.env.persist);
        a.g_inc_mem(r11, eoff);
        a.jmp(retry);
        a.bind(checks);
        efail
    }

    /// Past an integer version's entry checks: their failures in a row are
    /// none. Clobbers `r11`.
    fn entry_ok(&self, a: &mut X64Assembler, ri: usize) {
        if self.env.persist == 0 || ri >= super::super::global_persist::COUNTERS {
            return;
        }
        let r11 = super::super::call_seq::r::R11;
        a.g_movabs(r11, self.env.persist);
        a.g_store_imm32(r11, super::super::global_persist::entry_off(ri), 0);
    }

    /// At the back-edge of region `ri`'s double version (not a speculative
    /// region's, the register file not yet written back, the promoted
    /// registers live): when its integer version's entry failed for a
    /// value and asked to be tried again ([`Self::entry_retry`]), write the
    /// registers back and enter it; else fall through. Clobbers `rax`,
    /// `r11`.
    pub(super) fn retry_int(&self, a: &mut X64Assembler, ri: usize) -> Option<()> {
        let rg = &self.regions[ri];
        let entry = match rg.int.as_ref() {
            Some(p) if !rg.spec_int => p.entry?,
            _ => return Some(()),
        };
        if self.env.persist == 0 || ri >= super::super::global_persist::COUNTERS {
            return Some(());
        }
        let r11 = super::super::call_seq::r::R11;
        let eoff = super::super::global_persist::entry_off(ri);
        let stay = a.new_label();
        a.g_movabs(r11, self.env.persist);
        a.g_load(RAX, r11, eoff);
        a.test_rax_rax();
        a.jcc(CC_E, stay);
        a.g_mem(true, &[0x81], 7, r11, eoff);
        a.raw(&ENTRY_RETRIES.to_le_bytes());
        a.jcc(CC_AE, stay);
        self.flush(a, Some(ri));
        a.jmp(entry);
        a.bind(stay);
        Some(())
    }

    /// The entry of speculative region `ri`'s integer version (its header's
    /// label just bound): see the type's docs. A failed speculative check
    /// leaves for the plain copy, as the speculative version's own entry
    /// does; a failed int check (or a bail count at its limit) enters the
    /// speculative version.
    pub(super) fn spec_int_entry(&mut self, a: &mut X64Assembler, ri: usize) -> Option<()> {
        let head = self.regions[ri].head;
        self.cur_pc = head;
        let p = self.regions[ri].int.as_ref()?;
        let dbl = p.dbl_entry?;
        let copy = self.copy_label(head)?;
        let r11 = super::super::call_seq::r::R11;
        a.g_movabs(r11, self.env.persist);
        a.g_mem(
            true,
            &[0x81],
            7,
            r11,
            super::super::global_persist::counter_off(ri),
        );
        a.raw(&super::INT_BAIL_LIMIT.to_le_bytes());
        a.jcc(CC_AE, dbl);
        // A value not an int32 runs an iteration of the plain copy, whose
        // back-edge enters the region again (`entry_retry`).
        let efail = self.entry_retry(a, ri, dbl, copy);
        self.refresh_globals(a, ri, true);
        self.spec_globals_ready(a, ri, head)?;
        self.hoist_views(a, ri);
        // The carried bindings' registers start as the bindings, as in the
        // double version (their plan has them carried, so a bail into the
        // double version finds them equal to the bindings).
        self.load_carried_globals(a, ri);
        let p = self.regions[ri].int.as_ref()?;
        for s in &self.regions[ri].slots {
            if s.carried && p.ints.binary_search(&s.reg).is_err() {
                a.movsd_x_mem(s.xmm, disp(s.reg));
            }
        }
        let checks: Vec<u16> = self.regions[ri].spec.as_ref()?.checks.clone();
        for r in checks {
            if !self.is_int(r) {
                self.num_guard(a, Some(ri), r, copy);
            }
        }
        let p = self.regions[ri].int.as_ref()?;
        let carried: Vec<(u16, u8)> = p
            .slots
            .iter()
            .filter(|s| s.carried)
            .map(|s| (s.reg, s.gpr))
            .collect();
        let mut fails = Vec::new();
        for (r, gpr) in carried {
            a.load_rax(disp(r));
            a.exact_i32_check(RCX);
            let f = self.entry_fail(a, ri, r, efail, &mut fails);
            a.jcc(CC_NE, f);
            if gpr >= MEM_BASE {
                a.st32(self.home_disp(usize::from(gpr - MEM_BASE)), RCX);
            } else {
                a.mov32_rr(gpr, RCX);
            }
        }
        if !fails.is_empty() {
            let past = a.new_label();
            a.jmp(past);
            self.emit_entry_fails(a, ri, efail, &fails);
            a.bind(past);
        }
        self.entry_ok(a, ri);
        let inner = a.new_label();
        a.bind(inner);
        if let Some(p) = self.regions[ri].int.as_mut() {
            p.inner = Some(inner);
        }
        Some(())
    }

    /// In the entry into speculative nest `ri`'s fast version at its inner
    /// header `h` (`NumGen::emit_inner_entries`, the bindings validated and
    /// the views hoisted, the register file up to date): unless its bail
    /// count is at its limit, into its integer version there, with each
    /// int register live at `h` loaded from the register file (an exact
    /// int32, else on, into the double version, as without one), each
    /// other promoted one live there into its SSE register and checked as
    /// the double version's entry does (`checks`, a failure to `copy`).
    /// Emits nothing for a region without an integer version. Clobbers
    /// `rax`, `rcx`, `r11`, `xmm0` and the int GPRs.
    pub(super) fn int_inner_entry(
        &mut self,
        a: &mut X64Assembler,
        ri: usize,
        h: usize,
        checks: &[u16],
        copy: Label,
    ) {
        let rg = &self.regions[ri];
        let (Some(p), Some(Some(target))) = (rg.int.as_ref(), self.int_labels.get(h).copied())
        else {
            return;
        };
        if !rg.spec_int || self.env.persist == 0 || ri >= super::super::global_persist::COUNTERS {
            return;
        }
        let live: &[u16] = rg
            .live_inner
            .iter()
            .find(|x| x.0 == h)
            .map_or(&[], |x| x.1.as_slice());
        let ints: Vec<(u16, IK)> = p
            .slots
            .iter()
            .filter(|s| s.start <= h && h <= s.end && (s.carried || live.contains(&s.reg)))
            .map(|s| (s.reg, self.slot_ik(s)))
            .collect();
        let dbls: Vec<(u8, u16)> = rg
            .slots
            .iter()
            .filter(|s| s.start <= h && h <= s.end && p.ints.binary_search(&s.reg).is_err())
            .map(|s| (s.xmm, s.reg))
            .collect();
        let dbl = a.new_label();
        let r11 = super::super::call_seq::r::R11;
        a.g_movabs(r11, self.env.persist);
        a.g_mem(
            true,
            &[0x81],
            7,
            r11,
            super::super::global_persist::counter_off(ri),
        );
        a.raw(&super::INT_BAIL_LIMIT.to_le_bytes());
        a.jcc(CC_AE, dbl);
        let saved = self.imode.replace(ri);
        let pc = core::mem::replace(&mut self.cur_pc, h);
        for (r, k) in ints {
            a.load_rax(disp(r));
            a.exact_i32_check(RCX);
            a.jcc(CC_NE, dbl);
            match k {
                IK::Gpr(g) => a.mov32_rr(g, RCX),
                IK::Mem(d) => a.st32(d, RCX),
                IK::Imm(_) => {}
            }
        }
        for (x, r) in dbls {
            a.movsd_x_mem(x, disp(r));
        }
        for &r in checks {
            if !self.is_int(r) {
                self.num_guard(a, Some(ri), r, copy);
            }
        }
        self.imode = saved;
        self.cur_pc = pc;
        a.jmp(target);
        a.bind(dbl);
    }

    /// The entry at inner header `h` of numeric nest `ri` (a nest with no
    /// speculative version, on-stack replacement there) into its integer
    /// version there, the register file up to date: every int register
    /// live at `h` checked to be an exact int32 and loaded, every other
    /// promoted register live there loaded; any failed check falls through
    /// (the double version's entry follows). Its promoted global bindings
    /// are validated and loaded as at its own entry. Clobbers `rax`, `rcx`, `xmm`s and the int GPRs.
    pub(super) fn plain_int_inner_entry(&mut self, a: &mut X64Assembler, ri: usize, h: usize) {
        let rg = &self.regions[ri];
        let (Some(p), Some(Some(target))) = (rg.int.as_ref(), self.int_labels.get(h).copied())
        else {
            return;
        };
        if rg.spec.is_some() || rg.spec_int {
            return;
        }
        let Some(live) = rg.live_inner.iter().find(|x| x.0 == h).map(|x| &x.1) else {
            return;
        };
        let ints: Vec<(u16, IK)> = p
            .slots
            .iter()
            .filter(|s| s.start <= h && h <= s.end && live.contains(&s.reg))
            .map(|s| (s.reg, self.slot_ik(s)))
            .collect();
        let dbls: Vec<(u8, u16)> = rg
            .slots
            .iter()
            .filter(|s| s.start <= h && h <= s.end && p.ints.binary_search(&s.reg).is_err())
            .map(|s| (s.xmm, s.reg))
            .collect();
        let dbl = a.new_label();
        let saved = self.imode.replace(ri);
        let pc = core::mem::replace(&mut self.cur_pc, h);
        // (The bindings' validation first: its calls clobber the GPRs.)
        self.gvalidate(a, p, dbl);
        for (r, k) in ints {
            a.load_rax(disp(r));
            a.exact_i32_check(RCX);
            a.jcc(CC_NE, dbl);
            match k {
                IK::Gpr(g) => a.mov32_rr(g, RCX),
                IK::Mem(d) => a.st32(d, RCX),
                IK::Imm(_) => {}
            }
        }
        for (x, r) in dbls {
            a.movsd_x_mem(x, disp(r));
        }
        if let Some(p) = self.regions[ri].int.as_ref() {
            self.gload(a, p, dbl);
        }
        self.refresh_globals(a, ri, false);
        self.imode = saved;
        self.cur_pc = pc;
        a.jmp(target);
        a.bind(dbl);
    }

    /// Op `pc` of a speculative region's integer version (`labels`: the
    /// version's): as `emit_int_op` / the double emitter, else (`false`)
    /// the generic emitter's, the int registers saved around it.
    pub(super) fn emit_spec_int_op(
        &mut self,
        a: &mut X64Assembler,
        labels: &[Label],
        throw_exit: Label,
        pc: usize,
        ops: &[GenericOp],
        reg: Option<usize>,
    ) -> Option<bool> {
        // (The save slots kept from the op before, `int_clean`.)
        if !core::mem::take(&mut self.int_clean_ok)
            || self.is_target.get(pc).copied().unwrap_or(true)
        {
            self.int_clean.0 = 0;
        }
        self.int_clean.1 = self.int_clean.0;
        if self.emit_int_op(a, labels, throw_exit, pc, ops)? {
            return Some(true);
        }
        // An element or `.length` read of a hoisted view into an int
        // register: no walk (`ihoisted_get`), unless a later site relies on
        // the receiver cache this read fills (as the generic emitter's
        // own typed path does not either, `typed::relied_on`).
        match ops[pc] {
            GenericOp::GetElem { dst, obj, index }
                if self.is_int(dst)
                    && self.is_int(index)
                    && !self.rc_relied.get(pc).copied().unwrap_or(true)
                    && let Some(v) = self.hoisted_view(pc, obj) =>
            {
                self.ihoisted_get(a, dst, obj, index, v)?;
                return Some(true);
            }
            GenericOp::ArrayLen { dst, arr }
                if self.is_int(dst)
                    && !self.rc_relied.get(pc).copied().unwrap_or(true)
                    && let Some(v) = self.hoisted_view(pc, arr) =>
            {
                self.ihoisted_len(a, dst, arr, v)?;
                return Some(true);
            }
            _ => {}
        }
        // An element or `.length` read of a hoisted array into an int
        // register: inline from the hoisted slots, any miss (not the held
        // array, out of bounds, a hole, an element not an exact int32) the
        // generic emitter's code below, which runs the op from the start.
        // (Likewise, first, a hoisted view's element of any kind, read into
        // a register of the double version or written from an int or a
        // promoted one, `views`: its miss tries the hoisted array, if the
        // register has one too, then the generic code.)
        let view = |s: &Self, r: u16| {
            if s.rc_relied.get(pc).copied().unwrap_or(true) {
                return None;
            }
            s.hoisted_view(pc, r)
        };
        match ops[pc] {
            GenericOp::GetElem { dst, obj, index }
                if !self.is_int(dst)
                    && self.is_int(index)
                    && obj != index
                    && dst != index
                    && let Some(v) = view(self, obj) =>
            {
                self.ihoisted_view_get_any(a, dst, obj, index, v)?;
            }
            GenericOp::SetElem {
                obj, index, val, ..
            } if self.is_int(index)
                && obj != index
                && obj != val
                && let Some(v) = view(self, obj) =>
            {
                self.ihoisted_view_set(a, obj, index, val, v)?;
            }
            _ => {}
        }
        match ops[pc] {
            GenericOp::GetElem { dst, obj, index }
                if self.is_int(dst)
                    && self.is_int(index)
                    && obj != index
                    && let Some(s) = self.hoisted_array(pc, obj) =>
            {
                self.ihoisted_arr_get(a, dst, obj, index, s)?;
            }
            GenericOp::GetElem { dst, obj, index }
                if !self.is_int(dst)
                    && self.is_int(index)
                    && obj != index
                    && let Some(s) = self.hoisted_array(pc, obj) =>
            {
                self.ihoisted_arr_get_any(a, dst, obj, index, s)?;
            }
            GenericOp::ArrayLen { dst, arr }
                if self.is_int(dst)
                    && let Some(s) = self.hoisted_array(pc, arr) =>
            {
                self.ihoisted_arr_len(a, dst, arr, s)?;
            }
            GenericOp::SetElem {
                obj, index, val, ..
            } if self.is_int(index)
                && obj != index
                && obj != val
                && let Some(s) = self.hoisted_array(pc, obj) =>
            {
                self.ihoisted_arr_set(a, obj, index, val, s)?;
            }
            _ => {}
        }
        // `Math.imul` into an int register: the generic emitter's code finds
        // the built-in, then this emitter's computes (`emit_int_math`).
        match self.bc(pc) {
            // (Its built-in check clobbers no int GPR: `emit_got_ident`.)
            Some(Op::CallWithMethod { dst, .. }) if self.is_int(*dst) => {
                self.int_math = Some(pc);
                return Some(false);
            }
            Some(Op::CallMethod { dst, .. }) if self.is_int(*dst) => {
                self.int_save(a, pc);
                self.int_saved = true;
                self.int_reload = true;
                self.int_math = Some(pc);
                return Some(false);
            }
            _ => {}
        }
        // (An inlined call's guard reads only its callee, pinned; it stores
        // its int arguments into the register file itself: `guard_args`.)
        // (Nor does a compiled `for-in`'s key read or liveness check, whose
        // fast path reads the key index where this version keeps it,
        // `hidden_index`, and whose every other path leaves.)
        let forin_leaves =
            self.sa(pc) & super::SA_FORIN != 0 && self.in_spec(pc) && self.copy_label(pc).is_some();
        if !matches!(ops[pc], GenericOp::InlineGuard { .. }) && !forin_leaves {
            self.ishadow_uses(a, pc)?;
        }
        if self.emit_dbl(a, labels, throw_exit, pc, ops, reg)? {
            return Some(true);
        }
        if !super::generic_region_op(&ops[pc], self.bc(pc)?, self.sa(pc)) {
            return None;
        }
        // An element or `.length` read into an int register: the generic
        // emitter's read, then checked (`after_generic_op`).
        if let GenericOp::GetElem { dst, .. } | GenericOp::ArrayLen { dst, .. } = ops[pc]
            && self.is_int(dst)
        {
            self.int_post = Some(dst);
        }
        // (Code that keeps the int GPRs needs no save; an inlined call's
        // guard saves them itself, around what may clobber them:
        // `guard_keep`.)
        match ops[pc] {
            // (`rcx` / `rax` only.)
            GenericOp::InlineLeave | GenericOp::PlainPrims { .. } => {}
            GenericOp::InlineGuard { .. } => self.int_saved = true,
            // (A property op that keeps `rsi` and `rdi` when only they are
            // live: nothing to save, and its leaves find them in place.)
            // (Likewise a compiled `for-in`'s ops, `forin`.)
            GenericOp::GetProp { .. }
            | GenericOp::SetProp { .. }
            | GenericOp::ArrGet { .. }
            | GenericOp::GetElem { .. }
            | GenericOp::Interp { .. }
                if self.keeps_si_di(pc)
                    && self
                        .live_int_gprs(pc)
                        .iter()
                        .all(|&(g, _)| g == RSI || g == RDI) => {}
            _ => {
                self.int_save(a, pc);
                self.int_saved = true;
                self.int_reload = true;
            }
        }
        Some(false)
    }

    /// For the inlined call's guard at `pc`: the int GPRs it keeps (in a
    /// speculative region's integer version), as `(gpr, save slot)`.
    pub(crate) fn guard_keep(&self, pc: usize) -> Vec<(u8, i32)> {
        if !self.int_saved || !self.spec_int_mode() {
            return Vec::new();
        }
        self.live_int_gprs(pc)
            .into_iter()
            .map(|(g, i)| (g, self.int_save_disp(i)))
            .collect()
    }

    /// A bail of speculative region `ri`'s integer version at `pc`: the
    /// register file written back, counted, then op `pc` in the plain copy.
    pub(super) fn spec_int_bail(&self, a: &mut X64Assembler, ri: usize, pc: usize) {
        self.iflush(a, ri);
        let r11 = super::super::call_seq::r::R11;
        a.g_movabs(r11, self.env.persist);
        a.g_inc_mem(r11, super::super::global_persist::counter_off(ri));
        if let Some(l) = self.copy_label(pc) {
            self.lazy_pend(a, pc, true);
            a.jmp(l);
        }
    }
}

// --- GPR encoders (32-bit forms over any of the 16 registers) ---

impl X64Assembler {
    /// `opc r/m32(rm), r32(reg)` (`01` add, `29` sub, `21` and, `09` or, `31`
    /// xor, `39` cmp, `85` test, `89` mov).
    fn alu32_rr(&mut self, opc: u8, rm: u8, reg: u8) {
        let rex = 0x40 | ((reg >> 3) << 2) | (rm >> 3);
        if rex != 0x40 {
            self.code.push(rex);
        }
        self.code
            .extend_from_slice(&[opc, 0xc0 | ((reg & 7) << 3) | (rm & 7)]);
    }

    /// `mov r32(d), r32(s)` (nothing when `d == s`).
    fn mov32_rr(&mut self, d: u8, s: u8) {
        if d != s {
            self.alu32_rr(0x89, d, s);
        }
    }

    /// `test r32(x), r32(y)`.
    fn test32_rr(&mut self, x: u8, y: u8) {
        self.alu32_rr(0x85, x, y);
    }

    /// `imul r32(d), r32(s)`.
    fn imul32_rr(&mut self, d: u8, s: u8) {
        let rex = 0x40 | ((d >> 3) << 2) | (s >> 3);
        if rex != 0x40 {
            self.code.push(rex);
        }
        self.code
            .extend_from_slice(&[0x0f, 0xaf, 0xc0 | ((d & 7) << 3) | (s & 7)]);
    }

    /// `mov r32, imm32`.
    fn mov32_imm(&mut self, r: u8, v: i32) {
        if r >= 8 {
            self.code.push(0x41);
        }
        self.code.push(0xb8 + (r & 7));
        self.code.extend_from_slice(&v.to_le_bytes());
    }

    /// `op r32, imm32` (`81 /ext`: `4` and, `1` or, `6` xor).
    fn alu32_imm(&mut self, ext: u8, r: u8, v: i32) {
        if r >= 8 {
            self.code.push(0x41);
        }
        self.code
            .extend_from_slice(&[0x81, 0xc0 | (ext << 3) | (r & 7)]);
        self.code.extend_from_slice(&v.to_le_bytes());
    }

    /// `test r32, imm32`.
    fn test32_imm(&mut self, r: u8, v: i32) {
        if r >= 8 {
            self.code.push(0x41);
        }
        self.code.extend_from_slice(&[0xf7, 0xc0 | (r & 7)]);
        self.code.extend_from_slice(&v.to_le_bytes());
    }

    /// `cmp r32, imm32`.
    fn cmp32_imm(&mut self, r: u8, v: i32) {
        if r >= 8 {
            self.code.push(0x41);
        }
        self.code.extend_from_slice(&[0x81, 0xf8 | (r & 7)]);
        self.code.extend_from_slice(&v.to_le_bytes());
    }

    /// `add r32, imm8` (sign-extended).
    fn add32_imm8(&mut self, r: u8, v: i8) {
        if r >= 8 {
            self.code.push(0x41);
        }
        self.code
            .extend_from_slice(&[0x83, 0xc0 | (r & 7), v.to_le_bytes()[0]]);
    }

    /// `movsxd r64(d), r32(s)`.
    fn movsxd(&mut self, d: u8, s: u8) {
        self.code
            .extend_from_slice(&[0x48 | ((d >> 3) << 2) | (s >> 3), 0x63]);
        self.code.push(0xc0 | ((d & 7) << 3) | (s & 7));
    }

    /// `F7 /ext r32` (`2` not, `3` neg).
    fn grp3_32(&mut self, ext: u8, r: u8) {
        if r >= 8 {
            self.code.push(0x41);
        }
        self.code
            .extend_from_slice(&[0xf7, 0xc0 | (ext << 3) | (r & 7)]);
    }

    /// `D3 /ext r32, cl` (`4` shl, `5` shr, `7` sar). The count is taken mod 32,
    /// as JavaScript's shifts take it.
    fn shift32_cl(&mut self, ext: u8, r: u8) {
        if r >= 8 {
            self.code.push(0x41);
        }
        self.code
            .extend_from_slice(&[0xd3, 0xc0 | (ext << 3) | (r & 7)]);
    }

    /// `setcc r8` for `al`/`cl`/`dl`/`bl`.
    fn setcc_r8(&mut self, cc: u8, r: u8) {
        self.code
            .extend_from_slice(&[0x0f, 0x90 | cc, 0xc0 | (r & 7)]);
    }

    /// `xorpd x, x; cvtsi2sd xmm(x), r32`.
    fn cvtsi2sd32(&mut self, x: u8, gpr: u8) {
        self.sse_rr(0x66, 0x57, x, x);
        self.sse_rr_w(0xf2, false, 0x2a, x, gpr);
    }

    /// `cvttsd2si r32, xmm(x)`.
    fn cvttsd2si32(&mut self, gpr: u8, x: u8) {
        self.sse_rr_w(0xf2, false, 0x2c, gpr, x);
    }

    /// `mov r32, dword [rbp + disp]`.
    fn ld32(&mut self, r: u8, disp: i32) {
        self.g_mem(false, &[0x8b], r, super::super::call_seq::r::RBP, disp);
    }

    /// `mov dword [rbp + disp], r32`.
    fn st32(&mut self, disp: i32, r: u8) {
        self.g_mem(false, &[0x89], r, super::super::call_seq::r::RBP, disp);
    }

    /// `mov dword [rbp + disp], imm32`.
    fn st32_imm(&mut self, disp: i32, v: i32) {
        self.g_mem(false, &[0xc7], 0, super::super::call_seq::r::RBP, disp);
        self.code.extend_from_slice(&v.to_le_bytes());
    }

    /// `cmp r32, dword [rbp + disp]`.
    fn cmp32_mem(&mut self, r: u8, disp: i32) {
        self.g_mem(false, &[0x3b], r, super::super::call_seq::r::RBP, disp);
    }

    /// `movsxd r64, dword [rbp + disp]`.
    fn movsxd_mem(&mut self, r: u8, disp: i32) {
        self.g_mem(true, &[0x63], r, super::super::call_seq::r::RBP, disp);
    }

    /// `xorpd x, x; cvtsi2sd xmm(x), dword [rbp + disp]`.
    fn cvtsi2sd32_mem(&mut self, x: u8, disp: i32) {
        self.sse_rr(0x66, 0x57, x, x);
        self.code.push(0xf2);
        self.g_mem(
            false,
            &[0x0f, 0x2a],
            x,
            super::super::call_seq::r::RBP,
            disp,
        );
    }

    /// `cvttsd2si r32, [rbx + disp]`.
    fn cvttsd2si32_mem(&mut self, gpr: u8, disp: i32) {
        self.sse_rm(0xf2, false, 0x2c, gpr, disp);
    }

    /// `gpr32 = (int32) bits in rax`, flags `NE` unless `rax` is exactly that
    /// int32's Number (so `-0`, fractions, NaN, out-of-range values and
    /// non-Numbers all fail). Clobbers `xmm0`/`r11`.
    fn exact_i32_check(&mut self, gpr: u8) {
        self.movq_x_gpr(0, RAX);
        self.cvttsd2si32(gpr, 0);
        self.cvtsi2sd32(0, gpr);
        self.movq_gpr_x(R11, 0);
        self.cmp_rax_r11();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpr_encodings() {
        let mut a = X64Assembler::new();
        a.alu32_rr(0x01, RAX, 9); // add eax, r9d
        assert_eq!(a.code(), &[0x44, 0x01, 0xc8]);
        let mut a = X64Assembler::new();
        a.mov32_rr(10, RAX); // mov r10d, eax
        assert_eq!(a.code(), &[0x41, 0x89, 0xc2]);
        let mut a = X64Assembler::new();
        a.imul32_rr(RAX, 8); // imul eax, r8d
        assert_eq!(a.code(), &[0x41, 0x0f, 0xaf, 0xc0]);
        let mut a = X64Assembler::new();
        a.mov32_imm(9, 7); // mov r9d, 7
        assert_eq!(a.code(), &[0x41, 0xb9, 7, 0, 0, 0]);
        let mut a = X64Assembler::new();
        a.movsxd(RCX, 7); // movsxd rcx, edi
        assert_eq!(a.code(), &[0x48, 0x63, 0xcf]);
        let mut a = X64Assembler::new();
        a.movsxd(RAX, 10); // movsxd rax, r10d
        assert_eq!(a.code(), &[0x49, 0x63, 0xc2]);
        let mut a = X64Assembler::new();
        a.cvtsi2sd32(3, 8); // xorpd xmm3, xmm3; cvtsi2sd xmm3, r8d
        assert_eq!(
            a.code(),
            &[0x66, 0x0f, 0x57, 0xdb, 0xf2, 0x41, 0x0f, 0x2a, 0xd8]
        );
        let mut a = X64Assembler::new();
        a.cvttsd2si32_mem(6, 16); // cvttsd2si esi, [rbx+16]
        assert_eq!(a.code(), &[0xf2, 0x0f, 0x2c, 0xb3, 16, 0, 0, 0]);
        let mut a = X64Assembler::new();
        a.grp3_32(3, 9); // neg r9d
        assert_eq!(a.code(), &[0x41, 0xf7, 0xd9]);
        let mut a = X64Assembler::new();
        a.add32_imm8(RAX, -1); // add eax, -1
        assert_eq!(a.code(), &[0x83, 0xc0, 0xff]);
    }

    fn proto_of(src: &str, name: &str) -> crate::nbvm::FnProto {
        let program = crate::parser::Parser::parse_program(src).expect("parse");
        crate::nbvm::compile_program(&program)
            .expect("compile")
            .into_iter()
            .find(|p| p.name == name)
            .expect("function")
    }

    #[test]
    fn plan_picks_int_registers_and_wrapping_sums() {
        let p = proto_of(
            "function f(n){ let a = 0, b = 1.5, u = 0; for (let i = 0; i < n; i = i - -1) { a = (a + i * 3) | 0; b = b * 0.5 + 1; u = a >>> 1; } return a + b + u; }",
            "f",
        );
        let (ops, ..) = crate::jit::lower_nbvm_generic(&p).expect("generic");
        let regions = super::super::plan_regions(&ops, &p.ops, &[], p.n_regs);
        assert_eq!(regions.len(), 1, "{:#?}", p.ops);
        let ip = regions[0].int.as_ref().expect("an integer version");
        let dst_of = |pred: &dyn Fn(&Op) -> bool| {
            p.ops
                .iter()
                .find(|o| pred(o))
                .and_then(super::super::written_reg)
                .expect("op")
        };
        let mul3 = dst_of(&|o| matches!(o, Op::Mul { .. }));
        assert!(ip.ints.binary_search(&mul3).is_ok(), "i * 3 is an int");
        let fmul = p
            .ops
            .iter()
            .filter(|o| matches!(o, Op::Mul { .. }))
            .nth(1)
            .and_then(super::super::written_reg)
            .expect("b * 0.5");
        assert!(ip.ints.binary_search(&fmul).is_err(), "b * 0.5 is a double");
        let ushr = dst_of(&|o| matches!(o, Op::ValueBin { op, .. } if *op == crate::nbvm::VB_USHR));
        assert!(ip.ints.binary_search(&ushr).is_err(), ">>> makes a uint32");
        // `a + i * 3` only feeds `| 0`: it wraps instead of checking overflow.
        let add = p
            .ops
            .iter()
            .position(|o| matches!(o, Op::AddValue { .. }))
            .expect("+");
        assert!(ip.wraps.contains(&add));
        // `i - -1` (the counter) checks overflow.
        let step = p
            .ops
            .iter()
            .position(|o| matches!(o, Op::Sub { .. }))
            .expect("i - -1");
        assert!(!ip.wraps.contains(&step));
        let counter = super::super::written_reg(&p.ops[step]).expect("dst");
        assert!(
            ip.ints.binary_search(&counter).is_ok(),
            "the counter is an int"
        );
        // No two simultaneously live int registers share a GPR.
        for x in &ip.slots {
            for y in &ip.slots {
                if !core::ptr::eq(x, y) && x.gpr == y.gpr {
                    assert!(x.end <= y.start || y.end <= x.start);
                }
            }
        }
        // A loop over doubles only has no integer version.
        let q = proto_of(
            "function g(x){ let s = 0.5; for (let k = 0.5; k < 9.5; k = k + 1.5) { s = s * x; } return s; }",
            "g",
        );
        let (qops, ..) = crate::jit::lower_nbvm_generic(&q).expect("generic");
        let qr = super::super::plan_regions(&qops, &q.ops, &[], q.n_regs);
        assert!(qr.iter().all(|r| r.int.is_none()));
    }

    #[test]
    fn loop_nests_get_int_registers_in_gprs_and_frame_slots() {
        // A nest of element loops: one region, whose integer version has
        // both counters, the row offsets and the sum as int registers, more
        // of them live across the inner loop than there are GPRs.
        let p = proto_of(
            "function f(a, w, h){ let s = 0; for (let y = 0; y < h; y = y + 1) { const up = y * w, dn = (y + 1) * w, row = up + w; for (let x = 0; x < w; x = x + 1) { s = (s + a[up + x] + a[row + x] + a[dn + x] + (x & y)) | 0; } } return s; }",
            "f",
        );
        let (ops, ..) = crate::jit::lower_nbvm_generic(&p).expect("generic");
        let regions = super::super::plan_regions(&ops, &p.ops, &[], p.n_regs);
        let r = regions
            .iter()
            .find(|r| !r.loops.is_empty())
            .expect("a nest");
        assert!(r.spec_int, "a speculative nest's integer version");
        let ip = r.int.as_ref().expect("an integer version");
        // The counters (each `<`'s left operand) and the products.
        for o in &p.ops {
            match o {
                Op::Lt { a, .. } | Op::Compare { a, .. } => {
                    assert!(ip.ints.binary_search(a).is_ok(), "counter r{a} is an int");
                }
                Op::Mul { dst, .. } => {
                    assert!(
                        ip.ints.binary_search(dst).is_ok(),
                        "product r{dst} is an int"
                    );
                }
                _ => {}
            }
        }
        assert!(ip.n_mem > 0, "some int registers live in frame slots");
        // No two simultaneously live int registers share a GPR or a slot.
        for x in &ip.slots {
            for y in &ip.slots {
                if !core::ptr::eq(x, y) && x.gpr == y.gpr {
                    assert!(x.end <= y.start || y.end <= x.start);
                }
            }
        }
    }

    #[test]
    fn numeric_nests_are_one_region_with_an_integer_version() {
        // Counting loops with integer arithmetic alone (no element or
        // property access): one region over the nest, no speculative
        // version, an integer version keeping both counters and the sum
        // (more live values than GPRs: some in frame slots).
        let p = proto_of(
            "function f(n){ let s = 0, t = 1; for (let i = 0; i < n; i = i + 1) { const u = i * 3, v = i ^ 5; for (let j = 0; j < n; j = j + 1) { s = (s + i * j + (j ^ i) + u - v) | 0; t = (t + s) & 1023; } } return s + t; }",
            "f",
        );
        let (ops, ..) = crate::jit::lower_nbvm_generic(&p).expect("generic");
        let regions = super::super::plan_regions(&ops, &p.ops, &[], p.n_regs);
        assert_eq!(regions.len(), 1, "one region");
        let r = &regions[0];
        assert_eq!(r.loops.len(), 1, "over the nest");
        assert!(r.spec.is_none() && !r.spec_int);
        let ip = r.int.as_ref().expect("an integer version");
        for o in &p.ops {
            if let Op::Compare { a, .. } = o {
                assert!(ip.ints.binary_search(a).is_ok(), "counter r{a} is an int");
            }
        }
        assert!(ip.n_mem > 0, "some int registers live in frame slots");
        let (h, _) = r.loops[0];
        assert!(
            r.live_inner.iter().any(|x| x.0 == h),
            "its inner header's liveness"
        );
    }

    #[test]
    fn wrapping_needs_exact_intermediates() {
        // `h * 31 + i` stays below 2^53: both ops wrap. `k * k` (two int32s)
        // may not be exact as a double: it checks overflow (the sum over that
        // checked int32 still wraps).
        let p = proto_of(
            "function f(n){ let h = 0, k = 0; for (let i = 0; i < n; i = i - -1) { h = (h * 31 + i) | 0; k = (k * k + i) | 0; } return h + k; }",
            "f",
        );
        let (ops, ..) = crate::jit::lower_nbvm_generic(&p).expect("generic");
        let regions = super::super::plan_regions(&ops, &p.ops, &[], p.n_regs);
        let ip = regions[0].int.as_ref().expect("an integer version");
        let pcs = |pred: &dyn Fn(&Op) -> bool| -> Vec<usize> {
            (0..p.ops.len()).filter(|&pc| pred(&p.ops[pc])).collect()
        };
        let muls = pcs(&|o| matches!(o, Op::Mul { .. }));
        let adds = pcs(&|o| matches!(o, Op::AddValue { .. }));
        assert!(ip.wraps.contains(&muls[0]) && ip.wraps.contains(&adds[0]));
        assert!(!ip.wraps.contains(&muls[1]) && ip.wraps.contains(&adds[1]));
    }

    #[test]
    fn reciprocal_quotient_is_exact() {
        // The `imod_const` formula, checked at the edges of its range.
        let mut divisors: Vec<u32> = (1..=2000).collect();
        divisors.extend([65535, 65536, 65537, 1_000_000_007, 0x7fff_fffe, 0x7fff_ffff]);
        divisors.extend((0..31).map(|k| 1u32 << k));
        let mut ns: Vec<u64> = (0..5000).collect();
        ns.extend((0..31).flat_map(|k| [(1u64 << k) - 1, 1u64 << k, (1u64 << k) + 1]));
        ns.extend([0x7fff_ffff, 0x7fff_fffe, 0x7fff_0000, 1_999_999_999]);
        for &c in &divisors {
            let l = 32 - (c - 1).leading_zeros();
            let m: u64 = (1u64 << (31 + l)) / u64::from(c) + 1;
            assert!(m <= 1 << 32);
            for &n in ns
                .iter()
                .chain([u64::from(c) * 3 - 1, u64::from(c) * 7].iter())
            {
                if n >= 1 << 31 {
                    continue;
                }
                let q = (n * m) >> (31 + l);
                assert_eq!(q, n / u64::from(c), "n={n} c={c}");
            }
        }
    }

    /// The region of `name`'s one loop, planned.
    fn region_of(src: &str, name: &str) -> (crate::nbvm::FnProto, super::super::Region) {
        let p = proto_of(src, name);
        let (ops, ..) = crate::jit::lower_nbvm_generic(&p).expect("generic");
        let mut regions = super::super::plan_regions(&ops, &p.ops, &[], p.n_regs);
        assert_eq!(regions.len(), 1, "{:#?}", p.ops);
        (p, regions.remove(0))
    }

    #[test]
    fn speculative_plan_sums_int_elements() {
        // `s += a[i]` (an int index): the element and the sum are int
        // registers, the element checked when read.
        let (p, r) = region_of(
            "function f(a){ let s = 0; for (let i = 0; i < a.length; i = i + 1) s += a[i]; return s; }",
            "f",
        );
        assert!(r.spec_int, "a speculative integer version");
        let ip = r.int.as_ref().expect("an integer version");
        let get = p
            .ops
            .iter()
            .find_map(|o| match o {
                Op::GetKey { dst, .. } => Some(*dst),
                _ => None,
            })
            .expect("a[i]");
        assert!(ip.has_int(get), "the element is an int register");
        // Summed by a read by a String key: not (its generic code would
        // cost the int GPRs' saves for nothing).
        let (q, r2) = region_of(
            "function g(o){ let s = 0; for (let i = 0; i < 9; i = i + 1) s += o['k' + (i & 3)]; return s; }",
            "g",
        );
        let get2 = q
            .ops
            .iter()
            .find_map(|o| match o {
                Op::GetKey { dst, .. } => Some(*dst),
                _ => None,
            })
            .expect("o[ks]");
        assert!(r2.int.as_ref().is_none_or(|ip| !ip.has_int(get2)));
    }

    #[test]
    fn wide_plan_takes_index_and_identity_loops() {
        // A counter and its `& 3` index over a property read: an integer
        // version, though no element, guard, `for-in` or `Math.imul` op.
        let (_, r) = region_of(
            "function f(r, n){ let s = 0; for (let i = 0; i < n; i = i + 1) s += r[i & 3].v; return s; }",
            "f",
        );
        assert!(r.spec_int && r.int.is_some(), "an integer version");
        // An identity test of an element against a loop-invariant object
        // decides two distinct objects inline: an integer version too.
        let (q, r) = region_of(
            "function g(r, p, n){ let c = 0; for (let i = 0; i < n; i = i + 1) { if (r[i & 3].o === p) c = c + 1; } return c; }",
            "g",
        );
        assert!(
            r.int.is_some(),
            "an integer version for an identity loop: {:#?}",
            q.ops
        );
        // Elements compared by `===` are no int registers (two reads would
        // each look like one if the other is).
        let (p, r) = region_of(
            "function h(a, b, n){ let c = 0; for (let i = 0; i < n; i = i + 1) { if (a[i & 3] === b[i & 7]) c = c + 1 | 0; } return c; }",
            "h",
        );
        if let Some(ip) = r.int.as_ref() {
            for o in &p.ops {
                if let Op::GetKey { dst, .. } = o {
                    assert!(!ip.has_int(*dst), "an element compared by identity");
                }
            }
        }
    }

    #[test]
    fn index_copied_into_a_var_is_an_int_register() {
        // `var x = j & 63` (a register only element ops read): an int
        // register of the speculative version, its double location the
        // register file.
        let (p, r) = region_of(
            "function f(a, n){ var c = 0; for (var j = 0; j < n; j = j + 1) { var x = j & 63, y = (j * 7) & 63; var t = a[x]; a[x] = a[y]; a[y] = t; if (a[x].k < a[y].k) c = c + 1; } return c; }",
            "f",
        );
        let ip = r.int.as_ref().expect("an integer version");
        let key = p
            .ops
            .iter()
            .find_map(|o| match o {
                Op::GetKey { key, .. } => Some(*key),
                _ => None,
            })
            .expect("a[x]");
        assert!(ip.has_int(key), "the index register is an int register");
        assert!(
            !r.slots.iter().any(|s| s.reg == key),
            "and has no SSE register"
        );
    }

    #[test]
    fn const_i32_rejects_minus_zero_and_fractions() {
        assert_eq!(const_i32(NanBox::number(5.0)), Some(5));
        assert_eq!(const_i32(NanBox::number(-0.0)), None);
        assert_eq!(const_i32(NanBox::number(0.5)), None);
        assert_eq!(const_i32(NanBox::number(2_147_483_648.0)), None);
        assert_eq!(const_i32(NanBox::number(-2_147_483_648.0)), Some(i32::MIN));
        assert_eq!(const_i32(NanBox::number(f64::NAN)), None);
        assert_eq!(const_i32(NanBox::boolean(true)), None);
    }
}
