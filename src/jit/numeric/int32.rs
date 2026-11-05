//! **Int32 speculation** for promoted loop regions (`ROADMAP.md` §2.1).
//!
//! A region whose registers mostly hold small integers (loop counters, `| 0`
//! arithmetic, bit manipulation) gets a second, *integer version* of its body,
//! emitted after the double version. In it, the *int registers* (planned by
//! [`plan`]) live as int32 values in general-purpose registers
//! ([`GPRS`]) or, for a constant temporary, as an immediate; every other
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

/// The general-purpose registers int registers live in: caller-saved, and
/// never touched by the emitters' fast paths (`rax`/`rcx`/`rdx`/`r11` are
/// their scratch, `rbx`/`r15` are pinned).
const GPRS: [u8; 5] = [6, 7, 8, 9, 10];

/// How many times in a row an integer version's entry may find a
/// loop-carried value that is not an int32 and have it tried again after an
/// iteration elsewhere (`NumGen::entry_retry`).
const ENTRY_RETRIES: i32 = 8;

/// How many int GPRs there are (their save slots, `NumGen::int_save`).
pub(super) const N_GPRS: usize = GPRS.len();

/// x86 condition codes beyond the parent's.
const CC_O: u8 = 0x0;
const CC_S: u8 = 0x8;
const CC_L: u8 = 0xc;
const CC_GE: u8 = 0xd;
const CC_LE: u8 = 0xe;
const CC_G: u8 = 0xf;

const R11: u8 = 11;

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
    /// The promoted global bindings.
    pub(super) globals: Vec<PGlobal>,
    /// The integer version's entry (region entry jumps here first).
    pub(super) entry: Option<Label>,
    /// The double version's entry (its header loads).
    pub(super) dbl_entry: Option<Label>,
    /// Bound past the integer entry checks: in-version branches to the header.
    inner: Option<Label>,
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
    #[cfg(test)]
    pub(super) fn has_int(&self, r: u16) -> bool {
        self.ints.binary_search(&r).is_ok()
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
) -> Option<IntPlan> {
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
    let elem_reader_ok = |r: u16| {
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
                    Op::Move { dst, .. } => work.push(*dst),
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
    let starts_double = |r: u16| {
        bytecode.iter().enumerate().any(|(pc, op)| {
            outside(pc)
                && match op {
                    Op::LoadConst { dst, .. } => *dst == r && dbl_const(r),
                    Op::Move { dst, src } => *dst == r && dbl_const(*src),
                    _ => false,
                }
        })
    };
    let mut cset: Vec<u16> = info
        .iter()
        .map(|x| x.0)
        .filter(|&r| !(carried(r) && starts_double(r)))
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
                Op::GetKey { dst, .. } => (elems || spec) && elem_reader_ok(*dst),
                Op::ArrayLen { .. } => elems || spec,
                // A promoted global reads / writes its pseudo-register.
                Op::LoadGlobal { name, .. } => !spec && vreg_of(name).is_none_or(inn),
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
                        || ds.iter().any(|&pc| {
                            matches!(bytecode[pc], Op::LoadGlobal { .. } | Op::LoadConst { .. })
                                || elem_def(&bytecode[pc])
                        });
                    !speculative || useful(r, &mut du_d, &mut du_u)
                })
                .collect();
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
            cset.retain(|r| !failed.contains(r));
            continue;
        }
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
        if elems || spec {
            // A `>>>` result that is not held as bits is not an int32.
            let lost: Vec<u16> = body
                .iter()
                .enumerate()
                .filter_map(|(i, op)| match op {
                    Op::ValueBin { op, dst, .. }
                        if *op == VB_USHR && inn(*dst) && !wrap_bits.contains(&(head + i)) =>
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
            globals,
            entry: None,
            dbl_entry: None,
            inner: None,
        });
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
            .map(|s| IK::Gpr(s.gpr))
    }

    /// The GPR holding int register `r`, or `scratch` loaded with its constant.
    fn ival(&self, a: &mut X64Assembler, r: u16, scratch: u8) -> Option<u8> {
        Some(match self.ik(r)? {
            IK::Gpr(g) => g,
            IK::Imm(v) => {
                a.mov32_imm(scratch, v);
                scratch
            }
        })
    }

    /// `xmm(x) = (f64) int register r`.
    fn icvt(&self, a: &mut X64Assembler, x: u8, r: u16) -> Option<()> {
        match self.ik(r)? {
            IK::Gpr(g) => a.cvtsi2sd32(x, g),
            IK::Imm(v) => {
                a.movabs_rax(f64::from(v).to_bits() as i64);
                a.movq_x_gpr(x, RAX);
            }
        }
        Some(())
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
                a.cvtsi2sd32(0, s.gpr);
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
            if s.start <= pc && pc <= s.end && !self.is_greg(s.reg) {
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
        for s in &p.slots {
            if s.carried && !self.is_greg(s.reg) {
                a.load_rax(disp(s.reg));
                a.exact_i32_check(RCX);
                a.jcc(CC_NE, efail);
                a.mov32_rr(s.gpr, RCX);
            }
        }
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
                if let IK::Gpr(g) = self.ik(dst)? {
                    let Op::LoadConst { value, .. } = *bc else {
                        return None;
                    };
                    a.mov32_imm(g, const_i32(value)?);
                }
            }
            GenericOp::Move { dst, src } if ii(self, dst) => {
                // (A constant live range: the move of its constant, nothing.)
                let IK::Gpr(d) = self.ik(dst)? else {
                    return Some(true);
                };
                let s = self.ival(a, src, RAX)?;
                a.mov32_rr(d, s);
            }
            GenericOp::Move { dst, src } if ii(self, src) => {
                self.icvt(a, 0, src)?;
                self.store_xmm0(a, reg, dst);
            }
            GenericOp::ToNumeric { dst, src, .. } if ii(self, dst) => {
                let IK::Gpr(d) = self.ik(dst)? else {
                    return None;
                };
                let s = self.ival(a, src, RAX)?;
                a.mov32_rr(d, s);
            }
            GenericOp::Step { dst, src, dec, .. } if ii(self, dst) => {
                let IK::Gpr(d) = self.ik(dst)? else {
                    return None;
                };
                let s = self.ival(a, src, RAX)?;
                let t = if d != s { d } else { RAX };
                a.mov32_rr(t, s);
                a.add32_imm8(t, if dec { -1 } else { 1 });
                let bail = self.ibail(a);
                a.jcc(CC_O, bail);
                a.mov32_rr(d, t);
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
                match self.ik(cond)? {
                    IK::Gpr(g) => {
                        a.test32_rr(g, g);
                        self.jcc_to(a, labels, reg, CC_E, target);
                    }
                    IK::Imm(0) => self.jump(a, labels, reg, target),
                    IK::Imm(_) => {}
                }
            }
            GenericOp::CheckTdz { src, .. } if ii(self, src) => {}
            GenericOp::Jump { target } if target <= pc => {
                let ri = reg?;
                if pc != self.regions[ri].end || target != self.regions[ri].head {
                    return None;
                }
                let inner = self.regions[ri].int.as_ref()?.inner?;
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
                    let IK::Gpr(d) = self.ik(dst)? else {
                        return None;
                    };
                    let s = self.ival(a, src, RAX)?;
                    a.mov32_rr(RAX, s);
                    let bail = self.ibail(a);
                    a.test32_rr(RAX, RAX);
                    a.jcc(CC_E, bail); // -0
                    a.grp3_32(3, RAX); // neg eax
                    a.jcc(CC_O, bail); // -(-2^31)
                    a.mov32_rr(d, RAX);
                }
                Op::BitNot { dst, a: src } if ii(self, dst) => {
                    let IK::Gpr(d) = self.ik(dst)? else {
                        return None;
                    };
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
                    a.mov32_rr(d, RAX);
                }
                Op::LoadGlobal { dst, .. } if ii(self, dst) => {
                    let IK::Gpr(d) = self.ik(dst)? else {
                        return None;
                    };
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
                    a.mov32_rr(d, RCX);
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
        let IK::Gpr(d) = self.ik(dst)? else {
            return None;
        };
        let gx = self.ival(a, x, RAX)?;
        let gy = self.ival(a, y, RCX)?;
        let wrap = self.iplan().is_some_and(|p| p.wraps.contains(&self.cur_pc));
        // Computing in `d` itself is fine unless an operand lives there (a bail
        // must still see the operands).
        let t = if d != gx && d != gy { d } else { RAX };
        a.mov32_rr(t, gx);
        a.alu32_rr(opc, t, gy);
        if !wrap {
            let bail = self.ibail(a);
            a.jcc(CC_O, bail);
        }
        a.mov32_rr(d, t);
        Some(())
    }

    /// `dst = x * y` over int registers, bailing on overflow and on a zero
    /// product with a negative factor (`-0`).
    fn imul(&mut self, a: &mut X64Assembler, dst: u16, x: u16, y: u16) -> Option<()> {
        let IK::Gpr(d) = self.ik(dst)? else {
            return None;
        };
        let gx = self.ival(a, x, RDX)?;
        let gy = self.ival(a, y, RCX)?;
        if self.iplan().is_some_and(|p| p.wraps.contains(&self.cur_pc)) {
            // Wrapped (see `plan`): only the low 32 bits matter, `-0` included.
            let t = if d != gx && d != gy { d } else { RAX };
            a.mov32_rr(t, gx);
            a.imul32_rr(t, gy);
            a.mov32_rr(d, t);
            return Some(());
        }
        a.mov32_rr(RAX, gx);
        a.imul32_rr(RAX, gy);
        let bail = self.ibail(a);
        a.jcc(CC_O, bail);
        let ok = a.new_label();
        a.test32_rr(RAX, RAX);
        a.jcc(CC_NE, ok);
        a.mov32_rr(R11, gx);
        a.alu32_rr(0x09, R11, gy); // or r11d, y
        a.jcc(CC_S, bail);
        a.bind(ok);
        a.mov32_rr(d, RAX);
        Some(())
    }

    /// `dst = x % y` over int registers, bailing on a zero divisor (NaN) and on a
    /// zero remainder of a negative dividend (`-0`).
    fn imod(&mut self, a: &mut X64Assembler, dst: u16, x: u16, y: u16) -> Option<()> {
        let IK::Gpr(d) = self.ik(dst)? else {
            return None;
        };
        if let IK::Imm(c) = self.ik(y)?
            && c != 0
            && c != i32::MIN
        {
            return self.imod_const(a, d, x, c.unsigned_abs());
        }
        let gx = self.ival(a, x, RAX)?;
        a.movsxd(RAX, gx);
        let gy = self.ival(a, y, RCX)?;
        a.movsxd(RCX, gy);
        let bail = self.ibail(a);
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
        a.mov32_rr(d, RDX);
        Some(())
    }

    /// `gpr d = x % c` for a constant divisor `0 < c < 2^31` (the sign of a
    /// divisor does not matter: the remainder takes the dividend's), by a
    /// reciprocal multiplication instead of a division: for `0 <= n < 2^31`,
    /// `floor(n / c) = (n * m) >> (31 + l)` with `l = ceil(log2 c)` and
    /// `m = floor(2^(31+l) / c) + 1` (then `c <= 2^l` gives `2^(31+l) < m*c <=
    /// 2^(31+l) + 2^l`, the classic exactness condition; `m <= 2^32`, so the
    /// product stays below 2^63). Bails on `x = -2^31` (`|x|` out of range) and
    /// on a zero remainder of a negative dividend (`-0`).
    fn imod_const(&mut self, a: &mut X64Assembler, d: u8, x: u16, c: u32) -> Option<()> {
        let l = 32 - (c - 1).leading_zeros(); // ceil(log2 c); 0 for c = 1
        let m: u64 = (1u64 << (31 + l)) / u64::from(c) + 1;
        let bail = self.ibail(a);
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
        a.mov32_rr(d, RCX);
        Some(())
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
        let Some((target, false)) = self.fusable(pc, ops, dst) else {
            return Some(false);
        };
        match (self.ik(x)?, self.ik(y)?) {
            (IK::Gpr(g), IK::Imm(v)) | (IK::Imm(v), IK::Gpr(g)) => a.test32_imm(g, v),
            (IK::Gpr(g), IK::Gpr(h)) => a.test32_rr(g, h),
            (IK::Imm(_), IK::Imm(_)) => return Some(false),
        }
        self.jcc_to(a, labels, self.imode, CC_E, target);
        // (`fusable` without a store: nothing else reaches the test.)
        if let Some(k) = self.skip.get_mut(pc + 1) {
            *k = true;
        }
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
            let (g, v) = match (self.ik(x)?, self.ik(y)?) {
                (IK::Gpr(g), IK::Imm(v)) | (IK::Imm(v), IK::Gpr(g)) => (g, v),
                _ => (u8::MAX, 0),
            };
            if g != u8::MAX {
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
            let IK::Gpr(d) = self.ik(dst)? else {
                return None;
            };
            // A `>>>` result only where it is held as its bits
            // (`IntPlan::wrap_bits`): else it is planned never (a uint32).
            if vop == VB_USHR && !self.iplan()?.wrap_bits.contains(&pc) {
                return None;
            }
            a.mov32_rr(d, RAX);
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
        let store = fused.is_none_or(|(_, s)| s);
        let gx = self.ival(a, x, RAX)?;
        match self.ik(y)? {
            IK::Imm(v) => a.cmp32_imm(gx, v),
            IK::Gpr(gy) => a.alu32_rr(0x39, gx, gy), // cmp x, y
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
        if let Some((target, _)) = fused {
            self.jcc_to(a, labels, reg, branch_cc, target);
            if self.is_target.get(pc + 1).copied().unwrap_or(true) {
                a.jmp(*labels.get(pc + 2)?);
            } else if let Some(k) = self.skip.get_mut(pc + 1) {
                // Nothing else reaches the `JumpIfFalse`: fall through past it.
                *k = true;
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
    fn spec_int_mode(&self) -> bool {
        self.imode.is_some_and(|ri| self.regions[ri].spec_int)
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
    fn int_save(&self, a: &mut X64Assembler, pc: usize) {
        for (g, i) in self.live_int_gprs(pc) {
            a.g_store(super::super::call_seq::r::RBP, self.int_save_disp(i), g);
        }
    }

    /// The int GPRs live at `pc` back from their save slots.
    pub(super) fn int_restore(&self, a: &mut X64Assembler, pc: usize) {
        for (g, i) in self.live_int_gprs(pc) {
            a.g_load(g, super::super::call_seq::r::RBP, self.int_save_disp(i));
        }
    }

    /// After the generic emitter's code for an op (any op): in a speculative
    /// region's integer version, the int registers it saved come back.
    pub(crate) fn after_generic_op(&mut self, a: &mut X64Assembler) {
        self.int_saved = false;
        self.int_math = None;
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
        if let Some(dst) = self.int_post.take() {
            // The value read, where the double version keeps it: an exact
            // int32 into its int register, else (the op done) the rest of
            // the iteration in the plain copy, from the next op.
            let pc = self.cur_pc;
            let Some(IK::Gpr(d)) = self.ik(dst) else {
                return;
            };
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
            a.mov32_rr(d, RCX);
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
            }
        }
        a.imul32_rr(RAX, RCX);
        let IK::Gpr(d) = self.ik(*dst)? else {
            return None;
        };
        a.mov32_rr(d, RAX);
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
        for s in &p.slots {
            if s.carried {
                a.load_rax(disp(s.reg));
                a.exact_i32_check(RCX);
                a.jcc(CC_NE, efail);
                a.mov32_rr(s.gpr, RCX);
            }
        }
        self.entry_ok(a, ri);
        let inner = a.new_label();
        a.bind(inner);
        if let Some(p) = self.regions[ri].int.as_mut() {
            p.inner = Some(inner);
        }
        Some(())
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
        if !matches!(ops[pc], GenericOp::InlineGuard { .. }) {
            self.ishadow_uses(a, pc)?;
        }
        if self.emit_dbl(a, labels, throw_exit, pc, ops, reg)? {
            return Some(true);
        }
        if !super::generic_region_op(&ops[pc], self.bc(pc)?) {
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
