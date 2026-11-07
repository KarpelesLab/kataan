//! The generic tier's **receiver cache**: which property ops of a body keep
//! the heap slot address of their receiver in a frame slot for a later op on
//! the same object, so it skips the arena walk (handle check, bounds and
//! the family's cell-tag guards, `jit::handle`).
//!
//! Soundness is dynamic and does not depend on this plan. A cache *way* is a
//! pair of frame slots: the receiver's handle bits (the key) and the address
//! of its heap slot as a full walk proved it. The frame
//! prologue and every `call` the body makes empty all the keys
//! (`X64Assembler::clear_recv_cache`), and only Rust code (always behind a
//! call) can allocate, collect, move or free a heap slot. So a key equal to a
//! register's value means the same object, walked with no Rust code run
//! since: its slot address is still right. A key compare that fails just
//! walks again (out of line) and refills the way.
//!
//! The plan only decides where that is worth it: a site that a later site on
//! the same value can reuse *fills* a way after its walk, and a site that an
//! earlier one may have filled *checks* the way first. The estimate follows
//! the op stream: a call on an op's fast path, or a branch target, ends what
//! is known; a `Move` makes its destination the same value as its source.

use alloc::vec;
use alloc::vec::Vec;

use super::GenericOp;
use crate::nbvm::Op;

/// Per register of a body: whether it holds the objects of a **walk**, so a
/// way filled with it predicts the next one's heap slot
/// (`X64Assembler::p_fill_way`): a register that some op steps to a new
/// object of a sequence (an element read, `GetKey`; an iterator's value,
/// `IterNext`; a link of its own object, `r = r.link`, maybe through a
/// temporary; a copy of such a register) and that nothing else defines but
/// copies and constants (its initial value). Those visit a new object each
/// time, often laid out in order. Other receivers (a method's `this`, a few
/// objects taking turns, a register reused for another value) would mostly
/// miss the prediction, and pay its instructions for nothing. (Only a guess
/// of the profit: every prediction is checked.)
pub(super) fn walk_regs(bytecode: &[Op]) -> Vec<bool> {
    let (mut defs, mut uses) = (Vec::new(), Vec::new());
    let mut n = 0;
    for op in bytecode {
        super::numeric::def_use(op, &mut defs, &mut uses);
        if let Op::GetKey { dst, .. } | Op::GetProp { dst, .. } = *op {
            defs.push(dst);
        }
        for &r in defs.iter().chain(&uses) {
            n = n.max(usize::from(r) + 1);
        }
    }
    // `t = r.link` reads (for `r = t`), and the registers defined otherwise
    // than by a copy or a constant.
    let mut link_of: Vec<(u16, u16)> = Vec::new();
    let mut other = vec![false; n];
    let mut walk = vec![false; n];
    for op in bytecode {
        super::numeric::def_use(op, &mut defs, &mut uses);
        match *op {
            Op::GetKey { dst, .. } | Op::IterNext { dst, .. } => walk[usize::from(dst)] = true,
            Op::GetProp { dst, obj, .. } if dst == obj => walk[usize::from(dst)] = true,
            Op::GetProp { dst, obj, .. } => link_of.push((dst, obj)),
            _ => {}
        }
        let plain = matches!(op, Op::Move { .. } | Op::LoadConst { .. });
        for &d in &defs {
            let d = usize::from(d);
            // (An `IterNext` also defines its `done` and cursor.)
            let step = matches!(*op, Op::IterNext { dst, .. } if usize::from(dst) == d);
            if !plain && !step && !walk[d] {
                other[d] = true;
            }
        }
    }
    for op in bytecode {
        if let Op::Move { dst, src } = *op
            && link_of.contains(&(src, dst))
        {
            walk[usize::from(dst)] = true;
        }
    }
    // A copy of a walk register is one.
    loop {
        let mut changed = false;
        for op in bytecode {
            if let Op::Move { dst, src } = *op
                && walk[usize::from(src)]
                && !walk[usize::from(dst)]
            {
                walk[usize::from(dst)] = true;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    for (w, o) in walk.iter_mut().zip(&other) {
        *w &= !o;
    }
    walk
}

/// The most global binding addresses a body caches (each costs a frame slot
/// and a store in the prologue).
const MAX_GLOBAL_SLOTS: usize = 8;

/// A body's global binding cache plan: the slots it uses and, per op, the slot
/// of a `LoadGlobal` / `StoreGlobal` that uses one.
pub(super) struct GlobalPlan {
    /// How many slots the body uses.
    pub(super) slots: usize,
    /// Per op, its slot.
    pub(super) site: Vec<Option<usize>>,
}

/// Plans the global binding cache: the body's global reads and writes,
/// grouped by `(name, read | write)`; the (up to [`MAX_GLOBAL_SLOTS`]) most
/// frequent pairs get a slot each, those outside promoted loops (`in_region`)
/// first. A promoted loop's own accesses keep their own addresses
/// (`numeric`), but the loop's plain copy (`NumGen::set_copy`) runs them as
/// any other op of the body: through the cache, or a helper call each. A slot keeps the address an access validated across calls and
/// loop iterations, for as long as the global binding epoch it was validated
/// under is current (see `compile_generic`), so every access of a pair after
/// the first in an activation reuses it. A write's validated address would
/// also serve reads, but not the reverse, so the two kinds of one name take
/// separate slots.
pub(super) fn plan_globals(
    ops: &[GenericOp],
    bytecode: &[Op],
    in_region: impl Fn(usize) -> bool,
) -> GlobalPlan {
    let n = ops.len();
    let mut key = vec![None; n];
    for (pc, op) in ops.iter().enumerate() {
        if matches!(op, GenericOp::Interp { .. }) {
            key[pc] = match bytecode.get(pc) {
                Some(Op::LoadGlobal { name, .. }) => Some((name.as_str(), false)),
                Some(Op::StoreGlobal { name, .. }) => Some((name.as_str(), true)),
                _ => None,
            };
        }
    }
    // Per pair: its accesses outside promoted loops, and all of them.
    type Counts = (usize, usize);
    let mut groups: Vec<((&str, bool), Counts)> = Vec::new();
    for (pc, k) in key.iter().enumerate() {
        let Some(k) = *k else { continue };
        let outside = usize::from(!in_region(pc));
        match groups.iter_mut().find(|g| g.0 == k) {
            Some(g) => {
                g.1.0 += outside;
                g.1.1 += 1;
            }
            None => groups.push((k, (outside, 1))),
        }
    }
    groups.sort_by_key(|x| core::cmp::Reverse(x.1));
    groups.truncate(MAX_GLOBAL_SLOTS);
    let site = key
        .iter()
        .map(|k| {
            let k = (*k)?;
            groups.iter().position(|g| g.0 == k)
        })
        .collect();
    GlobalPlan {
        slots: groups.len(),
        site,
    }
}

/// Which `GenericOp::InitProp`s define a property of a **fresh literal**: an
/// object an `Op::NewObject` of this body made, in a register that nothing
/// but the object's own plain `InitProp`s has named since (with no branch
/// target in between, and fewer than [`crate::object::INLINE_SLOTS`] such
/// adds so far). No other code can have seen such an object (nothing read
/// the register, so the handle went nowhere): it is still the ordinary,
/// extensible, unfrozen, unflagged object `NewObject` made, live (the
/// register roots it), with at most that many properties — see
/// `X64Assembler::p_init_prop_fresh`. An `InitProp`'s value escapes into its
/// object, so a fresh object stored into another stops being fresh.
pub(super) fn fresh_inits(
    ops: &[GenericOp],
    bytecode: &[Op],
    osr_entries: &[usize],
    catch_targets: &[usize],
) -> Vec<bool> {
    let n = ops.len();
    let target = branch_targets(ops, osr_entries, catch_targets);
    let mut out = vec![false; n];
    // (register, plain adds so far)
    let live = core::cell::RefCell::new(Vec::<(u16, usize)>::new());
    for (pc, op) in ops.iter().enumerate() {
        if target[pc] {
            live.borrow_mut().clear();
        }
        if let GenericOp::InitProp {
            obj, val, plain, ..
        } = *op
        {
            let mut l = live.borrow_mut();
            if let Some(i) = l.iter().position(|e| e.0 == obj) {
                if plain && val != obj && l[i].1 < crate::object::INLINE_SLOTS {
                    out[pc] = true;
                    l[i].1 += 1;
                } else {
                    l.swap_remove(i);
                }
            }
            l.retain(|e| e.0 != val);
            continue;
        }
        let Some(b) = bytecode.get(pc) else {
            live.borrow_mut().clear();
            continue;
        };
        // Any other op naming a fresh object's register may let it escape
        // (or overwrites it).
        crate::bytecode::op_for_each_reg(b, &|r| live.borrow_mut().retain(|e| e.0 != r));
        if let Op::NewObject { dst } = b {
            live.borrow_mut().push((*dst, 0));
        }
    }
    out
}

/// One `SetProp` of a **fresh-object add chain** ([`fresh_adds`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct FreshAdd {
    /// The inlined `new`'s guard that allocated the receiver (whose
    /// receiver-cache way it filled).
    pub(super) guard: usize,
    /// How many adds of the chain come before this one.
    pub(super) k: u32,
    /// The chain's previous add, if any.
    pub(super) prev: Option<usize>,
}

/// The `SetProp`s that may add to the object an inlined `new`'s guard (or a
/// `super(…)`'s, into a base constructor) has just allocated (`this`, see
/// `call_seq::emit_inline_new_this`) while nothing but the chain's own
/// earlier `SetProp`s on it can have touched it: from the guard on, every op
/// naming `this` — or a register a `Move` copied it to (an inlined
/// constructor's result, the caller's `this` after `super(…)`) — is a
/// `SetProp` on it storing another register, a copy, a `CheckTdz` of it or
/// an `InitFields` on it (whose fast path does nothing; its slow path's
/// call empties the way); an `InlineLeave` or a super constructor's lookup
/// may come between. A branch target ends the chain, except the join after
/// an inlined call whose only other way in is falling through its fallback
/// call op (every path of which makes a call, which empties every way, and
/// the ops from there to the next site fill none): the chain goes on with
/// what the one branch in knew. The plan is static only; what holds at run
/// time is the emitter's to check (see `X64Assembler::p_add_fresh`): that
/// the guard's way still holds the object (no call — so no Rust code —
/// since the allocation, and every earlier add of the chain took its fast
/// path, whose other paths empty the way).
pub(super) fn fresh_adds(
    ops: &[GenericOp],
    bytecode: &[Op],
    osr_entries: &[usize],
    catch_targets: &[usize],
) -> Vec<Option<FreshAdd>> {
    /// (guard, the registers holding `this`, adds so far, last add)
    type Chain = (usize, Vec<u16>, u32, Option<usize>);
    let n = ops.len();
    let target = branch_targets(ops, osr_entries, catch_targets);
    let mut into = vec![0u32; n + 1];
    for op in ops {
        if let GenericOp::Jump { target: t } | GenericOp::JumpIfFalse { target: t, .. } = *op
            && t <= n
        {
            into[t] += 1;
        }
    }
    // The fallback ops of the body's inlined calls (each a call on every
    // path: a `new`, or a `super(…)` through the interpreter's op).
    let mut fallback = vec![false; n];
    for op in ops {
        if let GenericOp::InlineGuard { call, .. } = *op
            && let Some(c) = fallback.get_mut(call as usize)
        {
            *c = match ops[call as usize] {
                GenericOp::Construct { .. } => true,
                GenericOp::Interp { .. } => is_super_call(bytecode, call as usize),
                _ => false,
            };
        }
    }
    let entry = |i: usize| osr_entries.contains(&i) || catch_targets.contains(&i);
    let mut out = vec![None; n];
    // Per join, the chain at its one forward branch.
    let mut snap: Vec<Option<Chain>> = vec![None; n + 1];
    let mut cur: Option<Chain> = None;
    for (i, op) in ops.iter().enumerate() {
        if target[i] {
            let joined = snap[i].take();
            cur = match joined {
                Some(c) if into[i] == 1 && i > 0 && fallback[i - 1] && !entry(i) => Some(c),
                _ => None,
            };
        }
        if let GenericOp::InlineGuard {
            call, caps, n_caps, ..
        } = *op
        {
            let allocates = match ops.get(call as usize) {
                Some(GenericOp::Construct { .. }) => true,
                Some(GenericOp::Interp { .. }) => is_super_call(bytecode, call as usize),
                _ => false,
            };
            cur = if allocates {
                n_caps
                    .checked_sub(1)
                    .and_then(|k| caps.checked_add(k))
                    .map(|t| (i, vec![t], 0, None))
            } else {
                None
            };
            continue;
        }
        let Some((g, mut ts, k, prev)) = cur.take() else {
            continue;
        };
        let is_t = |r: u16| ts.contains(&r);
        match (*op, bytecode.get(i)) {
            (GenericOp::SetProp { obj, val, .. }, _) if is_t(obj) && !is_t(val) => {
                out[i] = Some(FreshAdd { guard: g, k, prev });
                cur = Some((g, ts, k + 1, Some(i)));
                continue;
            }
            (GenericOp::Move { dst, src }, _) => {
                if is_t(src) {
                    if !ts.contains(&dst) {
                        ts.push(dst);
                    }
                } else {
                    ts.retain(|&x| x != dst);
                }
            }
            (GenericOp::CheckTdz { .. } | GenericOp::InlineLeave, _) => {}
            (GenericOp::Jump { target: t }, _) if t > i && t <= n => {
                snap[t] = Some((g, ts, k, prev));
                continue;
            }
            (GenericOp::Interp { .. }, Some(Op::InitFields { ctor, .. })) if !is_t(*ctor) => {}
            (GenericOp::Interp { .. }, Some(Op::SuperCtor { dst, f })) if !is_t(*f) => {
                let d = *dst;
                ts.retain(|&x| x != d);
            }
            (GenericOp::Interp { .. } | GenericOp::PushHandler { .. }, _) => continue,
            (_, Some(b)) => {
                let names_t = core::cell::Cell::new(false);
                crate::bytecode::op_for_each_reg(b, &|r| {
                    if ts.contains(&r) {
                        names_t.set(true);
                    }
                });
                if names_t.get() {
                    continue;
                }
            }
            (_, None) => continue,
        }
        if let GenericOp::JumpIfFalse { target: t, .. } = *op
            && t <= n
        {
            // (Its target is no join this chain goes on at.)
            snap[t] = None;
        }
        if ts.is_empty() {
            continue;
        }
        cur = Some((g, ts, k, prev));
    }
    out
}

/// Which `JumpIfFalse`s an inlined call's guard just before takes over
/// (`call_seq::InlineEmit::fused_miss`): the guard's own test of its result,
/// branching forward, that nothing else reaches (no branch to it, not an
/// entry) and that is no receiver-cache site (`plain(i)`: its truthiness
/// test fills no way). Its guard then branches on a miss itself and falls
/// into the inlined body on a hit, so the test is emitted as nothing.
pub(super) fn fused_guard_tests(
    ops: &[GenericOp],
    osr_entries: &[usize],
    catch_targets: &[usize],
    plain: impl Fn(usize) -> bool,
) -> Vec<bool> {
    let n = ops.len();
    let target = branch_targets(ops, osr_entries, catch_targets);
    let mut out = vec![false; n];
    for i in 1..n {
        if let (GenericOp::InlineGuard { dst, .. }, GenericOp::JumpIfFalse { cond, target: t }) =
            (&ops[i - 1], &ops[i])
            && cond == dst
            && *t > i
            && *t <= n
            && !target[i]
            && plain(i)
        {
            out[i] = true;
        }
    }
    out
}

/// A `Not` the `JumpIfFalse` right after tests (`t = !x; if (!t) goto L`,
/// i.e. `if (!x) …`): the not's emitter branches on `x`'s truthiness itself
/// and the test is emitted as nothing. Only a forward branch that nothing
/// else reaches (no branch to it, not an entry), on a register other than
/// `x`. [`NotBranch::dead`]: no other op names `t`, so the boolean need not
/// be stored at all.
#[derive(Clone, Copy)]
pub(super) struct NotBranch {
    /// Where the test branches when `x` is truthy.
    pub(super) target: usize,
    /// Nothing reads the not's result but the test.
    pub(super) dead: bool,
}

/// Per op, its [`NotBranch`] fusion (on the `Not`; the `JumpIfFalse`
/// after it is marked in the second vector).
pub(super) fn not_branches(
    ops: &[GenericOp],
    bytecode: &[Op],
    osr_entries: &[usize],
    catch_targets: &[usize],
    eligible: impl Fn(usize) -> bool,
) -> (Vec<Option<NotBranch>>, Vec<bool>) {
    let n = ops.len();
    let target = branch_targets(ops, osr_entries, catch_targets);
    let mut nots = vec![None; n];
    let mut tests = vec![false; n];
    for i in 1..n {
        if let (GenericOp::Not { dst, a }, GenericOp::JumpIfFalse { cond, target: t }) =
            (ops[i - 1], ops[i])
            && cond == dst
            && dst != a
            && t > i
            && t <= n
            && !target[i]
            && eligible(i - 1)
            && eligible(i)
        {
            let dead = bytecode
                .iter()
                .enumerate()
                .all(|(j, o)| j == i - 1 || j == i || !crate::bytecode::op_mentions_reg(o, dst));
            nots[i - 1] = Some(NotBranch { target: t, dead });
            tests[i] = true;
        }
    }
    (nots, tests)
}

/// Which op indices (up to `ops.len()`) are branch targets or entries.
pub(super) fn branch_targets(
    ops: &[GenericOp],
    osr_entries: &[usize],
    catch_targets: &[usize],
) -> Vec<bool> {
    let n = ops.len();
    let mut target = vec![false; n + 1];
    for &t in osr_entries.iter().chain(catch_targets) {
        if t <= n {
            target[t] = true;
        }
    }
    for op in ops {
        if let GenericOp::Jump { target: t } | GenericOp::JumpIfFalse { target: t, .. } = *op
            && t <= n
        {
            target[t] = true;
        }
    }
    target
}

/// Whether bytecode op `pc` (a `GenericOp::Interp` or `CallMethod`) usually
/// runs without a call: a global access (the global binding cache), the
/// inline `RequireObjectCoercible` check, an inline `Math.f(x)`, or a
/// sloppy function's `this` binding (an object's is nothing to do).
fn usually_inline(bytecode: &[Op], pc: usize) -> bool {
    match bytecode.get(pc) {
        Some(
            Op::LoadGlobal { .. } | Op::StoreGlobal { .. } | Op::RequireObjectCoercible { .. },
        ) => true,
        Some(op @ Op::CallMethod { .. }) => super::inline_math(op).is_some(),
        Some(op @ Op::CallWithMethod { .. }) => super::inline_math_got(op).is_some(),
        // A sloppy function's `this` binding of an object.
        Some(Op::BindThis { .. }) => true,
        _ => false,
    }
}

/// What a way's fill proved of its receiver beyond the walk — fixed per way,
/// so a hit can skip those checks too (they hold until Rust code runs: only it
/// can turn an object into a dictionary or a host-exotic one, or give an array
/// side-table state).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Family {
    /// An ordinary shaped `Cell::Object` without the host-exotic flag.
    Obj,
    /// A `Cell::Array` without side-table state.
    Arr,
    /// A binding cell (`Cell::Box`).
    Cell,
}

/// The most ways a body uses per family (each way costs a store after every
/// call): `Obj` ways first, then `Arr`, then `Cell` ways.
const MAX_OBJ_WAYS: usize = 12;
const MAX_ARR_WAYS: usize = 1;
const MAX_CELL_WAYS: usize = 2;

/// How one property op uses the receiver cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct SitePlan {
    /// The way (its frame slots).
    pub(super) way: usize,
    /// What the way's fills prove (the way's family).
    pub(super) family: Family,
    /// Compare the receiver with the way's key first (a hit skips the walk).
    pub(super) check: bool,
    /// Record the walked receiver in the way.
    pub(super) fill: bool,
    /// In a speculative loop's fast version: an earlier site of the same
    /// iteration put this very receiver in the way, so it is still there
    /// (no compare: see [`plan`]). Outside it (the loop's plain copy) the
    /// site checks.
    pub(super) known: bool,
    /// For a [`known`](Self::known) site: the earlier `SetProp` of the same
    /// iteration on the same receiver whose fast path (all of whose
    /// continuations passed its flags-byte test, the others leaving for the
    /// loop's plain copy) this site follows with no Rust code in between —
    /// so the receiver's flags byte still passes that test.
    pub(super) flags_from: Option<usize>,
    /// For a [`known`](Self::known) site: the earlier site of the same
    /// iteration on the same receiver it follows (with no Rust code in
    /// between, and no other value in the way).
    pub(super) pred: Option<usize>,
}

/// A body's receiver-cache plan: the ways it uses and each op's site plan.
pub(super) struct RecvPlan {
    /// How many ways the body uses (`0`: none, and no clears).
    pub(super) ways: usize,
    /// Per op, its plan when it is a receiver site that uses the cache.
    pub(super) sites: Vec<Option<SitePlan>>,
}

/// The receiver register of an op whose inline fast path walks it (and so can
/// use the cache); `bc` is its bytecode op. (A truthiness test's is
/// [`plan`]'s to decide.)
fn site_recv(op: &GenericOp, bc: Option<&Op>) -> Option<(u16, Family)> {
    match *op {
        // An array iteration's by-index step walks the array (`iter + 1`).
        GenericOp::Interp { .. } => match bc {
            Some(Op::IterNext { iter, next, .. })
                if usize::from(*next) == usize::from(*iter) + 1 =>
            {
                Some((*next, Family::Arr))
            }
            _ => None,
        },
        GenericOp::GetProp { obj, .. }
        | GenericOp::SetProp { obj, .. }
        | GenericOp::InitProp {
            obj, plain: true, ..
        } => Some((obj, Family::Obj)),
        GenericOp::GetElem { obj, .. } | GenericOp::SetElem { obj, .. } => Some((obj, Family::Arr)),
        GenericOp::CallKey { recv, .. } => Some((recv, Family::Arr)),
        GenericOp::CellGet { cell, .. } | GenericOp::CellSet { cell, .. } => {
            Some((cell, Family::Cell))
        }
        GenericOp::ArrGet { arr, .. }
        | GenericOp::ArrSet { arr, .. }
        | GenericOp::ArrayLen { arr, .. } => Some((arr, Family::Arr)),
        _ => None,
    }
}

/// Whether bytecode op `pc` is a `super(…)` call whose inlined guard makes
/// `this` (`jit`'s `super_site`).
fn is_super_call(bytecode: &[Op], pc: usize) -> bool {
    matches!(bytecode.get(pc), Some(Op::SuperCallArgs { .. }))
}

/// The register an op writes, when it names one.
fn op_dst(op: &GenericOp) -> Option<u16> {
    match *op {
        GenericOp::Const { dst, .. }
        | GenericOp::Add { dst, .. }
        | GenericOp::Move { dst, .. }
        | GenericOp::GetProp { dst, .. }
        | GenericOp::GetMethod { dst, .. }
        | GenericOp::GetElem { dst, .. }
        | GenericOp::ArrayLen { dst, .. }
        | GenericOp::Arith { dst, .. }
        | GenericOp::Lt { dst, .. }
        | GenericOp::Compare { dst, .. }
        | GenericOp::NumAdd { dst, .. }
        | GenericOp::StrictEq { dst, .. }
        | GenericOp::ValueBin { dst, .. }
        | GenericOp::Not { dst, .. }
        | GenericOp::PlainPrims { dst, .. }
        | GenericOp::CallNative { dst, .. }
        | GenericOp::ToNumeric { dst, .. }
        | GenericOp::Step { dst, .. }
        | GenericOp::ArrGet { dst, .. }
        | GenericOp::CellGet { dst, .. }
        | GenericOp::CallMethod { dst, .. }
        | GenericOp::CallValue { dst, .. }
        | GenericOp::CallWithMethod { dst, .. }
        | GenericOp::CallKey { dst, .. }
        | GenericOp::Construct { dst, .. } => Some(dst),
        _ => None,
    }
}

/// Whether `op`'s result is usually another value each time it runs: an
/// element or property read, or a call's result (a global or captured
/// binding's read usually yields the same object again).
fn fresh_def(op: &GenericOp) -> bool {
    matches!(
        op,
        GenericOp::GetElem { .. }
            | GenericOp::ArrGet { .. }
            | GenericOp::GetProp { .. }
            | GenericOp::GetMethod { .. }
            | GenericOp::CallMethod { .. }
            | GenericOp::CallValue { .. }
            | GenericOp::CallWithMethod { .. }
            | GenericOp::CallKey { .. }
            | GenericOp::Construct { .. }
    )
}

/// Whether an op's fast path makes a call (which empties every way), or the
/// op may write registers it does not name.
fn ends_knowledge(op: &GenericOp) -> bool {
    matches!(
        op,
        GenericOp::Interp { .. }
            | GenericOp::CallNative { .. }
            | GenericOp::CallMethod { .. }
            | GenericOp::CallValue { .. }
            | GenericOp::CallWithMethod { .. }
            | GenericOp::CallKey { .. }
            | GenericOp::Construct { .. }
            | GenericOp::PushHandler { .. }
            | GenericOp::PopHandler
            | GenericOp::InitProp { plain: false, .. }
            | GenericOp::Ret { .. }
    )
}

/// The value id register `r` holds (a fresh one if it has none yet).
fn id_of(ids: &mut Vec<(u16, u32)>, next: &mut u32, r: u16) -> u32 {
    if let Some(&(_, id)) = ids.iter().find(|&&(x, _)| x == r) {
        return id;
    }
    *next += 1;
    ids.push((r, *next));
    *next
}

/// Per way, the layout every one of its sites compiles in, if they all
/// compile in the same one (`baked(op)`: the shape word an op's code compares
/// its receiver with). Such a way only ever holds an object of that layout:
///
/// - every fill of it is by one of those sites, whose code goes on with the
///   receiver only once its shape compared equal, and whose mismatch path
///   empties the way (and every other such way) before anything else;
/// - an object's shape changes only in Rust code (behind a call, which empties
///   every way) or in an inline add (`SetProp`'s add path, `InitProp`), after
///   which the emitter empties every such way (an add may hit an object
///   another way holds).
///
/// So a hit on such a way, at a site that compiles in the same layout, needs
/// no shape compare.
pub(super) fn shape_ways(
    plan: &RecvPlan,
    ops: &[GenericOp],
    baked: impl Fn(&GenericOp) -> Option<u64>,
) -> Vec<Option<u64>> {
    // `None`: no site yet; `Some(None)`: mixed; `Some(Some(s))`: all `s`.
    let mut out: Vec<Option<Option<u64>>> = vec![None; plan.ways];
    for (op, site) in ops.iter().zip(&plan.sites) {
        let Some(p) = site else { continue };
        // An inlined `new` / `super(…)`'s guard fills its way with the
        // fresh `this`, of no property's layout: such a way proves none.
        if let GenericOp::InlineGuard { call, .. } = *op
            && matches!(
                ops.get(call as usize),
                Some(GenericOp::Construct { .. } | GenericOp::Interp { .. })
            )
        {
            if let Some(slot) = out.get_mut(p.way) {
                *slot = Some(None);
            }
            continue;
        }
        // A truthiness test fills such a way only with an object of the
        // way's layout (`compile_generic`'s `truthy_rax!`), and so does an
        // inlined call's guard on its method's receiver
        // (`call_seq::emit_inline_guard`): they prove none.
        if matches!(
            op,
            GenericOp::Not { .. }
                | GenericOp::JumpIfFalse { .. }
                | GenericOp::InlineGuard { .. }
                | GenericOp::GetMethod { .. }
        ) {
            continue;
        }
        let Some(slot) = out.get_mut(p.way) else {
            continue;
        };
        let b = if p.family == Family::Obj {
            baked(op)
        } else {
            None
        };
        *slot = match (*slot, b) {
            (None, b) => Some(b),
            (Some(Some(s)), Some(b)) if s == b => Some(Some(s)),
            _ => Some(None),
        };
    }
    out.into_iter().map(Option::flatten).collect()
}

/// Plans the receiver cache for `ops` (see the module docs), given the global
/// binding cache's plan and which ops lie in promoted loops (whose global
/// accesses keep their own validated addresses); `osr_entries` and
/// `catch_targets` are entries besides the ops' own branch targets.
///
/// `in_spec` tells the ops of speculative promoted loops, whose fast version
/// runs no Rust code from its header to its back-edge: every slow path there
/// leaves for the loop's plain copy instead of rejoining it
/// (`numeric::NumGen::spec_bail`), and so does every inline property add.
/// So once a site of an iteration has the receiver in its way (a hit, or a
/// walk that filled it), a later site of the same iteration on the same
/// value finds it still there — unless a site between them used that way
/// for another value — and is [`known`](SitePlan::known): it takes the
/// way's slot address without comparing. `global_copy` names the global
/// reads there that copy a register (the same value).
#[allow(clippy::too_many_arguments)]
pub(super) fn plan(
    ops: &[GenericOp],
    bytecode: &[Op],
    globals: &GlobalPlan,
    in_region: impl Fn(usize) -> bool,
    in_spec: impl Fn(usize) -> bool,
    cold_guard: impl Fn(usize) -> bool,
    global_copy: impl Fn(usize) -> Option<u16>,
    truthy_site: impl Fn(usize, u16) -> bool,
    reused: impl Fn(usize) -> Option<u16>,
    osr_entries: &[usize],
    catch_targets: &[usize],
) -> RecvPlan {
    let n = ops.len();
    // An inlined `new`'s guard in a speculative loop whose miss leaves its
    // fast version (`numeric::guard_diamonds`): it fills its way with the
    // fresh `this` (for a base constructor; a derived one's fills nothing),
    // which the fresh adds after it compare (`fresh_adds`) — but it is no
    // later site's certain predecessor.
    let cold_new = |i: usize| {
        cold_guard(i)
            && matches!(ops.get(i), Some(GenericOp::InlineGuard { call, poly: 0, .. })
                if matches!(ops.get(*call as usize), Some(GenericOp::Construct { .. }))
                    || (matches!(ops.get(*call as usize), Some(GenericOp::Interp { .. }))
                        && is_super_call(bytecode, *call as usize)))
    };
    // An op's receiver (a truthiness test's only where the generic
    // emitter makes it, `truthy_site`).
    let site_recv = |op: &GenericOp, i: usize| match *op {
        GenericOp::Not { a: r, .. } | GenericOp::JumpIfFalse { cond: r, .. } => {
            truthy_site(i, r).then_some((r, Family::Obj))
        }
        // A read a promoted loop's fast version replaces by a register copy
        // makes no way access there (`numeric::plan_reuse`): a property's,
        // or a captured binding's (no later cell site may take its way as
        // filled).
        GenericOp::GetProp { .. } | GenericOp::CellGet { .. } if reused(i).is_some() => None,
        // A method read through the call site's memo (`GetMethodWay`).
        GenericOp::GetMethod { obj, .. } if !in_spec(i) => Some((obj, Family::Obj)),
        // An inlined method call's guard looks the method up on its
        // receiver (`call_seq::emit_inline_guard`) — but not in a
        // speculative loop, where a later site may take the way as filled
        // without comparing (`known`) while a guard's miss goes on in the
        // loop without filling it.
        // (A polymorphic site's later way takes the first way's callee
        // value and looks nothing up.)
        GenericOp::InlineGuard {
            call,
            caps,
            n_caps,
            poly,
            ..
        } if (!in_spec(i) || cold_new(i)) && poly & 0xff == 0 => match ops.get(call as usize)? {
            GenericOp::CallMethod { recv: r, .. }
            | GenericOp::GetProp { obj: r, .. }
            | GenericOp::SetProp { obj: r, .. } => Some((*r, Family::Obj)),
            // An inlined `new`'s guard allocates `this` (the register after
            // the captures) and fills its way with the fresh object; so does
            // an inlined `super(…)`'s, into a base constructor (a derived
            // one's `this` stays unbound, and fills nothing).
            GenericOp::Construct { .. } => Some((caps + n_caps.checked_sub(1)?, Family::Obj)),
            GenericOp::Interp { .. } if is_super_call(bytecode, call as usize) => {
                Some((caps + n_caps.checked_sub(1)?, Family::Obj))
            }
            _ => None,
        },
        _ => site_recv(op, bytecode.get(i)),
    };
    let target = branch_targets(ops, osr_entries, catch_targets);
    // Value ids: registers holding the same value (by `Move`) share one.
    let mut next_id = 0u32;
    let mut ids: Vec<(u16, u32)> = Vec::new();
    // The live values since the last call or join: (id, the op that last
    // walked it, whether every path here passed that op — else a join merged
    // it in from some of its predecessors only).
    let mut live: Vec<(u32, usize, bool)> = Vec::new();
    let mut fill = vec![false; n];
    let mut check = vec![false; n];
    let mut site_id = vec![0u32; n];
    // The registers holding a value made since the last join by an op that
    // usually makes a different one each time it runs (`fresh_def`), not yet
    // walked: a promoted loop's first site on one walks it at once rather
    // than first comparing it with what its way held last iteration.
    let mut fresh: Vec<u16> = Vec::new();
    let mut fresh_first = vec![false; n];
    // The property reads since the last op that may have changed a
    // property: (the receiver's value id, the key, the result's value id).
    // A later read of the same key on the same value likely gets the same
    // value (`this.items.length … this.items[i]`), so its result takes that
    // id: its sites then compare the way an earlier site on the first
    // result filled. Only a guess (a getter, or a store between, makes
    // another value), so such an id is never a site's certain predecessor
    // (`guessed`): a key compare decides every hit.
    let mut reads: Vec<(u32, &str, u32)> = Vec::new();
    // (No value id is `u32::MAX`.)
    const GLOBALS: u32 = u32::MAX;
    let mut guessed: Vec<u32> = Vec::new();
    // Per site: the previous site of the same iteration on its value, both
    // in a speculative loop.
    let mut pred: Vec<Option<usize>> = vec![None; n];
    // A branch target reached only by one forward branch (no fall-through
    // into it, no other branch, not an entry) knows what that branch knew:
    // `if (!p) return; p.x`, and which register held which value there
    // (the code it skips may have written them: the loop body of
    // `while (q.link) q = q.link; q.link = p`).
    //
    // A target reached only forward (no back-edge, not an entry) by several
    // paths may find a value walked on some of them: its sites compare the
    // way first (a key compare is always sound, and a miss just walks), but
    // none of them is `known`.
    let mut into = vec![0u32; n + 1];
    let mut back = vec![false; n + 1];
    for (i, op) in ops.iter().enumerate() {
        if let GenericOp::Jump { target: t } | GenericOp::JumpIfFalse { target: t, .. } = *op
            && t <= n
        {
            into[t] += 1;
            back[t] |= t <= i;
        }
    }
    let entry: Vec<usize> = osr_entries.iter().chain(catch_targets).copied().collect();
    #[allow(clippy::type_complexity)]
    let mut snap: Vec<Option<(Vec<(u32, usize, bool)>, Vec<u16>)>> = vec![None; n + 1];
    // Per target, the value ids and the property reads at its one forward
    // branch (`Some(None)` once a second one is seen).
    #[allow(clippy::type_complexity)]
    let mut snap_ids: Vec<Option<Option<(Vec<(u16, u32)>, Vec<(u32, &str, u32)>)>>> =
        vec![None; n + 1];
    // The union of two paths' states (see above).
    let merge = |fill: &mut Vec<bool>,
                 (mut l, mut f): (Vec<(u32, usize, bool)>, Vec<u16>),
                 (l2, f2): (Vec<(u32, usize, bool)>, Vec<u16>)| {
        for (id, at, _) in l2 {
            match l.iter_mut().find(|e| e.0 == id) {
                Some(e) => {
                    if e.1 != at {
                        // Both walks fill the way (a later site checks it).
                        fill[at] = true;
                        fill[e.1] = true;
                    }
                }
                None => l.push((id, at, false)),
            }
        }
        // (Nothing merged is certain: `known` follows one path's sites.)
        for e in &mut l {
            e.2 = false;
        }
        for r in f2 {
            if !f.contains(&r) {
                f.push(r);
            }
        }
        (l, f)
    };
    // The `this` an inlined `new`'s guard writes (a new value, which its
    // own site walks first).
    let new_this = |op: &GenericOp| match *op {
        GenericOp::InlineGuard {
            call, caps, n_caps, ..
        } if matches!(ops.get(call as usize), Some(GenericOp::Construct { .. }))
            || (matches!(ops.get(call as usize), Some(GenericOp::Interp { .. }))
                && is_super_call(bytecode, call as usize)) =>
        {
            n_caps.checked_sub(1).map(|k| caps + k)
        }
        _ => None,
    };
    for (i, op) in ops.iter().enumerate() {
        if let Some(t) = new_this(op) {
            ids.retain(|&(x, _)| x != t);
        }
        let mut joined_reads = None;
        if target[i] {
            let falls_in = i > 0
                && !matches!(
                    bytecode.get(i - 1),
                    Some(Op::Jump { .. } | Op::Return { .. } | Op::Throw { .. })
                );
            match snap[i].take() {
                Some((l, f)) if into[i] == 1 && !falls_in && !entry.contains(&i) => {
                    live = l;
                    fresh = f;
                    if let Some(Some((v, r))) = snap_ids[i].take() {
                        ids = v;
                        joined_reads = Some(r);
                    }
                }
                Some(st) if !back[i] && !entry.contains(&i) => {
                    let here = (core::mem::take(&mut live), core::mem::take(&mut fresh));
                    // One branch in, and a fall-in that knows no walked
                    // value (an inlined call's fallback `call` just before:
                    // `x = o.m(); if (!x)`): the registers name what they
                    // named on the branch's path, the only one with a way
                    // to reuse (none of it certain, see `merge`).
                    let branch_ids = falls_in && into[i] == 1 && here.0.is_empty();
                    let (l, f) = if falls_in {
                        merge(&mut fill, st, here)
                    } else {
                        // Several branches and no fall-in: `st` merged them.
                        st
                    };
                    live = l;
                    fresh = f;
                    if branch_ids && let Some(Some((v, r))) = snap_ids[i].take() {
                        ids = v;
                        joined_reads = Some(r);
                    }
                }
                // A loop's header, entered by falling in: a value walked
                // before the loop may still be in its way on the first
                // iteration (`q = this.queue; while (q.link) …`). Its sites
                // there compare the way first, and the walk before fills
                // it; nothing is certain across the back-edge (or after an
                // entry there, which finds every way empty).
                _ if back[i] && falls_in => {
                    for e in &mut live {
                        e.2 = false;
                    }
                    fresh.clear();
                }
                _ => {
                    live.clear();
                    fresh.clear();
                }
            }
            reads = joined_reads.unwrap_or_default();
        }
        if let Some((r, fam)) = site_recv(op, i) {
            let id = id_of(&mut ids, &mut next_id, r) * 3 + fam as u32;
            site_id[i] = id;
            // A truthiness test fills its way only when the value is an
            // object (of the way's layout, if it has one): it is never a
            // later site's certain predecessor, nor a certain one itself.
            // (Nor is an inlined `new`'s guard, see `cold_new`, nor a
            // captured binding's site: its walk fills the way only for a
            // binding cell, not for a captured parameter's array.)
            let truthy = matches!(op, GenericOp::Not { .. } | GenericOp::JumpIfFalse { .. })
                || (in_spec(i) && cold_new(i))
                || fam == Family::Cell;
            if let Some(e) = live.iter_mut().find(|e| e.0 == id) {
                fill[e.1] = true;
                check[i] = true;
                if e.2 && !truthy && in_spec(i) && in_spec(e.1) && !guessed.contains(&(id / 3)) {
                    pred[i] = Some(e.1);
                }
                // (Every site fills its way: from here on, it holds the
                // value on every path.)
                *e = (id, i, !truthy);
            } else {
                live.push((id, i, !truthy));
                fresh_first[i] = fresh.contains(&r);
            }
        }
        let inline = matches!(
            op,
            GenericOp::Interp { .. }
                | GenericOp::CallMethod { .. }
                | GenericOp::CallWithMethod { .. }
        ) && match bytecode.get(i) {
            // A global access without a cache slot calls a helper.
            Some(Op::LoadGlobal { .. } | Op::StoreGlobal { .. }) => {
                globals.site.get(i).copied().flatten().is_some() || in_region(i)
            }
            // A constructor's super constructor / "no instance fields"
            // through the site's identity memo (its miss calls a helper,
            // which empties every way — or, in a speculative loop's fast
            // version, whose `known` sites follow one path's walks, leaves
            // for the plain copy: `jit`'s `spec_bail`).
            Some(Op::SuperCtor { .. } | Op::InitFields { .. }) => true,
            _ => usually_inline(bytecode, i),
        };
        if ends_knowledge(op) && !inline {
            live.clear();
        }
        match *op {
            // An inlined call's guard writes its result and the inlined
            // body's captures (and makes no call).
            GenericOp::InlineGuard {
                dst, caps, n_caps, ..
            } => {
                let written = |x: u16| x == dst || (x >= caps && x - caps < n_caps);
                // (A `new`'s `this` keeps the id its site gave it.)
                let this = new_this(op);
                ids.retain(|&(x, _)| !written(x) || Some(x) == this);
                fresh.retain(|&x| !written(x));
            }
            GenericOp::Move { dst, src } => {
                let id = id_of(&mut ids, &mut next_id, src);
                ids.retain(|&(x, _)| x != dst);
                ids.push((dst, id));
                let f = fresh.contains(&src);
                fresh.retain(|&x| x != dst);
                if f {
                    fresh.push(dst);
                }
            }
            // A property read: a repeat of an earlier one (see `reads`), or a
            // fresh value.
            GenericOp::GetProp { dst, obj, .. }
                if reused(i).is_none()
                    && let Some(Op::GetProp { key, .. }) = bytecode.get(i) =>
            {
                let oid = id_of(&mut ids, &mut next_id, obj);
                let prev = reads
                    .iter()
                    .find(|r| r.0 == oid && r.1 == key.as_str())
                    .map(|r| r.2);
                ids.retain(|&(x, _)| x != dst);
                fresh.retain(|&x| x != dst);
                match prev {
                    Some(vid) => {
                        ids.push((dst, vid));
                        guessed.push(vid);
                    }
                    None => {
                        fresh.push(dst);
                        let vid = id_of(&mut ids, &mut next_id, dst);
                        reads.push((oid, key.as_str(), vid));
                    }
                }
            }
            _ => {
                if let Some(d) = op_dst(op) {
                    ids.retain(|&(x, _)| x != d);
                    fresh.retain(|&x| x != d);
                    if fresh_def(op) {
                        fresh.push(d);
                    }
                }
            }
        }
        match (op, bytecode.get(i)) {
            // A read of a value a register still holds is a copy of it
            // (`numeric::NumGen::global_copy`).
            (GenericOp::Interp { .. }, Some(Op::LoadGlobal { dst, .. }))
                if let Some(src) = global_copy(i) =>
            {
                let d = *dst;
                let id = id_of(&mut ids, &mut next_id, src);
                ids.retain(|&(x, _)| x != d);
                ids.push((d, id));
            }
            // A global binding is a property of a pseudo object (`GLOBALS`)
            // for `reads`: a repeated read (with no store or call between)
            // most likely gets the same value.
            (GenericOp::Interp { .. }, Some(Op::LoadGlobal { dst, name, .. })) => {
                let d = *dst;
                ids.retain(|&(x, _)| x != d);
                match reads
                    .iter()
                    .find(|r| r.0 == GLOBALS && r.1 == name.as_str())
                    .map(|r| r.2)
                {
                    Some(vid) => {
                        ids.push((d, vid));
                        guessed.push(vid);
                    }
                    None => {
                        let vid = id_of(&mut ids, &mut next_id, d);
                        reads.push((GLOBALS, name.as_str(), vid));
                    }
                }
            }
            (GenericOp::Interp { .. }, Some(Op::StoreGlobal { name, src, .. })) => {
                reads.retain(|r| !(r.0 == GLOBALS && r.1 == name.as_str()));
                let vid = id_of(&mut ids, &mut next_id, *src);
                reads.push((GLOBALS, name.as_str(), vid));
            }
            (GenericOp::Interp { .. }, Some(Op::RequireObjectCoercible { .. })) => {}
            // (They write their `dst` only, or no register: `InitFields`'
            // field initializers run in frames of their own.)
            (GenericOp::Interp { .. }, Some(Op::SuperCtor { dst, .. })) => {
                let d = *dst;
                ids.retain(|&(x, _)| x != d);
                fresh.retain(|&x| x != d);
            }
            (GenericOp::Interp { .. }, Some(Op::InitFields { .. })) => {}
            (GenericOp::Interp { .. } | GenericOp::PushHandler { .. }, _) => {
                // It may write any register.
                ids.clear();
                fresh.clear();
            }
            _ => {}
        }
        if let (GenericOp::Interp { .. }, Some(Op::LoadGlobal { dst, .. })) = (op, bytecode.get(i))
        {
            fresh.retain(|x| x != dst);
        }
        // (Anything but a read, a copy or plain arithmetic may store a
        // property, or run code that does.)
        if !matches!(
            op,
            GenericOp::GetProp { .. }
                | GenericOp::GetElem { .. }
                | GenericOp::ArrayLen { .. }
                | GenericOp::Move { .. }
                | GenericOp::Const { .. }
                | GenericOp::Nop
                | GenericOp::Not { .. }
                | GenericOp::JumpIfFalse { .. }
                | GenericOp::Compare { .. }
                | GenericOp::StrictEq { .. }
                | GenericOp::Lt { .. }
                | GenericOp::Arith { .. }
                | GenericOp::Add { .. }
                | GenericOp::NumAdd { .. }
                | GenericOp::ToNumeric { .. }
                | GenericOp::Step { .. }
                | GenericOp::SetProp { .. }
                | GenericOp::InlineGuard { .. }
                | GenericOp::InlineLeave
        ) && !matches!(
            (op, bytecode.get(i)),
            (
                GenericOp::Interp { .. },
                Some(
                    Op::RequireObjectCoercible { .. }
                        | Op::LoadGlobal { .. }
                        | Op::StoreGlobal { .. }
                )
            )
        ) {
            reads.clear();
        }
        // A property store most likely changes only its own key's value (an
        // accessor's or a `valueOf`'s code may do more: the reads are only
        // guesses, which a key compare decides).
        // A later read of the key on the same object most likely gets the
        // value stored (`h.cur = {…}; h.cur.x`).
        if let (
            GenericOp::SetProp { obj, val, .. },
            Some(Op::SetProp { key, .. } | Op::SetPropStrict { key, .. }),
        ) = (op, bytecode.get(i))
        {
            reads.retain(|r| r.1 != key.as_str());
            let oid = id_of(&mut ids, &mut next_id, *obj);
            let vid = id_of(&mut ids, &mut next_id, *val);
            reads.push((oid, key.as_str(), vid));
        }
        if let GenericOp::Jump { target: t } | GenericOp::JumpIfFalse { target: t, .. } = *op
            && t > i
            && t <= n
        {
            let st = (live.clone(), fresh.clone());
            snap[t] = Some(match snap[t].take() {
                Some(prev) => merge(&mut fill, prev, st),
                None => st,
            });
            snap_ids[t] = Some(match snap_ids[t].take() {
                None => Some((ids.clone(), reads.clone())),
                Some(_) => None,
            });
        }
    }
    // A value read afresh each iteration from the same object and key
    // (`this.items.length` in a loop's condition: the receiver register no
    // op of the loop writes, the key no op of it stores) is usually the
    // object it was last iteration: its first site is no fresh one.
    let loops = innermost_loops(ops);
    for i in 0..n {
        if !fresh_first[i] {
            continue;
        }
        let (Some((head, end)), Some((r, _))) = (loops[i], site_recv(&ops[i], i)) else {
            continue;
        };
        // The op that defined the site's receiver (the nearest write before
        // it in the stream).
        let Some(def) = (head..i)
            .rev()
            .find(|&j| writes(&ops[j], bytecode.get(j), r))
        else {
            continue;
        };
        let (GenericOp::GetProp { obj, .. }, Some(Op::GetProp { key, .. })) =
            (&ops[def], bytecode.get(def))
        else {
            continue;
        };
        let stable = (head..=end).all(|j| {
            !writes(&ops[j], bytecode.get(j), *obj)
                && !matches!(
                    bytecode.get(j),
                    Some(Op::SetProp { key: k, .. } | Op::SetPropStrict { key: k, .. }) if k == key
                )
        });
        if stable {
            fresh_first[i] = false;
        }
    }
    // In a promoted loop, a receiver is usually the one the same site saw
    // last iteration (a loop-invariant object, or one read afresh from the
    // same binding): every site there checks its way first and refills it —
    // except the first site on a value the iteration has just made (an
    // element or property read, a call's result), which is usually another
    // object than last time: it walks the value at once (and fills the way).
    for i in 0..n {
        if in_region(i) && site_recv(&ops[i], i).is_some() {
            fill[i] = true;
            check[i] = !fresh_first[i];
        }
    }
    // Likewise in any other loop's body (an outer loop, or one that calls):
    // a loop-invariant receiver read afresh each iteration (a global or
    // captured binding, an outer loop's array) is usually the one this site
    // walked last iteration, and when no call ran since its way still holds
    // it. A miss only costs the compare (the walk then refills the way).
    let mut in_loop = vec![false; n];
    for (end, op) in ops.iter().enumerate() {
        if let GenericOp::Jump { target: head } = *op
            && head <= end
        {
            for x in &mut in_loop[head..=end] {
                *x = true;
            }
        }
    }
    for i in 0..n {
        if in_loop[i] && !in_region(i) && !fresh_first[i] && site_recv(&ops[i], i).is_some() {
            fill[i] = true;
            check[i] = true;
        }
    }
    // Loop-carried receivers share a way (see `carried_aliases`).
    let alias = carried_aliases(ops, bytecode, &site_recv, &reused, &site_id);
    let gid = |i: usize| {
        let mut id = site_id[i];
        while let Some(&(_, to)) = alias.iter().find(|a| a.0 == id) {
            id = to;
        }
        id
    };
    // Ways, least recently used first per family, over the sites that use
    // the cache: `Obj` ways are `0..obj`, `Arr` ways follow them.
    // As many ways as the family has distinct values to cache, up to its cap.
    let uses = |fam: Family, cap: usize| {
        let mut seen: Vec<u32> = Vec::new();
        for i in 0..n {
            if (fill[i] || check[i])
                && site_recv(&ops[i], i).is_some_and(|(_, f)| f == fam)
                && !seen.contains(&gid(i))
            {
                seen.push(gid(i));
            }
        }
        seen.len().min(cap)
    };
    let obj_ways = uses(Family::Obj, MAX_OBJ_WAYS);
    let arr_ways = uses(Family::Arr, MAX_ARR_WAYS);
    let cell_ways = uses(Family::Cell, MAX_CELL_WAYS);
    let mut way_of: Vec<(u32, usize)> = Vec::new();
    let mut used = vec![0usize; obj_ways + arr_ways + cell_ways];
    let mut sites = vec![None; n];
    for i in 0..n {
        if !(fill[i] || check[i]) {
            continue;
        }
        let Some((_, family)) = site_recv(&ops[i], i) else {
            continue;
        };
        let range = match family {
            Family::Obj => 0..obj_ways,
            Family::Arr => obj_ways..obj_ways + arr_ways,
            Family::Cell => obj_ways + arr_ways..obj_ways + arr_ways + cell_ways,
        };
        let id = gid(i);
        let way = if let Some(&(_, w)) = way_of.iter().find(|&&(x, _)| x == id) {
            w
        } else {
            let w = range.min_by_key(|&w| used[w]).unwrap_or(0);
            way_of.retain(|&(_, x)| x != w);
            way_of.push((id, w));
            w
        };
        used[way] = i + 1;
        sites[i] = Some(SitePlan {
            way,
            family,
            check: check[i],
            fill: true,
            known: false,
            flags_from: None,
            pred: None,
        });
    }
    // More binding cells in one innermost loop than cell ways: each
    // iteration's sites would evict one another (every check a miss, then
    // the walk and a refill), so the cells past the first `cell_ways` of the
    // loop walk at once, with no way (`ctor_chain`'s inlined levels each
    // read their own parent class binding).
    {
        let inner = innermost_loops(ops);
        let mut per_loop: Vec<((usize, usize), Vec<u32>)> = Vec::new();
        for i in 0..n {
            let (Some(lp), Some(p)) = (inner[i], sites[i]) else {
                continue;
            };
            if p.family != Family::Cell {
                continue;
            }
            let id = gid(i);
            let ids = match per_loop.iter_mut().find(|(l, _)| *l == lp) {
                Some((_, ids)) => ids,
                None => {
                    per_loop.push((lp, Vec::new()));
                    &mut per_loop.last_mut().expect("just pushed").1
                }
            };
            if !ids.contains(&id) {
                ids.push(id);
            }
            if ids
                .iter()
                .position(|&x| x == id)
                .is_some_and(|k| k >= cell_ways)
            {
                sites[i] = None;
            }
        }
    }
    // A site is known when its predecessor of the iteration used the same
    // way and no site between them used it for another value.
    for i in 0..n {
        let (Some(p), Some(si)) = (pred[i], sites[i]) else {
            continue;
        };
        let same_way = sites[p].is_some_and(|sp| sp.way == si.way);
        let undisturbed = (p + 1..i)
            .all(|q| sites[q].is_none_or(|sq| sq.way != si.way || site_id[q] == site_id[i]));
        let flags_from = if matches!(ops[p], GenericOp::SetProp { .. }) {
            Some(p)
        } else {
            sites[p].and_then(|sp| sp.flags_from)
        };
        if same_way
            && undisturbed
            && site_id[p] == site_id[i]
            && let Some(x) = sites[i].as_mut()
        {
            x.known = true;
            x.flags_from = flags_from;
            x.pred = Some(p);
        }
    }
    let ways = obj_ways + arr_ways + cell_ways;
    RecvPlan { ways, sites }
}

/// An op's receiver-cache receiver, if it is a site (`plan`'s `site_recv`).
type SiteRecv<'a> = dyn Fn(&GenericOp, usize) -> Option<(u16, Family)> + 'a;

/// Which value ids should share a way because a loop carries one into the
/// other: a site near a loop's head on the value a register held on entry
/// to the iteration (`q.link` of `while (q.link) q = q.link`), where the
/// iteration ends with that register holding a value a later site of the
/// iteration walked (the truthiness test of the read `q.link`). That later
/// site then leaves the next iteration's receiver in the way the earlier
/// one checks. Pairs `(from, to)` of `site_id`s (with no cycles); a hint
/// for way assignment only — nothing relies on it for soundness (a key
/// compare decides every hit, and `known` compares real value ids).
fn carried_aliases(
    ops: &[GenericOp],
    bytecode: &[Op],
    site_recv: &SiteRecv<'_>,
    reused: &dyn Fn(usize) -> Option<u16>,
    site_id: &[u32],
) -> Vec<(u32, u32)> {
    // A value token: `ENTRY | r` for register `r`'s value on entry to the
    // iteration, else `pc + 1` of the op that made it.
    const ENTRY: u64 = 1 << 32;
    let mut alias: Vec<(u32, u32)> = Vec::new();
    let resolve = |alias: &[(u32, u32)], mut id: u32| {
        while let Some(&(_, to)) = alias.iter().find(|a| a.0 == id) {
            id = to;
        }
        id
    };
    for (end, op) in ops.iter().enumerate() {
        let GenericOp::Jump { target: head } = *op else {
            continue;
        };
        if head > end {
            continue;
        }
        let tok = core::cell::RefCell::new(Vec::<(u16, u64)>::new());
        let get = |r: u16| {
            tok.borrow()
                .iter()
                .find(|e| e.0 == r)
                .map_or(ENTRY | u64::from(r), |e| e.1)
        };
        let set = |r: u16, t: u64| {
            let mut v = tok.borrow_mut();
            v.retain(|e| e.0 != r);
            v.push((r, t));
        };
        // A global binding is a register too (past the real ones): a loop
        // that reads it at the top and stores it at the end
        // (`v = v.add(d)` at the top level) carries its value like one.
        let globals = core::cell::RefCell::new(Vec::<(alloc::string::String, u64)>::new());
        let global = |name: &str| {
            let mut g = globals.borrow_mut();
            if let Some((_, t)) = g.iter().find(|e| e.0 == name) {
                return *t;
            }
            let t = ENTRY | ((1 << 16) + g.len() as u64);
            g.push((name.into(), t));
            t
        };
        // Per global binding, the token it holds (its entry one until stored).
        let gtok = core::cell::RefCell::new(Vec::<(u64, u64)>::new());
        let get_g = |name: &str| {
            let e = global(name);
            gtok.borrow().iter().find(|x| x.0 == e).map_or(e, |x| x.1)
        };
        // The first site on each entry value, and the last site per token.
        let mut first: Vec<(u64, usize)> = Vec::new();
        let mut walked: Vec<(u64, usize)> = Vec::new();
        // The end of an inlined call jumps over its fallback call op (which
        // joins it after): the values are the inlined path's.
        let mut skip_to = 0;
        for (pc, op) in ops.iter().enumerate().take(end).skip(head) {
            if pc < skip_to {
                continue;
            }
            if let GenericOp::Jump { target } = *op
                && target == pc + 2
                && pc > 0
                && matches!(ops[pc - 1], GenericOp::InlineLeave)
            {
                skip_to = target;
                continue;
            }
            if let Some((r, _)) = site_recv(op, pc) {
                let t = get(r);
                if t & ENTRY != 0 && !first.iter().any(|f| f.0 == t) {
                    first.push((t, pc));
                }
                walked.retain(|w| w.0 != t);
                walked.push((t, pc));
            }
            let def = pc as u64 + 1;
            match (*op, bytecode.get(pc)) {
                (GenericOp::Interp { .. }, Some(Op::LoadGlobal { dst, name, .. })) => {
                    set(*dst, get_g(name));
                    continue;
                }
                // (A check names a register it only reads.)
                (GenericOp::Interp { .. }, Some(Op::RequireObjectCoercible { .. })) => continue,
                (GenericOp::Interp { .. }, Some(Op::StoreGlobal { name, src, .. })) => {
                    let (e, t) = (global(name), get(*src));
                    let mut g = gtok.borrow_mut();
                    g.retain(|x| x.0 != e);
                    g.push((e, t));
                    continue;
                }
                _ => {}
            }
            match *op {
                GenericOp::Move { dst, src } => set(dst, get(src)),
                GenericOp::GetProp { dst, .. } if let Some(src) = reused(pc) => {
                    set(dst, get(src));
                }
                GenericOp::InlineGuard {
                    dst, caps, n_caps, ..
                } => {
                    set(dst, def);
                    for c in caps..caps.saturating_add(n_caps) {
                        set(c, def);
                    }
                }
                GenericOp::Interp { .. } | GenericOp::PushHandler { .. } => {
                    if let Some(b) = bytecode.get(pc) {
                        crate::bytecode::op_for_each_reg(b, &|r| set(r, def));
                    }
                }
                _ => {
                    if let Some(d) = op_dst(op) {
                        set(d, def);
                    }
                }
            }
        }
        for (e, i) in first {
            // The entry value's register (or global binding) at the end.
            let t = match u16::try_from(e & !ENTRY) {
                Ok(r) => get(r),
                Err(_) => gtok.borrow().iter().find(|x| x.0 == e).map_or(e, |x| x.1),
            };
            if t == e {
                continue;
            }
            let Some(&(_, s)) = walked.iter().find(|w| w.0 == t) else {
                continue;
            };
            let (from, to) = (resolve(&alias, site_id[i]), resolve(&alias, site_id[s]));
            if from != to && from % 3 == to % 3 {
                alias.push((from, to));
            }
        }
    }
    alias
}

/// The innermost loop of each op, as `(header, back-edge)`: the shortest
/// span of a backward `Jump` that covers it.
fn innermost_loops(ops: &[GenericOp]) -> Vec<Option<(usize, usize)>> {
    let mut inner: Vec<Option<(usize, usize)>> = vec![None; ops.len()];
    for (end, op) in ops.iter().enumerate() {
        if let GenericOp::Jump { target: head } = *op
            && head <= end
        {
            for x in &mut inner[head..=end] {
                if x.is_none_or(|(h, e)| h <= head && e >= end) {
                    *x = Some((head, end));
                }
            }
        }
    }
    inner
}

/// Whether `op` (bytecode `bc`) may write register `r`.
pub(super) fn writes(op: &GenericOp, bc: Option<&Op>, r: u16) -> bool {
    match *op {
        GenericOp::InlineGuard {
            dst, caps, n_caps, ..
        } => r == dst || (r >= caps && r - caps < n_caps),
        // (A check or a global store names registers it only reads.)
        GenericOp::Interp { .. }
            if matches!(
                bc,
                Some(Op::RequireObjectCoercible { .. } | Op::StoreGlobal { .. })
            ) =>
        {
            false
        }
        GenericOp::Interp { .. } | GenericOp::PushHandler { .. } => bc.is_none_or(|b| {
            let hit = core::cell::Cell::new(false);
            crate::bytecode::op_for_each_reg(b, &|x| hit.set(hit.get() || x == r));
            hit.get()
        }),
        _ => op_dst(op) == Some(r),
    }
}

/// A `GetMethod`'s receiver-cache way (`call_seq::emit_get_method`): its
/// frame slots, whether to compare the key first and whether a walk fills
/// it (only with an object of the way's layout, if it proves one). A
/// `GetMethod` site proves no layout itself and is never a speculative
/// loop's (its inline lookup may go on without filling).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct GetMethodWay {
    pub(super) key: i32,
    pub(super) cell: i32,
    pub(super) check: bool,
    pub(super) fill: bool,
    pub(super) layout: Option<u64>,
}

/// A way that a `call` keeps instead of emptying it: its frame slots, the
/// family its fills prove and, for a layout-proving way (`shape_ways`), the
/// layout. See [`survivors`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct SurviveWay {
    /// The key slot (`[rbp + key]`).
    pub(super) key: i32,
    /// The heap slot address slot (`[rbp + cell]`).
    pub(super) cell: i32,
    /// What the way's fills prove.
    pub(super) family: Family,
    /// The layout a hit on the way proves (its sites skip their shape
    /// compare), if any.
    pub(super) shape: Option<u64>,
}

/// The most ops [`survivors`] follows from a call to the next sites.
const SURVIVE_SCAN: usize = 256;

/// Per op, the ways a call made there (on its fast or slow path) keeps
/// rather than empties: those a site likely uses again before the next call
/// (following the op stream from the op, around its innermost loop's
/// back-edge and along forward jumps). Outside loops, none.
///
/// What a way proves (that its key's object lives at the address in its
/// `cell` slot, of the way's family, and of its layout for a layout-proving
/// way) holds after a call only if the call neither moved the arena (the
/// address is the arena base plus the handle's index, and a key equal to a
/// register's value is a live handle to that very slot) nor retyped or
/// reshaped that cell (Rust code may: an add, a delete, a dictionary switch,
/// a host-exotic mark, a heap swap). So after the call the kept ways are
/// re-validated, never assumed (`X64Assembler::survive_reload`): a moved
/// arena empties them all, and each way's family checks (and layout compare)
/// run again on its cell, a failure emptying it. That is all a fill proved,
/// so a later hit may skip exactly what it skipped before.
///
/// `excluded` ways never survive: an inlined `new`'s way, whose fresh adds
/// take a hit as "no call since the allocation" (`fresh_adds`).
pub(super) fn survivors(plan: &RecvPlan, ops: &[GenericOp], excluded: &[usize]) -> Vec<Vec<usize>> {
    let n = ops.len();
    let mut out = vec![Vec::new(); n];
    if plan.ways == 0 {
        return out;
    }
    let inner = innermost_loops(ops);
    // Ops whose fast path makes a call: the scan stops there.
    let calls = |op: &GenericOp| {
        matches!(
            op,
            GenericOp::CallNative { .. }
                | GenericOp::CallMethod { .. }
                | GenericOp::CallValue { .. }
                | GenericOp::CallWithMethod { .. }
                | GenericOp::CallKey { .. }
                | GenericOp::Construct { .. }
                | GenericOp::Ret { .. }
        )
    };
    let mut seen = vec![false; n];
    let mut visited: Vec<usize> = Vec::new();
    for c in 0..n {
        let Some((head, end)) = inner[c] else {
            continue;
        };
        let mut ways: Vec<usize> = Vec::new();
        let mut j = c + 1;
        for _ in 0..SURVIVE_SCAN {
            if j > end || j < head || seen[j] {
                break;
            }
            seen[j] = true;
            visited.push(j);
            if let Some(p) = plan.sites[j]
                && p.check
                && !ways.contains(&p.way)
                && !excluded.contains(&p.way)
            {
                ways.push(p.way);
            }
            if calls(&ops[j]) {
                break;
            }
            j = match ops[j] {
                GenericOp::Jump { target } => target,
                _ => j + 1,
            };
        }
        for v in visited.drain(..) {
            seen[v] = false;
        }
        out[c] = ways;
    }
    out
}

impl super::X64Assembler {
    /// A way's fill after a full walk: `rax` = the receiver's bits, `rcx` =
    /// its heap slot (checked), into the way's `key` and `cell` slots.
    ///
    /// With a `delta` slot (a receiver of [`walk_regs`]), pinned code also
    /// **predicts** the slot (see `chase`'s stride
    /// prediction): a site whose receiver changes every time it runs, as a
    /// walk along a list or over an array of objects, usually finds the next
    /// one at the same distance from the last one as that one was from its
    /// predecessor. The way's `delta` slot keeps that distance; when the
    /// walked slot is the way's last one plus it, `rcx` is made the
    /// predicted address (the same value) rather than the walked one, so the
    /// loads through it no longer wait for the walk: the processor runs on
    /// past the compare's branch while the walk is still being computed. A
    /// wrong prediction just records the new distance. (`cell` and `delta`
    /// may hold anything: they only make a guess, which the compare checks.)
    /// Clobbers `r10` (pinned code only, where it holds no arena).
    pub(super) fn p_fill_way(&mut self, key: i32, cell: i32, delta: Option<i32>) {
        use super::call_seq::r;
        if let Some(delta) = delta.filter(|_| self.pinned()) {
            let hit = self.new_label();
            let back = self.new_label();
            self.g_load(r::R10, r::RBP, cell);
            self.g_add_mem(r::R10, r::RBP, delta);
            self.raw(&[0x4c, 0x39, 0xd1]); // cmp rcx, r10
            self.je(hit);
            self.g_mov(r::R10, r::RCX);
            self.g_mem(true, &[0x2b], r::R10, r::RBP, cell); // sub r10, [rbp + cell]
            self.g_store(r::RBP, delta, r::R10);
            self.jmp(back);
            self.bind(hit);
            self.g_mov(r::RCX, r::R10);
            self.bind(back);
        }
        self.g_store(r::RBP, key, r::RAX);
        self.g_store(r::RBP, cell, r::RCX);
    }

    /// Sets the ways the `call`s emitted next keep (see [`survivors`]).
    pub(super) fn set_recv_survive(&mut self, ways: Vec<SurviveWay>, layout: super::JitLayout) {
        self.recv_survive = ways;
        self.recv_survive_layout = Some(layout);
    }

    /// `cmp qword [base + disp], shape` for an equality jump: one compare
    /// with a sign-extended immediate when the layout's address fits one,
    /// else through `scratch`.
    pub(super) fn p_cmp_shape(&mut self, base: u8, disp: i32, shape: u64, scratch: u8) {
        match i32::try_from(shape as i64) {
            Ok(imm) => {
                self.g_mem(true, &[0x81], 7, base, disp);
                self.code.extend_from_slice(&imm.to_le_bytes());
            }
            Err(_) => {
                self.g_movabs(scratch, shape);
                self.g_cmp_mem(scratch, base, disp);
            }
        }
    }

    /// Sets the frame slot of the arena base as of the latest call (see the
    /// `recv_seen` field): a body with a way to keep needs it.
    pub(super) fn set_recv_seen(&mut self, slot: i32) {
        self.recv_seen = Some(slot);
    }

    /// Right after a `call` returns (its result in `rax`/`rdx`, which this
    /// keeps): reloads the pinned arena and re-validates the ways the call
    /// kept ([`survivors`]), emptying those that no longer hold; then notes
    /// the arena base. Clobbers `rcx`, `r11` and the flags.
    ///
    /// The noted base is what every key's cell was filled under: a cell is
    /// filled with the arena as it is (it moves only in Rust code, behind a
    /// call), and every call either empties a way or keeps it only when the
    /// base it reloads is the noted one. (`r12` itself does not survive the
    /// call: a pinned callee entered at its `exact_pinned` entry hands back
    /// garbage, see `call_seq::emit_native_entry`.)
    pub(super) fn survive_reload(&mut self) {
        use super::call_seq::r;
        let ways = core::mem::take(&mut self.recv_survive);
        match (self.recv_seen, self.recv_survive_layout) {
            (Some(seen), Some(l)) if self.pinned() => {
                self.pin_reload();
                if !ways.is_empty() {
                    let moved = self.new_label();
                    let done = self.new_label();
                    self.g_cmp_mem(super::pin::R12, r::RBP, seen);
                    self.jne(moved);
                    for w in &ways {
                        let next = self.new_label();
                        let bad = self.new_label();
                        // cmp qword [rbp + key], RECV_CACHE_EMPTY
                        self.code.extend_from_slice(&[0x48, 0x81, 0xbd]);
                        self.code.extend_from_slice(&w.key.to_le_bytes());
                        self.code
                            .extend_from_slice(&super::RECV_CACHE_EMPTY.to_le_bytes());
                        self.je(next);
                        self.g_load(r::RCX, r::RBP, w.cell);
                        // The family checks of a fill (`family_checks!`), the
                        // last one's flags left for the jump below.
                        match w.family {
                            // (Passing leaves ZF set.)
                            Family::Obj => self.p_object_not_exotic(&l, bad),
                            Family::Arr => {
                                self.cmp_byte_at_rcx_imm(l.off_cell_tag, l.cell_array_disc);
                                self.jne(bad);
                            }
                            Family::Cell => {
                                self.cmp_byte_at_rcx_imm(l.off_cell_tag, l.props.box_disc);
                            }
                        }
                        // (The last check jumps on success, past the empty.)
                        match w.family {
                            Family::Obj => {
                                if let Some(s) = w.shape {
                                    self.p_cmp_shape(r::RCX, l.off_shape, s, r::R11);
                                }
                                self.je(next);
                            }
                            Family::Arr => {
                                self.cmp_byte_at_rcx_imm(l.off_arr_side, 0);
                                self.je(next);
                            }
                            Family::Cell => self.je(next),
                        }
                        self.bind(bad);
                        self.store_empty(w.key);
                        self.bind(next);
                    }
                    self.jmp(done);
                    self.bind(moved);
                    for w in &ways {
                        self.store_empty(w.key);
                    }
                    self.bind(done);
                }
                self.g_store(r::RBP, seen, super::pin::R12);
            }
            _ => {
                for w in &ways {
                    self.store_empty(w.key);
                }
                self.pin_reload();
            }
        }
        self.recv_survive = ways;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    fn get(dst: u16, obj: u16, key: &str) -> Op {
        Op::GetProp {
            dst,
            obj,
            key: key.to_string(),
        }
    }

    /// `for (p = q; p; p = p.next) s += p.v`: `p` (and nothing else) walks.
    #[test]
    fn a_list_walk_register() {
        let ops = alloc::vec![
            Op::Move { dst: 8, src: 0 },
            Op::JumpIfFalse { cond: 8, target: 7 },
            get(9, 8, "v"),
            Op::AddValue { dst: 5, a: 5, b: 9 },
            get(11, 8, "next"),
            Op::Move { dst: 8, src: 11 },
            Op::Jump { target: 1 },
            Op::Return { src: 5 },
        ];
        let w = walk_regs(&ops);
        assert!(w[8]);
        assert!(!w[0] && !w[9] && !w[11] && !w[5]);
    }

    /// `o = a[i]` and its copies walk; a register also defined by another
    /// read (`p = this.queue`, a reused temporary) does not.
    #[test]
    fn element_reads_and_other_definitions() {
        let ops = alloc::vec![
            Op::GetKey {
                dst: 14,
                obj: 0,
                key: 9
            },
            Op::Move { dst: 15, src: 14 },
            get(16, 15, "x"),
            get(20, 1, "queue"),
            Op::Move { dst: 21, src: 20 },
            get(22, 21, "link"),
            Op::Move { dst: 21, src: 22 },
            get(21, 3, "other"),
        ];
        let w = walk_regs(&ops);
        assert!(w[14] && w[15]);
        assert!(!w[16] && !w[20] && !w[21] && !w[22]);
    }
}
