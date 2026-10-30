//! The generic tier's **receiver cache**: which property ops of a body keep
//! the heap slot address of their receiver in a frame slot for a later op on
//! the same object, so it skips the arena walk (handle check, bounds,
//! occupancy and generation guards).
//!
//! Soundness is dynamic and does not depend on this plan. A cache *way* is a
//! pair of frame slots: the receiver's handle bits (the key) and the address
//! of its live, occupied heap slot as a full walk proved it. The frame
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

/// Plans the global binding cache: the body's global reads and writes outside
/// promoted loops (`in_region`, which keep their own), grouped by `(name,
/// read | write)`; the (up to [`MAX_GLOBAL_SLOTS`]) most frequent pairs get a
/// slot each. A slot keeps the address an access validated across calls and
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
        if matches!(op, GenericOp::Interp { .. }) && !in_region(pc) {
            key[pc] = match bytecode.get(pc) {
                Some(Op::LoadGlobal { name, .. }) => Some((name.as_str(), false)),
                Some(Op::StoreGlobal { name, .. }) => Some((name.as_str(), true)),
                _ => None,
            };
        }
    }
    let mut groups: Vec<((&str, bool), usize)> = Vec::new();
    for &k in key.iter().flatten() {
        match groups.iter_mut().find(|g| g.0 == k) {
            Some(g) => g.1 += 1,
            None => groups.push((k, 1)),
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

/// Which op indices (up to `ops.len()`) are branch targets or entries.
fn branch_targets(ops: &[GenericOp], osr_entries: &[usize], catch_targets: &[usize]) -> Vec<bool> {
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
/// inline `RequireObjectCoercible` check, or an inline `Math.f(x)`.
fn usually_inline(bytecode: &[Op], pc: usize) -> bool {
    match bytecode.get(pc) {
        Some(
            Op::LoadGlobal { .. } | Op::StoreGlobal { .. } | Op::RequireObjectCoercible { .. },
        ) => true,
        Some(op @ Op::CallMethod { .. }) => super::inline_math(op).is_some(),
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
const MAX_OBJ_WAYS: usize = 3;
const MAX_ARR_WAYS: usize = 1;
const MAX_CELL_WAYS: usize = 1;

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
        // A truthiness test fills such a way only with an object of the
        // way's layout (`compile_generic`'s `truthy_rax!`): it proves none.
        if matches!(op, GenericOp::Not { .. } | GenericOp::JumpIfFalse { .. }) {
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
    global_copy: impl Fn(usize) -> Option<u16>,
    truthy_site: impl Fn(usize, u16) -> bool,
    reused: impl Fn(usize) -> bool,
    osr_entries: &[usize],
    catch_targets: &[usize],
) -> RecvPlan {
    let n = ops.len();
    // An op's receiver (a truthiness test's only where the generic
    // emitter makes it, `truthy_site`).
    let site_recv = |op: &GenericOp, i: usize| match *op {
        GenericOp::Not { a: r, .. } | GenericOp::JumpIfFalse { cond: r, .. } => {
            truthy_site(i, r).then_some((r, Family::Obj))
        }
        // A read a promoted loop's fast version replaces by a register copy
        // makes no way access there (`numeric::plan_reuse`).
        GenericOp::GetProp { .. } if reused(i) => None,
        // An inlined method call's guard looks the method up on its
        // receiver (`call_seq::emit_inline_guard`) — but not in a
        // speculative loop, where a later site may take the way as filled
        // without comparing (`known`) while a guard's miss goes on in the
        // loop without filling it.
        GenericOp::InlineGuard { call, .. } if !in_spec(i) => match ops.get(call as usize)? {
            GenericOp::CallMethod { recv: r, .. } | GenericOp::GetProp { obj: r, .. } => {
                Some((*r, Family::Obj))
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
    // walked it).
    let mut live: Vec<(u32, usize)> = Vec::new();
    let mut fill = vec![false; n];
    let mut check = vec![false; n];
    let mut site_id = vec![0u32; n];
    // The registers holding a value made since the last join by an op that
    // usually makes a different one each time it runs (`fresh_def`), not yet
    // walked: a promoted loop's first site on one walks it at once rather
    // than first comparing it with what its way held last iteration.
    let mut fresh: Vec<u16> = Vec::new();
    let mut fresh_first = vec![false; n];
    // Per site: the previous site of the same iteration on its value, both
    // in a speculative loop.
    let mut pred: Vec<Option<usize>> = vec![None; n];
    // A branch target reached only by one forward branch (no fall-through
    // into it, no other branch, not an entry) knows what that branch knew:
    // `if (!p) return; p.x`. (`ids` is not restored: a register the skipped
    // code wrote just loses its value's identity, which only costs a hit.)
    let mut into = vec![0u32; n + 1];
    for op in ops {
        if let GenericOp::Jump { target: t } | GenericOp::JumpIfFalse { target: t, .. } = *op
            && t <= n
        {
            into[t] += 1;
        }
    }
    let entry: Vec<usize> = osr_entries.iter().chain(catch_targets).copied().collect();
    #[allow(clippy::type_complexity)]
    let mut snap: Vec<Option<(Vec<(u32, usize)>, Vec<u16>)>> = vec![None; n + 1];
    for (i, op) in ops.iter().enumerate() {
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
                }
                _ => {
                    live.clear();
                    fresh.clear();
                }
            }
        }
        if let Some((r, fam)) = site_recv(op, i) {
            let id = id_of(&mut ids, &mut next_id, r) * 3 + fam as u32;
            site_id[i] = id;
            if let Some(e) = live.iter_mut().find(|e| e.0 == id) {
                fill[e.1] = true;
                check[i] = true;
                if in_spec(i) && in_spec(e.1) {
                    pred[i] = Some(e.1);
                }
                e.1 = i;
            } else {
                live.push((id, i));
                fresh_first[i] = fresh.contains(&r);
            }
        }
        let inline = matches!(op, GenericOp::Interp { .. } | GenericOp::CallMethod { .. })
            && match bytecode.get(i) {
                // A global access without a cache slot calls a helper.
                Some(Op::LoadGlobal { .. } | Op::StoreGlobal { .. }) => {
                    globals.site.get(i).copied().flatten().is_some() || in_region(i)
                }
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
                ids.retain(|&(x, _)| !written(x));
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
            (GenericOp::Interp { .. }, Some(Op::LoadGlobal { dst, .. })) => {
                let d = *dst;
                ids.retain(|&(x, _)| x != d);
            }
            (
                GenericOp::Interp { .. },
                Some(Op::StoreGlobal { .. } | Op::RequireObjectCoercible { .. }),
            ) => {}
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
        if let GenericOp::Jump { target: t } | GenericOp::JumpIfFalse { target: t, .. } = *op
            && t > i
            && t <= n
        {
            snap[t] = Some((live.clone(), fresh.clone()));
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
    // Ways, least recently used first per family, over the sites that use
    // the cache: `Obj` ways are `0..obj`, `Arr` ways follow them.
    // As many ways as the family has distinct values to cache, up to its cap.
    let uses = |fam: Family, cap: usize| {
        let mut seen: Vec<u32> = Vec::new();
        for i in 0..n {
            if (fill[i] || check[i])
                && site_recv(&ops[i], i).is_some_and(|(_, f)| f == fam)
                && !seen.contains(&site_id[i])
            {
                seen.push(site_id[i]);
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
        let id = site_id[i];
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
