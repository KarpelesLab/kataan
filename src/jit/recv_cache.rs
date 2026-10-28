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
}

/// The most ways a body uses per family (each way costs a store after every
/// call): `Obj` ways first, then `Arr` ways.
const MAX_OBJ_WAYS: usize = 3;
const MAX_ARR_WAYS: usize = 1;

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
}

/// A body's receiver-cache plan: the ways it uses and each op's site plan.
pub(super) struct RecvPlan {
    /// How many ways the body uses (`0`: none, and no clears).
    pub(super) ways: usize,
    /// Per op, its plan when it is a receiver site that uses the cache.
    pub(super) sites: Vec<Option<SitePlan>>,
}

/// The receiver register of an op whose inline fast path walks it (and so can
/// use the cache).
fn site_recv(op: &GenericOp) -> Option<(u16, Family)> {
    match *op {
        GenericOp::GetProp { obj, .. }
        | GenericOp::SetProp { obj, .. }
        | GenericOp::InitProp {
            obj, plain: true, ..
        } => Some((obj, Family::Obj)),
        GenericOp::GetElem { obj, .. } | GenericOp::SetElem { obj, .. } => Some((obj, Family::Arr)),
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
        | GenericOp::GetElem { dst, .. }
        | GenericOp::ArrayLen { dst, .. }
        | GenericOp::Arith { dst, .. }
        | GenericOp::Lt { dst, .. }
        | GenericOp::Compare { dst, .. }
        | GenericOp::NumAdd { dst, .. }
        | GenericOp::StrictEq { dst, .. }
        | GenericOp::ValueBin { dst, .. }
        | GenericOp::Not { dst, .. }
        | GenericOp::CallNative { dst, .. }
        | GenericOp::ToNumeric { dst, .. }
        | GenericOp::Step { dst, .. }
        | GenericOp::ArrGet { dst, .. }
        | GenericOp::CallMethod { dst, .. }
        | GenericOp::CallValue { dst, .. }
        | GenericOp::CallKey { dst, .. }
        | GenericOp::Construct { dst, .. } => Some(dst),
        _ => None,
    }
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

/// Plans the receiver cache for `ops` (see the module docs); `osr_entries` and
/// `catch_targets` are entries besides the ops' own branch targets.
pub(super) fn plan(ops: &[GenericOp], osr_entries: &[usize], catch_targets: &[usize]) -> RecvPlan {
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
    // Value ids: registers holding the same value (by `Move`) share one.
    let mut next_id = 0u32;
    let mut ids: Vec<(u16, u32)> = Vec::new();
    // The live values since the last call or join: (id, the op that last
    // walked it).
    let mut live: Vec<(u32, usize)> = Vec::new();
    let mut fill = vec![false; n];
    let mut check = vec![false; n];
    let mut site_id = vec![0u32; n];
    for (i, op) in ops.iter().enumerate() {
        if target[i] {
            live.clear();
        }
        if let Some((r, fam)) = site_recv(op) {
            let id = id_of(&mut ids, &mut next_id, r) * 2 + u32::from(fam == Family::Arr);
            site_id[i] = id;
            if let Some(e) = live.iter_mut().find(|e| e.0 == id) {
                fill[e.1] = true;
                check[i] = true;
                e.1 = i;
            } else {
                live.push((id, i));
            }
        }
        if ends_knowledge(op) {
            live.clear();
        }
        match *op {
            GenericOp::Move { dst, src } => {
                let id = id_of(&mut ids, &mut next_id, src);
                ids.retain(|&(x, _)| x != dst);
                ids.push((dst, id));
            }
            _ => {
                if let Some(d) = op_dst(op) {
                    ids.retain(|&(x, _)| x != d);
                }
            }
        }
        if matches!(op, GenericOp::Interp { .. } | GenericOp::PushHandler { .. }) {
            // It may write any register.
            ids.clear();
        }
    }
    // Ways, least recently used first per family, over the sites that use
    // the cache: `Obj` ways are `0..obj`, `Arr` ways follow them.
    // As many ways as the family has distinct values to cache, up to its cap.
    let uses = |fam: Family, cap: usize| {
        let mut seen: Vec<u32> = Vec::new();
        for i in 0..n {
            if (fill[i] || check[i])
                && site_recv(&ops[i]).is_some_and(|(_, f)| f == fam)
                && !seen.contains(&site_id[i])
            {
                seen.push(site_id[i]);
            }
        }
        seen.len().min(cap)
    };
    let obj_ways = uses(Family::Obj, MAX_OBJ_WAYS);
    let arr_ways = uses(Family::Arr, MAX_ARR_WAYS);
    let mut way_of: Vec<(u32, usize)> = Vec::new();
    let mut used = vec![0usize; obj_ways + arr_ways];
    let mut sites = vec![None; n];
    for i in 0..n {
        if !(fill[i] || check[i]) {
            continue;
        }
        let Some((_, family)) = site_recv(&ops[i]) else {
            continue;
        };
        let range = match family {
            Family::Obj => 0..obj_ways,
            Family::Arr => obj_ways..obj_ways + arr_ways,
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
        });
    }
    let ways = obj_ways + arr_ways;
    RecvPlan { ways, sites }
}
