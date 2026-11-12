//! Recompilation of generic-tier bodies whose property sites learned more
//! while they ran than they knew when they were compiled (`ROADMAP.md` §2.1).
//!
//! A body compiles its property sites from the inline caches the interpreter
//! warmed (`JitProto::compile_generic_warm`). Code reached only after the
//! compile — the later loops of top-level code or of a long function, entered
//! by on-stack replacement when its *first* loop got hot — starts with cold
//! sites, and code whose objects changed layout (a phase change) keeps
//! comparing with a layout that no longer arrives. Either way the site's own
//! runtime cache, which the compiled code keeps filling, now knows a stable
//! layout that the code does not have compiled in.
//!
//! The loop back-edge poll (`nbvm::jit_tier::jit_helper_backedge`) asks
//! [`JitProto::wants_recompile`] about the loop it polls for. If the answer is
//! yes, the body leaves at that loop's header with the register file up to
//! date (`NanBox::jit_exit_bits`); the VM compiles the function again from
//! [`JitProto::runtime_warm`] and continues in the new code at the same
//! header. Each recompilation counts against a small per-function budget
//! ([`MAX_GENERATION`]), so a site that keeps changing its mind cannot make
//! the function recompile forever.

use super::{GenericOp, JitProto};
use crate::shape::Shape;
use alloc::rc::Rc;
use alloc::vec::Vec;

std::thread_local! {
    /// Whether the generic body being compiled polls for its
    /// recompilation at its native-call entry: it has call sites with no
    /// callee to inline yet (`JitProto::note_cold_calls`). A body that
    /// only compiled callers call never meets the interpreter's entry check
    /// ([`JitProto::wants_recompile_on_entry`]) and, unless an on-stack
    /// replacement entered it, never leaves at a back-edge to be
    /// recompiled: a function compiled while its callers took only one
    /// path (a ray tracer's `trace` over the first rows, all sky) would
    /// keep that path's code. The entry counts calls down in the body's
    /// persistent array (`global_persist::ENTRY_POLL`) and at zero asks
    /// [`JitProto::entry_poll`].
    pub(crate) static ENTRY_POLL: core::cell::Cell<bool> = const { core::cell::Cell::new(false) };
}

/// Runs `f` (a generic compile) with `ENTRY_POLL` set to `on`.
pub fn with_entry_poll<R>(on: bool, f: impl FnOnce() -> R) -> R {
    let was = ENTRY_POLL.with(|e| e.replace(on));
    let r = f();
    ENTRY_POLL.with(|e| e.set(was));
    r
}

/// How many times the native-call entry's polls double the wait while no
/// cold call site has taken a callee (`ENTRY_FIRST << round` calls; then
/// they stay that far apart).
const ENTRY_ROUNDS: u8 = 10;

/// The wait (in `ENTRY_FIRST`s of calls) before the poll that checks that
/// the cold call sites that took a callee are still as many.
const ENTRY_SETTLE: u64 = 8;

/// The code size (bytes) up to which a body whose cold call sites all
/// took callees no inliner could use is recompiled anyway, to drop its
/// native-call entry's poll ([`JitProto::entry_poll`]).
const ENTRY_RECOMPILE_SMALL: usize = 16 * 1024;

/// How many times a function's generic body may be recompiled.
pub const MAX_GENERATION: u8 = 3;

/// Whether a call site that settled on a callee no recompilation could
/// inline is no reason to recompile its caller (`JitProto::wants_recompile`
/// and the entry polls take an `inlinable` test of the callee), while a large
/// body whose cold call sites all settled so is recompiled at its entry poll
/// when its property sites learned something since ([`JitProto::entry_poll`]);
/// `KATAAN_JIT_COLDINL=0` turns it off (read once per process).
#[must_use]
pub fn cold_inl_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("KATAAN_JIT_COLDINL").is_none_or(|v| v != "0"))
}

/// Whether an inlined callee's property sites carry what their caches
/// learned into the recompilation of the body that inlined them
/// ([`JitProto::inlined_warm`]); `KATAAN_JIT_INLWARM=0` turns it off (read
/// once per process). Off, a recompilation compiles an inlined callee's
/// sites from the callee's own feedback only (which a callee that only
/// ever runs inlined never improves), and the back-edge poll still asks
/// for recompilations on account of those sites.
#[must_use]
pub fn inl_warm_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("KATAAN_JIT_INLWARM").is_none_or(|v| v != "0"))
}

/// What a body's recompilation checks need, recorded at compile time.
pub(super) struct TierUp {
    /// `(pc, site)` of every `GetProp`/`SetProp`, by pc.
    pub(super) site_pcs: Vec<(usize, u16)>,
    /// Per site: whether it is a `SetProp`'s.
    pub(super) site_writes: Vec<bool>,
    /// Per site: its pc (`usize::MAX` for a site number no op has).
    pub(super) site_pc: Vec<usize>,
    /// Per loop header (ascending): the `(site, is a write)` of the property
    /// sites between it and its last back-edge.
    pub(super) loops: Vec<(usize, Vec<(u16, bool)>)>,
    /// Per site: the layout compiled into its code, if any, and the slot.
    pub(super) baked: Vec<Option<(Rc<Shape>, u32)>>,
    /// Per add site: the transition compiled into its code
    /// (`props::BakedAdd`), as the add's result shape and the new slot.
    pub(super) added: Vec<Option<(Rc<Shape>, u32)>>,
    /// Every loop's `(header, last back-edge)`, ascending.
    pub(super) extents: Vec<(usize, usize)>,
    /// Per site: whether its compiled-in layout had its slot as Number-only
    /// (`Shape::may_be_non_number`), which a speculative loop's fast
    /// version may rely on (`numeric::Spec::fields`): once the slot may hold
    /// anything, that version is never entered again, and a recompilation
    /// gives the loop one that does not rely on it.
    pub(super) num_slots: Vec<bool>,
    /// `(pc, call cache)` of the call sites that had no callee to inline
    /// when the body was compiled ([`JitProto::note_cold_calls`]): one whose
    /// cache has since settled on a single callee is worth a recompilation
    /// that may inline it.
    pub(super) cold_calls: Vec<(usize, u16)>,
    /// The call caches of the [`Self::cold_calls`] outside every loop (the
    /// ones [`JitProto::wants_recompile_on_entry`] checks).
    pub(super) entry_cold: Vec<u16>,
    /// `(pc, call cache)` of every inlined call's guard: one whose misses
    /// turned a speculative loop's fast version off
    /// (`call_seq::CallSite::spec_misses`) is worth a recompilation, which
    /// inlines what the call site has seen since.
    pub(super) guard_calls: Vec<(usize, u16)>,
    /// How many recompilations produced this body (`0`: the first compile).
    pub(super) generation: u8,
    /// `(header, region)` of the speculative loops: one whose fast version
    /// keeps leaving for its plain copy (`numeric`'s leave counter, in the
    /// persistent array) is worth a recompilation without it.
    pub(super) spec_regions: Vec<(usize, usize)>,
    /// The function's own header pcs of the loops this body was compiled
    /// with no speculative version for (inherited by its recompilations).
    pub(super) spec_off: Vec<usize>,
    /// `(header, end, region, sites)` of the loops whose integer version
    /// has overflow bails (`numeric::NumGen::demote_heads`): one that took any
    /// (its overflow-bail mask, in the persistent array, not empty) is
    /// worth a recompilation in which the sites' results are doubles and
    /// the rest of the loop stays integer.
    pub(super) int_regions: Vec<super::numeric::DemoteRegion>,
    /// The sites (the function's own pc, the register) whose values this
    /// body was compiled to keep as doubles (inherited by its
    /// recompilations).
    pub(super) int_demote: Vec<(usize, u16)>,
    /// The function's own element access sites' feedback this body was
    /// compiled with (`(own pc, feedback)`, `ic::elem_fb`; inherited by
    /// its recompilations, which may run while the interpreter's caches are
    /// out of the function's tier).
    pub(super) elem_kinds: Vec<(usize, u8)>,
    /// Whether a loop of the function's own encloses an inlined callee's
    /// loop (`numeric::leave_head`): an activation of this body can then
    /// be asked to leave at the next such loop's back-edge
    /// ([`JitProto::leaves_soon`]).
    pub(super) leave_heads: bool,
    /// Polls left until the next real check ([`CHECK_EVERY`]).
    pub(super) countdown: core::cell::Cell<u8>,
    /// Whether [`JitProto::wants_recompile_on_entry`] has said yes (it
    /// does once).
    pub(super) entry_asked: core::cell::Cell<bool>,
    /// How many cold call sites had taken a callee at the native-call
    /// entry's last poll ([`JitProto::entry_poll`]).
    pub(super) entry_seen: core::cell::Cell<u32>,
    /// The native-call entry's polls so far.
    pub(super) entry_round: core::cell::Cell<u8>,
    /// The callee key [`JitProto::wants_recompile_on_entry`] last found not
    /// inlinable (`call_seq::CallSite::EMPTY` for none).
    pub(super) not_inlinable: core::cell::Cell<u64>,
}

/// The flag of a demotion site ([`JitProto::int_demote`]) inside an inlined
/// callee: the rest is its pc in the inlining body's numbering, not the
/// function's own. A recompilation that inlines the same callees numbers
/// them the same way; one that does not finds another op there, or one that
/// does not name the register, and ignores the site. (Demoting a register
/// only makes an integer version keep that value as a double: whatever it
/// is applied to, the code stays exact.)
pub(super) const INLINED_SITE: usize = !(usize::MAX >> 1);

/// Whether `op` names register `r`.
pub(super) fn op_names(op: Option<&crate::nbvm::Op>, r: u16) -> bool {
    let hit = core::cell::Cell::new(false);
    if let Some(op) = op {
        crate::bytecode::op_for_each_reg(op, &|x| hit.set(hit.get() || x == r));
    }
    hit.get()
}

/// The hits a compiled-in layout counts as when it seeds a recompilation
/// (enough to be compiled in again).
const BAKED_HITS: u32 = 1 << 16;

/// One back-edge poll in this many checks the sites: a long loop polls
/// every thousand or so iterations, but a short-lived activation polls at its
/// first back-edge, and a function called in a hot loop would otherwise pay
/// for a check on every call.
const CHECK_EVERY: u8 = 8;

/// The callee key the call site whose cache is `cache` has settled on (one
/// way filled, the others empty), if it has.
fn settled_call(s: &super::GenericSites, cache: u16) -> Option<u64> {
    let c = s.calls.get(cache as usize)?;
    // SAFETY: the body that writes this cache (through its raw pointer) is
    // suspended in the caller's call, or not running, and no other borrow
    // of it is live.
    #[allow(unsafe_code)]
    let c = unsafe { &*c.get() };
    let mut keys = c
        .targets
        .iter()
        .map(|t| t.key)
        .filter(|&k| k != super::call_seq::CallSite::EMPTY);
    let k = keys.next()?;
    keys.next().is_none().then_some(k)
}

/// Whether the call site whose cache is `cache` has taken a callee an
/// inlining recompilation could use — one or a few, none of them function
/// `own` (which never inlines itself), all of them `inlinable` — if it has
/// taken any.
fn took_callee(
    s: &super::GenericSites,
    cache: u16,
    own: u32,
    inlinable: &dyn Fn(u64) -> bool,
) -> Option<bool> {
    let c = s.calls.get(cache as usize)?;
    // SAFETY: as in `settled_call`.
    #[allow(unsafe_code)]
    let c = unsafe { &*c.get() };
    let keys = || {
        c.targets
            .iter()
            .map(|t| t.key)
            .filter(|&k| k != super::call_seq::CallSite::EMPTY)
    };
    let n = keys().count();
    (n > 0).then(|| {
        n <= crate::ic::CalleeHint::POLY_MAX
            && keys().all(|k| k & 0xffff_ffff != u64::from(own) && inlinable(k))
    })
}

impl TierUp {
    /// The checks' data for lowered `ops`, whose sites `baked` has compiled
    /// in.
    pub(super) fn plan(
        ops: &[GenericOp],
        baked: Vec<Option<(Rc<Shape>, u32)>>,
        added: Vec<Option<(Rc<Shape>, u32)>>,
    ) -> Self {
        let mut site_pcs = Vec::new();
        let mut site_writes = Vec::new();
        let mut site_pc = Vec::new();
        for (pc, op) in ops.iter().enumerate() {
            if let GenericOp::GetProp { site, .. } | GenericOp::SetProp { site, .. } = *op {
                site_pcs.push((pc, site));
                if site_writes.len() <= site as usize {
                    site_writes.resize(site as usize + 1, false);
                    site_pc.resize(site as usize + 1, usize::MAX);
                }
                site_writes[site as usize] = matches!(op, GenericOp::SetProp { .. });
                site_pc[site as usize] = pc;
            }
        }
        // Each header's extent: up to its last back-edge.
        let mut ends: Vec<(usize, usize)> = Vec::new();
        for (pc, op) in ops.iter().enumerate() {
            if let GenericOp::Jump { target } | GenericOp::JumpIfFalse { target, .. } = *op
                && target <= pc
            {
                match ends.iter_mut().find(|(h, _)| *h == target) {
                    Some(e) => e.1 = e.1.max(pc),
                    None => ends.push((target, pc)),
                }
            }
        }
        ends.sort_unstable();
        let extents = ends.clone();
        let loops = ends
            .into_iter()
            .map(|(head, end)| {
                let sites = ops[head..=end]
                    .iter()
                    .filter_map(|op| match *op {
                        GenericOp::GetProp { site, .. } => Some((site, false)),
                        GenericOp::SetProp { site, .. } => Some((site, true)),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                (head, sites)
            })
            .filter(|(_, sites)| !sites.is_empty())
            .collect();
        Self {
            site_pcs,
            site_writes,
            site_pc,
            loops,
            baked,
            added,
            extents,
            num_slots: Vec::new(),
            cold_calls: Vec::new(),
            entry_cold: Vec::new(),
            spec_regions: Vec::new(),
            spec_off: Vec::new(),
            int_regions: Vec::new(),
            int_demote: Vec::new(),
            elem_kinds: Vec::new(),
            leave_heads: false,
            guard_calls: ops
                .iter()
                .enumerate()
                .filter_map(|(pc, op)| {
                    let GenericOp::InlineGuard { call, .. } = *op else {
                        return None;
                    };
                    let cache = match *ops.get(call as usize)? {
                        GenericOp::CallMethod { cache, .. }
                        | GenericOp::CallValue { cache, .. }
                        | GenericOp::CallWithMethod { cache, .. }
                        | GenericOp::Construct { cache, .. } => cache,
                        GenericOp::GetProp { call, .. } | GenericOp::SetProp { call, .. } => call,
                        _ => return None,
                    };
                    Some((pc, cache))
                })
                .collect(),
            generation: 0,
            countdown: core::cell::Cell::new(1),
            entry_asked: core::cell::Cell::new(false),
            entry_seen: core::cell::Cell::new(0),
            entry_round: core::cell::Cell::new(0),
            not_inlinable: core::cell::Cell::new(super::call_seq::CallSite::EMPTY),
        }
    }
}

impl JitProto {
    /// How many recompilations produced this body (`0` for a first compile
    /// or a non-generic body).
    #[must_use]
    pub fn generation(&self) -> u8 {
        self.sites.as_ref().map_or(0, |s| s.tier_up.generation)
    }

    /// Records the call sites, by the function's own pc, that had no callee
    /// feedback when this body was compiled (so none to inline): the back-edge
    /// poll recompiles the body once one of them settles on a single callee
    /// (see [`JitProto::wants_recompile`]).
    pub fn note_cold_calls(&mut self, own_pcs: &[usize]) {
        let body: Vec<usize> = own_pcs.iter().map(|&pc| self.body_pc(pc)).collect();
        let Some(s) = self.sites.as_mut() else {
            return;
        };
        s.tier_up.cold_calls = body
            .into_iter()
            .filter_map(|pc| {
                let i = s
                    .call_pcs
                    .binary_search_by_key(&pc, |&(p, _)| p as usize)
                    .ok()?;
                Some((pc, s.call_pcs[i].1))
            })
            .collect();
        let t = &mut s.tier_up;
        t.entry_cold = t
            .cold_calls
            .iter()
            .filter(|&&(pc, _)| {
                !t.extents
                    .iter()
                    .any(|&(head, end)| (head..=end).contains(&pc))
            })
            .map(|&(_, cache)| cache)
            .collect();
    }

    /// The function's own element access sites' feedback this body was
    /// compiled with (`(own pc, feedback)`, `ic::elem_fb`): what its
    /// recompilation starts from.
    #[must_use]
    pub fn elem_kinds(&self) -> Vec<(usize, u8)> {
        self.sites
            .as_ref()
            .map_or_else(Vec::new, |s| s.tier_up.elem_kinds.clone())
    }

    /// The function's own header pcs of the loops a recompilation of this
    /// body should give no speculative version: this body's own such loops,
    /// and those whose fast version kept leaving for its plain copy.
    #[must_use]
    pub fn spec_off(&self) -> Vec<usize> {
        let Some(s) = self.sites.as_ref() else {
            return Vec::new();
        };
        let t = &s.tier_up;
        let mut out = t.spec_off.clone();
        if s.deopt
            .as_ref()
            .and_then(|d| d.sink.as_ref())
            .is_some_and(|x| x.forced.get() >= super::sink::exit_limit())
            && !out.contains(&super::SINK_OFF)
        {
            out.push(super::SINK_OFF);
        }
        for &(h, ri) in &t.spec_regions {
            let pc = self.orig_pc(h);
            if pc != usize::MAX
                && s.persist.leaves(ri) >= super::numeric::SPEC_LEAVE_LIMIT
                && !out.contains(&pc)
            {
                out.push(pc);
            }
        }
        out
    }

    /// The sites (the function's own pc, the register) whose values a
    /// recompilation of this body should keep as doubles in its integer
    /// versions: this body's own such sites, and those whose overflow bails
    /// (or entry failures) it took. A site's pc must be the function's own,
    /// and its register one some own op of the loop names (an inlined
    /// callee's registers are numbered by the inlining, which a
    /// recompilation may do otherwise).
    #[must_use]
    pub fn int_demote(&self) -> Vec<(usize, u16)> {
        let Some(s) = self.sites.as_ref() else {
            return Vec::new();
        };
        let t = &s.tier_up;
        let mut out = t.int_demote.clone();
        let last = super::numeric::DEMOTE_BITS - 1;
        let ops = self.bytecode.as_deref().unwrap_or(&[]);
        for (head, end, ri, sites) in &t.int_regions {
            let mask = s.persist.demotes(*ri);
            for (k, &(pc, r)) in sites.iter().enumerate() {
                if mask & (1 << k.min(last)) == 0 {
                    continue;
                }
                let own = self.orig_pc(pc);
                let named = (*head..=*end).any(|q| {
                    let hit = core::cell::Cell::new(false);
                    if self.orig_pc(q) != usize::MAX
                        && let Some(op) = ops.get(q)
                    {
                        crate::bytecode::op_for_each_reg(op, &|x| hit.set(hit.get() || x == r));
                    }
                    hit.get()
                });
                if own != usize::MAX && named && !out.contains(&(own, r)) {
                    out.push((own, r));
                }
                // A site inside an inlined callee's loop (`CG` calling an
                // inlined `A.mul`): kept by its pc in this body's numbering,
                // which a recompilation applies only where its own splice
                // has an inlined op there naming the register (see
                // [`INLINED_SITE`]).
                if own == usize::MAX
                    && op_names(ops.get(pc), r)
                    && !out.contains(&(pc | INLINED_SITE, r))
                {
                    out.push((pc | INLINED_SITE, r));
                }
            }
        }
        out
    }

    /// [`Self::int_demote`]'s sites at the function's own pcs (what a caller
    /// that inlines the function applies to its spliced copy,
    /// `inline::Candidate::demote`).
    #[must_use]
    pub fn own_int_demote(&self) -> Vec<(usize, u16)> {
        let mut v = self.int_demote();
        v.retain(|&(pc, _)| pc & INLINED_SITE == 0);
        v
    }

    /// Whether an activation of this body that wants to be recompiled for
    /// a demotion ([`Self::wants_demote`]) can be asked to leave at the
    /// next back-edge of a loop of its own that encloses an inlined
    /// callee's loop (`numeric::leave_head`): it has such a loop.
    #[must_use]
    pub fn leaves_soon(&self) -> bool {
        self.sites.as_ref().is_some_and(|s| s.tier_up.leave_heads) && self.wants_demote()
    }

    /// Whether one of this body's integer versions found a value no int32
    /// that a recompilation would demote (its loop's back-edge poll asks
    /// [`JitProto::wants_recompile`] too), while the budget lasts.
    #[must_use]
    pub fn wants_demote(&self) -> bool {
        let Some(s) = self.sites.as_ref() else {
            return false;
        };
        let t = &s.tier_up;
        t.generation < MAX_GENERATION
            && t.int_regions
                .iter()
                .any(|(_, _, ri, _)| s.persist.demotes(*ri) != 0)
    }

    /// Whether this body's code runs bytecode `ops` (the stream its back-edge
    /// polls name).
    #[must_use]
    pub fn runs(&self, ops: *const crate::nbvm::Op) -> bool {
        self.bytecode
            .as_deref()
            .is_some_and(|b| core::ptr::eq(b.as_ptr(), ops))
    }

    /// Marks this body as the `generation`th recompilation of its function.
    pub fn set_generation(&mut self, generation: u8) {
        if let Some(s) = self.sites.as_mut() {
            s.tier_up.generation = generation;
        }
    }

    /// Whether recompiling would give the loop at header pc `header` better
    /// code: one of its property sites' runtime caches now holds a stable
    /// layout (the same test the compiler applies to the interpreter's) other
    /// than the one compiled in, if any — while the function's
    /// recompilation budget lasts. (A cold call site that settled counts
    /// only when its callee is `inlinable`, by its key: a recompilation
    /// could not inline any other.)
    #[must_use]
    pub fn wants_recompile(&self, header: usize, inlinable: &dyn Fn(u64) -> bool) -> bool {
        let Some(s) = self.sites.as_ref() else {
            return false;
        };
        let t = &s.tier_up;
        if t.generation >= MAX_GENERATION {
            return false;
        }
        // A loop whose integer version found a result no int32.
        if t.int_regions
            .iter()
            .any(|(h, _, ri, _)| *h == header && s.persist.demotes(*ri) != 0)
        {
            return true;
        }
        // Or such a loop of an inlined callee's inside it (which never polls
        // at a header of the function's own).
        if let Ok(i) = t.extents.binary_search_by_key(&header, |l| l.0) {
            let (head, end) = t.extents[i];
            if t.int_regions.iter().any(|(h, _, ri, _)| {
                (head..=end).contains(h)
                    && self.orig_pc(*h) == usize::MAX
                    && s.persist.demotes(*ri) != 0
            }) {
                return true;
            }
        }
        // A speculative loop whose fast version keeps leaving.
        if t.spec_regions
            .iter()
            .any(|&(h, ri)| h == header && s.persist.leaves(ri) >= super::numeric::SPEC_LEAVE_LIMIT)
        {
            return true;
        }
        let props = t.loops.binary_search_by_key(&header, |l| l.0).ok();
        let calls = t
            .extents
            .binary_search_by_key(&header, |l| l.0)
            .ok()
            .map(|i| t.extents[i])
            .filter(|&(head, end)| {
                t.cold_calls
                    .iter()
                    .any(|&(pc, _)| (head..=end).contains(&pc))
            });
        // A guard whose misses turned the loop's fast version off.
        if let Ok(i) = t.extents.binary_search_by_key(&header, |l| l.0) {
            let (head, end) = t.extents[i];
            if t.guard_calls.iter().any(|&(pc, cache)| {
                (head..=end).contains(&pc)
                    && s.calls.get(cache as usize).is_some_and(|c| {
                        // SAFETY: as for the property caches below.
                        #[allow(unsafe_code)]
                        let c = unsafe { &*c.get() };
                        c.spec_misses >= super::call_seq::SPEC_MISS_LIMIT
                    })
            }) {
                return true;
            }
        }
        if props.is_none() && calls.is_none() {
            return false;
        }
        let left = t.countdown.get() - 1;
        if left > 0 {
            t.countdown.set(left);
            return false;
        }
        t.countdown.set(CHECK_EVERY);
        if let Some((head, end)) = calls
            && t.cold_calls.iter().any(|&(pc, cache)| {
                (head..=end).contains(&pc) && settled_call(s, cache).is_some_and(inlinable)
            })
        {
            return true;
        }
        let Some(i) = props else {
            return false;
        };
        t.loops[i]
            .1
            .iter()
            .any(|&(site, write)| self.site_learned(site as usize, write))
    }

    /// Whether property site `site` (a write if `write`) learned something a
    /// recompilation would compile in: its compiled-in layout's slot, which
    /// the code relies on holding only Numbers, no longer does; or its cache
    /// now holds a stable layout other than the one compiled in (a write's
    /// by a plain store), which has the key at the cached slot.
    fn site_learned(&self, site: usize, write: bool) -> bool {
        let Some(s) = self.sites.as_ref() else {
            return false;
        };
        let t = &s.tier_up;
        // A Number-only slot the loop's code may rely on no longer is.
        if !write
            && t.num_slots.get(site).copied().unwrap_or(false)
            && let Some(Some((shape, slot))) = t.baked.get(site)
            && shape.may_be_non_number(*slot)
        {
            return true;
        }
        let (Some(cache), Some(key)) = (s.caches.get(site), s.keys.get(site)) else {
            return false;
        };
        // (An inlined callee's site whose origin is unknown: no
        // recompilation would compile what its cache learned in.)
        if inl_warm_enabled()
            && t.site_pc.get(site).is_some_and(|&pc| {
                pc != usize::MAX && self.orig_pc(pc) == usize::MAX && self.inl_src(pc).is_none()
            })
        {
            return false;
        }
        // SAFETY: the body that writes this cache (through its raw pointer)
        // is suspended in the poll's call, and no other borrow of it is live.
        #[allow(unsafe_code)]
        let Some((shape, slot, by_add)) = (unsafe { (*cache.get()).stable_memo() }) else {
            return false;
        };
        !(write && (by_add || shape.is_wrapper()))
            && !t
                .baked
                .get(site)
                .is_some_and(|b| b.as_ref().is_some_and(|(s, _)| Rc::ptr_eq(s, shape)))
            && shape.lookup(key) == Some(slot)
    }

    /// Whether any property site of this body learned something a
    /// recompilation would compile in ([`Self::site_learned`]).
    fn any_site_learned(&self) -> bool {
        let Some(s) = self.sites.as_ref() else {
            return false;
        };
        s.tier_up.site_pcs.iter().any(|&(_, site)| {
            self.site_learned(
                site as usize,
                s.tier_up
                    .site_writes
                    .get(site as usize)
                    .copied()
                    .unwrap_or(true),
            )
        })
    }

    /// Whether a call of this body from Rust code (an interpreted caller's)
    /// should recompile it first: a call site outside every loop that had
    /// no callee to inline when it was compiled has since settled on a
    /// single callee — which no back-edge poll would see (see
    /// [`JitProto::wants_recompile`]): a recursive builder `mk` compiled
    /// before its `new T(…)` ever ran gets its constructor inlined at its
    /// next call from the interpreter — while the recompilation budget
    /// lasts. (A recursive call, settled on the function `own` itself, is
    /// no reason: a function never inlines itself. A call from Rust code
    /// costs far more than the check, so every one checks, until the body
    /// has said yes once: a site that settled on a callee that could not
    /// be inlined is no cold site of the recompilation; nor is one whose
    /// callee is not `inlinable`, by its key.)
    #[must_use]
    pub fn wants_recompile_on_entry(&self, own: u32, inlinable: &dyn Fn(u64) -> bool) -> bool {
        let Some(s) = self.sites.as_ref() else {
            return false;
        };
        let t = &s.tier_up;
        if t.generation >= MAX_GENERATION || t.entry_cold.is_empty() || t.entry_asked.get() {
            return false;
        }
        // (A callee found not inlinable is remembered: this runs on every
        // call from Rust code.)
        let usable = |k: u64| {
            if k == t.not_inlinable.get() {
                return false;
            }
            let ok = inlinable(k);
            if !ok {
                t.not_inlinable.set(k);
            }
            ok
        };
        let yes = t.entry_cold.iter().any(|&cache| {
            settled_call(s, cache).is_some_and(|k| k & 0xffff_ffff != u64::from(own) && usable(k))
        });
        // (Nor does it ask again once each such site has taken callees:
        // the answer can no longer change.)
        let done = yes
            || (cold_inl_enabled()
                && t.entry_cold
                    .iter()
                    .all(|&cache| took_callee(s, cache, own, &|_| true).is_some()));
        t.entry_asked.set(done);
        yes
    }

    /// The native-call entry's recompilation poll ([`ENTRY_POLL`]), its
    /// countdown run out: whether function `own`'s body should be
    /// recompiled now (for its later calls; this one continues in this
    /// code). Yes once the cold call sites that took a callee the inliner
    /// could use (one, or a few: not the function itself) are as many as
    /// at the previous poll, and some: those sites have settled since this
    /// body was compiled, and the rest are no longer being reached. While
    /// none has, each poll waits twice as long as the one before (up to
    /// [`ENTRY_ROUNDS`] doublings); once some have, the next poll comes
    /// soon. All while the recompilation budget lasts. (A callee counts as
    /// usable only when `inlinable`, by its key.)
    #[must_use]
    pub fn entry_poll(&self, own: u32, inlinable: &dyn Fn(u64) -> bool) -> bool {
        let Some(s) = self.sites.as_ref() else {
            return false;
        };
        let t = &s.tier_up;
        if t.generation >= MAX_GENERATION {
            s.persist.set_entry_poll(u64::MAX);
            return false;
        }
        let (mut learned, mut open) = (0u32, 0u32);
        for &(_, cache) in &t.cold_calls {
            match took_callee(s, cache, own, inlinable) {
                Some(true) => learned += 1,
                Some(false) => {}
                None => open += 1,
            }
        }
        if learned > 0 && learned == t.entry_seen.get() {
            s.persist.set_entry_poll(u64::MAX);
            return true;
        }
        if learned == 0 && open == 0 {
            // Every cold site took callees, none an inliner could use (the
            // function itself, as a recursive `fib`'s second call; or too
            // many): nothing more to learn. A small body is recompiled once
            // more all the same — its recompilation has no cold sites,
            // so no poll costing every call; a large one stops polling,
            // unless its property sites learned since (a body only compiled
            // callers enter never polls at a back-edge for them).
            s.persist.set_entry_poll(u64::MAX);
            return self.code_len() <= ENTRY_RECOMPILE_SMALL
                || (cold_inl_enabled() && self.any_site_learned());
        }
        t.entry_seen.set(learned);
        let first = super::global_persist::ENTRY_FIRST;
        let next = if learned > 0 {
            first * ENTRY_SETTLE
        } else {
            let round = t.entry_round.get().saturating_add(1).min(ENTRY_ROUNDS);
            t.entry_round.set(round);
            first << round
        };
        s.persist.set_entry_poll(next);
        false
    }

    /// The address of this body's native-call entry poll countdown
    /// ([`ENTRY_POLL`]), which the entry passes its helper.
    #[must_use]
    pub fn entry_poll_addr(&self) -> u64 {
        self.sites
            .as_ref()
            .map_or(0, |s| s.persist.entry_poll_addr())
    }

    /// Where body pc `pc`'s op came from when an inlined callee brought it:
    /// `(function id, its pc there)` (`inline::Expanded::src_of`).
    #[must_use]
    pub fn inl_src(&self, pc: usize) -> Option<(u32, usize)> {
        let s = *self.pc_map.as_ref()?.2.get(pc)?;
        (s != super::inline::NO_SRC).then_some(((s >> 32) as u32, (s & 0xffff_ffff) as usize))
    }

    /// The memos the property-site caches of the callees this body inlined
    /// hold, as `(function id, its pc, memo, live)`, with the same rules as
    /// [`Self::runtime_warm`] (a compiled-in layout counts as settled unless
    /// the cache settled on another; `live`: the cache's own stable memo,
    /// not the compiled-in layout): what a recompilation that inlines the
    /// same callees compiles their spliced sites from, before their own
    /// feedback (which a callee that only runs inlined never improves). A
    /// site whose cache knows nothing stable is left out.
    #[must_use]
    pub fn inlined_warm(&self) -> Vec<(u32, usize, crate::ic::WarmEntry, bool)> {
        let Some(s) = self.sites.as_ref() else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for &(pc, site) in &s.tier_up.site_pcs {
            if self.orig_pc(pc) != usize::MAX {
                continue;
            }
            let (Some((f, fpc)), Some(c)) = (self.inl_src(pc), s.caches.get(site as usize)) else {
                continue;
            };
            // SAFETY: as in `wants_recompile` (or the body is not running).
            #[allow(unsafe_code)]
            let w = unsafe { (*c.get()).warm_entry() };
            let baked = s.tier_up.baked.get(site as usize).cloned().flatten();
            let added = s.tier_up.added.get(site as usize).cloned().flatten();
            let (e, live) = match (w, baked, added) {
                (Some(w), _, _) if w.is_stable() => (w, true),
                (_, Some((shape, slot)), _) => (
                    crate::ic::WarmEntry {
                        shape,
                        slot,
                        hits: BAKED_HITS,
                        misses: 0,
                        by_add: false,
                    },
                    false,
                ),
                (_, None, Some((shape, slot))) => (
                    crate::ic::WarmEntry {
                        shape,
                        slot,
                        hits: BAKED_HITS,
                        misses: 0,
                        by_add: true,
                    },
                    false,
                ),
                _ => continue,
            };
            out.push((f, fpc, e, live));
        }
        out
    }

    /// The memos this body's own property-site caches hold, by the
    /// function's own bytecode pc (the warm start of its recompilation; an
    /// inlined body's sites are left out). A site with a layout compiled
    /// in reports that layout as settled unless its cache settled on
    /// another: the compiled-in compare, not the cache, counts its hits
    /// (likewise a compiled-in add, as an add memo).
    #[must_use]
    pub fn runtime_warm(&self) -> Vec<Option<crate::ic::WarmEntry>> {
        let Some(s) = self.sites.as_ref() else {
            return Vec::new();
        };
        let n = s
            .tier_up
            .site_pcs
            .iter()
            .map(|&(pc, _)| self.orig_pc(pc))
            .filter(|&pc| pc != usize::MAX)
            .max()
            .map_or(0, |pc| pc + 1);
        let mut out = alloc::vec![None; n];
        for &(pc, site) in &s.tier_up.site_pcs {
            let pc = self.orig_pc(pc);
            let Some(c) = s.caches.get(site as usize).filter(|_| pc != usize::MAX) else {
                continue;
            };
            // SAFETY: as in `wants_recompile` (or the body is not running).
            #[allow(unsafe_code)]
            let w = unsafe { (*c.get()).warm_entry() };
            let baked = s.tier_up.baked.get(site as usize).cloned().flatten();
            let added = s.tier_up.added.get(site as usize).cloned().flatten();
            out[pc] = match (w, baked, added) {
                (Some(w), _, _) if w.is_stable() => Some(w),
                (_, Some((shape, slot)), _) => Some(crate::ic::WarmEntry {
                    shape,
                    slot,
                    hits: BAKED_HITS,
                    misses: 0,
                    by_add: false,
                }),
                // A compiled-in add counts no hits in the cache either (an
                // inlined copy of the body, or its recompilation, then
                // compiles it in again).
                (_, None, Some((shape, slot))) => Some(crate::ic::WarmEntry {
                    shape,
                    slot,
                    hits: BAKED_HITS,
                    misses: 0,
                    by_add: true,
                }),
                (w, None, None) => w,
            };
        }
        out
    }
}
