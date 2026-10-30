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

/// How many times a function's generic body may be recompiled.
pub const MAX_GENERATION: u8 = 3;

/// What a body's recompilation checks need, recorded at compile time.
pub(super) struct TierUp {
    /// `(pc, site)` of every `GetProp`/`SetProp`, by pc.
    pub(super) site_pcs: Vec<(usize, u16)>,
    /// Per loop header (ascending): the `(site, is a write)` of the property
    /// sites between it and its last back-edge.
    pub(super) loops: Vec<(usize, Vec<(u16, bool)>)>,
    /// Per site: the layout compiled into its code, if any, and the slot.
    pub(super) baked: Vec<Option<(Rc<Shape>, u32)>>,
    /// Per add site: the transition compiled into its code
    /// (`props::BakedAdd`), as the add's result shape and the new slot.
    pub(super) added: Vec<Option<(Rc<Shape>, u32)>>,
    /// How many recompilations produced this body (`0`: the first compile).
    pub(super) generation: u8,
    /// Polls left until the next real check ([`CHECK_EVERY`]).
    pub(super) countdown: core::cell::Cell<u8>,
}

/// The hits a compiled-in layout counts as when it seeds a recompilation
/// (enough to be compiled in again).
const BAKED_HITS: u32 = 1 << 16;

/// One back-edge poll in this many checks the sites: a long loop polls
/// every thousand or so iterations, but a short-lived activation polls at its
/// first back-edge, and a function called in a hot loop would otherwise pay
/// for a check on every call.
const CHECK_EVERY: u8 = 8;

impl TierUp {
    /// The checks' data for lowered `ops`, whose sites `baked` has compiled
    /// in.
    pub(super) fn plan(
        ops: &[GenericOp],
        baked: Vec<Option<(Rc<Shape>, u32)>>,
        added: Vec<Option<(Rc<Shape>, u32)>>,
    ) -> Self {
        let mut site_pcs = Vec::new();
        for (pc, op) in ops.iter().enumerate() {
            if let GenericOp::GetProp { site, .. } | GenericOp::SetProp { site, .. } = *op {
                site_pcs.push((pc, site));
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
            loops,
            baked,
            added,
            generation: 0,
            countdown: core::cell::Cell::new(1),
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
    /// recompilation budget lasts.
    #[must_use]
    pub fn wants_recompile(&self, header: usize) -> bool {
        let Some(s) = self.sites.as_ref() else {
            return false;
        };
        let t = &s.tier_up;
        if t.generation >= MAX_GENERATION {
            return false;
        }
        let Ok(i) = t.loops.binary_search_by_key(&header, |l| l.0) else {
            return false;
        };
        let left = t.countdown.get() - 1;
        if left > 0 {
            t.countdown.set(left);
            return false;
        }
        t.countdown.set(CHECK_EVERY);
        t.loops[i].1.iter().any(|&(site, write)| {
            let site = site as usize;
            let (Some(cache), Some(key)) = (s.caches.get(site), s.keys.get(site)) else {
                return false;
            };
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
        })
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
