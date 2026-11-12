//! Polymorphic property sites in the generic tier (`ROADMAP.md` §2.1).
//!
//! A property site compiles in at most one layout (`Baked`): the one its
//! cache settled on while it was interpreted. A site that sees two to four
//! layouts — `c.queued` over a work queue of several constraint classes, or
//! a site whose objects change layout from one phase of a program to the
//! next — then compared every other layout with its one-entry cache, whose
//! helper re-armed it on each change, and a body recompiled for the newer
//! layout dropped the older one (so one that alternates recompiled until its
//! budget ran out).
//!
//! Here such a site also compiles in up to [`MAX_WAYS`] further layouts
//! ([`PolyWay`]): a short chain of shape compares, each with its own fixed
//! slot, tried after the site's compiled-in layout misses (out of line) or,
//! for a site with none, before its cache. They come from the layouts its
//! cache held while it was interpreted (`ic::PolyHist`, which also counts
//! each one's hits, so the most used one becomes the primary layout:
//! [`choose`]) and, for a recompilation, from those the old body had
//! compiled in. Each is a fact of an immutable layout (the key is a data
//! property in that slot), exactly what the cache hit relies on, so a way
//! hit does what that hit does; any other layout still takes the cache path
//! and its helper (the megamorphic fallback). A site whose history keeps
//! evicting layouts gets no ways.
//!
//! A body compiled in one phase of a program and only entered by compiled
//! callers never leaves at a back-edge to be recompiled: a site whose cache
//! path keeps hitting (a layout of a later phase) asks for the body's
//! recompilation every so many hits ([`X64Assembler::p_site_count`]), which
//! the next back-edge poll grants while the budget lasts
//! (`jit_tier::site_request`).
//!
//! `KATAAN_JIT_POLY=0` turns the ways off, `KATAAN_JIT_SITEPOLL=0` the
//! requests (each read once per process).

use super::call_seq::r;
use super::{JitLayout, Label, X64Assembler};
use crate::shape::Shape;
use alloc::rc::Rc;
use alloc::vec::Vec;

/// The most layouts a site compiles in beyond its primary one.
pub(crate) const MAX_WAYS: usize = 3;

/// A site whose history evicted this many layouts is megamorphic: no ways.
const MEGA_EVICTIONS: u32 = 8;

/// Whether polymorphic sites compile their ways in.
#[must_use]
pub(crate) fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("KATAAN_JIT_POLY").is_none_or(|v| v != "0"))
}

/// One further layout compiled into a property site: a receiver whose shape
/// word is `shape` has the site's key as a data property in slot `slot`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PolyWay {
    pub shape: u64,
    pub slot: u32,
    /// Whether the layout had the slot as Number-only when compiled
    /// (`Shape::may_be_non_number`): a store of anything else then checks.
    pub num: bool,
}

/// The ways of a site: per site, up to [`MAX_WAYS`] (unused `None`).
pub(crate) type Ways = [Option<PolyWay>; MAX_WAYS];

/// Whether `shape` may be one of a site's ways: it has `key` as a data
/// property in slot `slot`, and is neither a primitive wrapper's nor
/// `[[IsHTMLDDA]]`'s layout.
fn usable(shape: &Rc<Shape>, slot: u32, key: &str) -> bool {
    shape.lookup(key) == Some(slot)
        && !shape.is_wrapper()
        && !shape.has_brand(crate::shape::brand_bit(crate::shape::BRAND_KEYS[1]))
}

/// A site's primary layout (if any) and its further layouts, as
/// [`choose`] answers.
pub(crate) type Choice = (Option<(Rc<Shape>, u32)>, Vec<(Rc<Shape>, u32)>);

/// What a property site of key `key` (a write's if `write`) compiles in
/// from its memo `w`: its primary layout (`Baked`), if any, and its further
/// layouts (at most [`MAX_WAYS`]). `legacy` is whether the memo alone
/// would compile its layout in (`WarmEntry::is_stable`, and no add's result
/// or primitive wrapper for a write), which is the answer when disabled, for
/// a megamorphic site (no ways) and for a site that held one usable layout.
/// Otherwise the layouts the site held, by the hits counted while it held
/// each: the most used one is the primary when it is settled as
/// `WarmEntry::is_stable` asks of a memo (not always the one the cache holds
/// now), the rest are ways; with none settled, the memo's layout stays the
/// primary if `legacy`, else all of them are ways and the site has no
/// primary. An add's site keeps its add (no primary), its other layouts
/// ways.
pub(crate) fn choose(w: &crate::ic::WarmEntry, key: &str, write: bool, legacy: bool) -> Choice {
    let old = || legacy.then(|| (Rc::clone(&w.shape), w.slot));
    if !enabled() || w.evictions >= MEGA_EVICTIONS || key.starts_with('\0') {
        return (old(), Vec::new());
    }
    let current = crate::ic::Seen {
        shape: Rc::clone(&w.shape),
        slot: w.slot,
        hits: w.tally,
        by_add: w.by_add,
    };
    let mut cands: Vec<crate::ic::Seen> = Vec::new();
    for e in core::iter::once(current).chain(w.others.iter().cloned()) {
        // (A merged memo's lists may name a layout twice: the first counts.)
        if usable(&e.shape, e.slot, key)
            && !(write && e.by_add)
            && !cands.iter().any(|c| Rc::ptr_eq(&c.shape, &e.shape))
        {
            cands.push(e);
        }
    }
    // (The sort is stable: of equally used layouts, the current one first.)
    cands.sort_by_key(|e| core::cmp::Reverse(e.hits));
    let take = |c: Vec<crate::ic::Seen>, skip: Option<&Rc<Shape>>| -> Vec<(Rc<Shape>, u32)> {
        c.into_iter()
            .filter(|e| skip.is_none_or(|s| !Rc::ptr_eq(s, &e.shape)))
            .take(MAX_WAYS)
            .map(|e| (e.shape, e.slot))
            .collect()
    };
    if write && w.by_add {
        let ways = take(cands, None);
        return (None, if ways.len() >= 2 { ways } else { Vec::new() });
    }
    if cands.len() < 2 {
        return (old(), Vec::new());
    }
    let best = &cands[0];
    let settled = best.hits >= 4 && best.hits >= w.misses.saturating_mul(4);
    if settled && !(write && best.shape.is_wrapper()) {
        let first = cands.remove(0);
        (Some((first.shape, first.slot)), take(cands, None))
    } else if legacy {
        (old(), take(cands, Some(&w.shape)))
    } else {
        (None, take(cands, None))
    }
}

/// The emitter's form of a site's further layouts.
pub(crate) fn ways_of(sel: &[(Rc<Shape>, u32)]) -> Ways {
    let mut w: Ways = [None; MAX_WAYS];
    for (slot, (s, k)) in w.iter_mut().zip(sel) {
        *slot = Some(PolyWay {
            shape: super::props::shape_word(s),
            slot: *k,
            num: !s.may_be_non_number(*k),
        });
    }
    w
}

/// Merges layout lists (first ones first), dropping repeats.
pub(crate) fn merge(lists: &[&[crate::ic::Seen]]) -> Vec<crate::ic::Seen> {
    let mut out: Vec<crate::ic::Seen> = Vec::new();
    for l in lists {
        for e in *l {
            if !out.iter().any(|o| Rc::ptr_eq(&o.shape, &e.shape)) {
                out.push(e.clone());
            }
        }
    }
    out
}

/// Every this many hits of a site's cache path (a layout its code does
/// not have compiled in), its body asks for a recompilation
/// ([`X64Assembler::p_site_count`]).
const SITE_REQ_MASK: u32 = (1 << 14) - 1;

impl X64Assembler {
    /// After a cache hit's count (`r11` the site's `PropertyCache`, whose
    /// hit count is at `hits`): every so many hits, the body's request —
    /// its bytecode's address `body` — for the next back-edge poll, which
    /// may have it recompiled (`jit_tier::site_request`; at `[r15 + req]`,
    /// `None` when no site asks). Clobbers `r8`.
    pub(crate) fn p_site_count(&mut self, hits: i32, req: Option<i32>, body: u64) {
        let Some(req) = req else {
            return;
        };
        let skip = self.new_label();
        // test dword [r11 + hits], SITE_REQ_MASK
        self.g_mem(false, &[0xf7], 0, r::R11, hits);
        self.code.extend_from_slice(&SITE_REQ_MASK.to_le_bytes());
        self.jne(skip);
        self.g_movabs(r::R8, body);
        self.g_store(r::R15, req, r::R8);
        self.bind(skip);
    }

    /// A read's ways, with the receiver's live heap slot in `rcx` (an
    /// ordinary shaped object): the first way whose layout the object has
    /// loads its slot into `rax` and goes to `hit`; with none, falls
    /// through. Clobbers `rax` (on a hit) and `r11`.
    pub(crate) fn p_poly_get(&mut self, l: &JitLayout, ways: &Ways, hit: Label) {
        for w in ways.iter().flatten() {
            let next = self.new_label();
            self.p_cmp_shape(r::RCX, l.off_shape, w.shape, r::R11);
            self.jne(next);
            self.p_obj_fixed_slot_load(l, w.slot);
            self.jmp(hit);
            self.bind(next);
        }
    }

    /// A read's ways for a site without a compiled-in layout: the first way
    /// whose layout the object in `rcx` has puts its slot index in `eax`
    /// and goes to `hit` (the cache hit's slot read); with none, falls
    /// through. Clobbers `rax` (on a hit) and `r11`.
    pub(crate) fn p_poly_slot(&mut self, l: &JitLayout, ways: &Ways, hit: Label) {
        for w in ways.iter().flatten() {
            let next = self.new_label();
            self.p_cmp_shape(r::RCX, l.off_shape, w.shape, r::R11);
            self.jne(next);
            self.g_movabs(r::RAX, u64::from(w.slot));
            self.jmp(hit);
            self.bind(next);
        }
    }

    /// A write's ways, with the receiver's live heap slot in `rcx` (an
    /// ordinary shaped object whose flags allow an in-place store, tested
    /// by the caller): the first way whose layout the object has stores the
    /// value (at `[rbx + val]`; `xmm`: in that SSE register instead, when
    /// `val_num`) in its slot, as the compiled-in layout's store does —
    /// the slot's record of Number-only values and the write barrier
    /// checked first unless the value is proven a Number (else to
    /// `helper`) — and goes to `done`; with none, falls through. Clobbers
    /// `rax`, `rdx`, `r8`–`r11`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn p_poly_set(
        &mut self,
        l: &JitLayout,
        realm_off: i32,
        ways: &Ways,
        (val, val_num, xmm): (i32, bool, Option<u8>),
        helper: Label,
        done: Label,
    ) {
        for w in ways.iter().flatten() {
            let next = self.new_label();
            self.p_cmp_shape(r::RCX, l.off_shape, w.shape, r::R11);
            self.jne(next);
            match (val_num, xmm) {
                (true, Some(x)) => super::numeric::NumGen::gpr_from_xmm(self, r::RDX, x),
                (true, None) => self.g_load(r::RDX, r::RBX, val),
                (false, _) => {
                    self.g_load(r::RAX, r::RBX, val);
                    if w.num {
                        self.p_field_rep_const(l, (r::RAX, r::R8), w.shape, w.slot, helper);
                    }
                    self.p_barrier_check(l, realm_off, helper);
                    self.g_load(r::RDX, r::RBX, val);
                }
            }
            self.p_obj_fixed_slot_store(l, w.slot);
            self.jmp(done);
            self.bind(next);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::Object;

    fn obj(keys: &[&str]) -> Object {
        let mut o = Object::new(Shape::root());
        for k in keys {
            o.set(k, crate::nanbox::NanBox::number(1.0));
        }
        o
    }

    fn seen(o: &Object, slot: u32, hits: u32) -> crate::ic::Seen {
        crate::ic::Seen {
            shape: Rc::clone(o.shape()),
            slot,
            hits,
            by_add: false,
        }
    }

    fn memo(
        o: &Object,
        (slot, tally, misses): (u32, u32, u32),
        others: Vec<crate::ic::Seen>,
    ) -> crate::ic::WarmEntry {
        crate::ic::WarmEntry {
            shape: Rc::clone(o.shape()),
            slot,
            hits: tally + others.iter().map(|e| e.hits).sum::<u32>(),
            misses,
            by_add: false,
            tally,
            others,
            evictions: 0,
        }
    }

    #[test]
    fn choose_takes_the_most_used_layout() {
        let a = obj(&["x", "q"]);
        let b = obj(&["q"]);
        let c = obj(&["y", "z", "q"]);
        let d = obj(&["w"]);
        // The cache holds `a`, used least; `b` is the most used.
        let others = alloc::vec![seen(&c, 2, 50), seen(&b, 0, 400), seen(&d, 0, 99)];
        let w = memo(&a, (1, 3, 20), others);
        let (p, ways) = choose(&w, "q", false, true);
        let (p, slot) = p.expect("primary");
        assert!(Rc::ptr_eq(&p, b.shape()) && slot == 0);
        // Most used first; a layout without the key is none.
        let slots: Vec<u32> = ways.iter().map(|e| e.1).collect();
        assert_eq!(slots, alloc::vec![2, 1]);
        // None settled: every layout is a way, no primary.
        let w = memo(&a, (1, 30, 100), alloc::vec![seen(&b, 0, 40)]);
        let (p, ways) = choose(&w, "q", true, false);
        assert!(p.is_none());
        assert_eq!(ways.len(), 2);
        assert!(Rc::ptr_eq(&ways[0].0, b.shape()));
        // A megamorphic site keeps the old answer.
        let mut m = memo(&a, (1, 300, 1), alloc::vec![seen(&b, 0, 40)]);
        m.evictions = MEGA_EVICTIONS;
        let (p, ways) = choose(&m, "q", false, true);
        assert!(p.is_some_and(|(s, _)| Rc::ptr_eq(&s, a.shape())) && ways.is_empty());
        // A wrong slot is no way.
        let w = memo(&a, (1, 300, 1), alloc::vec![seen(&b, 3, 40)]);
        let (p, ways) = choose(&w, "q", false, true);
        assert!(p.is_some() && ways.is_empty());
    }
}
