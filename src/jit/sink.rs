//! **Scalar replacement** (allocation sinking) for the generic tier: an
//! object an inlined `new` makes, of a small base constructor that only
//! stores its fields (`this.x = x; this.y = y; …`), and that never escapes
//! the compiled body, is not allocated at all. Its fields live in registers
//! of their own (past the body's), its field reads become copies of them,
//! and its constructor's stores become copies into them.
//!
//! # What runs instead
//!
//! The analysis ([`plan`]) works on the tier's own copy of the bytecode, with
//! its calls inlined (`inline`), and rewrites it into another program the
//! emitter compiles as usual:
//!
//! * the `new`'s guard ([`Op::InlineGuard`]) still identifies the
//!   constructor exactly as before (the call's `ConstructFast` record), but
//!   instead of allocating `this` it checks the site's [`SinkMemo`] — the
//!   prototype the record names, at the current prototype epoch, has a chain
//!   on which adding the constructor's keys to a fresh object is a plain add
//!   (no setter, no read-only property, no exotic object on it) — and puts a
//!   **stand-in** in `this`: the prototype object itself. Every register the
//!   object would be in holds the stand-in instead (so registers always hold
//!   real values, which the collector may scan);
//! * a store `this.k = v` of the constructor is `Move f_k = v`;
//! * a read `o.k` of one of its fields is `Move d = f_k`;
//! * a method read `o.m` (an `Op::GetMethod`, `m` not a field) reads `m` from
//!   the stand-in, whose own properties are the object's inherited ones: as
//!   long as the property found is a data property (no getter, which would
//!   see the stand-in as its receiver, and no proxy) — else the op leaves
//!   for the interpreter ([`SinkPlan::vget`]);
//! * a method call's guard looks its method up on the stand-in, the same way;
//! * coercibility and TDZ checks of it pass, as they would.
//!
//! Anything else that would observe the object — a store into another
//! object, an identity comparison, a call it is passed to that is not
//! inlined, a closure, `arguments`, a return out of the body, a property it
//! does not have, a truthiness test — and the object is not sunk (its site is
//! left alone). An object live at a loop header of the function's own (an
//! on-stack-replacement entry or exit) or in a handler is not sunk either:
//! the analysis joins such a point with "not virtual", and a read of what
//! the join could not tell apart sinks nothing.
//!
//! # Leaving the compiled code
//!
//! The body's deoptimization exits (`deopt`) let the interpreter finish the
//! activation from an op. The interpreter runs the body's bytecode as it was
//! before the rewrite ([`SinkPlan`] keeps the rewritten ops for the emitter
//! only), so before it runs, every object the analysis knows is virtual
//! there is **materialized** ([`MatObj`]): an ordinary object with the
//! stand-in as its `[[Prototype]]` and the fields stored so far as its own
//! data properties, in order, written into every register that holds it —
//! innermost first, so an object stored in another one's field is made
//! before it. The interpreter then sees exactly the frame the original
//! program would have had.
//!
//! Exits are taken
//!
//! * where an inlined call's guard misses and its fallback call op would
//!   read a virtual object (before the op: [`SinkPlan::exit_before`]), or
//!   where the fallback call's result would meet a virtual object at the
//!   join after the site (after the op, at the join:
//!   [`SinkPlan::exit_after`]);
//! * where a virtual method read finds no plain data property;
//! * under `KATAAN_JIT_DEOPT_STRESS`, every few eligible op starts.
//!
//! A body whose exits keep happening is recompiled without sinking
//! (`jit_tier::deopt`).

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use crate::nbvm::{Op, Reg};

/// The most sites one body sinks (a poisoned state is a bit mask of sites).
const MAX_SITES: usize = 63;
/// The most fields of one sunk object.
const MAX_FIELDS: usize = 12;
/// The most dataflow rounds (each a fixpoint) before giving up.
const MAX_ROUNDS: usize = 64;

/// A sunk object's guard memo (one per site, owned with the code, read by
/// the guard): the prototype `proto` (a raw handle) was found, at prototype
/// epoch `stamp`, to have a chain on which the constructor's adds are plain
/// (see the module docs), and the site's **stand-in** for it is in the word
/// at `word` (a GC-rooted word of the body's `LitCells`): an ordinary object
/// whose `[[Prototype]]` is `proto` and whose own properties are the
/// constructor's keys, in order, each `0` — the layout the site's objects
/// have, so every cache that saw them sees it the same way (its values are
/// never read: the sunk object's field reads are copies of its field
/// registers). `stamp` `0` never matches.
#[repr(C)]
#[derive(Debug)]
pub struct SinkMemo {
    /// The prototype epoch of the check.
    pub stamp: core::cell::Cell<u64>,
    /// The prototype's raw handle.
    pub proto: core::cell::Cell<u64>,
    /// The address of the stand-in's word.
    pub word: u64,
    /// The prototype epoch at which compiled code last read the stand-in
    /// from its word and walked it to its heap slot (`0`: never), which
    /// [`Self::standin`] and [`Self::cell`] then hold: the word changes only
    /// with the memo, and a cell moves only with the epoch, so the guard
    /// takes both from here while the epoch is this one
    /// (`call_seq::emit_inline_new_this_made`; `KATAAN_JIT_SINK_CELL=0`
    /// turns it off).
    pub cell_stamp: core::cell::Cell<u64>,
    /// The stand-in's handle bits, at [`Self::cell_stamp`].
    pub standin: core::cell::Cell<u64>,
    /// The stand-in's heap slot address, at [`Self::cell_stamp`].
    pub cell: core::cell::Cell<u64>,
}

impl SinkMemo {
    /// An empty memo whose stand-in goes in the word at `word`.
    #[must_use]
    pub fn new(word: u64) -> Self {
        Self {
            stamp: core::cell::Cell::new(0),
            proto: core::cell::Cell::new(0),
            word,
            cell_stamp: core::cell::Cell::new(0),
            standin: core::cell::Cell::new(0),
            cell: core::cell::Cell::new(0),
        }
    }
}

/// Whether a sunk guard takes its stand-in and the stand-in's heap slot
/// from its memo ([`SinkMemo::cell_stamp`]; `KATAAN_JIT_SINK_CELL=0` turns
/// it off).
#[must_use]
pub(crate) fn cell_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("KATAAN_JIT_SINK_CELL").is_none_or(|v| v != "0"))
}

/// One sunk `new` site.
#[derive(Clone, Debug)]
pub struct SinkSite {
    /// The guard's pc.
    pub guard: usize,
    /// The constructor's `this` register (where the guard puts the stand-in).
    pub this: Reg,
    /// The fallback `Op::Construct`'s pc.
    pub fallback: usize,
    /// The constructor's register at the fallback (its value there).
    pub ctor: Reg,
    /// The keys the constructor stores, in order.
    pub keys: Vec<String>,
    /// Each key's field register.
    pub fregs: Vec<Reg>,
    /// The register the guard also puts the stand-in in, which nothing
    /// else writes (where an exit finds the prototype).
    pub preg: Reg,
}

/// An object to materialize at an exit: site `site`'s, with its first `n`
/// fields stored, held by the registers `regs` (its field registers among
/// the other objects').
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MatObj {
    /// The site (index into [`SinkPlan::sites`]).
    pub site: u16,
    /// How many of its fields are stored.
    pub n: u8,
    /// Every register holding it (the stand-in).
    pub regs: Vec<Reg>,
}

/// A body's sinking: the rewritten bytecode and what the emitter and the
/// exits need.
#[derive(Debug)]
pub struct SinkPlan {
    /// The rewritten ops (the emitter's; the interpreter keeps the original).
    pub ops: Vec<Op>,
    /// The register count with the field registers.
    pub n_regs: usize,
    /// The sunk sites.
    pub sites: Vec<SinkSite>,
    /// Per pc: the site whose guard it is.
    pub sunk_guard: Vec<Option<u16>>,
    /// Per pc: a method read on a virtual object (its slow path leaves).
    pub vget: Vec<bool>,
    /// Per pc: a fallback call op that leaves before it runs.
    pub exit_before: Vec<bool>,
    /// Per pc: a fallback call op that leaves after it runs (at the next
    /// op, the join after its site).
    pub exit_after: Vec<bool>,
    /// Per pc: a proper tail call guard (`Op::TailCallGuard` and kin) that
    /// reads a virtual object, whose deep path leaves before it runs (the
    /// interpreter then makes the tail call with the objects made). Its
    /// shallow path does nothing, so the guarded call — inlined, reading
    /// the virtual objects as any inlined call does — runs compiled.
    pub deep_exit: Vec<bool>,
    /// Per pc: the objects to materialize at an exit at its start.
    pub mats: Vec<Vec<MatObj>>,
    /// Per pc of a fallback with [`Self::exit_after`]: the objects to
    /// materialize at that exit (as the fallback left them).
    pub mats_after: Vec<(usize, Vec<MatObj>)>,
    /// Per pc: the stored fields' registers of the objects held there (which
    /// an exit there reads: `numeric::EXTRA_USES`).
    pub held_fregs: Vec<Vec<Reg>>,
}

impl SinkPlan {
    /// The objects to materialize at an exit at body pc `pc` (`after`: the
    /// exit after the fallback op at `pc - 1`).
    #[must_use]
    pub fn mats_at(&self, pc: usize, after: bool) -> &[MatObj] {
        if after {
            let f = pc.wrapping_sub(1);
            return self
                .mats_after
                .iter()
                .find(|(p, _)| *p == f)
                .map_or(&[][..], |(_, m)| &m[..]);
        }
        self.mats.get(pc).map_or(&[][..], |m| &m[..])
    }
}

// A register's abstract value: `N` (not a virtual object), `V(s)` (site
// `s`'s current object: `VTAG | s`) or poisoned (`P(mask)`: the sites whose
// objects it may hold on some path, any read of which is unsupported).
const N: u64 = 0;
const VTAG: u64 = 1 << 63;

fn is_v(x: u64) -> bool {
    x & VTAG != 0
}

fn site_of(x: u64) -> usize {
    (x & 0xff) as usize
}

fn vmask(x: u64) -> u64 {
    if is_v(x) { 1u64 << site_of(x) } else { x }
}

fn join_reg(a: u64, b: u64) -> u64 {
    if a == b { a } else { vmask(a) | vmask(b) }
}

// A site's object's stored-field count: `DEAD` (none on this path), `k + 1`
// (its first `k` fields stored) or `CONFLICT` (paths disagree).
const DEAD: u8 = 0;
const CONFLICT: u8 = u8::MAX;

fn join_init(a: u8, b: u8) -> u8 {
    match (a, b) {
        (x, y) if x == y => x,
        (DEAD, y) => y,
        (x, DEAD) => x,
        _ => CONFLICT,
    }
}

#[derive(Clone, PartialEq, Eq)]
struct State {
    regs: Vec<u64>,
    init: Vec<u8>,
}

impl State {
    fn join(&mut self, o: &State) -> bool {
        let mut changed = false;
        for (a, &b) in self.regs.iter_mut().zip(&o.regs) {
            let j = join_reg(*a, b);
            changed |= j != *a;
            *a = j;
        }
        for (a, &b) in self.init.iter_mut().zip(&o.init) {
            let j = join_init(*a, b);
            changed |= j != *a;
            *a = j;
        }
        changed
    }
}

/// A candidate site, found by shape ([`candidates`]).
struct Cand {
    guard: usize,
    this: Reg,
    fallback: usize,
    ctor: Reg,
    /// `(pc, key, value register)` per store, in order.
    stores: Vec<(usize, String, Reg)>,
}

/// Whether `k` is a key a sunk object may have: an ordinary string key
/// (not an array index, which lives in elements, nor `__proto__`).
fn key_ok(k: &str) -> bool {
    !k.is_empty() && k != "__proto__" && !k.bytes().all(|b| b.is_ascii_digit())
}

/// The `InlineLeave` ending the inlined body that starts at `from`.
fn leave_of(ops: &[Op], from: usize) -> Option<usize> {
    let mut depth = 0usize;
    for (pc, op) in ops.iter().enumerate().skip(from) {
        match op {
            Op::InlineGuard { .. } => depth += 1,
            Op::InlineLeave if depth == 0 => return Some(pc),
            Op::InlineLeave => depth -= 1,
            _ => {}
        }
    }
    None
}

/// The inlined `new`s whose constructor body is straight-line field stores:
/// its guard, `JumpIfFalse` to the fallback, the body (`Move`s and constants
/// not naming `this`, `CheckTdz`s, distinct stores `this.k = v` of another
/// register, and the result's `Move r = this` last), its `InlineLeave`, the
/// jump past the fallback `Construct`.
fn candidates(ops: &[Op]) -> Vec<Cand> {
    let mut out = Vec::new();
    for (g, op) in ops.iter().enumerate() {
        let Op::InlineGuard {
            dst,
            call,
            callee,
            caps,
            n_caps,
            poly: 0,
            ..
        } = op
        else {
            continue;
        };
        let f = *call;
        let Some(Op::Construct { ctor, .. }) = ops.get(f) else {
            continue;
        };
        if ctor != callee || *n_caps == 0 {
            continue;
        }
        let this = caps + n_caps - 1;
        if !matches!(ops.get(g + 1), Some(Op::JumpIfFalse { cond, target }) if cond == dst && *target == f)
        {
            continue;
        }
        let Some(l) = leave_of(ops, g + 2) else {
            continue;
        };
        if l + 2 != f || !matches!(ops.get(l + 1), Some(Op::Jump { target }) if *target == f + 1) {
            continue;
        }
        let mut stores: Vec<(usize, String, Reg)> = Vec::new();
        let mut ok = true;
        let mut result_seen = false;
        for (pc, b) in ops.iter().enumerate().take(l).skip(g + 2) {
            if result_seen {
                ok = false;
                break;
            }
            match b {
                Op::SetProp { obj, key, src } | Op::SetPropStrict { obj, key, src }
                    if *obj == this =>
                {
                    if *src == this
                        || !key_ok(key)
                        || stores.iter().any(|(_, k, _)| k == key)
                        || stores.len() >= MAX_FIELDS
                    {
                        ok = false;
                        break;
                    }
                    stores.push((pc, key.clone(), *src));
                }
                Op::Move { dst, src } if *src == this && *dst != this => result_seen = true,
                Op::Move { dst, src } if *dst != this && *src != this => {}
                Op::LoadConst { dst, .. } if *dst != this => {}
                Op::CheckTdz { .. } => {}
                _ => {
                    ok = false;
                    break;
                }
            }
        }
        if ok && result_seen && !stores.is_empty() {
            out.push(Cand {
                guard: g,
                this,
                fallback: f,
                ctor: *ctor,
                stores,
            });
        }
    }
    out.truncate(MAX_SITES);
    out
}

/// The guard diamonds' fallback arms (`numeric::guard_diamonds`): a
/// `PlainPrims t` at `g`, `JumpIfFalse t → F` at `g + 1`, the fast arm's
/// `Jump E` at `F - 1`, and nothing but that test branching into `F..E`
/// (whose branches stay in it or go to `E`). Per pc of such an arm, its
/// first pc `F` (the outermost arm's, for nested ones).
///
/// The arm runs only when the guard's speculation fails (an argument that
/// is not a plain primitive), so an arm that would observe a virtual object
/// leaves at `F` instead ([`SinkPlan::exit_before`]): the interpreter runs
/// it, and the rest of the activation, with the objects made.
fn arms(ops: &[Op]) -> Vec<Option<usize>> {
    let n = ops.len();
    let mut out = vec![None; n];
    if !arms_enabled() {
        return out;
    }
    for (g, op) in ops.iter().enumerate() {
        let Op::PlainPrims { dst, .. } = op else {
            continue;
        };
        let Some(&Op::JumpIfFalse { cond, target: f }) = ops.get(g + 1) else {
            continue;
        };
        if cond != *dst || f <= g + 2 || f >= n {
            continue;
        }
        let Some(&Op::Jump { target: e }) = ops.get(f - 1) else {
            continue;
        };
        if e <= f || e > n || out[f..e].iter().any(Option::is_some) {
            continue;
        }
        let ok = ops.iter().enumerate().all(|(pc, o)| match o {
            Op::Jump { target } | Op::JumpIfFalse { target, .. } => {
                if pc == g + 1 {
                    true
                } else if (f..e).contains(&pc) {
                    (f + 1..=e).contains(target)
                } else {
                    !(f..e).contains(target)
                }
            }
            Op::PushHandler { target, .. } => !(f..e).contains(target),
            Op::InlineGuard { call, .. } => (f..e).contains(&pc) || !(f..e).contains(call),
            _ => true,
        });
        if ok {
            for x in &mut out[f..e] {
                *x = Some(f);
            }
        }
    }
    out
}

/// Per pc, the tracked registers (`slot`, `nt` of them) live at its start:
/// read on some path from there before being written, as bit masks by
/// tracked index. A catch target's live registers count as live everywhere
/// (any op may throw to it).
fn liveness(ops: &[Op], slot: &[u32], nt: usize) -> Vec<Vec<u64>> {
    let n = ops.len();
    let words = nt.div_ceil(64).max(1);
    let mut live = vec![vec![0u64; words]; n + 1];
    let catches: Vec<usize> = ops
        .iter()
        .filter_map(|o| match o {
            Op::PushHandler { target, .. } if *target < n => Some(*target),
            _ => None,
        })
        .collect();
    let idx = |r: u16| slot.get(r as usize).copied().filter(|&i| i != u32::MAX);
    let mut defs = Vec::new();
    let mut uses = Vec::new();
    let mut out = vec![0u64; words];
    let mut changed = true;
    while changed {
        changed = false;
        let mut caught = vec![0u64; words];
        for &c in &catches {
            for (a, b) in caught.iter_mut().zip(&live[c]) {
                *a |= b;
            }
        }
        for pc in (0..n).rev() {
            out.copy_from_slice(&caught);
            let succ = |t: usize, out: &mut [u64], live: &[Vec<u64>]| {
                if let Some(l) = live.get(t) {
                    for (a, b) in out.iter_mut().zip(l) {
                        *a |= b;
                    }
                }
            };
            match &ops[pc] {
                Op::Jump { target } => succ(*target, &mut out, &live),
                Op::JumpIfFalse { target, .. } => {
                    succ(pc + 1, &mut out, &live);
                    succ(*target, &mut out, &live);
                }
                Op::Return { .. }
                | Op::Throw { .. }
                | Op::ThrowTypeError { .. }
                | Op::ThrowReferenceError { .. }
                | Op::TailCall { .. }
                | Op::TailCallValue { .. }
                | Op::TailCallThis { .. } => {}
                _ => succ(pc + 1, &mut out, &live),
            }
            super::numeric::def_use(&ops[pc], &mut defs, &mut uses);
            for &d in &defs {
                if let Some(i) = idx(d) {
                    out[i as usize / 64] &= !(1u64 << (i % 64));
                }
            }
            for &u in &uses {
                if let Some(i) = idx(u) {
                    out[i as usize / 64] |= 1u64 << (i % 64);
                }
            }
            if out != live[pc] {
                live[pc].copy_from_slice(&out);
                changed = true;
            }
        }
    }
    live
}

/// Whether a guard diamond's fallback arm may leave instead of observing a
/// virtual object ([`arms`]; `KATAAN_JIT_SINK_ARMS=0` turns it off).
fn arms_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("KATAAN_JIT_SINK_ARMS").is_none_or(|v| v != "0"))
}

/// Whether a proper tail call guard's deep path may leave instead of
/// observing a virtual object ([`SinkPlan::deep_exit`];
/// `KATAAN_JIT_SINK_TAIL=0` turns it off).
fn tail_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("KATAAN_JIT_SINK_TAIL").is_none_or(|v| v != "0"))
}

/// What one dataflow round found.
struct Round {
    /// Per pc: the in-state (`None`: unreached).
    states: Vec<Option<State>>,
    /// Sites found unsinkable.
    fail: u64,
    /// Fallbacks that must leave before running (newly found).
    before: Vec<usize>,
    /// Per pc: a virtual method read.
    vget: Vec<bool>,
    /// (Debugging: the pcs where sites failed, and their masks.)
    fails: Vec<(usize, u64)>,
}

/// The analysis' fixed inputs.
struct Ctx<'a> {
    ops: &'a [Op],
    n_regs: usize,
    cands: &'a [Cand],
    /// Per site: its field registers.
    fregs: &'a [Vec<Reg>],
    /// Per pc: the site whose guard it is.
    guard_of: &'a [Option<usize>],
    /// Per pc: the site whose store it is, and the field index.
    store_of: &'a [Option<(usize, usize)>],
    /// Per pc: whether it is an inlined call's fallback op.
    fallback: &'a [bool],
    /// Per pc in a guard diamond's fallback arm ([`arms`]): the arm's
    /// first pc, where an exit is taken instead when an op of the arm
    /// would observe a virtual object.
    arm_of: &'a [Option<usize>],
    /// Per pc: an `Op::GetMethod` (the tier holds it as a `GetProp`).
    method_gets: &'a [bool],
    /// The sloppy ordinary callees' keys.
    legacy: &'a [u64],
    /// Entries (own loop headers, catch targets): joined with all-`N`.
    entries: &'a [usize],
    /// The handlers' catch targets.
    catches: &'a [usize],
    /// The outermost function's parameter count.
    n_params: usize,
    /// Per register, its index in a state (`u32::MAX`: a register that
    /// never holds a sunk object, always `N`).
    slot: &'a [u32],
}

impl Ctx<'_> {
    /// Register `r`'s value in `s`.
    fn at(&self, s: &State, r: Reg) -> u64 {
        match self.slot.get(r as usize) {
            Some(&i) if i != u32::MAX => s.regs[i as usize],
            _ => N,
        }
    }

    fn field_of(&self, s: usize, key: &str) -> Option<usize> {
        self.cands[s].stores.iter().position(|(_, k, _)| k == key)
    }

    /// One dataflow fixpoint with sites `active`, the given exits, and
    /// `handler` the state every catch target starts from.
    fn round(&self, active: u64, before: &[bool], after: &[bool], handler: &State) -> Round {
        let n = self.ops.len();
        let ns = self.cands.len();
        let empty = State {
            regs: vec![N; self.n_regs],
            init: vec![DEAD; ns],
        };
        let mut states: Vec<Option<State>> = vec![None; n + 1];
        states[0] = Some(empty.clone());
        for &e in self.entries {
            if e < n {
                states[e] = Some(empty.clone());
            }
        }
        for &c in self.catches {
            if c < n {
                match &mut states[c] {
                    Some(s) => {
                        s.join(handler);
                    }
                    None => states[c] = Some(handler.clone()),
                }
            }
        }
        let mut fail = 0u64;
        let mut fails: Vec<(usize, u64)> = Vec::new();
        let mut new_before: Vec<usize> = Vec::new();
        let mut vget = vec![false; n];
        let mut defs = Vec::new();
        let mut uses = Vec::new();
        let mut succ: Vec<usize> = Vec::with_capacity(2);
        let mut dirty = vec![true; n];
        let mut changed = true;
        // (The op's out-state, reused.)
        let mut s = empty.clone();
        while changed {
            changed = false;
            for pc in 0..n {
                if !dirty[pc] {
                    continue;
                }
                dirty[pc] = false;
                let Some(src) = states[pc].as_ref() else {
                    continue;
                };
                // (A fallback arm that leaves at its start: the interpreter
                // runs it.)
                let arm = self.arm_of[pc];
                if before[pc] && arm == Some(pc) {
                    continue;
                }
                s.regs.copy_from_slice(&src.regs);
                s.init.copy_from_slice(&src.init);
                let mut leaves = false;
                let f0 = fail;
                self.transfer(
                    pc,
                    &mut s,
                    active,
                    (&mut fail, &mut vget, &mut leaves),
                    (&mut defs, &mut uses),
                );
                // An op of a fallback arm that would observe an object: the
                // arm leaves at its start instead.
                if let Some(f) = arm
                    && fail != f0
                    && !before[f]
                {
                    fail = f0;
                    new_before.push(f);
                    continue;
                }
                if fail != f0 && debug() && fails.len() < 32 {
                    fails.push((pc, fail & !f0));
                }
                if leaves && !before[pc] {
                    new_before.push(pc);
                }
                if before[pc] || after[pc] || leaves {
                    continue;
                }
                succ.clear();
                match &self.ops[pc] {
                    Op::Jump { target } => succ.push(*target),
                    Op::JumpIfFalse { target, .. } => {
                        succ.push(pc + 1);
                        succ.push(*target);
                    }
                    Op::Return { .. }
                    | Op::Throw { .. }
                    | Op::ThrowTypeError { .. }
                    | Op::ThrowReferenceError { .. }
                    | Op::TailCall { .. }
                    | Op::TailCallValue { .. }
                    | Op::TailCallThis { .. } => {}
                    _ => succ.push(pc + 1),
                }
                for &t in &succ {
                    if t > n {
                        continue;
                    }
                    let c = match &mut states[t] {
                        Some(x) => x.join(&s),
                        None => {
                            states[t] = Some(s.clone());
                            true
                        }
                    };
                    if c && t < n {
                        dirty[t] = true;
                        changed = true;
                    }
                }
            }
        }
        new_before.sort_unstable();
        new_before.dedup();
        Round {
            states,
            fail,
            before: new_before,
            vget,
            fails,
        }
    }

    /// `ops[pc]`'s effect on `s`, failing the sites whose objects it would
    /// observe in a way the rewrite cannot keep; `leaves`: a fallback call op
    /// that reads a virtual object (an exit before it).
    #[allow(clippy::too_many_lines)]
    fn transfer(
        &self,
        pc: usize,
        s: &mut State,
        active: u64,
        (fail, vget, leaves): (&mut u64, &mut [bool], &mut bool),
        (defs, uses): (&mut Vec<u16>, &mut Vec<u16>),
    ) {
        let get = |s: &State, r: Reg| self.at(s, r);
        let op = &self.ops[pc];
        // A plain read: a virtual object or a poisoned register is observed.
        let read = |s: &State, r: Reg, fail: &mut u64| {
            let x = get(s, r);
            if x != N {
                *fail |= vmask(x);
            }
        };
        let set = |s: &mut State, r: Reg, v: u64| match self.slot.get(r as usize) {
            Some(&i) if i != u32::MAX => s.regs[i as usize] = v,
            _ => debug_assert!(v == N, "an untracked register holds a sunk object"),
        };
        // A sunk site's guard: a new object (the last one is gone: every
        // register still holding it is poisoned).
        if let Some(site) = self.guard_of[pc].filter(|&k| active & (1 << k) != 0) {
            let Op::InlineGuard {
                dst,
                args,
                caps,
                n_caps,
                key,
                ..
            } = op
            else {
                return;
            };
            for &a in args {
                let x = get(s, a);
                if is_v(x) && self.legacy.contains(key) {
                    *fail |= vmask(x);
                } else if !is_v(x) && x != N {
                    *fail |= x;
                }
            }
            let tag = VTAG | site as u64;
            for x in &mut s.regs {
                if *x == tag {
                    *x = 1 << site;
                }
            }
            for i in 0..*n_caps {
                set(s, caps + i, N);
            }
            set(s, *dst, N);
            set(s, self.cands[site].this, tag);
            s.init[site] = 1;
            return;
        }
        match op {
            Op::Move { dst, src } => {
                let x = get(s, *src);
                set(s, *dst, x);
            }
            Op::GetProp { dst, obj, key } if is_v(get(s, *obj)) => {
                let v = get(s, *obj);
                let site = site_of(v);
                match self.field_of(site, key) {
                    Some(i) => {
                        let init = s.init[site];
                        if init == CONFLICT || (init as usize) < i + 2 {
                            *fail |= 1 << site;
                            set(s, *dst, N);
                        } else {
                            let f = get(s, self.fregs[site][i]);
                            set(s, *dst, f);
                        }
                    }
                    None if self.method_gets[pc] && key_ok(key) => {
                        vget[pc] = true;
                        set(s, *dst, N);
                    }
                    None => {
                        *fail |= 1 << site;
                        set(s, *dst, N);
                    }
                }
            }
            Op::SetProp { obj, src, .. } | Op::SetPropStrict { obj, src, .. }
                if is_v(get(s, *obj)) =>
            {
                let site = site_of(get(s, *obj));
                let x = get(s, *src);
                match self.store_of[pc] {
                    Some((st, i)) if st == site && s.init[site] as usize == i + 1 => {
                        if x != N && (!is_v(x) || site_of(x) == site) {
                            *fail |= vmask(x) | 1 << site;
                        }
                        set(s, self.fregs[site][i], x);
                        s.init[site] += 1;
                    }
                    _ => {
                        *fail |= (1 << site) | vmask(x);
                    }
                }
            }
            Op::RequireObjectCoercible { src, .. } | Op::CheckTdz { src } => {
                let x = get(s, *src);
                if !is_v(x) && x != N {
                    *fail |= x;
                }
            }
            Op::InlineGuard {
                dst,
                call,
                key,
                callee,
                args,
                caps,
                n_caps,
                ..
            } => {
                // The guard reads its callee (a method call's receiver, whose
                // method it looks up) and its arguments (the activation's,
                // which a sloppy callee's `f.arguments` would read).
                let c = get(s, *callee);
                match self.ops.get(*call) {
                    Some(Op::CallMethod { recv, key: k, .. })
                        if recv == callee && is_v(c) && key_ok(k) =>
                    {
                        if self.field_of(site_of(c), k).is_some() {
                            *fail |= vmask(c);
                        }
                    }
                    _ => read(s, *callee, fail),
                }
                for &a in args {
                    let x = get(s, a);
                    if is_v(x) && self.legacy.contains(key) {
                        *fail |= vmask(x);
                    } else if !is_v(x) && x != N {
                        *fail |= x;
                    }
                }
                for i in 0..*n_caps {
                    set(s, caps + i, N);
                }
                set(s, *dst, N);
            }
            Op::Jump { .. } | Op::InlineLeave => {}
            // A proper tail call guard does nothing while the call depth is
            // shallow; deeper, it leaves before it runs, the objects it
            // reads made ([`SinkPlan::deep_exit`]). (A poisoned register is
            // no object an exit could make.)
            Op::TailCallGuard { .. } | Op::TailSpreadGuard { .. } | Op::TailMethodGuard { .. }
                if tail_enabled() =>
            {
                super::numeric::def_use(op, defs, uses);
                for &u in uses.iter() {
                    let x = get(s, u);
                    if !is_v(x) && x != N {
                        *fail |= x;
                    }
                }
            }
            _ => {
                super::numeric::def_use(op, defs, uses);
                if self.fallback[pc] {
                    // A guarded call op: one reading a virtual object
                    // leaves before it runs (a poisoned one is no
                    // object the interpreter could be given).
                    for &u in uses.iter() {
                        let x = get(s, u);
                        if is_v(x) {
                            *leaves = true;
                        } else if x != N {
                            *fail |= x;
                        }
                    }
                    if *leaves {
                        return;
                    }
                } else {
                    for &u in uses.iter() {
                        read(s, u, fail);
                    }
                }
                for &d in defs.iter() {
                    set(s, d, N);
                }
            }
        }
        // No virtual object in a parameter register of the outermost
        // function (what its legacy `f.arguments` could read).
        for r in 0..self.n_params.min(self.slot.len()) {
            let x = self.at(s, r as Reg);
            if is_v(x) {
                *fail |= vmask(x);
            }
        }
    }
}

/// Plans the sinking of `ops` (a body's bytecode as the tier compiles it,
/// with its inlined calls spliced in; `n_regs` registers, `n_params` of
/// them the outermost function's parameters; `method_gets` its
/// `Op::GetMethod`s, held as `GetProp`s; `legacy` the sloppy ordinary
/// inlined callees' keys; `orig_of` per pc the function's own pc, or
/// `u32::MAX` inside an inlined body, empty when unspliced). `None` when no
/// site sinks.
#[must_use]
pub(crate) fn plan(
    ops: &[Op],
    n_regs: usize,
    n_params: usize,
    method_gets: &[bool],
    legacy: &[u64],
    orig_of: &[u32],
    rejected: &[usize],
) -> Option<SinkPlan> {
    let n = ops.len();
    if ops.iter().any(|o| {
        matches!(
            o,
            Op::DirectEval { .. } | Op::MakeArguments { .. } | Op::Yield { .. } | Op::Await { .. }
        )
    }) {
        return None;
    }
    let mut cands = candidates(ops);
    cands.retain(|c| !rejected.contains(&c.guard));
    if debug() {
        std::eprintln!(
            "[sink] {} candidates at {:?}",
            cands.len(),
            cands.iter().map(|c| c.guard).collect::<Vec<_>>()
        );
    }
    if cands.is_empty() {
        return None;
    }
    // Field registers, past the body's.
    let mut next = n_regs;
    let mut fregs: Vec<Vec<Reg>> = Vec::with_capacity(cands.len());
    for c in &cands {
        let mut f = Vec::with_capacity(c.stores.len());
        for _ in &c.stores {
            f.push(Reg::try_from(next).ok()?);
            next += 1;
        }
        fregs.push(f);
    }
    let mut pregs: Vec<Reg> = Vec::with_capacity(cands.len());
    for _ in &cands {
        pregs.push(Reg::try_from(next).ok()?);
        next += 1;
    }
    let total = next;
    let mut guard_of = vec![None; n];
    let mut store_of = vec![None; n];
    for (k, c) in cands.iter().enumerate() {
        guard_of[c.guard] = Some(k);
        for (i, (pc, _, _)) in c.stores.iter().enumerate() {
            store_of[*pc] = Some((k, i));
        }
    }
    let mut fallback = vec![false; n];
    let mut catches: Vec<usize> = Vec::new();
    let mut entries: Vec<usize> = Vec::new();
    for (pc, op) in ops.iter().enumerate() {
        match op {
            Op::InlineGuard { call, .. } if *call < n => fallback[*call] = true,
            Op::PushHandler { target, .. } => catches.push(*target),
            Op::Jump { target } | Op::JumpIfFalse { target, .. }
                if *target <= pc && orig_of.get(*target).is_none_or(|&o| o != u32::MAX) =>
            {
                entries.push(*target);
            }
            _ => {}
        }
    }
    catches.sort_unstable();
    catches.dedup();
    entries.extend(catches.iter().copied());
    entries.sort_unstable();
    entries.dedup();
    // The registers that may ever hold a sunk object (or a poisoned one):
    // the sites' `this`, field and stand-in registers, and whatever copies
    // or field reads take from them.
    let mut slot = vec![u32::MAX; total];
    let mut tracked: Vec<Reg> = Vec::new();
    {
        let add = |r: Reg, slot: &mut Vec<u32>, tracked: &mut Vec<Reg>| {
            if let Some(x) = slot.get_mut(r as usize)
                && *x == u32::MAX
            {
                *x = tracked.len() as u32;
                tracked.push(r);
                return true;
            }
            false
        };
        for (k, c) in cands.iter().enumerate() {
            add(c.this, &mut slot, &mut tracked);
            add(pregs[k], &mut slot, &mut tracked);
            for &f in &fregs[k] {
                add(f, &mut slot, &mut tracked);
            }
        }
        let mut changed = true;
        while changed {
            changed = false;
            for op in ops {
                let (dst, src) = match op {
                    Op::Move { dst, src } | Op::GetProp { dst, obj: src, .. } => (*dst, *src),
                    _ => continue,
                };
                if slot.get(src as usize).is_some_and(|&x| x != u32::MAX) {
                    changed |= add(dst, &mut slot, &mut tracked);
                }
            }
        }
    }
    let arm_of = arms(ops);
    let cx = Ctx {
        ops,
        n_regs: tracked.len(),
        cands: &cands,
        fregs: &fregs,
        guard_of: &guard_of,
        store_of: &store_of,
        fallback: &fallback,
        arm_of: &arm_of,
        method_gets,
        legacy,
        entries: &entries,
        catches: &catches,
        n_params,
        slot: &slot,
    };
    let all: u64 = if cands.len() >= 64 {
        u64::MAX >> 1
    } else {
        (1u64 << cands.len()) - 1
    };
    let mut active = all;
    let mut before = vec![false; n];
    let mut after = vec![false; n];
    let mut handler = State {
        regs: vec![N; tracked.len()],
        init: vec![DEAD; cands.len()],
    };
    let mut rounds = 0;
    let mut live: Option<Vec<Vec<u64>>> = None;
    let r = loop {
        rounds += 1;
        if rounds > MAX_ROUNDS || active == 0 {
            return None;
        }
        let r = cx.round(active, &before, &after, &handler);
        let mut fail = r.fail;
        // An object the paths into a point disagree about how far its
        // constructor got, or whose stored fields' registers were
        // poisoned, cannot be made there.
        for st in r.states.iter().flatten() {
            let held = st
                .regs
                .iter()
                .filter(|&&x| is_v(x))
                .fold(0u64, |m, &x| m | 1u64 << site_of(x));
            if held == 0 {
                continue;
            }
            for (k, &init) in st.init.iter().enumerate() {
                if held & (1u64 << k) == 0 {
                    continue;
                }
                if init == CONFLICT {
                    fail |= 1 << k;
                    continue;
                }
                for &f in fregs[k].iter().take((init as usize).saturating_sub(1)) {
                    let x = cx.at(st, f);
                    if x != N && !is_v(x) {
                        fail |= 1 << k;
                    }
                }
            }
        }
        // A fallback arm that leaves at its start: the interpreter runs the
        // arm and the rest, which the analysis did not follow from there,
        // so no register live there may hold a stale stand-in.
        for pc in (0..n).filter(|&pc| before[pc] && arm_of[pc] == Some(pc)) {
            if let Some(st) = r.states[pc].as_ref() {
                let live = live.get_or_insert_with(|| liveness(ops, &slot, tracked.len()));
                for (i, &x) in st.regs.iter().enumerate() {
                    if x != N && !is_v(x) && live[pc][i / 64] & (1u64 << (i % 64)) != 0 {
                        fail |= x;
                    }
                }
            }
        }
        let mut again = false;
        for &b in &r.before {
            if !before[b] {
                before[b] = true;
                again = true;
            }
        }
        // A fallback whose result meets a virtual object at the join after
        // its site: leave after it instead.
        for f in 0..n {
            if !fallback[f] || before[f] || after[f] || f + 1 >= n {
                continue;
            }
            let (Some(sf), Some(se)) = (r.states[f].as_ref(), r.states[f + 1].as_ref()) else {
                continue;
            };
            let mut out = sf.clone();
            let mut d = Vec::new();
            let mut u = Vec::new();
            super::numeric::def_use(&ops[f], &mut d, &mut u);
            for &x in &d {
                if let Some(&i) = slot.get(x as usize).filter(|&&i| i != u32::MAX) {
                    out.regs[i as usize] = N;
                }
            }
            let conflicts = out
                .regs
                .iter()
                .zip(&se.regs)
                .any(|(&a, &b)| a == N && b != N && !is_v(b) && b & active != 0);
            if conflicts {
                after[f] = true;
                again = true;
            }
        }
        // The catch targets start from every state a throw could leave
        // (any register that ever holds a virtual object there poisoned).
        if !catches.is_empty() {
            let mut h = handler.clone();
            for st in r.states.iter().flatten() {
                for (a, &b) in h.regs.iter_mut().zip(&st.regs) {
                    *a |= vmask(b);
                }
            }
            if h != handler {
                handler = h;
                again = true;
            }
        }
        if again {
            continue;
        }
        // (The exits settled: what still fails, fails.)
        if fail & active != 0 {
            if debug() {
                std::eprintln!(
                    "[sink] round {rounds}: fail {:#x} (active {active:#x}) at {:?}",
                    fail & active,
                    r.fails
                );
                for &(pc, m) in &r.fails {
                    std::eprintln!("[sink]   {pc} {m:#x}: {:?}", ops.get(pc));
                }
            }
            active &= !fail;
            before.fill(false);
            after.fill(false);
            handler.regs.fill(N);
            handler.init.fill(DEAD);
            continue;
        }
        break r;
    };
    // The plan: sites in their final numbering.
    let mut index = vec![u16::MAX; cands.len()];
    let mut sites = Vec::new();
    for (k, c) in cands.iter().enumerate() {
        if active & (1 << k) != 0 {
            index[k] = u16::try_from(sites.len()).ok()?;
            sites.push(SinkSite {
                guard: c.guard,
                this: c.this,
                fallback: c.fallback,
                ctor: c.ctor,
                keys: c.stores.iter().map(|(_, k, _)| k.clone()).collect(),
                fregs: fregs[k].clone(),
                preg: pregs[k],
            });
        }
    }
    if sites.is_empty() {
        return None;
    }
    let mats_of = |st: &State| -> Option<Vec<MatObj>> {
        let mut objs: Vec<(usize, MatObj)> = Vec::new();
        for (k, &init) in st.init.iter().enumerate() {
            if active & (1 << k) == 0 {
                continue;
            }
            let tag = VTAG | k as u64;
            let regs: Vec<Reg> = (0..st.regs.len())
                .filter(|&i| st.regs[i] == tag)
                .map(|i| tracked[i])
                .collect();
            if regs.is_empty() {
                continue;
            }
            objs.push((
                k,
                MatObj {
                    site: index[k],
                    n: init.checked_sub(1)?,
                    regs,
                },
            ));
        }
        // Innermost first: an object one of whose stored fields holds
        // another comes after it.
        let mut out: Vec<MatObj> = Vec::with_capacity(objs.len());
        let mut done = vec![false; objs.len()];
        while out.len() < objs.len() {
            let mut progress = false;
            for i in 0..objs.len() {
                if done[i] {
                    continue;
                }
                let (k, m) = &objs[i];
                let ready = fregs[*k].iter().take(m.n as usize).all(|&f| {
                    let x = cx.at(st, f);
                    !is_v(x)
                        || objs
                            .iter()
                            .enumerate()
                            .all(|(j, (k2, _))| VTAG | *k2 as u64 != x || done[j])
                });
                if ready {
                    done[i] = true;
                    out.push(m.clone());
                    progress = true;
                }
            }
            if !progress {
                return None;
            }
        }
        Some(out)
    };
    let mut mats = vec![Vec::new(); n];
    for (pc, st) in r.states.iter().enumerate().take(n) {
        if let Some(st) = st {
            mats[pc] = mats_of(st)?;
        }
    }
    let mut mats_after = Vec::new();
    for f in (0..n).filter(|&f| after[f]) {
        let Some(sf) = r.states[f].as_ref() else {
            continue;
        };
        let mut out = sf.clone();
        let mut d = Vec::new();
        let mut u = Vec::new();
        super::numeric::def_use(&ops[f], &mut d, &mut u);
        for &x in &d {
            if let Some(&i) = slot.get(x as usize).filter(|&&i| i != u32::MAX) {
                out.regs[i as usize] = N;
            }
        }
        mats_after.push((f, mats_of(&out)?));
    }
    let mut held_fregs: Vec<Vec<Reg>> = vec![Vec::new(); n];
    for (pc, ms) in mats.iter().enumerate() {
        for m in ms {
            held_fregs[pc].extend(sites[m.site as usize].fregs.iter().take(m.n as usize));
        }
    }
    for (f, ms) in &mats_after {
        if let Some(h) = held_fregs.get_mut(f + 1) {
            for m in ms {
                h.extend(sites[m.site as usize].fregs.iter().take(m.n as usize));
            }
        }
    }
    // The rewrite.
    let mut out_ops = ops.to_vec();
    let mut sunk_guard = vec![None; n];
    for (k, c) in cands.iter().enumerate() {
        if active & (1 << k) == 0 {
            continue;
        }
        sunk_guard[c.guard] = Some(index[k]);
        for (i, (pc, _, src)) in c.stores.iter().enumerate() {
            out_ops[*pc] = Op::Move {
                dst: fregs[k][i],
                src: *src,
            };
        }
    }
    // The tail call guards that read a virtual object leave on their deep
    // path.
    let mut deep_exit = vec![false; n];
    {
        let mut d = Vec::new();
        let mut u = Vec::new();
        for pc in 0..n {
            let Some(st) = r.states[pc].as_ref() else {
                continue;
            };
            if !matches!(
                ops[pc],
                Op::TailCallGuard { .. } | Op::TailSpreadGuard { .. } | Op::TailMethodGuard { .. }
            ) {
                continue;
            }
            super::numeric::def_use(&ops[pc], &mut d, &mut u);
            deep_exit[pc] = u.iter().any(|&x| {
                let v = cx.at(st, x);
                is_v(v) && active & (1 << site_of(v)) != 0
            });
        }
    }
    let mut vget = vec![false; n];
    for pc in 0..n {
        let Some(st) = r.states[pc].as_ref() else {
            continue;
        };
        if let Op::GetProp { dst, obj, key } = &ops[pc] {
            let x = cx.at(st, *obj);
            if is_v(x) && active & (1 << site_of(x)) != 0 {
                let k = site_of(x);
                match cx.field_of(k, key) {
                    Some(i) => {
                        out_ops[pc] = Op::Move {
                            dst: *dst,
                            src: fregs[k][i],
                        };
                    }
                    None => vget[pc] = r.vget[pc],
                }
            }
        }
    }
    Some(SinkPlan {
        ops: out_ops,
        n_regs: total,
        sites,
        sunk_guard,
        vget,
        exit_before: before,
        exit_after: after,
        deep_exit,
        mats,
        mats_after,
        held_fregs,
    })
}

std::thread_local! {
    /// Whether the body being compiled planned any sinking (else a failed
    /// compile is not worth another try without).
    pub(crate) static PLANNED: core::cell::Cell<bool> = const { core::cell::Cell::new(false) };
    /// The guard pcs of the sites the emitter could not place an exit for
    /// in the body being compiled ([`reject`]), which its next try leaves
    /// out.
    pub(crate) static REJECTED: core::cell::RefCell<Vec<usize>> =
        const { core::cell::RefCell::new(Vec::new()) };
}

/// Records that the emitter cannot place an exit at body pc `pc` (or a
/// sunk guard there): every site whose object an exit there would make, or
/// whose guard it is, is rejected ([`REJECTED`]).
pub(crate) fn reject(p: &SinkPlan, pc: usize) {
    REJECTED.with(|r| {
        let mut r = r.borrow_mut();
        let mut add = |k: usize| {
            if let Some(s) = p.sites.get(k)
                && !r.contains(&s.guard)
            {
                r.push(s.guard);
            }
        };
        if let Some(Some(k)) = p.sunk_guard.get(pc) {
            add(*k as usize);
        }
        for m in p.mats_at(pc, false).iter().chain(p.mats_at(pc, true)) {
            add(m.site as usize);
        }
        if let Some(ms) = p.mats.get(pc + 1) {
            for m in ms {
                add(m.site as usize);
            }
        }
    });
}

/// Whether `KATAAN_JIT_SINK=log` (the analysis' reasons).
fn debug() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("KATAAN_JIT_SINK").is_some_and(|v| v == "log"))
}

/// Whether `KATAAN_JIT_LOG` is on (read once per process).
#[must_use]
pub(crate) fn log() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("KATAAN_JIT_LOG").is_some_and(|v| v != "0"))
}

/// Whether the generic tier sinks objects (`KATAAN_JIT_SINK=0` turns it
/// off; read once per process).
#[must_use]
pub(crate) fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("KATAAN_JIT_SINK").is_none_or(|v| v != "0"))
}

/// The forced exits (not counting `KATAAN_JIT_DEOPT_STRESS`'s) after which a
/// body is recompiled without sinking.
pub(crate) const EXIT_LIMIT: u64 = 64;

/// [`EXIT_LIMIT`], or `KATAAN_JIT_SINK_EXITS` (a test knob: a small limit
/// makes bodies retire their sinking early). Read once per process.
#[must_use]
pub(crate) fn exit_limit() -> u64 {
    static N: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("KATAAN_JIT_SINK_EXITS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n| n > 0)
            .unwrap_or(EXIT_LIMIT)
    })
}

/// What a body's exits need of its sinking (owned by its `DeoptInfo`).
pub struct SinkExits {
    /// The plan (its rewritten ops dropped: the code has them).
    pub plan: SinkPlan,
    /// Per site, its guard's memo.
    pub memos: alloc::boxed::Box<[SinkMemo]>,
    /// The body's literal words, whose `lit_base + k`th holds site `k`'s
    /// stand-in.
    pub lits: alloc::rc::Rc<crate::realm::LitCells>,
    /// See [`Self::lits`].
    pub lit_base: usize,
    /// Per site, its fallback `Construct`'s call site (whose
    /// `ConstructFast` record names the prototype the memo is checked for).
    pub calls: Vec<*const super::CallSite>,
    /// Per exit-before fallback op at the function's own level: its own pc
    /// (the call op the interpreter runs there).
    pub own_calls: Vec<(usize, usize)>,
    /// Per method call fallback that leaves before it runs: its receiver
    /// register and its site's chain cache, which the exit fills for the
    /// receiver's stand-in (the inlined guard looks the method up there, and
    /// only the call op, which never runs compiled now, fills it).
    pub fills: Vec<(usize, Reg, *mut super::props::MethodSite)>,
    /// The forced exits taken.
    pub forced: core::cell::Cell<u64>,
}

impl SinkExits {
    /// What site `pc`'s guard needs (`call_seq::InlineEmit::sink`).
    #[must_use]
    pub fn guard_emit(&self, pc: usize) -> Option<(*const SinkMemo, u16)> {
        let k = (*self.plan.sunk_guard.get(pc)?)? as usize;
        Some((
            core::ptr::from_ref(self.memos.get(k)?),
            self.plan.sites.get(k)?.preg,
        ))
    }

    /// The function's own pc of the own-level fallback op at body pc `pc`.
    #[must_use]
    pub fn own_call(&self, pc: usize) -> Option<usize> {
        self.own_calls.iter().find(|e| e.0 == pc).map(|e| e.1)
    }
}
