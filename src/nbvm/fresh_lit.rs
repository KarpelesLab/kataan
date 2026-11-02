//! A compile-time bytecode pass over **fresh literals** and dead register
//! writes (`ROADMAP.md` §2.1).
//!
//! An object or array literal nothing else has seen yet is plain data: until
//! its register reaches an op that can pass it on, no code can change it or
//! tell when it was made. So, within a basic block:
//!
//! * a read of a key the literal has (`o.a`, `a[1]` by a constant in range,
//!   `a.length`) is the value the literal was built from, while that value's
//!   register still holds it: the read becomes a `Move` (or a `LoadConst`);
//! * when that leaves the literal unread, its allocation goes: it is deleted
//!   if nothing reads its register later, and otherwise, when only the arm of
//!   the `if` ending the block reads it, moved into that arm.
//!
//! Then, in a function with a loop or a rewritten literal, any `LoadConst` or
//! `Move` whose register is dead goes (a script's completion-value writes that
//! a later statement overwrites, the constants those reads named), and the
//! stream is closed up over them.
//!
//! Moving and deleting ops keeps every other op's meaning: only `Jump`,
//! `JumpIfFalse` and `PushHandler` name a pc, the pass renumbers the first two,
//! and it leaves any function with a handler alone (any op could throw to it,
//! which makes "dead" and "unread" more than a question of the stream's order).
//! Nor does it touch code whose registers are read other than by naming them
//! or that keeps them across a suspension: generators, async functions, class
//! constructors, a mapped `arguments` object, direct `eval`.

use super::{FnProto, Op, Reg};
use crate::nanbox::NanBox;
use alloc::vec;
use alloc::vec::Vec;

/// The largest `ops × registers` table the liveness analysis builds.
const MAX_LIVENESS_BITS: usize = 1 << 24;

/// How many ops the liveness analysis may visit in all for one function's
/// literals, so that a huge function of many small blocks with literals stays
/// linear to compile. (Each analysis visits every op a few times.)
const LITERAL_LIVENESS_BUDGET: usize = 1 << 18;

/// How many rounds of dead-write removal a function gets (each round can
/// expose the writes that only fed the last round's).
const DEAD_WRITE_ROUNDS: usize = 4;

/// Runs the pass over freshly compiled functions.
pub(super) fn optimize(protos: &mut [FnProto]) {
    for p in protos {
        optimize_one(p);
    }
}

fn optimize_one(p: &mut FnProto) {
    if p.is_async || p.is_generator || p.class_ctor || p.derived || p.ops.is_empty() {
        return;
    }
    if p.ops.iter().any(|o| {
        matches!(
            o,
            Op::PushHandler { .. } | Op::DirectEval { .. } | Op::MakeArguments { .. }
        )
    }) {
        return;
    }
    let has_loop = p.ops.iter().enumerate().any(|(pc, o)| match o {
        Op::Jump { target } | Op::JumpIfFalse { target, .. } => *target <= pc,
        _ => false,
    });
    let has_lit = p
        .ops
        .iter()
        .any(|o| matches!(o, Op::NewObjectLit { .. } | Op::NewArrayFrom { .. }));
    if !has_loop && !has_lit {
        return;
    }
    if p.ops.len().saturating_mul(p.n_regs) > MAX_LIVENESS_BITS {
        return;
    }
    let mut changed = false;
    if has_lit {
        let leaders = leaders_of(&p.ops);
        let mut nops = vec![false; p.ops.len()];
        let mut budget = LITERAL_LIVENESS_BUDGET;
        let mut start = 0;
        while start < p.ops.len() {
            let end = block_end(&p.ops, &leaders, start);
            if p.ops[start..=end]
                .iter()
                .any(|o| matches!(o, Op::NewObjectLit { .. } | Op::NewArrayFrom { .. }))
            {
                changed |= literals_in_block(p, &leaders, &mut nops, &mut budget, (start, end));
            }
            start = end + 1;
        }
        if nops.iter().any(|&n| n) {
            remove(&mut p.ops, &nops);
        }
    }
    if changed || has_loop {
        drop_dead_writes(p);
    }
}

/// The pcs a branch can reach.
fn leaders_of(ops: &[Op]) -> Vec<bool> {
    let mut l = vec![false; ops.len() + 1];
    for op in ops {
        if let Op::Jump { target } | Op::JumpIfFalse { target, .. } = op
            && let Some(x) = l.get_mut(*target)
        {
            *x = true;
        }
    }
    l
}

/// Whether control never simply falls through `op`.
fn ends_block(op: &Op) -> bool {
    matches!(
        op,
        Op::Jump { .. }
            | Op::JumpIfFalse { .. }
            | Op::Return { .. }
            | Op::Throw { .. }
            | Op::ThrowTypeError { .. }
            | Op::ThrowReferenceError { .. }
    )
}

/// The last pc of the basic block starting at `start`.
fn block_end(ops: &[Op], leaders: &[bool], start: usize) -> usize {
    let mut e = start;
    while e + 1 < ops.len() && !ends_block(&ops[e]) && !leaders[e + 1] {
        e += 1;
    }
    e
}

/// `op`'s successors.
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

/// The registers `op` writes and reads. An op not modelled here counts as
/// reading and writing every register it names (`sure` false: the liveness
/// analysis must not take a maybe-write as a write).
struct Access {
    defs: Vec<Reg>,
    uses: Vec<Reg>,
    /// The `defs` surely happen (so a value before the op is dead).
    sure: bool,
}

fn access(op: &Op) -> Access {
    let mut a = Access {
        defs: Vec::new(),
        uses: Vec::new(),
        sure: true,
    };
    match op {
        Op::LoadConst { dst, .. } | Op::LoadGlobal { dst, .. } => a.defs.push(*dst),
        Op::Move { dst, src }
        | Op::ToNumeric { dst, src }
        | Op::Step { dst, src, .. }
        | Op::Neg { dst, a: src }
        | Op::BitNot { dst, a: src }
        | Op::Not { dst, a: src } => {
            a.defs.push(*dst);
            a.uses.push(*src);
        }
        Op::Add { dst, a: x, b }
        | Op::AddValue { dst, a: x, b }
        | Op::Sub { dst, a: x, b }
        | Op::Mul { dst, a: x, b }
        | Op::Div { dst, a: x, b }
        | Op::Mod { dst, a: x, b }
        | Op::Lt { dst, a: x, b }
        | Op::StrictEq { dst, a: x, b }
        | Op::ValueBin { dst, a: x, b, .. }
        | Op::Compare { dst, a: x, b, .. } => {
            a.defs.push(*dst);
            a.uses.extend([*x, *b]);
        }
        Op::JumpIfFalse { cond, .. } => a.uses.push(*cond),
        Op::Jump { .. } => {}
        Op::GetProp { dst, obj, .. } | Op::ArrayLen { dst, arr: obj } => {
            a.defs.push(*dst);
            a.uses.push(*obj);
        }
        Op::GetKey { dst, obj, key } => {
            a.defs.push(*dst);
            a.uses.extend([*obj, *key]);
        }
        Op::Return { src } | Op::StoreGlobal { src, .. } | Op::CheckTdz { src } => {
            a.uses.push(*src);
        }
        Op::NewObjectLit { dst, lit } => {
            a.uses.extend(lit.srcs.iter().copied());
            a.defs.push(*dst);
        }
        Op::NewArrayFrom { dst, elems } => {
            a.uses.extend(elems.iter().copied());
            a.defs.push(*dst);
        }
        _ => {
            let out = core::cell::RefCell::new(Vec::new());
            crate::bytecode::op_for_each_reg(op, &|r| out.borrow_mut().push(r));
            let named = out.into_inner();
            a.defs.clone_from(&named);
            a.uses = named;
            a.sure = false;
        }
    }
    a
}

/// A literal being followed through a block.
struct Lit {
    pc: usize,
    /// `(key, register)`; an array's keys are its indices (`key` empty).
    members: Vec<(Option<alloc::string::String>, Reg)>,
    is_array: bool,
    /// Per member: its register was written since.
    clobbered: Vec<bool>,
    /// Passed on, or read in a way that was not rewritten.
    escaped: bool,
    /// A register holding it was overwritten (it cannot be moved past that).
    overwritten: bool,
    /// Its own `Move`s, `(pc, dst)`.
    moves: Vec<(usize, Reg)>,
    /// The registers holding it, current.
    regs: Vec<Reg>,
    /// The literals it was made from.
    children: Vec<usize>,
    /// How many literals were made from it.
    parents: u32,
}

/// Follows the literals made in the block `start..=end`; rewrites reads, then
/// deletes or sinks what that left unread. Whether anything changed.
fn literals_in_block(
    p: &mut FnProto,
    leaders: &[bool],
    nops: &mut [bool],
    budget: &mut usize,
    (start, end): (usize, usize),
) -> bool {
    let n_regs = p.n_regs;
    let mut holds: Vec<Option<usize>> = vec![None; n_regs];
    let mut consts: Vec<Option<f64>> = vec![None; n_regs];
    let mut lits: Vec<Lit> = Vec::new();
    let mut changed = false;

    fn escape(lits: &mut [Lit], i: usize) {
        if !lits[i].escaped {
            lits[i].escaped = true;
            for c in lits[i].children.clone() {
                escape(lits, c);
            }
        }
    }
    fn kill(lits: &mut [Lit], holds: &mut [Option<usize>], consts: &mut [Option<f64>], r: Reg) {
        let ru = r as usize;
        if ru >= holds.len() {
            return;
        }
        consts[ru] = None;
        if let Some(i) = holds[ru].take() {
            lits[i].regs.retain(|&x| x != r);
            lits[i].overwritten = true;
        }
        for l in lits.iter_mut() {
            for (m, c) in l.members.iter().zip(l.clobbered.iter_mut()) {
                if m.1 == r {
                    *c = true;
                }
            }
        }
    }

    for pc in start..=end {
        // A read of a fresh literal that its members answer.
        let rewrite = match &p.ops[pc] {
            Op::GetProp { dst, obj, key } => {
                holds.get(*obj as usize).copied().flatten().and_then(|i| {
                    let l = &lits[i];
                    if l.escaped {
                        None
                    } else if l.is_array {
                        (key == "length").then(|| Op::LoadConst {
                            dst: *dst,
                            value: NanBox::number(l.members.len() as f64),
                        })
                    } else {
                        let k = l
                            .members
                            .iter()
                            .position(|m| m.0.as_deref() == Some(key.as_str()))?;
                        (!l.clobbered[k]).then(|| Op::Move {
                            dst: *dst,
                            src: l.members[k].1,
                        })
                    }
                })
            }
            Op::ArrayLen { dst, arr } => {
                holds.get(*arr as usize).copied().flatten().and_then(|i| {
                    let l = &lits[i];
                    (l.is_array && !l.escaped).then(|| Op::LoadConst {
                        dst: *dst,
                        value: NanBox::number(l.members.len() as f64),
                    })
                })
            }
            Op::GetKey { dst, obj, key } => {
                holds.get(*obj as usize).copied().flatten().and_then(|i| {
                    let l = &lits[i];
                    if !l.is_array || l.escaped {
                        return None;
                    }
                    let n = consts.get(*key as usize).copied().flatten()?;
                    // (`-0` reads index 0 as well: `ToPropertyKey(-0)` is "0".)
                    if !(n >= 0.0 && n < l.members.len() as f64 && (n as usize) as f64 == n) {
                        return None;
                    }
                    let k = n as usize;
                    (!l.clobbered[k]).then(|| Op::Move {
                        dst: *dst,
                        src: l.members[k].1,
                    })
                })
            }
            _ => None,
        };
        if let Some(op) = rewrite {
            p.ops[pc] = op;
            changed = true;
        }
        match &p.ops[pc] {
            Op::LoadConst { dst, value } => {
                let (dst, n) = (*dst, value.as_number());
                kill(&mut lits, &mut holds, &mut consts, dst);
                if let Some(c) = consts.get_mut(dst as usize) {
                    *c = n;
                }
            }
            Op::Move { dst, src } => {
                let (dst, src) = (*dst, *src);
                if dst == src {
                    continue;
                }
                let held = holds.get(src as usize).copied().flatten();
                let cv = consts.get(src as usize).copied().flatten();
                kill(&mut lits, &mut holds, &mut consts, dst);
                if let Some(c) = consts.get_mut(dst as usize) {
                    *c = cv;
                }
                if let Some(i) = held
                    && (dst as usize) < n_regs
                {
                    holds[dst as usize] = Some(i);
                    lits[i].regs.push(dst);
                    lits[i].moves.push((pc, dst));
                }
            }
            op @ (Op::NewObjectLit { .. } | Op::NewArrayFrom { .. }) => {
                let (dst, members, is_array): (Reg, Vec<_>, bool) = match op {
                    Op::NewObjectLit { dst, lit } => (
                        *dst,
                        lit.keys
                            .iter()
                            .cloned()
                            .map(Some)
                            .zip(lit.srcs.iter().copied())
                            .collect(),
                        false,
                    ),
                    Op::NewArrayFrom { dst, elems } => {
                        (*dst, elems.iter().map(|&r| (None, r)).collect(), true)
                    }
                    _ => unreachable!("matched above"),
                };
                let mut children = Vec::new();
                for m in &members {
                    if let Some(c) = holds.get(m.1 as usize).copied().flatten()
                        && !children.contains(&c)
                    {
                        children.push(c);
                    }
                }
                let writes_member = members.iter().any(|m| m.1 == dst);
                kill(&mut lits, &mut holds, &mut consts, dst);
                let id = lits.len();
                for &c in &children {
                    lits[c].parents += 1;
                    if writes_member {
                        // (Its register is gone from under the new literal:
                        // not followed.)
                        escape(&mut lits, c);
                    }
                }
                if !writes_member && (dst as usize) < n_regs {
                    let n = members.len();
                    lits.push(Lit {
                        pc,
                        members,
                        is_array,
                        clobbered: vec![false; n],
                        escaped: false,
                        overwritten: false,
                        moves: Vec::new(),
                        regs: vec![dst],
                        children,
                        parents: 0,
                    });
                    holds[dst as usize] = Some(id);
                }
            }
            op => {
                let a = access(op);
                for &u in &a.uses {
                    if let Some(i) = holds.get(u as usize).copied().flatten() {
                        escape(&mut lits, i);
                    }
                }
                for &d in &a.defs {
                    kill(&mut lits, &mut holds, &mut consts, d);
                }
            }
        }
    }
    if lits.iter().all(|l| l.escaped || l.overwritten) || *budget < p.ops.len() {
        return changed;
    }
    *budget -= p.ops.len();
    // What the rewrites left of the literals' readers beyond the block.
    let Some(live) = Liveness::of(&p.ops, n_regs) else {
        return changed;
    };
    let last = &p.ops[end];
    let arm = match last {
        Op::JumpIfFalse { target, .. }
            if *target > end + 1
                && *target <= p.ops.len()
                && (end + 2..*target).all(|q| !leaders[q])
                && !leaders[end + 1] =>
        {
            Some(*target)
        }
        _ => None,
    };
    let mut sunk: Vec<usize> = Vec::new();
    let mut succ = Vec::new();
    successors(last, end, &mut succ);
    // Parents first: a literal stays while one that refers to it stays, and
    // cannot be deleted under one that moved.
    #[derive(Clone, Copy, PartialEq)]
    enum Fate {
        Stays,
        Deleted,
        Sunk,
    }
    let mut fate = vec![Fate::Stays; lits.len()];
    for i in (0..lits.len()).rev() {
        let l = &lits[i];
        if l.escaped || l.overwritten {
            continue;
        }
        let referrers: Vec<Fate> = lits
            .iter()
            .enumerate()
            .filter(|(j, o)| *j != i && o.children.contains(&i))
            .map(|(j, _)| fate[j])
            .collect();
        if referrers.contains(&Fate::Stays) {
            continue;
        }
        let read_after = |at: usize| l.regs.iter().any(|&r| live.get(at, r));
        if !referrers.contains(&Fate::Sunk) && succ.iter().all(|&s| !read_after(s)) {
            nops[l.pc] = true;
            for &(m, _) in &l.moves {
                nops[m] = true;
            }
            fate[i] = Fate::Deleted;
            changed = true;
        } else if let Some(t) = arm
            // Its members still hold their values at the arm.
            && !l.clobbered.contains(&true)
            && !read_after(t)
        {
            sunk.push(l.pc);
            // A copy only the rewritten reads named is dead now.
            for &(m, r) in &l.moves {
                if l.regs.contains(&r) && live.get(end + 1, r) {
                    sunk.push(m);
                } else {
                    nops[m] = true;
                }
            }
            fate[i] = Fate::Sunk;
            changed = true;
        }
    }
    if !sunk.is_empty() {
        // The block's other ops close up, its control op among them, and the
        // moved ones follow it in their own order, right before the arm: no
        // pc outside the block changes, and no branch targets one inside.
        sunk.sort_unstable();
        let first = sunk[0];
        let mut kept = Vec::with_capacity(end + 1 - first);
        let mut moved = Vec::with_capacity(sunk.len());
        let mut kept_nops = Vec::with_capacity(end + 1 - first);
        let mut moved_nops = Vec::with_capacity(sunk.len());
        #[allow(clippy::needless_range_loop)] // (the op and its mark, same index)
        for pc in first..=end {
            let op = core::mem::replace(&mut p.ops[pc], Op::Jump { target: 0 });
            if sunk.binary_search(&pc).is_ok() {
                moved.push(op);
                moved_nops.push(nops[pc]);
            } else {
                kept.push(op);
                kept_nops.push(nops[pc]);
            }
        }
        for (k, (op, n)) in kept
            .into_iter()
            .zip(kept_nops)
            .chain(moved.into_iter().zip(moved_nops))
            .enumerate()
        {
            p.ops[first + k] = op;
            nops[first + k] = n;
        }
    }
    changed
}

/// Backward register liveness of a stream without handlers: live-in per pc.
struct Liveness {
    words: usize,
    bits: Vec<u64>,
}

impl Liveness {
    fn of(ops: &[Op], n_regs: usize) -> Option<Self> {
        let n = ops.len();
        let words = n_regs.div_ceil(64).max(1);
        if n.saturating_mul(words).saturating_mul(64) > MAX_LIVENESS_BITS.saturating_mul(4) {
            return None;
        }
        let accesses: Vec<Access> = ops.iter().map(access).collect();
        let mut bits = vec![0u64; n * words];
        let mut out = vec![0u64; words];
        let mut succ = Vec::with_capacity(2);
        loop {
            let mut changed = false;
            for pc in (0..n).rev() {
                out.fill(0);
                successors(&ops[pc], pc, &mut succ);
                for &t in &succ {
                    if t < n {
                        for w in 0..words {
                            out[w] |= bits[t * words + w];
                        }
                    }
                }
                let a = &accesses[pc];
                if a.sure {
                    for &d in &a.defs {
                        if (d as usize) < n_regs {
                            out[d as usize / 64] &= !(1u64 << (d % 64));
                        }
                    }
                }
                for &u in &a.uses {
                    if (u as usize) < n_regs {
                        out[u as usize / 64] |= 1u64 << (u % 64);
                    }
                }
                let row = &mut bits[pc * words..(pc + 1) * words];
                if row != out.as_slice() {
                    row.copy_from_slice(&out);
                    changed = true;
                }
            }
            if !changed {
                return Some(Self { words, bits });
            }
        }
    }

    /// Whether `r` is live on entry to `pc`.
    fn get(&self, pc: usize, r: Reg) -> bool {
        let i = pc * self.words + r as usize / 64;
        self.bits
            .get(i)
            .is_some_and(|w| w & (1u64 << (r % 64)) != 0)
    }
}

/// Removes every `LoadConst`/`Move` whose register nothing reads afterwards.
fn drop_dead_writes(p: &mut FnProto) {
    for _ in 0..DEAD_WRITE_ROUNDS {
        let Some(live) = Liveness::of(&p.ops, p.n_regs) else {
            return;
        };
        let n = p.ops.len();
        let mut nops = vec![false; n];
        let mut any = false;
        let mut succ = Vec::new();
        #[allow(clippy::needless_range_loop)] // (the op and its mark, same index)
        for pc in 0..n {
            let dst = match &p.ops[pc] {
                Op::LoadConst { dst, .. } => *dst,
                Op::Move { dst, .. } => *dst,
                _ => continue,
            };
            successors(&p.ops[pc], pc, &mut succ);
            if !succ.iter().any(|&s| live.get(s, dst)) {
                nops[pc] = true;
                any = true;
            }
        }
        if !any {
            return;
        }
        remove(&mut p.ops, &nops);
    }
}

/// Deletes the marked ops, renumbering the branches. A stream's last op is
/// never a plain fall-through off its end, and a removed op is never one that
/// transfers control, so every target maps to the next op that stays.
fn remove(ops: &mut Vec<Op>, nops: &[bool]) {
    let mut map = Vec::with_capacity(nops.len() + 1);
    let mut k = 0;
    for &gone in nops {
        map.push(k);
        if !gone {
            k += 1;
        }
    }
    map.push(k);
    let mut i = 0;
    ops.retain_mut(|op| {
        let gone = nops[i];
        i += 1;
        if !gone && let Op::Jump { target } | Op::JumpIfFalse { target, .. } = op {
            *target = map.get(*target).copied().unwrap_or(k);
        }
        !gone
    });
}
