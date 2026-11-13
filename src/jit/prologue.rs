//! Which registers a generic body's native-call entry must set to
//! `undefined` (`call_seq::emit_native_entry`).
//!
//! The interpreter starts every register past the activation's inputs
//! `undefined`, and the entry used to store it into each of them: a store per
//! register per call (over a hundred in a large loop body). A register the
//! body writes before any read of it, on every path, needs none of them: no
//! op can tell what it held before. What remains is what *else* looks at the
//! window before that write — the collector, which roots every word below the
//! register stack's top. That is safe without the store too: the stack never
//! holds a handle to a collected object (see `JitStack`), so such a register
//! holds, at worst, a value an earlier activation left there, whose object the
//! collector then keeps for one more cycle. Nothing else reads a native-entered
//! window: such an activation never leaves for the interpreter (only one an
//! interpreter entered does), every op names the registers it reads (the
//! verifier's operand list), and a throw out of the body ends the window (one
//! caught in it resumes at its catch pc, covered below).
//!
//! The analysis is a forward must-write data flow over the bytecode: `W(pc)`,
//! the registers written on every path from the entry to `pc`. A register
//! qualifies when every op that reads it has it in `W`. Every op that is not a
//! jump also flows into the next one (a spurious edge after a `return` only
//! shrinks `W`). A catch pc is entered from wherever the body may throw, which
//! is anywhere: its `W` is what every op has in its own (plus the catch
//! register the unwinding writes). An op counts as a write of its destination
//! only when it always writes it when it completes, and only as compiled (a
//! `Nop` writes nothing); every other register an op names counts as read.

use crate::nbvm::{Op, Reg};
use alloc::vec::Vec;

/// The register `op` always writes when it completes and does not read
/// (unless it also names it as an operand), for the ops whose compiled form
/// does exactly that.
fn writes(op: &Op) -> Option<Reg> {
    match op {
        Op::LoadConst { dst, .. }
        | Op::Move { dst, .. }
        | Op::Add { dst, .. }
        | Op::Sub { dst, .. }
        | Op::Mul { dst, .. }
        | Op::Div { dst, .. }
        | Op::Mod { dst, .. }
        | Op::AddValue { dst, .. }
        | Op::ValueBin { dst, .. }
        | Op::Compare { dst, .. }
        | Op::Lt { dst, .. }
        | Op::StrictEq { dst, .. }
        | Op::Not { dst, .. }
        | Op::Neg { dst, .. }
        | Op::BitNot { dst, .. }
        | Op::TypeOf { dst, .. }
        | Op::ToNumeric { dst, .. }
        | Op::Step { dst, .. }
        | Op::LoadGlobal { dst, .. }
        | Op::GetProp { dst, .. }
        | Op::GetMethod { dst, .. }
        | Op::GetKey { dst, .. }
        | Op::GetElem { dst, .. }
        | Op::ArrayLen { dst, .. }
        | Op::CellGet { dst, .. }
        | Op::NewObject { dst }
        | Op::NewArray { dst, .. }
        | Op::NewArrayFrom { dst, .. }
        | Op::NewString { dst, .. }
        | Op::NewBox { dst }
        | Op::MakeClosure { dst, .. }
        | Op::Call { dst, .. }
        | Op::CallValue { dst, .. }
        | Op::CallMethod { dst, .. }
        | Op::CallKey { dst, .. }
        | Op::CallWithMethod { dst, .. }
        | Op::Construct { dst, .. }
        | Op::CallNative { dst, .. }
        | Op::PlainPrims { dst, .. }
        | Op::InlineGuard { dst, .. } => Some(*dst),
        _ => None,
    }
}

/// Whether [`entry_fill`] counts a dropped dead store as touching nothing,
/// an inlined call's guard as writing the captures, and a return or throw
/// as going nowhere (`KATAAN_JIT_PFILL=0` keeps the coarser analysis, read
/// once per process).
fn precise_fill() -> bool {
    #[cfg(feature = "std")]
    {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("KATAAN_JIT_PFILL").is_none_or(|v| v != "0"))
    }
    #[cfg(not(feature = "std"))]
    {
        true
    }
}

/// For a native-call entry over `n_regs` registers of which `0..first` (and
/// `also`) are its inputs, always stored: per register, whether the entry
/// must still store `undefined` in it. `nop(pc)` says op `pc` compiled to
/// nothing. Conservative throughout: `true` for every register when in doubt.
pub(super) fn entry_fill(
    bytecode: &[Op],
    nop: &dyn Fn(usize) -> bool,
    n_regs: usize,
    first: usize,
    also: Option<usize>,
) -> Vec<bool> {
    let n = bytecode.len();
    let mut need = alloc::vec![true; n_regs];
    if n == 0 || n_regs <= first {
        return need;
    }
    // Per op: the registers it reads (`reads[at[pc]..at[pc + 1]]`), the one
    // it writes, and where it goes besides the next op.
    let mut reads: Vec<Reg> = Vec::new();
    let mut at: Vec<usize> = Vec::with_capacity(n + 1);
    let mut write: Vec<Option<Reg>> = Vec::with_capacity(n);
    // An inlined call's guard also writes the callee's captures (`caps`).
    let mut caps_of: Vec<Option<(Reg, u16)>> = Vec::with_capacity(n);
    let mut catch: Vec<(usize, Reg)> = Vec::new();
    let seen = core::cell::RefCell::new(Vec::new());
    let precise = precise_fill();
    for (pc, op) in bytecode.iter().enumerate() {
        seen.borrow_mut().clear();
        // A dead store `drop_dead_stores` dropped touches nothing: no op
        // reads its destination, and its source only reaches that.
        let dropped = precise && nop(pc) && matches!(op, Op::LoadConst { .. } | Op::Move { .. });
        if !dropped {
            crate::bytecode::op_for_each_reg(op, &|r| seen.borrow_mut().push(r));
        }
        let mut seen = seen.borrow_mut();
        let w = if nop(pc) { None } else { writes(op) };
        if let Some(d) = w
            && let Some(i) = seen.iter().position(|&r| r == d)
        {
            // The destination itself; a second mention is a read.
            seen.swap_remove(i);
        }
        // The guard's captures: written (when it holds, and only the
        // inlined body after it reads them; any other register of the
        // callee's window it initializes with ops of its own).
        let caps = match *op {
            Op::InlineGuard { caps, n_caps, .. } if precise && !nop(pc) && n_caps > 0 => {
                for k in 0..n_caps {
                    let c = caps.wrapping_add(k);
                    if let Some(i) = seen.iter().position(|&r| r == c) {
                        seen.swap_remove(i);
                    }
                }
                Some((caps, n_caps))
            }
            _ => None,
        };
        if seen.iter().chain(w.as_ref()).any(|&r| r as usize >= n_regs)
            || caps.is_some_and(|(c, k)| c as usize + k as usize > n_regs)
        {
            return need;
        }
        match *op {
            Op::Jump { target } | Op::JumpIfFalse { target, .. } if target > n => return need,
            Op::PushHandler { target, reg } => {
                if target >= n {
                    return need;
                }
                catch.push((target, reg));
            }
            _ => {}
        }
        at.push(reads.len());
        reads.extend_from_slice(&seen);
        write.push(w);
        caps_of.push(caps);
    }
    at.push(reads.len());
    // `W(pc)` on entry to each op, `words` bits per op (registers set:
    // written on every path), from "everything" down to the greatest
    // fixed point; an op no path reaches keeps everything.
    let words = n_regs.div_ceil(64);
    let full = |i: usize| -> u64 {
        let rem = n_regs - 64 * i;
        if rem >= 64 { !0 } else { (1u64 << rem) - 1 }
    };
    let mut w: Vec<u64> = (0..n * words).map(|k| full(k % words)).collect();
    let mut entry: Vec<u64> = alloc::vec![0; words];
    for r in (0..first).chain(also).filter(|&r| r < n_regs) {
        entry[r / 64] |= 1 << (r % 64);
    }
    w[..words].copy_from_slice(&entry);
    let mut out: Vec<u64> = alloc::vec![0; words];
    let mut all: Vec<u64> = alloc::vec![0; words];
    loop {
        let mut changed = false;
        // `s`'s `W` meets `out` (what a predecessor leaves).
        let mut meet = |w: &mut [u64], s: usize, out: &[u64]| {
            for (a, b) in w[s * words..(s + 1) * words].iter_mut().zip(out) {
                let v = *a & *b;
                changed |= v != *a;
                *a = v;
            }
        };
        for pc in 0..n {
            out.copy_from_slice(&w[pc * words..(pc + 1) * words]);
            if let Some(d) = write[pc] {
                out[d as usize / 64] |= 1 << (d % 64);
            }
            if let Some((c, k)) = caps_of[pc] {
                for r in c as usize..c as usize + k as usize {
                    out[r / 64] |= 1 << (r % 64);
                }
            }
            let (falls, target) = match bytecode[pc] {
                Op::Jump { target } => (false, Some(target)),
                Op::JumpIfFalse { target, .. } => (true, Some(target)),
                // (An op that never goes on to the next.)
                Op::Return { .. }
                | Op::Throw { .. }
                | Op::ThrowTypeError { .. }
                | Op::ThrowReferenceError { .. }
                    if precise =>
                {
                    (false, None)
                }
                _ => (true, None),
            };
            if falls && pc + 1 < n {
                meet(&mut w, pc + 1, &out);
            }
            if let Some(t) = target.filter(|&t| t < n) {
                meet(&mut w, t, &out);
            }
        }
        // A catch pc: what every op has (a throw may come from any), and
        // the catch register.
        if !catch.is_empty() {
            all.iter_mut().enumerate().for_each(|(i, a)| *a = full(i));
            for pc in 0..n {
                for (a, b) in all.iter_mut().zip(&w[pc * words..(pc + 1) * words]) {
                    *a &= *b;
                }
            }
            for &(t, reg) in &catch {
                out.copy_from_slice(&all);
                out[reg as usize / 64] |= 1 << (reg % 64);
                meet(&mut w, t, &out);
            }
        }
        if !changed {
            break;
        }
    }
    // A register qualifies when every read of it is after a write.
    let mut ok = alloc::vec![true; n_regs];
    for pc in 0..n {
        for &r in &reads[at[pc]..at[pc + 1]] {
            let r = r as usize;
            if w[pc * words + r / 64] >> (r % 64) & 1 == 0 {
                ok[r] = false;
            }
        }
    }
    for r in first..n_regs {
        need[r] = !ok[r];
    }
    need
}

#[cfg(test)]
mod tests {
    use super::entry_fill;
    use crate::nanbox::NanBox;
    use crate::nbvm::Op;
    use alloc::vec::Vec;

    fn konst(dst: u16) -> Op {
        Op::LoadConst {
            dst,
            value: NanBox::number(1.0),
        }
    }

    fn fill(ops: &[Op], n_regs: usize, first: usize) -> Vec<bool> {
        entry_fill(ops, &|_| false, n_regs, first, None)
    }

    #[test]
    fn written_before_read_needs_no_store() {
        let ops = [konst(2), Op::Move { dst: 3, src: 2 }, Op::Return { src: 3 }];
        assert_eq!(fill(&ops, 5, 2), [true, true, false, false, false]);
    }

    #[test]
    fn a_path_without_the_write_needs_it() {
        // r2 is written only when r0 is truthy, then read.
        let ops = [
            Op::JumpIfFalse { cond: 0, target: 2 },
            konst(2),
            Op::Return { src: 2 },
        ];
        assert_eq!(fill(&ops, 3, 1), [true, false, true]);
    }

    #[test]
    fn a_loop_carried_read_needs_it() {
        // r2 = r2 + r3 in a loop whose first iteration reads r2 unwritten.
        let ops = [
            konst(3),
            Op::AddValue { dst: 2, a: 2, b: 3 },
            Op::JumpIfFalse { cond: 0, target: 1 },
            Op::Return { src: 2 },
        ];
        assert_eq!(fill(&ops, 4, 1), [true, false, true, false]);
    }

    #[test]
    fn a_catch_reads_only_what_every_op_has() {
        // r2 is written before the try, r3 inside it; the catch reads both.
        let ops = [
            konst(2),
            Op::PushHandler { target: 5, reg: 4 },
            konst(3),
            Op::PopHandler,
            Op::Jump { target: 7 },
            Op::AddValue { dst: 5, a: 2, b: 3 },
            Op::Return { src: 5 },
            Op::Return { src: 4 },
        ];
        let f = fill(&ops, 6, 2);
        // (Conservative: the catch assumes a throw from anywhere, even
        // before r2's write.)
        assert!(f[2] && f[3], "unwritten at some op that may throw");
        assert!(f[4], "named by the PushHandler before its write");
        assert!(!f[5]);
    }

    #[test]
    fn a_dropped_dead_store_touches_nothing() {
        // `let x;` whose binding lives elsewhere: r2 = undefined, r3 = r2,
        // both dropped (nothing reads r3); r4 is read after its write.
        let ops = [
            Op::LoadConst {
                dst: 2,
                value: NanBox::undefined(),
            },
            Op::Move { dst: 3, src: 2 },
            konst(4),
            Op::Return { src: 4 },
        ];
        let f = entry_fill(&ops, &|pc| pc < 2, 5, 2, None);
        assert_eq!(f, [true, true, false, false, false]);
    }

    #[test]
    fn a_throw_goes_nowhere() {
        // The op after a throw is reached by the jump only, after r2's write.
        let ops = [
            Op::JumpIfFalse { cond: 0, target: 3 },
            konst(3),
            Op::Throw { src: 3 },
            konst(2),
            Op::Return { src: 2 },
        ];
        assert_eq!(fill(&ops, 4, 1), [true, false, false, false]);
    }

    #[test]
    fn an_inline_guard_writes_the_captures() {
        let ops = [
            Op::InlineGuard {
                dst: 2,
                call: 4,
                key: 0,
                callee: 0,
                args: Vec::new(),
                caps: 3,
                n_caps: 2,
                poly: 0,
            },
            Op::JumpIfFalse { cond: 2, target: 4 },
            Op::AddValue { dst: 5, a: 3, b: 4 },
            Op::Return { src: 5 },
            Op::Return { src: 0 },
        ];
        assert_eq!(fill(&ops, 6, 1), [true, false, false, false, false, false]);
    }

    #[test]
    fn a_nop_writes_nothing() {
        let ops = [konst(2), Op::Return { src: 2 }];
        let f = entry_fill(&ops, &|pc| pc == 0, 3, 1, None);
        assert_eq!(f, [true, false, true]);
    }
}
