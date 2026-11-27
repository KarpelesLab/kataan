//! `Op::ArrayUnpack`: an array destructuring's values read at once, shared
//! by the interpreter loop and the compiled tier's helper.
//!
//! `[a, b = 0, ...r] = v` runs `GetIterator(v)`, one `IteratorStep` per
//! element (each followed by its default and its target's write) and an
//! `IteratorClose` when the pattern stops before the end. When `v` is a
//! plain array with no hole whose iteration is the built-in one
//! (`iter_fast::plain_array_values`), every step reads a present own
//! element, so nothing a step does can be observed, and nothing can change
//! the array between the steps either when the pattern's defaults and
//! targets run no code (the compiler emits the op only then): the steps
//! may all run first. Without a rest element a pattern no longer than the
//! array leaves the iterator unfinished (no step found the end), and its
//! close must find no `return` to call (`iter_fast::close_is_noop`).
//! Anything else changes
//! nothing and reports `false`, for the protocol's own path.

use super::{Ctx, Reg, iter_fast};
use crate::nanbox::NanBox;

/// Reads the values of `[dsts…, ...rest] = regs[src]` into those registers
/// when the destructuring can be done at once (see the module docs).
pub(super) fn unpack(
    ctx: &mut Ctx,
    regs: &mut [NanBox],
    src: Reg,
    dsts: &[Reg],
    rest: Option<Reg>,
) -> bool {
    let v = regs[src as usize];
    let n = dsts.len();
    // (`can_open` also publishes its memo to compiled code, whose inline
    // form of this op reads it.)
    if !iter_fast::can_open(&mut ctx.realm, v) {
        return false;
    }
    let Some(len) = iter_fast::plain_spread_elems(&ctx.realm, v).map(<[NanBox]>::len) else {
        return false;
    };
    if rest.is_none() && len >= n && !iter_fast::close_is_noop(&mut ctx.realm) {
        return false;
    }
    // (Found plain just now: the memo it read is current.)
    let Some(elems) = iter_fast::plain_spread_elems(&ctx.realm, v) else {
        return false;
    };
    for (i, d) in dsts.iter().enumerate() {
        regs[*d as usize] = elems.get(i).copied().unwrap_or(NanBox::undefined());
    }
    if let Some(r) = rest {
        let tail = elems.get(n..).unwrap_or(&[]).to_vec();
        let h = ctx.realm.new_array_from_iter(tail.into_iter());
        regs[r as usize] = NanBox::handle(h.to_raw());
    }
    true
}
