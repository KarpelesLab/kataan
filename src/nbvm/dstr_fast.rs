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
    // (A close of an unfinished iteration must find nothing to call:
    // asked first, so the values are read in one pass below.)
    if rest.is_none() && !iter_fast::close_is_noop(&mut ctx.realm) {
        // (Only a pattern no longer than the array leaves it unfinished.)
        match iter_fast::plain_spread_elems(&ctx.realm, v) {
            Some(e) if e.len() < n => {}
            _ => return false,
        }
    }
    // The common case, a warm memo already published to compiled code (whose
    // inline form of this op reads it), in one lookup; else `can_open`
    // (which also publishes it) first.
    if !(iter_fast::open_published(&ctx.realm)
        && iter_fast::plain_spread_elems(&ctx.realm, v).is_some())
        && !iter_fast::can_open(&mut ctx.realm, v)
    {
        return false;
    }
    let Some(elems) = iter_fast::plain_spread_elems(&ctx.realm, v) else {
        return false;
    };
    for (i, d) in dsts.iter().enumerate() {
        regs[*d as usize] = elems.get(i).copied().unwrap_or(NanBox::undefined());
    }
    if let Some(r) = rest {
        let tail = super::ArgBuf::from_slice(elems.get(n..).unwrap_or(&[]));
        let h = ctx.realm.new_array_from_slice(&tail);
        regs[r as usize] = NanBox::handle(h.to_raw());
    }
    true
}
