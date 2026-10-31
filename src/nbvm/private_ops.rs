//! `Op::PrivateGet` / `Op::PrivateSet` / `Op::PrivateIn` / `Op::PrivateDefine`, shared by the
//! interpreter loop and the compiled tier's helper (`jit_helper_exec_op`) so
//! the two cannot diverge: the site's layout memo (`Realm::private_cached_get`
//! and friends, exact hits only), else the general brand-checked path.

use super::{
    Ctx, FnProto, Op, VmError, is_object_value, private_define, private_get, private_set, vm_error,
};
use crate::heap::Handle;
use crate::nanbox::NanBox;

/// Runs the private-element op `op` (one of the three above; anything else
/// is a no-op) over the registers `regs`.
pub(super) fn exec(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    op: &Op,
    regs: &mut [NanBox],
) -> Result<(), VmError> {
    match op {
        Op::PrivateGet {
            dst,
            obj,
            key,
            cache,
        } => {
            let (o, k) = (regs[*obj as usize], regs[*key as usize]);
            regs[*dst as usize] = match ctx.realm.private_cached_get(o, k, cache) {
                Some(v) => v,
                None => private_get(ctx, funcs, o, k)?,
            };
        }
        Op::PrivateSet {
            obj,
            key,
            src,
            cache,
        } => {
            let (o, k, v) = (
                regs[*obj as usize],
                regs[*key as usize],
                regs[*src as usize],
            );
            if !ctx.realm.private_cached_set(o, k, v, cache) {
                private_set(ctx, funcs, o, k, v)?;
            }
        }
        Op::PrivateIn {
            dst,
            obj,
            key,
            cache,
        } => {
            let (o, k) = (regs[*obj as usize], regs[*key as usize]);
            if let Some(b) = ctx.realm.private_cached_has(o, k, cache) {
                regs[*dst as usize] = NanBox::boolean(b);
                return Ok(());
            }
            match o
                .as_handle()
                .map(Handle::from_raw)
                .filter(|_| is_object_value(&ctx.realm, o))
            {
                Some(h) => {
                    let ks = ctx.realm.to_display_string(k);
                    regs[*dst as usize] = NanBox::boolean(ctx.realm.has_own(h, &ks));
                }
                None => {
                    let e = vm_error(
                        ctx,
                        "TypeError",
                        "Cannot use 'in' operator to search for a private field in a non-object",
                    );
                    return Err(VmError::Thrown(e));
                }
            }
        }
        Op::PrivateDefine { obj, key, src } => {
            let (o, k, v) = (
                regs[*obj as usize],
                regs[*key as usize],
                regs[*src as usize],
            );
            private_define(ctx, o, k, &[("", v)])?;
        }
        _ => {}
    }
    Ok(())
}

/// A private name's key text, copied out of the realm without allocating
/// when it is short (the usual `"\0#x@v7f"`), so that realm calls taking the
/// key need not hold a borrow of the realm's string.
pub(super) struct KeyText {
    buf: [u8; 64],
    len: usize,
    owned: Option<alloc::string::String>,
}

impl KeyText {
    /// The text of the private name `k` (its display string).
    pub(super) fn of(realm: &crate::realm::Realm, k: NanBox) -> Self {
        let mut t = Self {
            buf: [0; 64],
            len: 0,
            owned: None,
        };
        match k
            .as_handle()
            .and_then(|h| realm.string_leaf_bytes(Handle::from_raw(h)))
            .filter(|b| b.len() <= t.buf.len() && core::str::from_utf8(b).is_ok())
        {
            Some(b) => {
                t.buf[..b.len()].copy_from_slice(b);
                t.len = b.len();
            }
            None => t.owned = Some(realm.to_display_string(k)),
        }
        t
    }

    /// The key followed by `suffix` (an accessor's `":g"` / `":s"` slot).
    pub(super) fn with_suffix(&self, suffix: &str) -> Self {
        let mut t = Self {
            buf: self.buf,
            len: self.len,
            owned: None,
        };
        if self.owned.is_none() && self.len + suffix.len() <= t.buf.len() {
            t.buf[self.len..self.len + suffix.len()].copy_from_slice(suffix.as_bytes());
            t.len += suffix.len();
        } else {
            t.owned = Some(alloc::format!("{}{suffix}", self.as_str()));
        }
        t
    }

    /// The key text.
    pub(super) fn as_str(&self) -> &str {
        match &self.owned {
            Some(s) => s,
            // Copied from a `str` check above (`of` / `with_suffix`).
            None => core::str::from_utf8(&self.buf[..self.len]).unwrap_or(""),
        }
    }
}
