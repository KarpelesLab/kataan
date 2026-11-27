//! `ToPrimitive` of an ordinary object operand in the VM, for a hosted
//! operator (`+obj`, `obj * 2`, `a < b`, `x + obj`, …) whose object carries
//! its own conversion methods (`[Symbol.toPrimitive]`, or a `valueOf` /
//! `toString` a class or literal defines).
//!
//! Those operators are otherwise the host's whole (`host_binary` /
//! `host_unary`), which re-enters the VM for each method it calls. Here the
//! lookups are the realm's ordinary chain walk ([`Realm::ordinary_chain_get`],
//! which answers only over ordinary objects, so a `[[Get]]` it answers runs no
//! code) and each call is `call_closure`'s (a VM function in the VM, anything
//! else through the host's `Call`): exactly ToPrimitive (ECMA-262 §7.1.1) and
//! OrdinaryToPrimitive (§7.1.1.1), step for step. A step this walk cannot
//! take exactly (a getter, a Proxy or other exotic object on the chain, a
//! callable it cannot recognize cheaply, a non-callable `@@toPrimitive`)
//! hands the whole conversion back to the host when nothing has run yet, and
//! otherwise takes that one step through the host (`[[Get]]`, `Call`).

use super::{Ctx, FnProto, HostError, VmError, call_closure_pub, vm_error, with_host};
use crate::heap::Handle;
use crate::nanbox::{NanBox, Unpacked};
use crate::realm::{ChainHit, Realm};

/// ToPrimitive's preferred type.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Hint {
    Default,
    Number,
    String,
}

impl Hint {
    fn text(self) -> &'static str {
        match self {
            Hint::Default => "default",
            Hint::Number => "number",
            Hint::String => "string",
        }
    }
}

/// Whether the VM converts object operands itself (`KATAAN_VM_TOPRIM=0`
/// leaves them all to the host; read once per process).
fn enabled() -> bool {
    #[cfg(feature = "std")]
    {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("KATAAN_VM_TOPRIM").is_none_or(|v| v != "0"))
    }
    #[cfg(not(feature = "std"))]
    {
        true
    }
}

/// Whether `v` is an operand this module converts: an ordinary object cell
/// (not host-exotic) in a hosted single-realm run.
fn candidate(ctx: &Ctx, v: NanBox) -> bool {
    ctx.host.is_some()
        && !ctx.realm.multi_realm
        && v.as_handle().is_some_and(|h| {
            let h = Handle::from_raw(h);
            ctx.realm.is_object_cell(h) && !ctx.realm.is_host_exotic(h)
        })
}

/// Whether the value `v` is an Object (for a conversion method's result).
fn is_object(realm: &Realm, v: NanBox) -> bool {
    super::is_object_value(realm, v)
}

/// Whether the value `m` is callable, when that is cheap to tell: `Some`
/// for a non-handle (never), a VM function or a native / host / bound
/// native cell (always); `None` for any other object (a bound function
/// object, a Proxy, `%Function.prototype%`, …).
fn callable(realm: &Realm, m: NanBox) -> Option<bool> {
    let Some(h) = m.as_handle().map(Handle::from_raw) else {
        return Some(false);
    };
    if realm.is_vm_function(h) || realm.is_callable_cell(h) {
        return Some(true);
    }
    if realm.is_string_handle(h) || realm.symbol_at(h).is_some() || realm.is_bigint(h) {
        return Some(false);
    }
    None
}

/// The data value `[[Get]](o, key)` reads, when the ordinary walk answers it
/// without running code (`undefined` for an absent key); `None` otherwise.
fn plain_get(realm: &Realm, o: Handle, key: &str) -> Option<NanBox> {
    match realm.ordinary_chain_get(o, key)?.0 {
        ChainHit::Data(v) => Some(v),
        ChainHit::Absent => Some(NanBox::undefined()),
        ChainHit::Accessor(..) => None,
    }
}

/// `ToPrimitive(v, hint)` for an object operand `v` (see the module docs):
/// `Some` with its result or throw; `None`, having run nothing, when the
/// host is to convert it. `regs`: the calling frame's register file, lent
/// to the methods called (as `call_closure_pub`'s).
pub(super) fn to_primitive(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    v: NanBox,
    hint: Hint,
    regs: Option<&[NanBox]>,
) -> Option<Result<NanBox, VmError>> {
    if !enabled() || !candidate(ctx, v) {
        return None;
    }
    let o = Handle::from_raw(v.as_handle()?);
    // 1.a-b: GetMethod(input, @@toPrimitive).
    let exotic = {
        let key = ctx.realm.to_primitive_key()?;
        plain_get(&ctx.realm, o, key)?
    };
    if !matches!(exotic.unpack(), Unpacked::Undefined | Unpacked::Null) {
        // (A non-callable one is the host's TypeError.)
        if callable(&ctx.realm, exotic) != Some(true) {
            return None;
        }
        let h = NanBox::handle(ctx.realm.new_string(hint.text()).to_raw());
        return Some(call(ctx, funcs, regs, exotic, &[h], v).and_then(|r| {
            if is_object(&ctx.realm, r) {
                Err(convert_error(ctx))
            } else {
                Ok(r)
            }
        }));
    }
    // 1.c-d: OrdinaryToPrimitive(input, hint), `default` taken as `number`.
    let order = if hint == Hint::String {
        ["toString", "valueOf"]
    } else {
        ["valueOf", "toString"]
    };
    let mut ran = false;
    for name in order {
        let method = match plain_get(&ctx.realm, o, name) {
            Some(m) => m,
            // A getter or an exotic object on the chain: the host's whole
            // conversion, while nothing has run; else this `[[Get]]`.
            None if !ran => return None,
            None => {
                let k = ctx.realm.key_value(name);
                match with_host(ctx, |h| h.get_with_receiver(v, k, v))
                    .unwrap_or(Err(HostError::Fault))
                {
                    Ok(m) => m,
                    Err(e) => return Some(Err(VmError::from(e))),
                }
            }
        };
        match callable(&ctx.realm, method) {
            Some(false) => continue,
            Some(true) => {}
            None if !ran => return None,
            // (A value the cheap test cannot place is the host's `Call`.)
            None => {}
        }
        // `%Object.prototype.valueOf%` returns its (object) receiver: no
        // result, and nothing observable happens.
        if method
            .as_handle()
            .is_some_and(|m| ctx.realm.native_at(Handle::from_raw(m)) == Some(OBJ_PROTO_VALUEOF))
        {
            continue;
        }
        ran = true;
        match call(ctx, funcs, regs, method, &[], v) {
            Ok(r) if !is_object(&ctx.realm, r) => return Some(Ok(r)),
            Ok(_) => {}
            Err(e) => return Some(Err(e)),
        }
    }
    Some(Err(convert_error(ctx)))
}

/// `Call(f, this, args)`: a VM function with a compiled body entered
/// directly (`jit_tier::call_compiled_with`), anything else as
/// `call_closure_pub` calls it.
fn call(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    regs: Option<&[NanBox]>,
    f: NanBox,
    args: &[NanBox],
    this: NanBox,
) -> Result<NanBox, VmError> {
    #[cfg(all(feature = "jit", jit_native))]
    if let Some(r) = super::jit_tier::call_compiled_with(ctx, funcs, regs, f, this, args) {
        return r;
    }
    call_closure_pub(ctx, funcs, regs, f, args, this)
}

/// `%Object.prototype.valueOf%`'s native id.
const OBJ_PROTO_VALUEOF: u16 = crate::nbexec::N_OBJ_PROTO_VALUEOF;

/// The `TypeError` of a conversion that produced no primitive.
fn convert_error(ctx: &mut Ctx) -> VmError {
    VmError::Thrown(vm_error(
        ctx,
        "TypeError",
        "Cannot convert object to primitive value",
    ))
}

/// For a hosted binary operator over `a` and `b` whose operands both go
/// through ToPrimitive with `hint`, first `a` then `b`: the operands with
/// those this module converts replaced by their primitives, converting left
/// to right and stopping (leaving the rest to the host) at the first operand
/// it cannot, or once a converted left operand is a Symbol or a BigInt (whose
/// ToNumeric/ToString may throw before the right operand is converted: the
/// host's order). `None` when nothing was converted.
pub(super) fn operands(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    a: NanBox,
    b: NanBox,
    hint: Hint,
    regs: Option<&[NanBox]>,
) -> Option<Result<(NanBox, NanBox), VmError>> {
    let a_obj = candidate(ctx, a);
    if !a_obj && !candidate(ctx, b) {
        return None;
    }
    let x = if a_obj {
        match to_primitive(ctx, funcs, a, hint, regs)? {
            Ok(x) => x,
            Err(e) => return Some(Err(e)),
        }
    } else if is_object(&ctx.realm, a) {
        // A left operand the host converts: so is the right one.
        return None;
    } else {
        a
    };
    let stop = x.as_handle().is_some_and(|h| {
        let h = Handle::from_raw(h);
        ctx.realm.symbol_at(h).is_some() || ctx.realm.is_bigint(h)
    });
    if stop || !candidate(ctx, b) {
        return a_obj.then_some(Ok((x, b)));
    }
    // The converted left operand is rooted while the right one's methods run.
    let mark = ctx.frame_shadow.len();
    ctx.frame_shadow.push(x);
    let y = to_primitive(ctx, funcs, b, hint, regs);
    ctx.frame_shadow.truncate(mark);
    match y {
        Some(Ok(y)) => Some(Ok((x, y))),
        Some(Err(e)) => Some(Err(e)),
        None => a_obj.then_some(Ok((x, b))),
    }
}
