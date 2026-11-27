//! Iterator helpers (`map` / `filter` / `take` / `drop` results) stepped in
//! the VM, and the consuming `Iterator.prototype` methods (`reduce`,
//! `toArray`, `forEach`, `some`, `every`, `find`) run over them.
//!
//! A helper's `next` was the host's (`Interp::iter_helper_next`): each step
//! of a chain like `Iterator.from(gen).filter(p).map(f).take(n)` crossed to
//! the host once per level, stepped the level below through the host's
//! iterator protocol (a result object per level), and called back into the
//! VM for every callback and the generator's resumption.
//!
//! Here a helper is stepped from its internal slots exactly as the host's
//! `iter_helper_step` does, kind by kind: the same running / done marks, the
//! same counter and limit updates, the same closing of the underlying
//! iterator when a callback throws or a `take` runs out (`VmHost::iter_close`).
//! What it pulls from is stepped by the cheapest exact means: another helper
//! with the intrinsic `next` recursively, a VM generator with the intrinsic
//! `next` by its direct resumption (`gen_fast::delegate_next`), anything else
//! by the host's own `IteratorStep` (`VmHost::iter_step`). Callbacks that are
//! VM functions are called here; a helper whose callback is anything else, a
//! `flatMap`, a helper that is done or running, or one not made by the host's
//! `make_iter_helper` is left to the host. No result object is made below
//! the consumer.

use super::{Ctx, FnProto, VmError, gen_fast, with_host, with_host_regs};
use crate::heap::Handle;
use crate::nanbox::NanBox;
use crate::nbexec::{
    HELPER_COUNTER, HELPER_DONE, HELPER_FN, HELPER_INNER, HELPER_KIND, HELPER_LIMIT, HELPER_NEXT,
    HELPER_RUNNING, HELPER_SOURCE, N_GEN_NEXT, N_ITER_HELPER_NEXT,
};
use crate::realm::Realm;

/// Whether helpers are stepped in the VM (`KATAAN_ITER_HELPER_VM=0`: off).
fn enabled() -> bool {
    #[cfg(feature = "std")]
    {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("KATAAN_ITER_HELPER_VM").is_none_or(|v| v != "0"))
    }
    #[cfg(not(feature = "std"))]
    {
        true
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Map,
    Filter,
    Take,
    Drop,
}

/// A helper's slots, read when it can be stepped here.
#[derive(Clone, Copy)]
struct Level {
    kind: Kind,
    src: NanBox,
    next: NanBox,
    /// The callback (`undefined` for `take` / `drop`).
    f: NanBox,
}

fn is_native(realm: &Realm, v: NanBox, id: u16) -> bool {
    v.as_handle()
        .is_some_and(|h| realm.native_at(Handle::from_raw(h)) == Some(id))
}

/// Whether `f` is a VM function this module calls itself (not a class
/// constructor, whose call throws: the host's).
fn vm_callable(realm: &Realm, funcs: &[FnProto], f: NanBox) -> bool {
    f.as_handle()
        .and_then(|h| realm.vm_function(Handle::from_raw(h)))
        .is_some_and(|(id, _)| funcs.get(id as usize).is_some_and(|p| !p.class_ctor))
}

/// The keys of a `map` / `filter` helper's layout, and of a `take` / `drop`
/// one, once stepped (its running mark made); neither has a done mark or a
/// flatMap's inner iterator.
const FN_KEYS: [&str; 7] = [
    HELPER_KIND,
    HELPER_SOURCE,
    HELPER_NEXT,
    HELPER_FN,
    HELPER_LIMIT,
    HELPER_COUNTER,
    HELPER_RUNNING,
];
const LIM_KEYS: [&str; 6] = [
    HELPER_KIND,
    HELPER_SOURCE,
    HELPER_NEXT,
    HELPER_LIMIT,
    HELPER_COUNTER,
    HELPER_RUNNING,
];
const ABSENT: [&str; 2] = [HELPER_DONE, HELPER_INNER];

/// A helper slot rewritten per step.
#[derive(Clone, Copy)]
enum Field {
    Limit,
    Counter,
    Running,
}

impl Field {
    fn key(self) -> &'static str {
        match self {
            Field::Limit => HELPER_LIMIT,
            Field::Counter => HELPER_COUNTER,
            Field::Running => HELPER_RUNNING,
        }
    }
    /// Its index in [`FN_KEYS`]; one less in [`LIM_KEYS`].
    fn fn_index(self) -> usize {
        match self {
            Field::Limit => 4,
            Field::Counter => 5,
            Field::Running => 6,
        }
    }
}

/// The value of `field` on the helper `h` (by slot under a learned layout).
fn get(realm: &Realm, h: Handle, field: Field) -> Option<NanBox> {
    if let Some(v) = realm
        .iter_helper_fn_layout
        .as_ref()
        .and_then(|l| l.peek(realm, h))
    {
        return Some(v[field.fn_index()]);
    }
    if let Some(v) = realm
        .iter_helper_lim_layout
        .as_ref()
        .and_then(|l| l.peek(realm, h))
    {
        return Some(v[field.fn_index() - 1]);
    }
    realm.get_property(h, field.key())
}

/// `set_hidden_property(h, field, v)` (by slot under a learned layout).
fn set(realm: &mut Realm, h: Handle, field: Field, v: NanBox) {
    if let Some(l) = realm.iter_helper_fn_layout.take() {
        let hit = l.peek(realm, h).is_some();
        if hit {
            l.write_checked(realm, h, field.fn_index(), field.key(), v);
        }
        realm.iter_helper_fn_layout = Some(l);
        if hit {
            return;
        }
    }
    if let Some(l) = realm.iter_helper_lim_layout.take() {
        let hit = l.peek(realm, h).is_some();
        if hit {
            l.write_checked(realm, h, field.fn_index() - 1, field.key(), v);
        }
        realm.iter_helper_lim_layout = Some(l);
        if hit {
            return;
        }
    }
    realm.set_hidden_property(h, field.key(), v);
}

fn kind_of(realm: &Realm, k: NanBox) -> Option<Kind> {
    Some(
        match realm.string_leaf_bytes(Handle::from_raw(k.as_handle()?))? {
            b"map" => Kind::Map,
            b"filter" => Kind::Filter,
            b"take" => Kind::Take,
            b"drop" => Kind::Drop,
            _ => return None,
        },
    )
}

/// The slots of the helper `h` when this module steps it: a `map` /
/// `filter` / `take` / `drop` helper that is neither done nor running, whose
/// callback (if any) is a VM function. `None` for anything else.
fn level(ctx: &mut Ctx, funcs: &[FnProto], h: Handle) -> Option<Level> {
    let realm = &ctx.realm;
    // A learned layout: every slot at once (it has no done mark and no
    // inner iterator).
    let fast = if let Some([k, src, next, f, _, _, run]) = realm
        .iter_helper_fn_layout
        .as_ref()
        .and_then(|l| l.peek(realm, h))
    {
        Some((k, src, next, f, run))
    } else if let Some([k, src, next, _, _, run]) = realm
        .iter_helper_lim_layout
        .as_ref()
        .and_then(|l| l.peek(realm, h))
    {
        Some((k, src, next, NanBox::undefined(), run))
    } else {
        None
    };
    if let Some((k, src, next, f, run)) = fast {
        let kind = kind_of(realm, k)?;
        if realm.truthy(run) || src.as_handle().is_none() {
            return None;
        }
        if matches!(kind, Kind::Map | Kind::Filter) && !vm_callable(realm, funcs, f) {
            return None;
        }
        return Some(Level { kind, src, next, f });
    }
    let kind = kind_of(realm, realm.get_property(h, HELPER_KIND)?)?;
    if realm.get_property(h, HELPER_DONE).is_some()
        || realm
            .get_property(h, HELPER_RUNNING)
            .is_some_and(|v| realm.truthy(v))
        || realm.get_property(h, HELPER_INNER).is_some()
    {
        return None;
    }
    let src = realm.get_property(h, HELPER_SOURCE)?;
    src.as_handle()?;
    let next = realm.get_property(h, HELPER_NEXT)?;
    let fnful = matches!(kind, Kind::Map | Kind::Filter);
    let f = if fnful {
        let f = realm.get_property(h, HELPER_FN)?;
        if !vm_callable(realm, funcs, f) {
            return None;
        }
        f
    } else {
        NanBox::undefined()
    };
    // Learn this layout for the next steps (once the running mark exists).
    if realm.get_property(h, HELPER_RUNNING).is_some() {
        if fnful {
            if let Some(l) =
                crate::realm::hidden_layout::HiddenLayout::learn(realm, h, FN_KEYS, &ABSENT)
            {
                ctx.realm.iter_helper_fn_layout = Some(l);
            }
        } else if let Some(l) =
            crate::realm::hidden_layout::HiddenLayout::learn(realm, h, LIM_KEYS, &ABSENT)
        {
            ctx.realm.iter_helper_lim_layout = Some(l);
        }
    }
    Some(Level { kind, src, next, f })
}

/// The helper `v` (its handle and slots) when `next` is the intrinsic
/// `%IteratorHelperPrototype%.next` and this module steps it: a `drop` still
/// skipping only when it can skip exactly (see [`skips_exactly`]).
fn helper_of(ctx: &mut Ctx, funcs: &[FnProto], v: NanBox, next: NanBox) -> Option<(Handle, Level)> {
    helper_of_at(ctx, funcs, v, next, 0)
}

fn helper_of_at(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    v: NanBox,
    next: NanBox,
    depth: u32,
) -> Option<(Handle, Level)> {
    if !enabled() || ctx.host.is_none() || ctx.realm.multi_realm || depth > 16 {
        return None;
    }
    if !is_native(&ctx.realm, next, N_ITER_HELPER_NEXT) {
        return None;
    }
    let h = Handle::from_raw(v.as_handle()?);
    let lv = level(ctx, funcs, h)?;
    if lv.kind == Kind::Drop
        && limit_of(&ctx.realm, h) > 0.0
        && !skips_exactly(ctx, funcs, lv.src, lv.next, depth)
    {
        return None;
    }
    Some((h, lv))
}

/// Whether skipping a step of `it` (whose `next` is `next`) without reading
/// its value is exact here: a helper or generator this module steps itself
/// (the host's `IteratorStep` would read `value`, where `drop` reads only
/// `done`).
fn skips_exactly(ctx: &mut Ctx, funcs: &[FnProto], it: NanBox, next: NanBox, depth: u32) -> bool {
    if helper_of_at(ctx, funcs, it, next, depth + 1).is_some() {
        return true;
    }
    is_native(&ctx.realm, next, N_GEN_NEXT) && gen_fast::direct_target_ok(ctx, funcs, it)
}

fn limit_of(realm: &Realm, h: Handle) -> f64 {
    get(realm, h, Field::Limit)
        .and_then(|n| n.as_number())
        .unwrap_or(0.0)
}

/// `IteratorStep` + `IteratorValue` of `it` with its captured `next`: the
/// value, or `None` once done. `want`: whether the value is read (a `drop`
/// skip reads `done` only).
pub(super) fn step_any(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    regs: &[NanBox],
    it: NanBox,
    next: NanBox,
    want: bool,
) -> Result<Option<NanBox>, VmError> {
    if let Some((h, lv)) = helper_of(ctx, funcs, it, next) {
        return helper_next(ctx, funcs, regs, h, lv);
    }
    let u = NanBox::undefined();
    if let Some(r) = gen_fast::delegate_next(ctx, funcs, regs, it, next, u) {
        return match r? {
            gen_fast::GenOut::Yield(v) => Ok(Some(v)),
            gen_fast::GenOut::Done(_) => Ok(None),
            gen_fast::GenOut::Delegated(r) => {
                let d = super::get_str_plain(ctx, r, "done")?;
                if ctx.realm.truthy(d) {
                    return Ok(None);
                }
                if !want {
                    return Ok(Some(u));
                }
                super::get_str_plain(ctx, r, "value").map(Some)
            }
        };
    }
    match with_host_regs(ctx, regs, |h| h.iter_step(it, next)) {
        Some(r) => r.map_err(VmError::from),
        None => Err(VmError::Unsupported),
    }
}

/// `IteratorClose(it)` on a normal completion (the host's).
fn close(ctx: &mut Ctx, it: NanBox) -> Result<(), VmError> {
    match with_host(ctx, |h| h.iter_close(it)) {
        Some(r) => r.map_err(VmError::from),
        None => Err(VmError::Unsupported),
    }
}

/// Calls the helper callback `f(v, counter)`; on a throw the underlying
/// iterator `src` is closed (its own error dropped) and the throw re-raised.
fn call_cb(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    regs: &[NanBox],
    src: NanBox,
    f: NanBox,
    args: &[NanBox],
) -> Result<NanBox, VmError> {
    match super::call_closure_compiled(ctx, funcs, Some(regs), f, args, NanBox::undefined()) {
        Ok(v) => Ok(v),
        Err(VmError::Thrown(e)) => {
            let _ = close(ctx, src);
            Err(VmError::Thrown(e))
        }
        Err(e) => Err(e),
    }
}

/// `%IteratorHelperPrototype%.next` of the helper `h` (slots `lv`): the
/// value, or `None` once done — the host's `iter_helper_next` without the
/// result object.
fn helper_next(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    regs: &[NanBox],
    h: Handle,
    lv: Level,
) -> Result<Option<NanBox>, VmError> {
    let hb = NanBox::handle(h.to_raw());
    let mark = ctx.realm.host_roots.vals.len();
    // (The helper, its source and callback, and a slot for the value a
    // filter holds across its callback.)
    ctx.realm
        .host_roots
        .vals
        .extend_from_slice(&[hb, lv.src, lv.next, lv.f, NanBox::undefined()]);
    set(&mut ctx.realm, h, Field::Running, NanBox::boolean(true));
    let r = helper_body(ctx, funcs, regs, h, lv, mark + 4);
    set(&mut ctx.realm, h, Field::Running, NanBox::boolean(false));
    ctx.realm.host_roots.vals.truncate(mark);
    if !matches!(r, Ok(Some(_))) {
        ctx.realm
            .set_hidden_property(h, HELPER_DONE, NanBox::boolean(true));
    }
    r
}

fn counter_incr(realm: &mut Realm, h: Handle) -> f64 {
    let c = get(realm, h, Field::Counter)
        .and_then(|n| n.as_number())
        .unwrap_or(0.0);
    set(realm, h, Field::Counter, NanBox::number(c + 1.0));
    c
}

/// The host's `iter_helper_step` for the four kinds.
fn helper_body(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    regs: &[NanBox],
    h: Handle,
    lv: Level,
    held: usize,
) -> Result<Option<NanBox>, VmError> {
    let Level { kind, src, next, f } = lv;
    match kind {
        Kind::Take => {
            let rem = limit_of(&ctx.realm, h);
            if rem <= 0.0 {
                close(ctx, src)?;
                return Ok(None);
            }
            set(&mut ctx.realm, h, Field::Limit, NanBox::number(rem - 1.0));
            step_any(ctx, funcs, regs, src, next, true)
        }
        Kind::Drop => {
            // (`helper_of` proved the skips exact.)
            let mut rem = limit_of(&ctx.realm, h);
            while rem > 0.0 {
                if step_any(ctx, funcs, regs, src, next, false)?.is_none() {
                    return Ok(None);
                }
                rem -= 1.0;
            }
            set(&mut ctx.realm, h, Field::Limit, NanBox::number(0.0));
            step_any(ctx, funcs, regs, src, next, true)
        }
        Kind::Map => match step_any(ctx, funcs, regs, src, next, true)? {
            Some(v) => {
                let c = counter_incr(&mut ctx.realm, h);
                call_cb(ctx, funcs, regs, src, f, &[v, NanBox::number(c)]).map(Some)
            }
            None => Ok(None),
        },
        Kind::Filter => loop {
            match step_any(ctx, funcs, regs, src, next, true)? {
                Some(v) => {
                    ctx.realm.host_roots.vals[held] = v;
                    let c = counter_incr(&mut ctx.realm, h);
                    let keep = call_cb(ctx, funcs, regs, src, f, &[v, NanBox::number(c)])?;
                    if ctx.realm.truthy(keep) {
                        return Ok(Some(v));
                    }
                }
                None => return Ok(None),
            }
        },
    }
}

/// `helper.next()` at a call site that proved the method is the intrinsic:
/// the iterator result. `None` (nothing done) for the host's call.
pub(super) fn site_next(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    regs: &[NanBox],
    recv: NanBox,
    method: NanBox,
) -> Option<Result<NanBox, VmError>> {
    let (h, lv) = helper_of(ctx, funcs, recv, method)?;
    Some(helper_next(ctx, funcs, regs, h, lv).map(|step| {
        let (v, done) = match step {
            Some(v) => (v, false),
            None => (NanBox::undefined(), true),
        };
        NanBox::handle(ctx.realm.new_iter_result(v, done).to_raw())
    }))
}

/// A `for-of` / destructuring step over the helper `it` with the intrinsic
/// `next`: the value or `None` once done; `None` from the outer `Option`
/// (nothing done) for the general step.
pub(super) fn for_of_step(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    regs: &[NanBox],
    it: NanBox,
    next: NanBox,
) -> Option<Result<Option<NanBox>, VmError>> {
    let (h, lv) = helper_of(ctx, funcs, it, next)?;
    Some(helper_next(ctx, funcs, regs, h, lv))
}

/// The consuming `Iterator.prototype` methods served here.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Consume {
    Reduce,
    ToArray,
    ForEach,
    Some,
    Every,
    Find,
}

impl Consume {
    pub(super) fn of(name: &[u8]) -> Option<Self> {
        Some(match name {
            b"reduce" => Consume::Reduce,
            b"toArray" => Consume::ToArray,
            b"forEach" => Consume::ForEach,
            b"some" => Consume::Some,
            b"every" => Consume::Every,
            b"find" => Consume::Find,
            _ => return None,
        })
    }
}

/// `this.reduce(f, init)` / `toArray()` / `forEach(f)` / `some(f)` /
/// `every(f)` / `find(f)` (the host's `iterator_proto_helper`) when `this`
/// is a helper or VM generator whose `next` is found without running code
/// (the intrinsic one) and the callback is a VM function: the loop runs
/// here. `None` (nothing done) for the host's call.
pub(super) fn consume(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    regs: &[NanBox],
    this: NanBox,
    op: Consume,
    args: &[NanBox],
) -> Option<Result<NanBox, VmError>> {
    if !enabled() || ctx.host.is_none() || ctx.realm.multi_realm {
        return None;
    }
    let th = Handle::from_raw(this.as_handle()?);
    // GetIteratorDirect's `Get(this, "next")`, when it runs no code.
    let next = match ctx.realm.ordinary_chain_get(th, "next")?.0 {
        crate::realm::ChainHit::Data(n) => n,
        _ => return None,
    };
    let steppable = helper_of(ctx, funcs, this, next).is_some()
        || (is_native(&ctx.realm, next, N_GEN_NEXT)
            && gen_fast::direct_target_ok(ctx, funcs, this));
    if !steppable {
        return None;
    }
    let f = args.first().copied().unwrap_or(NanBox::undefined());
    if op != Consume::ToArray && !vm_callable(&ctx.realm, funcs, f) {
        return None;
    }
    let mark = ctx.realm.host_roots.vals.len();
    ctx.realm.host_roots.vals.extend_from_slice(&[
        this,
        next,
        f,
        args.get(1).copied().unwrap_or(NanBox::undefined()),
    ]);
    let r = consume_loop(ctx, funcs, regs, this, next, f, op, args, mark + 3);
    ctx.realm.host_roots.vals.truncate(mark);
    Some(r)
}

#[allow(clippy::too_many_arguments)]
fn consume_loop(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    regs: &[NanBox],
    this: NanBox,
    next: NanBox,
    f: NanBox,
    op: Consume,
    args: &[NanBox],
    slot: usize,
) -> Result<NanBox, VmError> {
    let u = NanBox::undefined();
    let call = |ctx: &mut Ctx, argv: &[NanBox]| -> Result<NanBox, VmError> {
        match super::call_closure_compiled(ctx, funcs, Some(regs), f, argv, u) {
            Ok(v) => Ok(v),
            Err(VmError::Thrown(e)) => {
                let _ = close(ctx, this);
                Err(VmError::Thrown(e))
            }
            Err(e) => Err(e),
        }
    };
    match op {
        Consume::ToArray => {
            let base = ctx.realm.host_roots.vals.len();
            loop {
                match step_any(ctx, funcs, regs, this, next, true) {
                    Ok(Some(v)) => ctx.realm.host_roots.vals.push(v),
                    Ok(None) => break,
                    Err(e) => {
                        ctx.realm.host_roots.vals.truncate(base);
                        return Err(e);
                    }
                }
            }
            let out = ctx.realm.host_roots.vals.split_off(base);
            Ok(NanBox::handle(ctx.realm.new_array(out).to_raw()))
        }
        Consume::ForEach => {
            let mut i = 0u64;
            while let Some(v) = step_any(ctx, funcs, regs, this, next, true)? {
                call(ctx, &[v, NanBox::number(i as f64)])?;
                i += 1;
            }
            Ok(u)
        }
        Consume::Some | Consume::Every | Consume::Find => {
            let mut i = 0u64;
            while let Some(v) = step_any(ctx, funcs, regs, this, next, true)? {
                ctx.realm.host_roots.vals[slot] = v;
                let t = call(ctx, &[v, NanBox::number(i as f64)])?;
                let t = ctx.realm.truthy(t);
                match op {
                    Consume::Every if !t => {
                        close(ctx, this)?;
                        return Ok(NanBox::boolean(false));
                    }
                    Consume::Some if t => {
                        close(ctx, this)?;
                        return Ok(NanBox::boolean(true));
                    }
                    Consume::Find if t => {
                        close(ctx, this)?;
                        return Ok(v);
                    }
                    _ => {}
                }
                i += 1;
            }
            Ok(match op {
                Consume::Every => NanBox::boolean(true),
                Consume::Some => NanBox::boolean(false),
                _ => u,
            })
        }
        Consume::Reduce => {
            let mut i = 0u64;
            let mut acc = if args.len() >= 2 {
                args[1]
            } else {
                match step_any(ctx, funcs, regs, this, next, true)? {
                    Some(v) => {
                        i = 1;
                        v
                    }
                    None => {
                        let e = with_host(ctx, |h| {
                            h.make_error(
                                "TypeError",
                                "Reduce of empty iterator with no initial value",
                            )
                        })
                        .ok_or(VmError::Unsupported)?;
                        return Err(VmError::Thrown(e));
                    }
                }
            };
            ctx.realm.host_roots.vals[slot] = acc;
            while let Some(v) = step_any(ctx, funcs, regs, this, next, true)? {
                acc = call(ctx, &[acc, v, NanBox::number(i as f64)])?;
                ctx.realm.host_roots.vals[slot] = acc;
                i += 1;
            }
            Ok(acc)
        }
    }
}
