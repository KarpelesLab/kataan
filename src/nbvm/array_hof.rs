//! The array higher-order built-ins, run inside the VM.
//!
//! `arr.map(f)` with a VM closure `f` used to be a delegated host call: the
//! VM lent its realm to the host, the host's `Array.prototype.map` looped, and
//! every element's callback re-entered the VM through a fresh context
//! (`call_vm_function`): two realm swaps, a context built and dropped, and the
//! generic host call path, per element. Here the loop runs in the calling VM
//! context instead, and each callback is an ordinary VM call
//! ([`call_closure`]), which takes the tiered / compiled path like any other.
//!
//! The loops follow the ECMA-262 algorithms (23.1.3) step by step. What makes
//! them cheap is only how each step is *performed*:
//!
//! * The method is recognized by identity: the receiver's `key` resolves,
//!   through an ordinary chain of data properties, to the built-in
//!   `%Array.prototype%` function of that name. A replaced, deleted or
//!   shadowed method, an accessor or an exotic object on the way, all miss.
//! * `LengthOfArrayLike(O)` of a dense array (no sparse tail) is its element
//!   count, read once.
//! * Per index, `HasProperty(O, k)` + `Get(O, k)` of a *present dense slot* is
//!   the slot itself (an index accessor lives in the side table over a hole).
//!   Anything else — a hole, an index past a callback's truncation — is asked
//!   of the host with its full `[[HasProperty]]` / `[[Get]]` (the prototype
//!   chain, inherited getters, proxies). The check is repeated per element, so
//!   a callback that mutates the array is observed exactly as the spec says.
//! * `ArraySpeciesCreate(O, n)` is taken here only when it is the default
//!   `ArrayCreate(n)`: `O.constructor` is a data property holding `%Array%`
//!   whose `@@species` is still the built-in getter (which returns `this`).
//!   Those reads have no side effects, so doing them up front is exact.
//!
//! Everything else (a native or bound callback, a typed array, a subclass, a
//! custom species, more than one realm) returns `None`: the host's built-in
//! runs as before.
//!
//! GC: the loop publishes the caller's registers plus the receiver, callback,
//! `thisArg` and the result / accumulator into [`Ctx::frame_shadow`] (the same
//! protocol as a JS-to-JS call), so a collection inside a callback sees every
//! value the loop still holds.

use super::{Ctx, FnProto, HostError, Realm, VmError, call_closure, host_get, vm_error, with_host};
use crate::heap::Handle;
use crate::nanbox::NanBox;
use crate::nbexec::{N_ARRAY_PROTO_FN, N_BASE_ARRAY, N_TYPED_ARRAY_SPECIES};
use alloc::vec::Vec;

/// The built-ins this module runs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Hof {
    ForEach,
    Map,
    Filter,
    Some,
    Every,
    Find,
    FindIndex,
    FindLast,
    FindLastIndex,
    Reduce,
    ReduceRight,
    Sort,
}

/// The built-in `Array.prototype` method named `name`, if it is one of these.
pub(super) fn hof_for(name: &[u8]) -> Option<Hof> {
    Some(match name {
        b"forEach" => Hof::ForEach,
        b"map" => Hof::Map,
        b"filter" => Hof::Filter,
        b"some" => Hof::Some,
        b"every" => Hof::Every,
        b"find" => Hof::Find,
        b"findIndex" => Hof::FindIndex,
        b"findLast" => Hof::FindLast,
        b"findLastIndex" => Hof::FindLastIndex,
        b"reduce" => Hof::Reduce,
        b"reduceRight" => Hof::ReduceRight,
        b"sort" => Hof::Sort,
        _ => return None,
    })
}

/// `key` looked up along `h`'s prototype chain when every object on the way
/// is ordinary and holds no accessor for it: `Ok(Some(v))` for a data
/// property, `Ok(None)` when absent. `Err(())` when the lookup could run code
/// or be exotic (an accessor, a proxy, a module namespace, …).
fn chain_data(realm: &Realm, h: Handle, key: &str) -> Result<Option<NanBox>, ()> {
    let mut cur = Some(h);
    let mut depth = 0usize;
    while let Some(c) = cur {
        if !realm.is_ordinary_get_cell(c) || realm.is_host_exotic(c) {
            return Err(());
        }
        if realm.accessor(c, key).is_some() {
            return Err(());
        }
        if let Some(v) = realm.get_property(c, key) {
            return Ok(Some(v));
        }
        depth += 1;
        if depth > 64 {
            return Err(());
        }
        cur = realm.object_proto(c);
    }
    Ok(None)
}

/// Whether `ArraySpeciesCreate(O, n)` for the array `h` is the default
/// `ArrayCreate(n)`, decided with no observable effect (see the module docs).
fn default_species(realm: &Realm, h: Handle) -> bool {
    let Ok(Some(c)) = chain_data(realm, h, "constructor") else {
        return false;
    };
    let Some(ch) = c.as_handle().map(Handle::from_raw) else {
        return false;
    };
    if realm.native_at(ch) != Some(N_BASE_ARRAY) {
        return false;
    }
    let Some(key) = realm.vm_array_iter().species_key.as_deref() else {
        return false;
    };
    realm.accessor(ch, key).is_some_and(|(g, _)| {
        g.as_handle()
            .is_some_and(|g| realm.native_at(Handle::from_raw(g)) == Some(N_TYPED_ARRAY_SPECIES))
    })
}

/// `recv[key](...args)` when `recv` is a plain array, `key` resolves to one
/// of the built-ins above, and the callback is a VM closure; `None` to take
/// the ordinary (host) call path. This resolves the method itself (an
/// ordinary data-property chain walk); a call site that has already proven
/// which intrinsic it calls enters [`run`] directly (`native_call`).
pub(super) fn try_call(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    regs: &[NanBox],
    recv: NanBox,
    key: &str,
    args: &[NanBox],
) -> Option<Result<NanBox, VmError>> {
    let h = Handle::from_raw(recv.as_handle()?);
    ctx.realm.array_elements(h)?;
    let op = hof_for(key.as_bytes())?;
    // The method must be the built-in `Array.prototype[key]` itself.
    let m = chain_data(&ctx.realm, h, key).ok()??;
    let (id, target) = ctx
        .realm
        .bound_native_at(Handle::from_raw(m.as_handle()?))?;
    if id != N_ARRAY_PROTO_FN || ctx.realm.string_value(target).as_deref() != Some(key) {
        return None;
    }
    run(ctx, funcs, regs, recv, op, args)
}

/// The built-in `op` called with `this = recv` and `args` — the caller has
/// established that the method called *is* that intrinsic — when `recv` is
/// a plain array, the callback a VM closure, and (`map`/`filter`) the species
/// the default; `None` to take the ordinary (host) call path. `regs` are the
/// calling frame's registers, published for the GC while the loop runs.
pub(super) fn run(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    regs: &[NanBox],
    recv: NanBox,
    op: Hof,
    args: &[NanBox],
) -> Option<Result<NanBox, VmError>> {
    let h = Handle::from_raw(recv.as_handle()?);
    let len = ctx.realm.array_elements(h)?.len();
    let cb = args.first().copied()?;
    if ctx.realm.multi_realm
        || !cb
            .as_handle()
            .is_some_and(|c| ctx.realm.is_vm_function(Handle::from_raw(c)))
        // A sparse tail (`length` beyond the dense storage) is the host's.
        || ctx.realm.array_length(h) != Some(len)
    {
        return None;
    }
    if matches!(op, Hof::Map | Hof::Filter) && !default_species(&ctx.realm, h) {
        return None;
    }
    // `sort` collects then writes back; here only over an array whose every
    // index in range is a present, ordinary data slot (the host's precise
    // `SortIndexedProperties` path takes the rest).
    if op == Hof::Sort
        && (ctx.realm.array_has_index_overrides(h)
            || ctx
                .realm
                .array_elements(h)
                .is_some_and(|e| e.iter().any(|v| v.is_hole())))
    {
        return None;
    }
    let this_arg = args.get(1).copied().unwrap_or(NanBox::undefined());
    // Publish the caller's registers and everything the loop holds.
    let mark = ctx.frame_shadow.len();
    ctx.frame_shadow.extend_from_slice(regs);
    ctx.frame_shadow.extend_from_slice(&[recv, cb, this_arg]);
    let slot = ctx.frame_shadow.len();
    ctx.frame_shadow.push(NanBox::undefined());
    ctx.frames_published += 1;
    let mut l = Loop {
        callee: Callee::new(ctx, funcs, cb),
        h,
        recv,
        this_arg,
        slot,
    };
    let r = match op {
        Hof::Sort => l.sort(ctx, len),
        Hof::Reduce | Hof::ReduceRight => l.reduce(ctx, len, op == Hof::ReduceRight, args),
        _ => l.iterate(ctx, len, op),
    };
    ctx.frames_published -= 1;
    ctx.frame_shadow.truncate(mark);
    Some(r)
}

/// The callback, and how [`Callee::invoke`] calls it.
struct Callee<'f> {
    funcs: &'f [FnProto],
    cb: NanBox,
    /// The callback's function, when it is one of `funcs` and callable (not a
    /// class constructor): the candidate for the direct compiled calls.
    #[cfg_attr(
        not(all(feature = "jit", target_os = "linux", target_arch = "x86_64")),
        allow(dead_code)
    )]
    proto: Option<&'f FnProto>,
    /// The prepared direct call into its compiled generic body, once it has one.
    #[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
    compiled: Option<super::jit_tier::CompiledCallback>,
    /// Calls that found no compiled body to enter; past [`COMPILED_MISS_LIMIT`]
    /// the loop stops looking (a callee that has not compiled by then — the
    /// tier-up threshold is a handful of calls — will not).
    #[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
    misses: u32,
}

/// See [`Callee::misses`].
#[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
const COMPILED_MISS_LIMIT: u32 = 64;

impl<'f> Callee<'f> {
    fn new(ctx: &Ctx, funcs: &'f [FnProto], cb: NanBox) -> Self {
        let proto = cb
            .as_handle()
            .and_then(|c| ctx.realm.vm_function(Handle::from_raw(c)))
            .and_then(|(id, _)| funcs.get(id as usize))
            .filter(|p| !p.class_ctor);
        Callee {
            funcs,
            cb,
            proto,
            #[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
            compiled: None,
            #[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
            misses: 0,
        }
    }

    /// `Call(callbackfn, this, args)`. A callee with a compiled generic body
    /// is entered directly (as a compiled caller enters it), and a hot one
    /// with Int/Float-tier code runs on its arguments alone (as
    /// `call_with_inner` runs it); anything else is the ordinary VM call. The
    /// loop's frame is already published, so the callee may collect.
    #[inline]
    fn invoke(&mut self, ctx: &mut Ctx, args: &[NanBox], this: NanBox) -> Result<NanBox, VmError> {
        #[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
        if let Some(proto) = self.proto {
            if let Some(c) = &mut self.compiled {
                return c.call(ctx, self.funcs, this, args);
            }
            if ctx.call_depth < ctx.realm.limits.max_call_depth
                && ctx.realm.vm_total_depth < ctx.realm.limits.max_call_depth
                && let Some(v) = super::unboxed_fast_call(proto, args)
            {
                return Ok(v);
            }
            if self.misses < COMPILED_MISS_LIMIT {
                if let Some(mut c) =
                    super::jit_tier::CompiledCallback::new(ctx, self.funcs, self.cb)
                {
                    let r = c.call(ctx, self.funcs, this, args);
                    self.compiled = Some(c);
                    return r;
                }
                self.misses += 1;
            }
        }
        call_closure(ctx, self.funcs, self.cb, args, this)
    }
}

/// One running built-in: the receiver, the callback, and the published root
/// slot (`ctx.frame_shadow[slot]`) holding the result or accumulator.
struct Loop<'f> {
    callee: Callee<'f>,
    h: Handle,
    recv: NanBox,
    this_arg: NanBox,
    slot: usize,
}

impl Loop<'_> {
    /// `HasProperty(O, k)` (when `probe`) then `Get(O, k)`: `None` for an
    /// absent index.
    #[inline]
    fn element(&self, ctx: &mut Ctx, k: usize, probe: bool) -> Result<Option<NanBox>, VmError> {
        if let Some(v) = ctx
            .realm
            .array_elements(self.h)
            .and_then(|e| e.get(k).copied())
            .filter(|v| !v.is_hole())
        {
            return Ok(Some(v));
        }
        self.element_slow(ctx, k, probe)
    }

    /// [`Self::element`] for an index that is not a present dense slot: the
    /// host's full `[[HasProperty]]` / `[[Get]]`.
    #[cold]
    fn element_slow(
        &self,
        ctx: &mut Ctx,
        k: usize,
        probe: bool,
    ) -> Result<Option<NanBox>, VmError> {
        let key = NanBox::handle(ctx.realm.new_string(&alloc::format!("{k}")).to_raw());
        if probe {
            let recv = self.recv;
            let has = with_host(ctx, |host| host.relational(false, key, recv))
                .unwrap_or(Err(HostError::Fault))
                .map_err(VmError::from)?;
            if !ctx.realm.truthy(has) {
                return Ok(None);
            }
        }
        host_get(ctx, self.recv, key).map(Some)
    }

    /// `Call(callbackfn, thisArg, « kValue, k, O »)`.
    #[inline]
    fn call(&mut self, ctx: &mut Ctx, v: NanBox, k: usize) -> Result<NanBox, VmError> {
        let a = [v, NanBox::number(k as f64), self.recv];
        self.callee.invoke(ctx, &a, self.this_arg)
    }

    /// `forEach`, `map`, `filter`, `some`, `every` and the `find` family.
    fn iterate(&mut self, ctx: &mut Ctx, len: usize, op: Hof) -> Result<NanBox, VmError> {
        // `ArraySpeciesCreate` (default, checked by the caller): `map` gets
        // `ArrayCreate(len)` (all holes), `filter` `ArrayCreate(0)`.
        let out = match op {
            Hof::Map => Some(ctx.realm.new_array(alloc::vec![NanBox::hole(); len])),
            Hof::Filter => Some(ctx.realm.new_array(Vec::new())),
            _ => None,
        };
        if let Some(o) = out {
            ctx.frame_shadow[self.slot] = NanBox::handle(o.to_raw());
        }
        // The `find` family reads every index with a plain `Get` (no
        // `HasProperty`), the rest skip absent indices.
        let probe = !matches!(
            op,
            Hof::Find | Hof::FindIndex | Hof::FindLast | Hof::FindLastIndex
        );
        let backwards = matches!(op, Hof::FindLast | Hof::FindLastIndex);
        for n in 0..len {
            let k = if backwards { len - 1 - n } else { n };
            let v = match self.element(ctx, k, probe)? {
                Some(v) => v,
                None if probe => continue,
                None => NanBox::undefined(),
            };
            let r = self.call(ctx, v, k)?;
            match op {
                Hof::ForEach => {}
                Hof::Map => {
                    // CreateDataPropertyOrThrow(A, k, mappedValue): `A` is fresh
                    // and unreachable from user code, so a plain store.
                    ctx.realm.set_element(out.expect("map result"), k, r);
                }
                Hof::Filter => {
                    if ctx.realm.truthy(r) {
                        // CreateDataPropertyOrThrow(A, to, kValue), through the
                        // write barrier (`A` may have aged during the callbacks).
                        let a = out.expect("filter result");
                        let to = ctx.realm.array_dense_len(a).unwrap_or(0);
                        ctx.realm.set_element(a, to, v);
                    }
                }
                Hof::Some => {
                    if ctx.realm.truthy(r) {
                        return Ok(NanBox::boolean(true));
                    }
                }
                Hof::Every => {
                    if !ctx.realm.truthy(r) {
                        return Ok(NanBox::boolean(false));
                    }
                }
                Hof::Find | Hof::FindLast => {
                    if ctx.realm.truthy(r) {
                        return Ok(v);
                    }
                }
                Hof::FindIndex | Hof::FindLastIndex => {
                    if ctx.realm.truthy(r) {
                        return Ok(NanBox::number(k as f64));
                    }
                }
                Hof::Reduce | Hof::ReduceRight | Hof::Sort => unreachable!(),
            }
        }
        Ok(match op {
            Hof::Map | Hof::Filter => ctx.frame_shadow[self.slot],
            Hof::Some => NanBox::boolean(false),
            Hof::Every => NanBox::boolean(true),
            Hof::FindIndex | Hof::FindLastIndex => NanBox::number(-1.0),
            _ => NanBox::undefined(),
        })
    }

    /// `reduce` / `reduceRight` (23.1.3.24 / .25).
    fn reduce(
        &mut self,
        ctx: &mut Ctx,
        len: usize,
        right: bool,
        args: &[NanBox],
    ) -> Result<NanBox, VmError> {
        let at = |n: usize| if right { len - 1 - n } else { n };
        let mut n = 0;
        let mut acc = if args.len() >= 2 {
            args[1]
        } else {
            // The first present element seeds the accumulator.
            loop {
                if n >= len {
                    let e = vm_error(
                        ctx,
                        "TypeError",
                        "Reduce of empty array with no initial value",
                    );
                    return Err(VmError::Thrown(e));
                }
                let k = at(n);
                n += 1;
                if let Some(v) = self.element(ctx, k, true)? {
                    break v;
                }
            }
        };
        while n < len {
            let k = at(n);
            n += 1;
            let Some(v) = self.element(ctx, k, true)? else {
                continue;
            };
            ctx.frame_shadow[self.slot] = acc;
            let a = [acc, v, NanBox::number(k as f64), self.recv];
            acc = self.callee.invoke(ctx, &a, NanBox::undefined())?;
        }
        Ok(acc)
    }

    /// `sort(comparefn)` (23.1.3.30) over a hole-free, override-free array:
    /// `SortIndexedProperties` collects every element, sorts the list, and the
    /// result is written back with `Set(obj, j, v, true)`.
    fn sort(&mut self, ctx: &mut Ctx, len: usize) -> Result<NanBox, VmError> {
        let items: Vec<NanBox> = ctx
            .realm
            .array_elements(self.h)
            .map(|e| e[..len.min(e.len())].to_vec())
            .unwrap_or_default();
        // The list's values stay rooted for the whole sort (it only permutes
        // them).
        ctx.frame_shadow.extend_from_slice(&items);
        // `undefined` sorts after everything, never passed to the comparator.
        let undefined_count = items.iter().filter(|v| v.is_undefined()).count();
        let mut v: Vec<NanBox> = items.into_iter().filter(|v| !v.is_undefined()).collect();
        let mut buf = alloc::vec![NanBox::undefined(); v.len()];
        self.sort_run(ctx, &mut v, &mut buf)?;
        v.extend(core::iter::repeat_n(NanBox::undefined(), undefined_count));
        // Write back. A comparator may have changed the array: a slot that is
        // still a present ordinary data element takes a plain store, anything
        // else the host's full `[[Set]]` (which throws on failure).
        for (j, x) in v.into_iter().enumerate() {
            let plain = ctx
                .realm
                .array_elements(self.h)
                .is_some_and(|e| e.get(j).is_some_and(|s| !s.is_hole()))
                && !ctx.realm.array_index_has_override(self.h, j);
            if plain {
                ctx.realm.set_element(self.h, j, x);
            } else {
                let key = NanBox::handle(ctx.realm.new_string(&alloc::format!("{j}")).to_raw());
                let recv = self.recv;
                with_host(ctx, |host| host.set_member(recv, key, x, true))
                    .unwrap_or(Err(HostError::Fault))
                    .map_err(VmError::from)?;
            }
        }
        Ok(self.recv)
    }

    /// `SortCompare(x, y)` with the user comparator: `ToNumber` of its result,
    /// `NaN` taken as `+0`.
    fn compare(&mut self, ctx: &mut Ctx, x: NanBox, y: NanBox) -> Result<f64, VmError> {
        let r = self.callee.invoke(ctx, &[x, y], NanBox::undefined())?;
        let n = match r.as_number() {
            Some(n) => n,
            None => {
                let n = super::host_unary_any(ctx, crate::ast::UnaryOp::Plus, r)?;
                n.as_number().unwrap_or(f64::NAN)
            }
        };
        Ok(if n.is_nan() { 0.0 } else { n })
    }

    /// Stable merge sort of `v` (`buf` is same-length scratch): the same
    /// algorithm, and so the same comparator call sequence, as the host's
    /// `Interp::sort_run` — insertion sort for short runs, and merges that take
    /// from the left on ties.
    fn sort_run(
        &mut self,
        ctx: &mut Ctx,
        v: &mut [NanBox],
        buf: &mut [NanBox],
    ) -> Result<(), VmError> {
        let n = v.len();
        if n <= 1 {
            return Ok(());
        }
        if n <= 16 {
            for i in 1..n {
                let mut j = i;
                while j > 0 && self.compare(ctx, v[j - 1], v[j])? > 0.0 {
                    v.swap(j - 1, j);
                    j -= 1;
                }
            }
            return Ok(());
        }
        let mid = n / 2;
        {
            let (left, right) = v.split_at_mut(mid);
            let (bl, br) = buf.split_at_mut(mid);
            self.sort_run(ctx, left, bl)?;
            self.sort_run(ctx, right, br)?;
        }
        let (mut i, mut j, mut k) = (0usize, mid, 0usize);
        while i < mid && j < n {
            if self.compare(ctx, v[i], v[j])? <= 0.0 {
                buf[k] = v[i];
                i += 1;
            } else {
                buf[k] = v[j];
                j += 1;
            }
            k += 1;
        }
        buf[k..k + (mid - i)].copy_from_slice(&v[i..mid]);
        k += mid - i;
        buf[k..k + (n - j)].copy_from_slice(&v[j..n]);
        v.copy_from_slice(&buf[..n]);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::nbexec::Interp;
    use crate::parser::Parser;

    /// Runs `src` hosted and renders its completion value.
    fn run(src: &str) -> alloc::string::String {
        let program = Parser::parse_program(src).expect("parse");
        let mut interp = Interp::new();
        let value = interp.run(&program).expect("exec");
        interp.realm().to_display_string(value)
    }

    #[test]
    fn callbacks_see_live_mutation_and_holes() {
        // Holes are skipped; an index the prototype supplies is visited.
        assert_eq!(
            run("[1,,3].map(x => x * 2).length + ':' + (1 in [1,,3].map(x => x))"),
            "3:false"
        );
        assert_eq!(
            run(
                "Array.prototype[1] = 'P'; var r = [1,,3].map(x => x).join(); delete Array.prototype[1]; r"
            ),
            "1,P,3"
        );
        // `length` is read once; a push is not visited, a pop is.
        assert_eq!(
            run(
                "var a = [1,2,3], s = []; a.forEach(x => { s.push(x); a.push(9); }); s.join() + '/' + a.length"
            ),
            "1,2,3/6"
        );
        assert_eq!(
            run("var a = [1,2,3,4], s = []; a.forEach(x => { s.push(x); a.pop(); }); s.join()"),
            "1,2"
        );
        // A getter installed mid-iteration fires.
        assert_eq!(
            run(
                "var a = [1,2,3]; a.map((x, i) => { if (i === 0) Object.defineProperty(a, 2, { get() { return 'G'; } }); return x; }).join()"
            ),
            "1,2,G"
        );
        // `find` reads holes with a plain `Get` (no skipping).
        assert_eq!(run("var n = 0; [1,,3].find(() => { n++; }); n"), "3");
    }

    #[test]
    fn results_this_and_throws() {
        assert_eq!(
            run("[1,2].map(function (x) { return x + this.k; }, { k: 5 }).join()"),
            "6,7"
        );
        assert_eq!(run("[0,1,2,3].filter(x => x % 2).join()"), "1,3");
        assert_eq!(run("[1,2,3,4].reduce((a, b) => a + b)"), "10");
        assert_eq!(
            run("['a','b','c'].reduceRight((a, b, i) => a + b + i)"),
            "cb1a0"
        );
        assert_eq!(
            run("try { [].reduce((a, b) => a) } catch (e) { e instanceof TypeError }"),
            "true"
        );
        assert_eq!(
            run("try { [1,2].map(x => { throw x * 7; }) } catch (e) { e }"),
            "7"
        );
        assert_eq!(
            run("try { [1].map(class {}) } catch (e) { e instanceof TypeError }"),
            "true"
        );
        assert_eq!(run("[1,2,3,4].findLastIndex(x => x % 2 === 1)"), "2");
        assert_eq!(
            run("[1,2].some(x => x > 1) + ',' + [1,2].every(x => x > 1)"),
            "true,false"
        );
    }

    #[test]
    fn non_default_species_and_replaced_methods_take_the_host() {
        assert_eq!(
            run("class A extends Array {}; A.from([1,2]).map(x => x) instanceof A"),
            "true"
        );
        assert_eq!(
            run(
                "var o = Array.prototype.map; Array.prototype.map = function () { return 'mine'; }; var r = [1].map(x => x); Array.prototype.map = o; r"
            ),
            "mine"
        );
    }

    #[test]
    fn sort_with_a_comparator() {
        assert_eq!(
            run("[5,1,4,2,3,9,8,7,6,10,11,12,13,14,15,16,17,18].sort((a, b) => a - b).join()"),
            "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18"
        );
        assert_eq!(
            run("[3,undefined,1,2].sort((a, b) => b - a).join()"),
            "3,2,1,"
        );
        // Abrupt comparator: the array is left as it was.
        assert_eq!(
            run("var a = [3,1,2]; try { a.sort(() => { throw 0; }); } catch (e) {} a.join()"),
            "3,1,2"
        );
        // A result object is ToNumber'd; an inconsistent comparator is safe.
        assert_eq!(
            run("[3,1,2].sort((a, b) => ({ valueOf() { return a - b; } })).join()"),
            "1,2,3"
        );
        assert_eq!(
            run(
                "var s = 1, a = []; for (var i = 0; i < 300; i++) a.push(i); a.sort(() => (s = s * 16807 % 2147483647) % 3 - 1); a.length + ':' + a.reduce((x, y) => x + y)"
            ),
            "300:44850"
        );
        // Mutation by the comparator: the sorted list is written back with `Set`.
        assert_eq!(
            run(
                "var a = [3,1,2], n = 0; a.sort((x, y) => { if (n++ === 0) a.push(9); return x - y; }); a.join()"
            ),
            "1,2,3,9"
        );
        assert_eq!(
            run("var a = [3,1,2]; a.sort((x, y) => { a.length = 0; return x - y; }); a.join()"),
            "1,2,3"
        );
        assert_eq!(
            run(
                "var a = [3,1,2]; try { a.sort((x, y) => { Object.freeze(a); return x - y; }); } catch (e) { e instanceof TypeError }"
            ),
            "true"
        );
    }
}
