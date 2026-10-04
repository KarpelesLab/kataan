//! The generator and async-function **intrinsics** the bytecode VM's
//! suspended frames use (`ROADMAP.md` §2.0): `%GeneratorPrototype%`,
//! `%GeneratorFunction.prototype%`, `%AsyncFunction.prototype%`,
//! `%AsyncGeneratorPrototype%` / `%AsyncGeneratorFunction.prototype%`, iterator
//! result objects, and `%AsyncFromSyncIteratorPrototype%` (async `yield*` /
//! `for await` over a sync iterator). The frames themselves — registers, pc,
//! handlers — live in the VM (`crate::nbvm`, `VM_GEN` / `VM_ASYNC` / `VM_AGEN`).

use super::*;

impl Interp {
    /// The shared `%GeneratorPrototype%`: `next`/`return`/`throw` (length 1, each
    /// dispatching on `this`'s generator frame) and `[Symbol.toStringTag]`
    /// "Generator", inheriting `%IteratorPrototype%`. Created once, cached on the
    /// `Iterator` constructor.
    pub(crate) fn generator_prototype(&mut self) -> Option<Handle> {
        let iter_ctor = self
            .current
            .get("Iterator")
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)?;
        const CACHE: &str = "\u{0}genproto";
        if let Some(gp) = self
            .realm
            .get_property(iter_ctor, CACHE)
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
        {
            return Some(gp);
        }
        let iter_proto = self
            .realm
            .get_property(iter_ctor, "prototype")
            .and_then(|p| p.as_handle())
            .map(Handle::from_raw)?;
        let gp = self.realm.new_object_with_proto(Some(iter_proto));
        for (name, nid) in [
            ("next", N_GEN_NEXT),
            ("return", N_GEN_RETURN),
            ("throw", N_GEN_THROW),
        ] {
            let f = self.realm.new_native(nid);
            self.install_fn_name_length(f, name, 1);
            self.realm
                .set_property(gp, name, NanBox::handle(f.to_raw()));
            self.realm.mark_hidden(gp, name);
        }
        self.install_to_string_tag(gp, "Generator");
        self.realm
            .set_hidden_property(iter_ctor, CACHE, NanBox::handle(gp.to_raw()));
        Some(gp)
    }

    /// `%GeneratorFunction.prototype%` — an ordinary object inheriting
    /// `%Function.prototype%`, whose own `prototype` data property is
    /// `%GeneratorPrototype%` (so `Object.getPrototypeOf(g).prototype` resolves to
    /// it), with `[Symbol.toStringTag]` "GeneratorFunction". A sync generator
    /// function's `[[Prototype]]` is set to this (via `set_native_proto`), which
    /// `object_proto` honors ahead of the `%Function.prototype%` fallback. Cached on
    /// the `Iterator` constructor.
    pub(crate) fn generator_function_prototype(&mut self) -> Option<Handle> {
        let iter_ctor = self
            .current
            .get("Iterator")
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)?;
        const CACHE: &str = "\u{0}genfnproto";
        if let Some(gfp) = self
            .realm
            .get_property(iter_ctor, CACHE)
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
        {
            return Some(gfp);
        }
        let fn_proto = self
            .current
            .get("Function")
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .and_then(|c| self.realm.get_property(c, "prototype"))
            .and_then(|p| p.as_handle())
            .map(Handle::from_raw);
        let gfp = self.realm.new_object_with_proto(fn_proto);
        if let Some(gp) = self.generator_prototype() {
            self.realm
                .set_property(gfp, "prototype", NanBox::handle(gp.to_raw()));
            self.realm.mark_hidden(gfp, "prototype");
            self.realm.set_readonly_property(gfp, "prototype");
            // `%GeneratorPrototype%.constructor` is `%GeneratorFunction.prototype%`
            // (this `gfp`): { writable:false, enumerable:false, configurable:true }.
            self.realm
                .set_property(gp, "constructor", NanBox::handle(gfp.to_raw()));
            self.realm.mark_hidden(gp, "constructor");
            self.realm.set_readonly_property(gp, "constructor");
        }
        self.install_to_string_tag(gfp, "GeneratorFunction");
        // `%GeneratorFunction%` — the constructor, reachable as
        // `Object.getPrototypeOf(function*(){}).constructor`. Its own `[[Prototype]]`
        // is `%Function%`; `prototype` is `%GeneratorFunction.prototype%`
        // { w:false,e:false,c:false }; the prototype's `constructor` points back
        // { w:false,e:false,c:true }.
        let gf = self.realm.new_native(N_GENERATOR_FUNCTION_CTOR);
        // `GetFunctionRealm` tagging: a `%GeneratorFunction%` built lazily while
        // running inside a `$262.createRealm()` realm belongs to *that* realm — so a
        // cross-realm `Reflect.construct(otherRealm.GeneratorFunction, …)` enters the
        // constructor's realm, giving the created function's `.prototype` object and
        // body that realm's `%GeneratorPrototype%` / globals (CreateDynamicFunction
        // step 19 `realmF`). Untagged (main realm) leaves the fast path untouched.
        if let Some(idx) = self.cur_realm {
            self.fn_realm.insert(gf.to_raw(), idx);
        }
        self.install_fn_name_length(gf, "GeneratorFunction", 1);
        self.realm
            .set_property(gf, "prototype", NanBox::handle(gfp.to_raw()));
        self.realm.mark_hidden(gf, "prototype");
        self.realm.set_readonly_property(gf, "prototype");
        self.realm.set_non_configurable_property(gf, "prototype");
        if let Some(fn_ctor) = self
            .current
            .get("Function")
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
        {
            self.realm.set_native_proto(gf, fn_ctor);
        }
        self.realm
            .set_property(gfp, "constructor", NanBox::handle(gf.to_raw()));
        self.realm.mark_hidden(gfp, "constructor");
        self.realm.set_readonly_property(gfp, "constructor");
        self.realm
            .set_hidden_property(iter_ctor, CACHE, NanBox::handle(gfp.to_raw()));
        Some(gfp)
    }

    /// `%AsyncFunction.prototype%` — an ordinary object inheriting
    /// `%Function.prototype%` with `[Symbol.toStringTag]` "AsyncFunction"
    /// ({ w:false, e:false, c:true }). A (non-generator) `async function`'s
    /// `[[Prototype]]` is set to this via `set_native_proto`, so
    /// `Object.prototype.toString.call(asyncFn)` yields "[object AsyncFunction]"
    /// (the tag is read through the prototype chain — including a proxy wrapper).
    /// It has no own `prototype` (async functions are not constructable);
    /// `.constructor` intentionally still resolves up to `%Function%` (the
    /// `AsyncFunction === Function` conflation), so only the tag is added. Cached
    /// on the `Iterator` constructor.
    pub(crate) fn async_function_prototype(&mut self) -> Option<Handle> {
        let iter_ctor = self
            .current
            .get("Iterator")
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)?;
        const CACHE: &str = "\u{0}asyncfnproto";
        if let Some(h) = self
            .realm
            .get_property(iter_ctor, CACHE)
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
        {
            return Some(h);
        }
        let fn_proto = self
            .current
            .get("Function")
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .and_then(|c| self.realm.get_property(c, "prototype"))
            .and_then(|p| p.as_handle())
            .map(Handle::from_raw);
        let afp = self.realm.new_object_with_proto(fn_proto);
        self.install_to_string_tag(afp, "AsyncFunction");
        // `%AsyncFunction%` — the constructor, reachable as
        // `Object.getPrototypeOf(async function(){}).constructor`. Its own
        // `[[Prototype]]` is `%Function%`; `prototype` is `%AsyncFunction.prototype%`
        // { w:false, e:false, c:false }; the prototype's `constructor` points back
        // { w:false, e:false, c:true }. Distinct from `%Function%` so
        // `asyncFn.constructor.prototype[@@toStringTag]` targets THIS prototype
        // (the `Object.prototype.toString` tag), not `%Function.prototype%`.
        let af = self.realm.new_native(N_ASYNC_FUNCTION_CTOR);
        // `GetFunctionRealm` tagging (see `%GeneratorFunction%`): a lazily-built
        // `%AsyncFunction%` belongs to the realm it was built in.
        if let Some(idx) = self.cur_realm {
            self.fn_realm.insert(af.to_raw(), idx);
        }
        self.install_fn_name_length(af, "AsyncFunction", 1);
        self.realm
            .set_property(af, "prototype", NanBox::handle(afp.to_raw()));
        self.realm.mark_hidden(af, "prototype");
        self.realm.set_readonly_property(af, "prototype");
        self.realm.set_non_configurable_property(af, "prototype");
        if let Some(fn_ctor) = self
            .current
            .get("Function")
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
        {
            self.realm.set_native_proto(af, fn_ctor);
        }
        self.realm
            .set_property(afp, "constructor", NanBox::handle(af.to_raw()));
        self.realm.mark_hidden(afp, "constructor");
        self.realm.set_readonly_property(afp, "constructor");
        self.realm
            .set_hidden_property(iter_ctor, CACHE, NanBox::handle(afp.to_raw()));
        Some(afp)
    }

    /// `%AsyncIteratorPrototype%` — `[Symbol.asyncIterator]` returns `this`,
    /// inheriting `%Object.prototype%`. Cached on the `Iterator` constructor.
    fn async_iterator_prototype(&mut self) -> Option<Handle> {
        let iter_ctor = self
            .current
            .get("Iterator")
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)?;
        const CACHE: &str = "\u{0}asynciterproto";
        if let Some(h) = self
            .realm
            .get_property(iter_ctor, CACHE)
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
        {
            return Some(h);
        }
        let obj_proto = self
            .current
            .get("Object")
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .and_then(|c| self.realm.get_property(c, "prototype"))
            .and_then(|p| p.as_handle())
            .map(Handle::from_raw);
        let aip = self.realm.new_object_with_proto(obj_proto);
        let self_iter = self.realm.new_native(N_ITERATOR_PROTO_SELF);
        self.install_fn_name_length(self_iter, "[Symbol.asyncIterator]", 0);
        let sym = self.well_known_symbol("asyncIterator");
        let key = self.member_key(sym);
        self.realm
            .set_hidden_property(aip, &key, NanBox::handle(self_iter.to_raw()));
        // `%AsyncIteratorPrototype%[@@asyncDispose]` (length 0).
        let dispose = self.realm.new_native(N_ASYNC_ITERATOR_DISPOSE);
        self.install_fn_name_length(dispose, "[Symbol.asyncDispose]", 0);
        let dsym = self.well_known_symbol("asyncDispose");
        let dkey = self.member_key(dsym);
        self.realm
            .set_property(aip, &dkey, NanBox::handle(dispose.to_raw()));
        self.realm.mark_hidden(aip, &dkey);
        self.realm
            .set_hidden_property(iter_ctor, CACHE, NanBox::handle(aip.to_raw()));
        Some(aip)
    }

    /// `%AsyncGeneratorPrototype%` — `next`/`return`/`throw` (length 1, each
    /// dispatching on `this`'s frame and wrapping the result in a promise) and
    /// `[Symbol.toStringTag]` "AsyncGenerator", inheriting `%AsyncIteratorPrototype%`.
    pub(crate) fn async_generator_prototype(&mut self) -> Option<Handle> {
        let iter_ctor = self
            .current
            .get("Iterator")
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)?;
        const CACHE: &str = "\u{0}asyncgenproto";
        if let Some(h) = self
            .realm
            .get_property(iter_ctor, CACHE)
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
        {
            return Some(h);
        }
        let aip = self.async_iterator_prototype();
        let agp = self.realm.new_object_with_proto(aip);
        for (name, nid) in [
            ("next", N_ASYNC_GEN_NEXT),
            ("return", N_ASYNC_GEN_RETURN),
            ("throw", N_ASYNC_GEN_THROW),
        ] {
            let f = self.realm.new_native(nid);
            self.install_fn_name_length(f, name, 1);
            self.realm
                .set_property(agp, name, NanBox::handle(f.to_raw()));
            self.realm.mark_hidden(agp, name);
        }
        self.install_to_string_tag(agp, "AsyncGenerator");
        self.realm
            .set_hidden_property(iter_ctor, CACHE, NanBox::handle(agp.to_raw()));
        Some(agp)
    }

    /// `%AsyncGeneratorFunction.prototype%` — own `prototype` =
    /// `%AsyncGeneratorPrototype%`, `[Symbol.toStringTag]` "AsyncGeneratorFunction",
    /// inheriting `%Function.prototype%`. An `async function*`'s `[[Prototype]]` is
    /// set to this via `set_native_proto`.
    pub(crate) fn async_generator_function_prototype(&mut self) -> Option<Handle> {
        let iter_ctor = self
            .current
            .get("Iterator")
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)?;
        const CACHE: &str = "\u{0}asyncgenfnproto";
        if let Some(h) = self
            .realm
            .get_property(iter_ctor, CACHE)
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
        {
            return Some(h);
        }
        let fn_proto = self
            .current
            .get("Function")
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .and_then(|c| self.realm.get_property(c, "prototype"))
            .and_then(|p| p.as_handle())
            .map(Handle::from_raw);
        let agfp = self.realm.new_object_with_proto(fn_proto);
        if let Some(agp) = self.async_generator_prototype() {
            self.realm
                .set_property(agfp, "prototype", NanBox::handle(agp.to_raw()));
            self.realm.mark_hidden(agfp, "prototype");
            self.realm.set_readonly_property(agfp, "prototype");
            // `%AsyncGeneratorPrototype%.constructor` is
            // `%AsyncGeneratorFunction.prototype%` (this `agfp`):
            // { writable:false, enumerable:false, configurable:true }.
            self.realm
                .set_property(agp, "constructor", NanBox::handle(agfp.to_raw()));
            self.realm.mark_hidden(agp, "constructor");
            self.realm.set_readonly_property(agp, "constructor");
        }
        self.install_to_string_tag(agfp, "AsyncGeneratorFunction");
        // `%AsyncGeneratorFunction%` — the constructor, reachable as
        // `Object.getPrototypeOf(async function*(){}).constructor`.
        let agf = self.realm.new_native(N_ASYNC_GENERATOR_FUNCTION_CTOR);
        // `GetFunctionRealm` tagging (see `%GeneratorFunction%`): a lazily-built
        // `%AsyncGeneratorFunction%` belongs to the realm it was built in.
        if let Some(idx) = self.cur_realm {
            self.fn_realm.insert(agf.to_raw(), idx);
        }
        self.install_fn_name_length(agf, "AsyncGeneratorFunction", 1);
        self.realm
            .set_property(agf, "prototype", NanBox::handle(agfp.to_raw()));
        self.realm.mark_hidden(agf, "prototype");
        self.realm.set_readonly_property(agf, "prototype");
        self.realm.set_non_configurable_property(agf, "prototype");
        if let Some(fn_ctor) = self
            .current
            .get("Function")
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
        {
            self.realm.set_native_proto(agf, fn_ctor);
        }
        self.realm
            .set_property(agfp, "constructor", NanBox::handle(agf.to_raw()));
        self.realm.mark_hidden(agfp, "constructor");
        self.realm.set_readonly_property(agfp, "constructor");
        self.realm
            .set_hidden_property(iter_ctor, CACHE, NanBox::handle(agfp.to_raw()));
        Some(agfp)
    }

    pub(crate) fn gen_result(&mut self, value: NanBox, done: bool) -> NanBox {
        let r = self.realm.new_object();
        self.realm.set_property(r, "value", value);
        self.realm.set_property(r, "done", NanBox::boolean(done));
        NanBox::handle(r.to_raw())
    }

    /// `AsyncFromSyncIteratorContinuation(result, promiseCapability,
    /// syncIteratorRecord, true)` (27.1.4.4) for the `result` object a *sync*
    /// iterator just produced: reads `done`/`value`, wraps the value with
    /// `PromiseResolve(%Promise%, value)`, and returns the promise that settles
    /// with `CreateIterResultObject(unwrapped, done)`.
    ///
    /// The wrapper promise is a real link in the chain — awaiting the returned
    /// promise therefore costs the two microtask turns the spec prescribes, not
    /// one. An abrupt `PromiseResolve` (step 6) closes the sync iterator when the
    /// result was not `done`, then propagates.
    fn async_from_sync_continuation(
        &mut self,
        iter: Handle,
        result: Result<NanBox, ExecError>,
    ) -> Result<Handle, ExecError> {
        self.async_from_sync_continuation_with(iter, result, true)
    }

    /// [`Self::async_from_sync_continuation`] with an explicit
    /// `closeOnRejection` (false for `return`).
    fn async_from_sync_continuation_with(
        &mut self,
        iter: Handle,
        result: Result<NanBox, ExecError>,
        close_on_rejection: bool,
    ) -> Result<Handle, ExecError> {
        let result = result?;
        let Some(rh) = self.as_object_handle(result) else {
            return Err(self.type_error("iterator result is not an object"));
        };
        let done = self.read_member(rh, "done")?;
        let done = self.realm.truthy(done);
        let value = self.read_member(rh, "value")?;
        let w = match self.promise_resolve_checked(value) {
            Ok(w) => w,
            Err(e) => {
                // `closeOnRejection` is true here (this is `next`, not `return`):
                // a non-done result closes the sync iterator before rejecting.
                if !done && close_on_rejection {
                    let _ = self.iterator_close(iter);
                }
                return Err(e);
            }
        };
        let state = self.realm.new_array(alloc::vec![
            NanBox::boolean(done),
            NanBox::handle(iter.to_raw())
        ]);
        let on_f = self.realm.new_bound_native(N_ASYNC_FROM_SYNC_UNWRAP, state);
        // Step 10: `onRejected` exists only for a non-done result; for a done one
        // the rejection simply passes through to the capability.
        let on_r = if done || !close_on_rejection {
            NanBox::undefined()
        } else {
            let f = self.realm.new_bound_native(N_ASYNC_FROM_SYNC_CLOSE, state);
            NanBox::handle(f.to_raw())
        };
        Ok(self.register_then(w, NanBox::handle(on_f.to_raw()), on_r, false))
    }

    /// `%AsyncFromSyncIteratorPrototype%.next` over a sync iterator: pull one
    /// `next()` and run [`Self::async_from_sync_continuation`] on it, **always**
    /// returning a promise. Every abrupt completion inside is an
    /// `IfAbruptRejectPromise` — it rejects the returned promise rather than
    /// throwing at the call site, so the caller's `Await` still costs its tick and
    /// the error surfaces one turn later (observable, and what `for await` over a
    /// poisoned sync iterator relies on).
    fn async_from_sync_next(&mut self, iter: Handle, next: NanBox) -> Result<Handle, ExecError> {
        let iter_val = NanBox::handle(iter.to_raw());
        let result = self.call_with_this(next, iter_val, &[]);
        match self.async_from_sync_continuation(iter, result) {
            Ok(p) => Ok(p),
            Err(ExecError::Throw(e)) => {
                let p = self.fresh_promise();
                self.settle(p, e, false);
                Ok(p)
            }
            // A non-throw fatal (stack overflow, resource limit) is not a JS
            // completion and must keep unwinding.
            Err(other) => Err(other),
        }
    }

    /// [`Self::async_from_sync_next`] passing `args` to the sync `next`.
    pub(crate) fn async_from_sync_next_args(
        &mut self,
        iter: Handle,
        next: NanBox,
        args: &[NanBox],
    ) -> Result<Handle, ExecError> {
        let iter_val = NanBox::handle(iter.to_raw());
        let result = self.call_with_this(next, iter_val, args);
        match self.async_from_sync_continuation(iter, result) {
            Ok(p) => Ok(p),
            Err(ExecError::Throw(e)) => {
                let p = self.fresh_promise();
                self.settle(p, e, false);
                Ok(p)
            }
            Err(other) => Err(other),
        }
    }

    /// `%AsyncFromSyncIteratorPrototype%.throw` (`is_throw`) / `.return`
    /// (27.1.4.2.2–3) over the sync iterator `iter` for the bytecode VM's async
    /// `yield*`: always a promise; abrupt completions reject it.
    pub(crate) fn async_from_sync_resume(
        &mut self,
        iter: Handle,
        is_throw: bool,
        value: NanBox,
    ) -> Result<Handle, ExecError> {
        let r = self.async_from_sync_resume_inner(iter, is_throw, value);
        match r {
            Ok(p) => Ok(p),
            Err(ExecError::Throw(e)) => {
                let p = self.fresh_promise();
                self.settle(p, e, false);
                Ok(p)
            }
            Err(other) => Err(other),
        }
    }

    fn async_from_sync_resume_inner(
        &mut self,
        iter: Handle,
        is_throw: bool,
        value: NanBox,
    ) -> Result<Handle, ExecError> {
        let iter_val = NanBox::handle(iter.to_raw());
        let method = self.read_member(iter, if is_throw { "throw" } else { "return" })?;
        if method.is_undefined() || method.is_null() {
            if is_throw {
                // No `throw`: close the sync iterator (a normal completion), then
                // reject with a TypeError.
                self.iterator_close(iter)?;
                return Err(self.type_error("The iterator does not provide a 'throw' method"));
            }
            let res = self.gen_result(value, true);
            let p = self.fresh_promise();
            self.settle(p, res, true);
            return Ok(p);
        }
        let result = self.call_with_this(method, iter_val, &[value])?;
        if self.as_object_handle(result).is_none() {
            return Err(self.type_error("iterator result is not an object"));
        }
        self.async_from_sync_continuation_with(iter, Ok(result), is_throw)
    }

    /// [`Self::async_from_sync_next`] for the bytecode VM's `for await`.
    pub(crate) fn async_from_sync_next_pub(
        &mut self,
        iter: Handle,
        next: NanBox,
    ) -> Result<Handle, ExecError> {
        self.async_from_sync_next(iter, next)
    }

    /// `%GeneratorPrototype%.next/return/throw` (`mode` 0 / 2 / 1): resumes the
    /// generator's suspended VM frame.
    pub(crate) fn vm_gen_resume(
        &mut self,
        this: NanBox,
        mode: u8,
        value: NanBox,
    ) -> Result<NanBox, ExecError> {
        let Some(h) = this.as_handle().map(Handle::from_raw) else {
            return Err(self.type_error("Generator method called on non-object"));
        };
        match self.vm_table.clone() {
            Some(table) if self.realm.get_property(h, crate::nbvm::VM_GEN).is_some() => {
                crate::nbvm::resume_vm_generator(self, &table, this, mode, value)
                    .map_err(super::vm_to_exec)
            }
            _ => Err(self.type_error("Generator method called on a non-generator")),
        }
    }

    /// `%AsyncGeneratorPrototype%.next/throw/return` (`kind` 0 / 1 / 2):
    /// AsyncGeneratorEnqueue on the generator's VM request queue. Always a
    /// promise: a `this` that is not an async generator rejects it.
    pub(crate) fn vm_agen_request(
        &mut self,
        this: NanBox,
        kind: u8,
        value: NanBox,
    ) -> Result<NanBox, ExecError> {
        let Some(h) = this.as_handle().map(Handle::from_raw) else {
            return Ok(self.rejected_type_error("Generator method called on non-object"));
        };
        match self.vm_table.clone() {
            Some(table) if self.realm.get_property(h, crate::nbvm::VM_AGEN).is_some() => {
                crate::nbvm::vm_agen_request(self, &table, this, kind, value)
                    .map_err(super::vm_to_exec)
            }
            _ => Ok(self.rejected_type_error("Generator method called on a non-async-generator")),
        }
    }

    /// A fresh promise already rejected with a `TypeError` carrying `msg`.
    fn rejected_type_error(&mut self, msg: &str) -> NanBox {
        let p = self.fresh_promise();
        let m = self.new_str(msg);
        let e = self.make_error(N_TYPE_ERROR, Some(m));
        self.settle(p, e, false);
        NanBox::handle(p.to_raw())
    }
}
