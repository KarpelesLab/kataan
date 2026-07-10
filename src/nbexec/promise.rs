use super::*;

/// A `PromiseCapability` record: the dependent promise plus its resolve/reject
/// functions, as produced by `NewPromiseCapability(C)`.
#[derive(Clone, Copy)]
pub(crate) struct PromiseCapability {
    pub promise: NanBox,
    pub resolve: NanBox,
    pub reject: NanBox,
}

impl<'a> Interp<'a> {
    /// Builds an iterator object over a generator's eagerly-collected `values`:
    /// a hidden buffer array plus a `next()` cursor, recognized by `for-of`,
    /// spread, and a `next()` method.
    pub(crate) fn make_generator(&mut self, values: Vec<NanBox>) -> NanBox {
        self.make_generator_with_return(values, NanBox::undefined())
    }

    /// Like [`make_generator`], but with the generator's `return` value (surfaced
    /// once, with `done: true`, after the yields are exhausted).
    pub(crate) fn make_generator_with_return(
        &mut self,
        values: Vec<NanBox>,
        ret: NanBox,
    ) -> NanBox {
        let obj = self.realm.new_object();
        let buf = self.realm.new_array(values);
        self.realm
            .set_hidden_property(obj, GEN_BUF, NanBox::handle(buf.to_raw()));
        self.realm
            .set_hidden_property(obj, GEN_IDX, NanBox::number(0.0));
        self.realm.set_hidden_property(obj, GEN_RET, ret);
        // Expose `next`/`return` as real (non-enumerable) methods so the iterator
        // is a valid GetIteratorDirect target — readable once and callable by the
        // lazy ES2025 iterator helpers — and `typeof gen.next === "function"`.
        let next = self.realm.new_native(N_GEN_ITER_NEXT);
        self.realm
            .set_property(obj, "next", NanBox::handle(next.to_raw()));
        self.realm.mark_hidden(obj, "next");
        let ret_fn = self.realm.new_native(N_GEN_ITER_RETURN);
        self.realm
            .set_property(obj, "return", NanBox::handle(ret_fn.to_raw()));
        self.realm.mark_hidden(obj, "return");
        // Link the iterator's `[[Prototype]]` to `%IteratorPrototype%` so it is an
        // `instanceof Iterator`, inherits the helper methods and `Symbol.dispose`,
        // and `Object.getPrototypeOf(arrayIterator)`'s chain reaches it (reflection
        // tests). Falls back to the default object proto before Iterator is set up.
        if let Some(iter_proto) = self
            .current
            .get("Iterator")
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .and_then(|c| self.realm.get_property(c, "prototype"))
            .and_then(|p| p.as_handle())
            .map(Handle::from_raw)
        {
            self.realm.set_object_proto(obj, Some(iter_proto));
        }
        NanBox::handle(obj.to_raw())
    }

    /// The lazily-created built-in iterator prototype for `@@toStringTag` `tag`
    /// (`"Array Iterator"`, `"String Iterator"`, `"Map Iterator"`,
    /// `"Set Iterator"`, `"RegExp String Iterator"`). It chains to
    /// `%IteratorPrototype%` and carries an inherited `next` (the shared eager
    /// iterator advance) plus the tag — so a built-in iterator's prototype is the
    /// real `%XIteratorPrototype%` with `next` *on the prototype*, not the
    /// instance (per spec / reflection tests).
    pub(crate) fn builtin_iterator_proto(&mut self, tag: &'static str) -> Handle {
        if let Some(&p) = self.builtin_iter_protos.get(tag) {
            return p;
        }
        let proto = self.realm.new_object();
        // Chain to `%IteratorPrototype%` (so the helper methods, `@@iterator`,
        // and `@@dispose` are inherited).
        if let Some(iter_proto) = self
            .current
            .get("Iterator")
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .and_then(|c| self.realm.get_property(c, "prototype"))
            .and_then(|p| p.as_handle())
            .map(Handle::from_raw)
        {
            self.realm.set_object_proto(proto, Some(iter_proto));
        }
        // `next` lives on the prototype (length 0, non-enumerable, writable,
        // configurable) and reads the receiver's buffer/index slots.
        let next = self.realm.new_native(N_GEN_ITER_NEXT);
        self.install_fn_name_length(next, "next", 0);
        self.realm
            .set_property(proto, "next", NanBox::handle(next.to_raw()));
        self.realm.mark_hidden(proto, "next");
        // `Class.prototype[Symbol.toStringTag] = tag` (non-writable, non-enumerable,
        // configurable) — `Object.prototype.toString` and the tests read it.
        self.install_to_string_tag(proto, tag);
        self.builtin_iter_protos.insert(tag, proto);
        proto
    }

    /// A **live** Set/Map iterator over the collection at `coll` (`kind`: 0 keys,
    /// 1 values, 2 entries). Unlike [`make_builtin_iterator`], it holds no
    /// snapshot — each `next()` re-reads the collection, so entries added or
    /// removed during iteration are observed (per spec `%MapIteratorPrototype%`).
    pub(crate) fn make_live_collection_iterator(
        &mut self,
        coll: Handle,
        kind: u8,
        tag: &'static str,
    ) -> NanBox {
        let proto = self.builtin_iterator_proto(tag);
        let obj = self.realm.new_object();
        self.realm.set_object_proto(obj, Some(proto));
        self.realm
            .set_hidden_property(obj, GEN_COLL, NanBox::handle(coll.to_raw()));
        self.realm
            .set_hidden_property(obj, GEN_KIND, NanBox::number(f64::from(kind)));
        NanBox::handle(obj.to_raw())
    }

    /// A **live** typed-array iterator over `ta` (`kind`: 0 keys, 1 values, 2
    /// entries). Each `next()` re-reads the live length and elements, so a
    /// resizable-buffer grow/shrink or an element write mid-iteration is observed.
    pub(crate) fn make_live_typed_iterator(&mut self, ta: Handle, kind: u8) -> NanBox {
        let proto = self.builtin_iterator_proto("Array Iterator");
        let obj = self.realm.new_object();
        self.realm.set_object_proto(obj, Some(proto));
        self.realm
            .set_hidden_property(obj, GEN_TA, NanBox::handle(ta.to_raw()));
        self.realm
            .set_hidden_property(obj, GEN_KIND, NanBox::number(f64::from(kind)));
        self.realm
            .set_hidden_property(obj, GEN_IDX, NanBox::number(0.0));
        NanBox::handle(obj.to_raw())
    }

    /// A **live** plain-array iterator over `arr` (`kind`: 0 keys, 1 values, 2
    /// entries). Each `next()` re-reads the array's current `length` and `Get`s the
    /// element at the cursor, so elements appended/assigned after the iterator was
    /// created are observed (per spec `CreateArrayIterator`).
    pub(crate) fn make_live_array_iterator(&mut self, arr: Handle, kind: u8) -> NanBox {
        let proto = self.builtin_iterator_proto("Array Iterator");
        let obj = self.realm.new_object();
        self.realm.set_object_proto(obj, Some(proto));
        self.realm
            .set_hidden_property(obj, GEN_ARR, NanBox::handle(arr.to_raw()));
        self.realm
            .set_hidden_property(obj, GEN_KIND, NanBox::number(f64::from(kind)));
        self.realm
            .set_hidden_property(obj, GEN_IDX, NanBox::number(0.0));
        NanBox::handle(obj.to_raw())
    }

    /// Builds a built-in iterator over `values` (a snapshot) whose `[[Prototype]]`
    /// is the `%XIteratorPrototype%` for `tag` (so `next` is inherited, not an own
    /// property). The receiver carries the same buffer/index slots a generator
    /// uses, so the shared `next` (`gen_iter_next`) advances it.
    pub(crate) fn make_builtin_iterator(
        &mut self,
        values: Vec<NanBox>,
        tag: &'static str,
    ) -> NanBox {
        let proto = self.builtin_iterator_proto(tag);
        let obj = self.realm.new_object();
        self.realm.set_object_proto(obj, Some(proto));
        let buf = self.realm.new_array(values);
        self.realm
            .set_hidden_property(obj, GEN_BUF, NanBox::handle(buf.to_raw()));
        self.realm
            .set_hidden_property(obj, GEN_IDX, NanBox::number(0.0));
        self.realm
            .set_hidden_property(obj, GEN_RET, NanBox::undefined());
        NanBox::handle(obj.to_raw())
    }

    // --- promises ---

    /// Settles the promise at `handle` (no-op if already settled), queuing its
    /// reactions as microtasks.
    pub(crate) fn settle(&mut self, handle: Handle, value: NanBox, fulfilled: bool) {
        use crate::cell::PromiseStatus::{Fulfilled, Pending, Rejected};
        let Some(state) = self.realm.promise_state(handle) else {
            return;
        };
        let reactions = {
            let mut s = state.borrow_mut();
            if s.status != Pending {
                return;
            }
            s.status = if fulfilled { Fulfilled } else { Rejected };
            s.value = value;
            core::mem::take(&mut s.reactions)
        };
        for r in reactions {
            let handler = if fulfilled {
                r.on_fulfilled
            } else {
                r.on_rejected
            };
            self.microtasks.push(Job {
                handler,
                value,
                result: r.result,
                fulfilled,
                finally: r.finally,
                thenable: None,
            });
        }
    }

    /// Resolves `handle` with `value`, adopting it if `value` is itself a
    /// promise/thenable (chain on its settlement). Mirrors the Promise Resolve
    /// Function (25.6.1.3.2).
    pub(crate) fn resolve_with(&mut self, handle: Handle, value: NanBox) {
        // Self-resolution: resolving a promise with itself is a TypeError
        // rejection ("Chaining cycle detected").
        if value.as_handle() == Some(handle.to_raw()) {
            let m = self.new_str("Chaining cycle detected for promise");
            let e = self.make_error(N_TYPE_ERROR, Some(m));
            self.settle(handle, e, false);
            return;
        }
        // A non-object resolution value fulfills directly.
        let Some(vh) = value.as_handle().map(Handle::from_raw) else {
            self.settle(handle, value, true);
            return;
        };
        // `Let then = Get(resolution, "then")` — through `read_member` so a `then`
        // *accessor* getter runs (and a throwing getter rejects the promise).
        let then = match self.read_member(vh, "then") {
            Ok(t) => t,
            Err(ExecError::Throw(e)) => {
                self.settle(handle, e, false);
                return;
            }
            // A non-throw abrupt completion: settle to avoid leaving it pending.
            Err(_) => {
                self.settle(handle, value, true);
                return;
            }
        };
        // A non-callable `then` fulfills with the value as-is.
        if !self.is_callable_value(then) {
            self.settle(handle, value, true);
            return;
        }
        // A real promise with the intrinsic `then` is adopted directly (cheaper and
        // keeps identity for the common case).
        if self.realm.promise_state(vh).is_some()
            && self
                .current
                .get("Promise")
                .and_then(|p| p.as_handle())
                .map(Handle::from_raw)
                .and_then(|pc| self.realm.get_property(pc, "prototype"))
                .and_then(|pp| pp.as_handle())
                .and_then(|pp| self.realm.get_property(Handle::from_raw(pp), "then"))
                == Some(then)
        {
            let on_f = self.realm.new_bound_native(N_RESOLVE, handle);
            let on_r = self.realm.new_bound_native(N_REJECT, handle);
            self.register_then(
                vh,
                NanBox::handle(on_f.to_raw()),
                NanBox::handle(on_r.to_raw()),
                false,
            );
            return;
        }
        // A thenable (any object with a callable `then`): EnqueuePromiseResolve-
        // ThenableJob — call `then(resolve, reject)` as a microtask so ordering
        // matches the spec (one extra tick before the thenable's `then` runs).
        let on_f = self.realm.new_bound_native(N_RESOLVE, handle);
        let on_r = self.realm.new_bound_native(N_REJECT, handle);
        self.microtasks.push(Job {
            handler: then,
            value,
            result: handle,
            fulfilled: true,
            finally: false,
            thenable: Some((NanBox::handle(on_f.to_raw()), NanBox::handle(on_r.to_raw()))),
        });
    }

    /// `PromiseResolve(%Promise%, value)`: if `value` is already a promise, return
    /// it unchanged (same identity); otherwise wrap it in a fresh promise resolved
    /// with `value` (adopting a thenable). Used by `await` to obtain the promise
    /// whose settlement resumes an async coroutine.
    pub(crate) fn promise_resolve(&mut self, value: NanBox) -> Handle {
        if let Some(raw) = value.as_handle()
            && self.realm.promise_state(Handle::from_raw(raw)).is_some()
        {
            return Handle::from_raw(raw);
        }
        let p = self.fresh_promise();
        self.resolve_with(p, value);
        p
    }

    /// `SpeciesConstructor(O, %Promise%)` — the constructor used by `then`/`finally`
    /// to build their result promise. Reads `O.constructor`; if `undefined`, uses
    /// `%Promise%`; then reads `[Symbol.species]` (using the constructor itself when
    /// that is `undefined`/`null`). A non-constructor result is a `TypeError`.
    pub(crate) fn promise_species_constructor(&mut self, o: Handle) -> Result<NanBox, ExecError> {
        let default_c = self.current.get("Promise").unwrap_or(NanBox::undefined());
        let ctor = self.read_member(o, "constructor")?;
        if matches!(ctor.unpack(), Unpacked::Undefined) {
            return Ok(default_c);
        }
        if !self.is_object_value(ctor) {
            return Err(self.type_error("Promise constructor is not an object"));
        }
        let ch = ctor.as_handle().map(Handle::from_raw).unwrap();
        let species_sym = self.well_known_symbol("species");
        let key = self.member_key(species_sym);
        let species = self.read_member(ch, &key)?;
        if matches!(species.unpack(), Unpacked::Undefined | Unpacked::Null) {
            return Ok(default_c);
        }
        if !self.is_constructor(species) {
            return Err(self.type_error("Promise [Symbol.species] is not a constructor"));
        }
        Ok(species)
    }

    /// Dispatches `Promise.prototype.{then,catch,finally}` on the *raw* receiver
    /// `recv` (a primitive number/boolean/string/symbol or an object), used by the
    /// `N_ARRAY_PROTO_FN` fast-path so these Promise-specific methods are not
    /// intercepted by the generic array-like / primitive-wrapper handlers.
    pub(crate) fn promise_proto_dispatch(
        &mut self,
        name: &str,
        recv: NanBox,
        args: &[NanBox],
    ) -> Result<NanBox, ExecError> {
        let arg = |i: usize| args.get(i).copied().unwrap_or(NanBox::undefined());
        match name {
            // `then` brand-checks `IsPromise(this)`; a non-promise (incl. any
            // primitive) is a `TypeError`.
            "then" => {
                let h = recv
                    .as_handle()
                    .map(Handle::from_raw)
                    .filter(|h| self.realm.promise_state(*h).is_some());
                match h {
                    Some(h) => self.perform_promise_then_method(h, arg(0), arg(1)),
                    None => Err(self.type_error("Promise.prototype.then called on a non-promise")),
                }
            }
            // `catch` is `return Invoke(this, "then", [undefined, onRejected])`.
            // `GetV` does `ToObject(this)` for the property lookup (so a primitive
            // receiver with a `then` on its wrapper-prototype is honored), then
            // `Call(then, this, …)` uses the *original* receiver as `this`.
            "catch" => {
                let obj = self.coerce_to_object(recv);
                let objh = obj.as_handle().map(Handle::from_raw).ok_or_else(|| {
                    self.type_error("Promise.prototype.catch called on null or undefined")
                })?;
                let then = self.read_member(objh, "then")?;
                self.call_with_this(then, recv, &[NanBox::undefined(), arg(0)])
            }
            // `finally` requires an Object receiver (SpeciesConstructor reads
            // `this.constructor`); a primitive is a `TypeError`.
            "finally" => {
                let h = recv.as_handle().map(Handle::from_raw).filter(|_| {
                    // `is_object_value` excludes primitive-string handles etc.
                    self.is_object_value(recv)
                });
                match h {
                    Some(h) => self.promise_finally(h, recv, arg(0)),
                    None => {
                        Err(self.type_error("Promise.prototype.finally called on a non-object"))
                    }
                }
            }
            _ => Ok(NanBox::undefined()),
        }
    }

    /// `Promise.prototype.then(onFulfilled, onRejected)` — the brand-checked,
    /// species-aware spec algorithm. `this` must be a promise (else a `TypeError`);
    /// the result promise is built from `SpeciesConstructor(this, %Promise%)`.
    pub(crate) fn perform_promise_then_method(
        &mut self,
        handle: Handle,
        on_f: NanBox,
        on_r: NanBox,
    ) -> Result<NanBox, ExecError> {
        // Brand check: `this` must have a `[[PromiseState]]` internal slot.
        if self.realm.promise_state(handle).is_none() {
            return Err(self.type_error("Promise.prototype.then called on a non-promise"));
        }
        let c = self.promise_species_constructor(handle)?;
        // Fast path: the species is the intrinsic Promise — use the cheap reaction
        // registration that returns a native promise directly.
        if self.current.get("Promise").and_then(|v| v.as_handle()) == c.as_handle() {
            let result = self.register_then(handle, on_f, on_r, false);
            return Ok(NanBox::handle(result.to_raw()));
        }
        // Foreign species: build its capability, then bridge an internal reaction
        // promise to it — when the internal `then` result settles, drive the
        // foreign capability's resolve/reject (so the user's `C`-built promise is
        // what `then` returns, observing its constructor/executor).
        let cap = self.new_promise_capability(c)?;
        let bridge = self.register_then(handle, on_f, on_r, false);
        self.register_then(bridge, cap.resolve, cap.reject, false);
        Ok(cap.promise)
    }

    pub(crate) fn register_then(
        &mut self,
        handle: Handle,
        on_f: NanBox,
        on_r: NanBox,
        finally: bool,
    ) -> Handle {
        use crate::cell::PromiseStatus::{Fulfilled, Pending};
        let result = self.fresh_promise();
        let state = self.realm.promise_state(handle).expect("a promise");
        let settled = {
            let s = state.borrow();
            match s.status {
                Pending => None,
                status => Some((status == Fulfilled, s.value)),
            }
        };
        match settled {
            None => state.borrow_mut().reactions.push(crate::cell::Reaction {
                on_fulfilled: on_f,
                on_rejected: on_r,
                result,
                finally,
            }),
            Some((fulfilled, value)) => {
                let handler = if fulfilled { on_f } else { on_r };
                self.microtasks.push(Job {
                    handler,
                    value,
                    result,
                    fulfilled,
                    finally,
                    thenable: None,
                });
            }
        }
        result
    }

    /// Drains the microtask queue (the event loop), running each promise
    /// reaction to completion.
    pub(crate) fn drain_microtasks(&mut self) -> Result<(), ExecError> {
        while !self.microtasks.is_empty() {
            self.run_one_microtask()?;
        }
        Ok(())
    }

    /// Runs the earliest-due `setTimeout` macrotask (least `delay`, ties by
    /// insertion order). A no-op when none are pending.
    pub(crate) fn run_one_macrotask(&mut self) -> Result<(), ExecError> {
        let Some(idx) = self
            .macrotasks
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.delay.total_cmp(&b.delay).then(a.seq.cmp(&b.seq)))
            .map(|(i, _)| i)
        else {
            return Ok(());
        };
        let t = self.macrotasks.remove(idx);
        self.call(t.callback, &t.args)?;
        Ok(())
    }

    /// Runs the event loop to quiescence: drain all microtasks, then run the
    /// earliest-due `setTimeout` macrotask (draining microtasks after each), until
    /// both queues are empty.
    pub(crate) fn run_event_loop(&mut self) -> Result<(), ExecError> {
        self.drain_microtasks()?;
        while !self.macrotasks.is_empty() {
            self.run_one_macrotask()?;
            self.drain_microtasks()?;
        }
        Ok(())
    }

    /// Runs the next queued promise reaction.
    pub(crate) fn run_one_microtask(&mut self) -> Result<(), ExecError> {
        let job = self.microtasks.remove(0);
        // PromiseResolveThenableJob: `handler` is the thenable's `then` method;
        // call `then.call(thenable, resolve, reject)`. A throw rejects via the
        // job's `reject` (the bound resolve/reject of `result`).
        if let Some((resolve, reject)) = job.thenable {
            // `PromiseResolveThenableJob` (25.6.2.2) installs a *fresh* pair of
            // resolving functions with their own `[[AlreadyResolved]]` flag. The
            // engine's resolve/reject natives share the promise-level flag (already
            // committed `true` by the outer resolve that scheduled this job), so
            // reset it here to model the fresh pair: a `resolve(...)` *inside* the
            // thenable's `then` commits it, and the step-3a reject after an abrupt
            // completion is then ignored — while a `then` that throws *without*
            // resolving still rejects the promise.
            if let Some(st) = self.realm.promise_state(job.result) {
                st.borrow_mut().already_resolved = false;
            }
            let args = [resolve, reject];
            if let Err(ExecError::Throw(e)) = self.call_with_this(job.handler, job.value, &args) {
                let already = self
                    .realm
                    .promise_state(job.result)
                    .is_some_and(|st| st.borrow().already_resolved);
                if !already {
                    self.settle(job.result, e, false);
                }
            }
            return Ok(());
        }
        if job.finally
            && job
                .handler
                .as_handle()
                .map(Handle::from_raw)
                .is_some_and(|h| self.is_callable(h))
        {
            // `finally`: run the callback (no args), then pass the original
            // value/rejection through (a throw from the callback overrides it).
            match self.call(job.handler, &[]) {
                Ok(_) => {
                    if job.fulfilled {
                        self.resolve_with(job.result, job.value);
                    } else {
                        self.settle(job.result, job.value, false);
                    }
                }
                Err(ExecError::Throw(e)) => self.settle(job.result, e, false),
                Err(other) => return Err(other),
            }
        } else if job
            .handler
            .as_handle()
            .map(Handle::from_raw)
            .is_some_and(|h| self.is_callable(h))
        {
            match self.call(job.handler, &[job.value]) {
                Ok(v) => self.resolve_with(job.result, v),
                Err(ExecError::Throw(e)) => self.settle(job.result, e, false),
                Err(other) => return Err(other),
            }
        } else if job.fulfilled {
            // Passthrough: settle with the same status/value.
            self.resolve_with(job.result, job.value);
        } else {
            self.settle(job.result, job.value, false);
        }
        Ok(())
    }

    /// `await value` — for a promise, drains microtasks until it settles (this
    /// model has no timers, so all promises settle via the queue), then yields
    /// its value or throws its rejection. A non-promise passes through.
    pub(crate) fn await_value(&mut self, value: NanBox) -> Result<NanBox, ExecError> {
        use crate::cell::PromiseStatus::{Fulfilled, Pending, Rejected};
        // `Await(value)` is `PromiseResolve(%Promise%, value)` then drive to
        // settlement. A real promise is awaited directly; a *thenable* (an object
        // with a callable `then`) is adopted — its `then` is called (one extra
        // tick) and we await the result; a primitive passes straight through. Only
        // objects are wrapped, so the common primitive await stays allocation-free.
        let state = match value
            .as_handle()
            .and_then(|raw| self.realm.promise_state(Handle::from_raw(raw)))
        {
            Some(s) => s,
            None if value.as_handle().is_some() => {
                // A potential thenable: PromiseResolve adopts it (reading `.then`
                // exactly once via `resolve_with`); a non-thenable object fulfills
                // synchronously, so this adds no microtask for the ordinary case.
                let p = self.promise_resolve(value);
                self.realm
                    .promise_state(p)
                    .expect("fresh promise has state")
            }
            None => return Ok(value), // a primitive
        };
        // Make progress on the event loop until the promise settles: drain
        // microtasks first, then run a `setTimeout` macrotask if still pending (so an
        // `await` / `Promise.all` on a timer-backed promise observes its value).
        while state.borrow().status == Pending
            && (!self.microtasks.is_empty() || !self.macrotasks.is_empty())
        {
            if self.microtasks.is_empty() {
                self.run_one_macrotask()?;
            } else {
                self.run_one_microtask()?;
            }
        }
        let s = state.borrow();
        match s.status {
            Fulfilled => Ok(s.value),
            Rejected => Err(ExecError::Throw(s.value)),
            Pending => Ok(NanBox::undefined()), // never settles
        }
    }

    /// `NewPromiseCapability(C)` — produces `{ promise, resolve, reject }` where
    /// `promise` is `new C(executor)` and `executor(resolve, reject)` captured the
    /// pair. For the intrinsic `%Promise%` this is the fast path (a fresh promise
    /// plus its bound resolve/reject natives); for any other constructor `C` it
    /// runs the full spec algorithm so a subclass / custom thenable participates.
    ///
    /// `C` must be a constructor (else a `TypeError`); the executor must be called
    /// exactly once with two callable arguments (else a `TypeError`).
    pub(crate) fn new_promise_capability(
        &mut self,
        c: NanBox,
    ) -> Result<PromiseCapability, ExecError> {
        // Fast path: `C` is the realm's own `Promise` constructor.
        if self.current.get("Promise").and_then(|v| v.as_handle()) == c.as_handle() {
            let promise = self.fresh_promise();
            let resolve = self.realm.new_bound_native(N_RESOLVE, promise);
            let reject = self.realm.new_bound_native(N_REJECT, promise);
            // A Promise resolve/reject function has `length: 1`, `name: ""`.
            self.install_fn_name_length(resolve, "", 1);
            self.install_fn_name_length(reject, "", 1);
            return Ok(PromiseCapability {
                promise: NanBox::handle(promise.to_raw()),
                resolve: NanBox::handle(resolve.to_raw()),
                reject: NanBox::handle(reject.to_raw()),
            });
        }
        // `C` must be a constructor.
        if !self.is_constructor(c) {
            return Err(self.type_error("Promise capability requires a constructor"));
        }
        // Build a GetCapabilitiesExecutor bound to a fresh state object, then
        // `new C(executor)`. The executor stores `(resolve, reject)` into the state.
        let state = self.realm.new_object();
        let executor = self
            .realm
            .new_bound_native(N_PROMISE_CAPABILITY_EXECUTOR, state);
        // GetCapabilitiesExecutor is an anonymous function of length 2.
        self.install_fn_name_length(executor, "", 2);
        let promise = self.construct(c, &[NanBox::handle(executor.to_raw())])?;
        let resolve = self
            .realm
            .get_property(state, PCAP_RESOLVE)
            .unwrap_or(NanBox::undefined());
        let reject = self
            .realm
            .get_property(state, PCAP_REJECT)
            .unwrap_or(NanBox::undefined());
        // The executor must have set both to callable values (10.2.x: a capability
        // whose resolve/reject is not callable is a TypeError).
        if !self.is_callable_value(resolve) || !self.is_callable_value(reject) {
            return Err(self.type_error("Promise resolve or reject function is not callable"));
        }
        Ok(PromiseCapability {
            promise,
            resolve,
            reject,
        })
    }

    /// Calls a capability's resolve/reject with `arg`.
    pub(crate) fn capability_resolve(
        &mut self,
        cap: &PromiseCapability,
        arg: NanBox,
    ) -> Result<(), ExecError> {
        self.call(cap.resolve, &[arg])?;
        Ok(())
    }
    pub(crate) fn capability_reject(
        &mut self,
        cap: &PromiseCapability,
        arg: NanBox,
    ) -> Result<(), ExecError> {
        self.call(cap.reject, &[arg])?;
        Ok(())
    }

    /// `Invoke(promise, "then", [onFulfilled, onRejected])` — the spec hook used
    /// by the combinators (so a subclassed/foreign promise's own `then` runs).
    pub(crate) fn invoke_then(
        &mut self,
        promise: NanBox,
        on_f: NanBox,
        on_r: NanBox,
    ) -> Result<(), ExecError> {
        let Some(h) = promise.as_handle().map(Handle::from_raw) else {
            return Err(self.type_error("Promise.then called on a non-object"));
        };
        let then = self.read_member(h, "then")?;
        self.call_with_this(then, promise, &[on_f, on_r])?;
        Ok(())
    }

    // --- Promise combinators (spec-faithful: NewPromiseCapability + per-element
    // resolve closures + Invoke(C,"resolve")/Invoke(p,"then")). ---

    /// Reads `Get(C, "resolve")` and requires it callable (else a `TypeError`).
    /// Shared by every combinator (PerformPromiseAll step "Let promiseResolve …").
    fn combinator_resolve_fn(&mut self, c: NanBox) -> Result<NanBox, ExecError> {
        let Some(ch) = c.as_handle().map(Handle::from_raw) else {
            return Err(self.type_error("Promise combinator called on a non-object"));
        };
        let resolve = self.read_member(ch, "resolve")?;
        if !self.is_callable_value(resolve) {
            return Err(self.type_error("Promise.resolve is not a function"));
        }
        Ok(resolve)
    }

    /// Allocates a per-element state object holding the shared accounting (a
    /// remaining-count cell, the values/result container, the capability) plus this
    /// element's index. `extra` keys (e.g. an errors array) are added by the caller.
    fn combinator_state(
        &mut self,
        remaining: Handle,
        container: NanBox,
        cap: &PromiseCapability,
        index: NanBox,
    ) -> Handle {
        let state = self.realm.new_object();
        self.realm
            .set_hidden_property(state, PCOMB_REMAINING, NanBox::handle(remaining.to_raw()));
        self.realm
            .set_hidden_property(state, PCOMB_VALUES, container);
        self.realm
            .set_hidden_property(state, PCOMB_CAP, cap.promise);
        self.realm
            .set_hidden_property(state, PCOMB_RESOLVE, cap.resolve);
        self.realm
            .set_hidden_property(state, PCOMB_REJECT, cap.reject);
        self.realm.set_hidden_property(state, PCOMB_INDEX, index);
        self.realm
            .set_hidden_property(state, PCOMB_CALLED, NanBox::boolean(false));
        state
    }

    /// Builds a combinator resolve/reject element function (a bound native over the
    /// element `state`), with the spec-mandated `length: 1` and `name: ""` own
    /// properties (a Promise resolve/reject function has length 1).
    fn make_element_fn(&mut self, id: u16, state: Handle) -> NanBox {
        let f = self.realm.new_bound_native(id, state);
        self.install_fn_name_length(f, "", 1);
        NanBox::handle(f.to_raw())
    }

    /// A mutable single-element "remaining count" cell (a one-element array used as
    /// a boxed integer the element closures share and decrement).
    fn count_cell(&mut self, initial: f64) -> Handle {
        self.realm.new_array(alloc::vec![NanBox::number(initial)])
    }
    fn cell_get(&mut self, cell: Handle) -> f64 {
        self.realm.get_element(cell, 0).as_number().unwrap_or(0.0)
    }
    fn cell_set(&mut self, cell: Handle, v: f64) {
        self.realm.set_element(cell, 0, NanBox::number(v));
    }

    /// Reads the shared parts of an element-state object.
    fn state_parts(&mut self, state: Handle) -> (Handle, NanBox, PromiseCapability, usize) {
        let remaining = self
            .realm
            .get_property(state, PCOMB_REMAINING)
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .expect("remaining cell");
        let container = self
            .realm
            .get_property(state, PCOMB_VALUES)
            .unwrap_or(NanBox::undefined());
        let cap = PromiseCapability {
            promise: self
                .realm
                .get_property(state, PCOMB_CAP)
                .unwrap_or(NanBox::undefined()),
            resolve: self
                .realm
                .get_property(state, PCOMB_RESOLVE)
                .unwrap_or(NanBox::undefined()),
            reject: self
                .realm
                .get_property(state, PCOMB_REJECT)
                .unwrap_or(NanBox::undefined()),
        };
        let index = self
            .realm
            .get_property(state, PCOMB_INDEX)
            .and_then(|v| v.as_number())
            .unwrap_or(0.0) as usize;
        (remaining, container, cap, index)
    }

    /// Whether the element closure at `state` has already been called (each
    /// resolve/reject element function runs its body at most once).
    fn element_already_called(&mut self, state: Handle) -> bool {
        if self
            .realm
            .get_property(state, PCOMB_CALLED)
            .is_some_and(|v| self.realm.truthy(v))
        {
            return true;
        }
        self.realm
            .set_hidden_property(state, PCOMB_CALLED, NanBox::boolean(true));
        false
    }

    /// `Promise.all` Resolve Element.
    pub(crate) fn promise_all_element(
        &mut self,
        state: Handle,
        value: NanBox,
    ) -> Result<NanBox, ExecError> {
        if self.element_already_called(state) {
            return Ok(NanBox::undefined());
        }
        let (remaining, container, cap, index) = self.state_parts(state);
        if let Some(arr) = container.as_handle().map(Handle::from_raw) {
            self.realm.set_element(arr, index, value);
        }
        let n = self.cell_get(remaining) - 1.0;
        self.cell_set(remaining, n);
        if n == 0.0 {
            self.capability_resolve(&cap, container)?;
        }
        Ok(NanBox::undefined())
    }

    /// `Promise.allSettled` Resolve/Reject Element (`fulfilled` selects which).
    pub(crate) fn promise_allsettled_element(
        &mut self,
        state: Handle,
        value: NanBox,
        fulfilled: bool,
    ) -> Result<NanBox, ExecError> {
        if self.element_already_called(state) {
            return Ok(NanBox::undefined());
        }
        let (remaining, container, cap, index) = self.state_parts(state);
        let obj = self.realm.new_object();
        if fulfilled {
            let s = self.new_str("fulfilled");
            self.realm.set_property(obj, "status", s);
            self.realm.set_property(obj, "value", value);
        } else {
            let s = self.new_str("rejected");
            self.realm.set_property(obj, "status", s);
            self.realm.set_property(obj, "reason", value);
        }
        if let Some(arr) = container.as_handle().map(Handle::from_raw) {
            self.realm
                .set_element(arr, index, NanBox::handle(obj.to_raw()));
        }
        let n = self.cell_get(remaining) - 1.0;
        self.cell_set(remaining, n);
        if n == 0.0 {
            self.capability_resolve(&cap, container)?;
        }
        Ok(NanBox::undefined())
    }

    /// `Promise.any` Reject Element.
    pub(crate) fn promise_any_element(
        &mut self,
        state: Handle,
        reason: NanBox,
    ) -> Result<NanBox, ExecError> {
        if self.element_already_called(state) {
            return Ok(NanBox::undefined());
        }
        let (remaining, container, cap, index) = self.state_parts(state);
        // `container` is the errors array for `any`.
        if let Some(arr) = container.as_handle().map(Handle::from_raw) {
            self.realm.set_element(arr, index, reason);
        }
        let n = self.cell_get(remaining) - 1.0;
        self.cell_set(remaining, n);
        if n == 0.0 {
            let agg = self.make_aggregate_error(container);
            self.capability_reject(&cap, agg)?;
        }
        Ok(NanBox::undefined())
    }

    /// `Promise.allKeyed` Resolve Element: stores `value` at the captured key of
    /// the result object (pre-created, so order is preserved).
    pub(crate) fn promise_all_keyed_element(
        &mut self,
        state: Handle,
        value: NanBox,
    ) -> Result<NanBox, ExecError> {
        if self.element_already_called(state) {
            return Ok(NanBox::undefined());
        }
        let (remaining, container, cap, _) = self.state_parts(state);
        let key = self
            .realm
            .get_property(state, PCOMB_INDEX)
            .unwrap_or(NanBox::undefined());
        if let Some(obj) = container.as_handle().map(Handle::from_raw) {
            let name = self.member_key(key);
            self.set_or_throw(obj, key, &name, value)?;
        }
        let n = self.cell_get(remaining) - 1.0;
        self.cell_set(remaining, n);
        if n == 0.0 {
            self.capability_resolve(&cap, container)?;
        }
        Ok(NanBox::undefined())
    }

    /// `Promise.allSettledKeyed` Fulfill / Reject Element: stores a
    /// `{status, value|reason}` record at the captured key.
    pub(crate) fn promise_allsettled_keyed_element(
        &mut self,
        state: Handle,
        value: NanBox,
        fulfilled: bool,
    ) -> Result<NanBox, ExecError> {
        if self.element_already_called(state) {
            return Ok(NanBox::undefined());
        }
        let (remaining, container, cap, _) = self.state_parts(state);
        let key = self
            .realm
            .get_property(state, PCOMB_INDEX)
            .unwrap_or(NanBox::undefined());
        let record = self.realm.new_object();
        if fulfilled {
            let s = self.new_str("fulfilled");
            self.realm.set_property(record, "status", s);
            self.realm.set_property(record, "value", value);
        } else {
            let s = self.new_str("rejected");
            self.realm.set_property(record, "status", s);
            self.realm.set_property(record, "reason", value);
        }
        if let Some(obj) = container.as_handle().map(Handle::from_raw) {
            let name = self.member_key(key);
            self.set_or_throw(obj, key, &name, NanBox::handle(record.to_raw()))?;
        }
        let n = self.cell_get(remaining) - 1.0;
        self.cell_set(remaining, n);
        if n == 0.0 {
            self.capability_resolve(&cap, container)?;
        }
        Ok(NanBox::undefined())
    }

    /// Builds an `AggregateError` whose `errors` is `errors_arr`.
    fn make_aggregate_error(&mut self, errors_arr: NanBox) -> NanBox {
        let msg = self.new_str("All promises were rejected");
        // Use the realm's AggregateError constructor when present so the prototype
        // chain and `instanceof AggregateError` are correct.
        if let Some(ctor) = self.current.get("AggregateError") {
            let empty = self.realm.new_array(Vec::new());
            if let Ok(e) = self.construct(ctor, &[NanBox::handle(empty.to_raw()), msg]) {
                if let Some(eh) = e.as_handle().map(Handle::from_raw) {
                    self.realm.set_property(eh, "errors", errors_arr);
                }
                return e;
            }
        }
        let agg = self.realm.new_object();
        let name = self.new_str("AggregateError");
        self.realm.set_property(agg, "name", name);
        let msg = self.new_str("All promises were rejected");
        self.realm.set_property(agg, "message", msg);
        self.realm.set_property(agg, "errors", errors_arr);
        NanBox::handle(agg.to_raw())
    }

    /// `Promise.all(iterable)` (this = `C`).
    pub(crate) fn perform_promise_all(
        &mut self,
        c: NanBox,
        iterable: NanBox,
    ) -> Result<NanBox, ExecError> {
        let cap = self.new_promise_capability(c)?;
        let result = self.perform_promise_all_inner(c, iterable, &cap, false);
        self.finish_combinator(cap, result)
    }

    /// `Promise.allSettled(iterable)` (this = `C`).
    pub(crate) fn perform_promise_all_settled(
        &mut self,
        c: NanBox,
        iterable: NanBox,
    ) -> Result<NanBox, ExecError> {
        let cap = self.new_promise_capability(c)?;
        let result = self.perform_promise_all_inner(c, iterable, &cap, true);
        self.finish_combinator(cap, result)
    }

    /// `Promise.allKeyed(obj)` / `Promise.allSettledKeyed(obj)` (this = `C`) —
    /// the *await-dictionary* proposal. Like `all` / `allSettled`, but the input
    /// is an object whose own *enumerable* keys (String then Symbol, in
    /// `[[OwnPropertyKeys]]` order) name the results: the returned promise
    /// fulfils with a **null-prototype** object carrying the same keys.
    pub(crate) fn perform_promise_all_keyed(
        &mut self,
        c: NanBox,
        obj: NanBox,
        settled: bool,
    ) -> Result<NanBox, ExecError> {
        let cap = self.new_promise_capability(c)?;
        let result = self.perform_promise_all_keyed_inner(c, obj, &cap, settled);
        self.finish_combinator(cap, result)
    }

    fn perform_promise_all_keyed_inner(
        &mut self,
        c: NanBox,
        obj: NanBox,
        cap: &PromiseCapability,
        settled: bool,
    ) -> Result<NanBox, ExecError> {
        let promise_resolve = self.combinator_resolve_fn(c)?;
        if !self.is_object_value(obj) {
            return Err(self.type_error("Promise.allKeyed called on a non-object"));
        }
        let oh = obj.as_handle().map(Handle::from_raw).unwrap();
        // `[[OwnPropertyKeys]]` (String then Symbol order; a proxy `ownKeys` trap
        // is honored), then `[[GetOwnProperty]]` per key — skip a key reported
        // absent or non-enumerable.
        let mut entries: Vec<(NanBox, String)> = Vec::new();
        for key in self.own_property_keys_values(oh)? {
            let name = self.member_key(key);
            let desc = self.descriptor_of(oh, &name)?;
            if matches!(desc.unpack(), Unpacked::Undefined) {
                continue;
            }
            let enumerable = desc
                .as_handle()
                .map(Handle::from_raw)
                .and_then(|dh| self.realm.get_property(dh, "enumerable"))
                .is_some_and(|v| self.realm.truthy(v));
            if enumerable {
                entries.push((key, name));
            }
        }
        // The result is a fresh null-prototype object. Pre-create every key (in
        // input order) so a later out-of-order settlement updates the value in
        // place without disturbing key order.
        let result = self.realm.new_object_with_proto(None);
        let result_box = NanBox::handle(result.to_raw());
        for (key, name) in &entries {
            self.set_or_throw(result, *key, name, NanBox::undefined())?;
        }
        let remaining = self.count_cell((entries.len() + 1) as f64);
        for (key, name) in entries {
            let value = self.read_member(oh, &name)?;
            let next = self.call_with_this(promise_resolve, c, &[value])?;
            let state = self.combinator_state(remaining, result_box, cap, key);
            if settled {
                let on_f = self.make_element_fn(N_PROMISE_ALLSETTLEDKEYED_FULFILL, state);
                let on_r = self.make_element_fn(N_PROMISE_ALLSETTLEDKEYED_REJECT, state);
                self.invoke_then(next, on_f, on_r)?;
            } else {
                let on_f = self.make_element_fn(N_PROMISE_ALLKEYED_ELEMENT, state);
                self.invoke_then(next, on_f, cap.reject)?;
            }
        }
        let n = self.cell_get(remaining) - 1.0;
        self.cell_set(remaining, n);
        if n == 0.0 {
            self.capability_resolve(cap, result_box)?;
        }
        Ok(cap.promise)
    }

    /// Shared `all` / `allSettled` body. On an abrupt completion the caller
    /// rejects the capability (IfAbruptRejectPromise).
    fn perform_promise_all_inner(
        &mut self,
        c: NanBox,
        iterable: NanBox,
        cap: &PromiseCapability,
        settled: bool,
    ) -> Result<NanBox, ExecError> {
        let promise_resolve = self.combinator_resolve_fn(c)?;
        // Iterate LAZILY (spec PerformPromiseAll): the per-element `promiseResolve`
        // call and `then` invocation are interleaved with `IteratorStep`, so an
        // abrupt completion there must `IteratorClose` the still-open iterator.
        let values = self.realm.new_array(alloc::vec![]);
        let values_box = NanBox::handle(values.to_raw());
        let remaining = self.count_cell(1.0);
        let it = self.get_iter_object(iterable)?;
        let next_m = self.read_member(it, "next")?;
        let mut i = 0usize;
        loop {
            let item = match self.iter_step(it, next_m)? {
                Some(v) => v,
                None => break,
            };
            self.realm.set_element(values, i, NanBox::undefined());
            let cur = self.cell_get(remaining);
            self.cell_set(remaining, cur + 1.0);
            let step = self.perform_promise_all_element(
                c,
                promise_resolve,
                item,
                remaining,
                values_box,
                cap,
                i,
                settled,
            );
            if let Err(e) = step {
                let _ = self.iterator_close(it);
                return Err(e);
            }
            i += 1;
        }
        // Decrement the initial +1: if every input already settled synchronously
        // (count back to 0) resolve now.
        let n = self.cell_get(remaining) - 1.0;
        self.cell_set(remaining, n);
        if n == 0.0 {
            self.capability_resolve(cap, values_box)?;
        }
        Ok(cap.promise)
    }

    /// One element of `all`/`allSettled`: `promiseResolve(item)` then subscribe the
    /// element resolve/reject functions. Split out so the iterator can be closed on
    /// an abrupt completion here.
    #[allow(clippy::too_many_arguments)]
    fn perform_promise_all_element(
        &mut self,
        c: NanBox,
        promise_resolve: NanBox,
        item: NanBox,
        remaining: Handle,
        values_box: NanBox,
        cap: &PromiseCapability,
        i: usize,
        settled: bool,
    ) -> Result<(), ExecError> {
        let next = self.call_with_this(promise_resolve, c, &[item])?;
        let state = self.combinator_state(remaining, values_box, cap, NanBox::number(i as f64));
        if settled {
            let on_f = self.make_element_fn(N_PROMISE_ALLSETTLED_FULFILL, state);
            let on_r = self.make_element_fn(N_PROMISE_ALLSETTLED_REJECT, state);
            self.invoke_then(next, on_f, on_r)?;
        } else {
            let on_f = self.make_element_fn(N_PROMISE_ALL_ELEMENT, state);
            self.invoke_then(next, on_f, cap.reject)?;
        }
        Ok(())
    }

    /// `Promise.race(iterable)` (this = `C`).
    pub(crate) fn perform_promise_race(
        &mut self,
        c: NanBox,
        iterable: NanBox,
    ) -> Result<NanBox, ExecError> {
        let cap = self.new_promise_capability(c)?;
        let result = self.perform_promise_race_inner(c, iterable, &cap);
        self.finish_combinator(cap, result)
    }

    fn perform_promise_race_inner(
        &mut self,
        c: NanBox,
        iterable: NanBox,
        cap: &PromiseCapability,
    ) -> Result<NanBox, ExecError> {
        let promise_resolve = self.combinator_resolve_fn(c)?;
        // Lazy iteration + IteratorClose on an abrupt `promiseResolve`/`then`.
        let it = self.get_iter_object(iterable)?;
        let next_m = self.read_member(it, "next")?;
        loop {
            let item = match self.iter_step(it, next_m)? {
                Some(v) => v,
                None => break,
            };
            if let Err(e) = self.perform_promise_race_element(c, promise_resolve, item, cap) {
                let _ = self.iterator_close(it);
                return Err(e);
            }
        }
        Ok(cap.promise)
    }

    /// One element of `race`: `promiseResolve(item)` then forward directly to the
    /// capability's resolve/reject (the first to settle wins).
    fn perform_promise_race_element(
        &mut self,
        c: NanBox,
        promise_resolve: NanBox,
        item: NanBox,
        cap: &PromiseCapability,
    ) -> Result<(), ExecError> {
        let next = self.call_with_this(promise_resolve, c, &[item])?;
        self.invoke_then(next, cap.resolve, cap.reject)?;
        Ok(())
    }

    /// `Promise.any(iterable)` (this = `C`).
    pub(crate) fn perform_promise_any(
        &mut self,
        c: NanBox,
        iterable: NanBox,
    ) -> Result<NanBox, ExecError> {
        let cap = self.new_promise_capability(c)?;
        let result = self.perform_promise_any_inner(c, iterable, &cap);
        self.finish_combinator(cap, result)
    }

    fn perform_promise_any_inner(
        &mut self,
        c: NanBox,
        iterable: NanBox,
        cap: &PromiseCapability,
    ) -> Result<NanBox, ExecError> {
        let promise_resolve = self.combinator_resolve_fn(c)?;
        // Lazy iteration + IteratorClose on an abrupt `promiseResolve`/`then`.
        let errors = self.realm.new_array(alloc::vec![]);
        let errors_box = NanBox::handle(errors.to_raw());
        let remaining = self.count_cell(1.0);
        let it = self.get_iter_object(iterable)?;
        let next_m = self.read_member(it, "next")?;
        let mut i = 0usize;
        loop {
            let item = match self.iter_step(it, next_m)? {
                Some(v) => v,
                None => break,
            };
            self.realm.set_element(errors, i, NanBox::undefined());
            let cur = self.cell_get(remaining);
            self.cell_set(remaining, cur + 1.0);
            let step = self.perform_promise_any_element(
                c,
                promise_resolve,
                item,
                remaining,
                errors_box,
                cap,
                i,
            );
            if let Err(e) = step {
                let _ = self.iterator_close(it);
                return Err(e);
            }
            i += 1;
        }
        let n = self.cell_get(remaining) - 1.0;
        self.cell_set(remaining, n);
        if n == 0.0 {
            let agg = self.make_aggregate_error(errors_box);
            self.capability_reject(cap, agg)?;
        }
        Ok(cap.promise)
    }

    /// One element of `any`: `promiseResolve(item)` then subscribe fulfillment to
    /// the capability resolve (first wins) and rejection to the any-element fn.
    #[allow(clippy::too_many_arguments)]
    fn perform_promise_any_element(
        &mut self,
        c: NanBox,
        promise_resolve: NanBox,
        item: NanBox,
        remaining: Handle,
        errors_box: NanBox,
        cap: &PromiseCapability,
        i: usize,
    ) -> Result<(), ExecError> {
        let next = self.call_with_this(promise_resolve, c, &[item])?;
        let state = self.combinator_state(remaining, errors_box, cap, NanBox::number(i as f64));
        let on_r = self.make_element_fn(N_PROMISE_ANY_ELEMENT, state);
        self.invoke_then(next, cap.resolve, on_r)?;
        Ok(())
    }

    /// `Promise.prototype.finally(onFinally)` — species-aware. When `onFinally`
    /// is not callable the call degenerates to `then(onFinally, onFinally)`. When
    /// it is callable, two thunks (Then Finally / Catch Finally) run it on either
    /// settlement and then re-thread the original value/reason through
    /// `C.resolve(result).then(valueThunk)`.
    pub(crate) fn promise_finally(
        &mut self,
        handle: Handle,
        recv: NanBox,
        on_finally: NanBox,
    ) -> Result<NanBox, ExecError> {
        // `this` must be an Object (the spec's RequireInternalSlot-free `finally`
        // still does `Let C = SpeciesConstructor(promise, %Promise%)`, which reads
        // `promise.constructor` and so needs an object receiver).
        if !self.is_object_value(recv) {
            return Err(self.type_error("Promise.prototype.finally called on a non-object"));
        }
        let c = self.promise_species_constructor(handle)?;
        let then = self.read_member(handle, "then")?;
        if !self.is_callable_value(on_finally) {
            // Non-callable: pass it through as both handlers.
            return self.call_with_this(then, recv, &[on_finally, on_finally]);
        }
        // Build the two finally closures over a shared state (onFinally + C).
        let state = self.realm.new_object();
        self.realm
            .set_hidden_property(state, PFIN_ONFINALLY, on_finally);
        self.realm.set_hidden_property(state, PFIN_CTOR, c);
        let then_finally = self.realm.new_bound_native(N_PROMISE_THEN_FINALLY, state);
        let catch_finally = self.realm.new_bound_native(N_PROMISE_CATCH_FINALLY, state);
        self.install_fn_name_length(then_finally, "", 1);
        self.install_fn_name_length(catch_finally, "", 1);
        self.call_with_this(
            then,
            recv,
            &[
                NanBox::handle(then_finally.to_raw()),
                NanBox::handle(catch_finally.to_raw()),
            ],
        )
    }

    /// A `finally` Then/Catch closure body: run `onFinally()`, then build
    /// `promise = C.resolve(result)` and return `promise.then(valueThunk)`
    /// (`thenmode` selects whether the thunk returns the value or re-throws it).
    pub(crate) fn promise_finally_thunk(
        &mut self,
        state: Handle,
        value: NanBox,
        then_mode: bool,
    ) -> Result<NanBox, ExecError> {
        let on_finally = self
            .realm
            .get_property(state, PFIN_ONFINALLY)
            .unwrap_or(NanBox::undefined());
        let c = self
            .realm
            .get_property(state, PFIN_CTOR)
            .unwrap_or(NanBox::undefined());
        let result = self.call(on_finally, &[])?;
        // `promise = PromiseResolve(C, result)` via `C.resolve(result)`.
        let resolve = self.combinator_resolve_fn(c)?;
        let promise = self.call_with_this(resolve, c, &[result])?;
        // `valueThunk = () => value` (Then Finally) or `() => { throw value }`
        // (Catch Finally), bound to a state object carrying the captured value.
        let thunk_state = self.realm.new_object();
        self.realm
            .set_hidden_property(thunk_state, PFIN_VALUE, value);
        let id = if then_mode {
            N_PROMISE_VALUE_THUNK
        } else {
            N_PROMISE_THROW_THUNK
        };
        let thunk = self.realm.new_bound_native(id, thunk_state);
        self.install_fn_name_length(thunk, "", 0);
        // `return Invoke(promise, "then", [valueThunk])`.
        let Some(ph) = promise.as_handle().map(Handle::from_raw) else {
            return Err(self.type_error("Promise.resolve did not return an object"));
        };
        let then = self.read_member(ph, "then")?;
        self.call_with_this(then, promise, &[NanBox::handle(thunk.to_raw())])
    }

    /// IfAbruptRejectPromise: a synchronous throw during a combinator's body
    /// rejects the capability and returns its promise instead of propagating.
    fn finish_combinator(
        &mut self,
        cap: PromiseCapability,
        result: Result<NanBox, ExecError>,
    ) -> Result<NanBox, ExecError> {
        match result {
            Ok(p) => Ok(p),
            Err(ExecError::Throw(e)) => {
                self.capability_reject(&cap, e)?;
                Ok(cap.promise)
            }
            Err(other) => Err(other),
        }
    }

    /// Allocates a fresh pending promise whose `[[Prototype]]` is the realm's
    /// `Promise.prototype` (so `getPrototypeOf(p) === Promise.prototype`,
    /// `p instanceof Promise`, and the inherited `Symbol.toStringTag` resolve).
    pub(crate) fn fresh_promise(&mut self) -> Handle {
        let p = self.realm.new_promise();
        if let Some(proto) = self
            .current
            .get("Promise")
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .and_then(|c| self.realm.get_property(c, "prototype"))
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
        {
            self.realm.set_native_proto(p, proto);
        }
        p
    }
}
