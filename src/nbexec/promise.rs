use super::*;

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
            });
        }
    }

    /// Resolves `handle` with `value`, adopting it if `value` is itself a
    /// promise (chain on its settlement).
    pub(crate) fn resolve_with(&mut self, handle: Handle, value: NanBox) {
        let inner = value
            .as_handle()
            .map(Handle::from_raw)
            .filter(|h| self.realm.promise_state(*h).is_some());
        if let Some(inner) = inner {
            // Adopt: when `inner` settles, settle `handle` the same way.
            let on_f = self.realm.new_bound_native(N_RESOLVE, handle);
            let on_r = self.realm.new_bound_native(N_REJECT, handle);
            self.register_then(
                inner,
                NanBox::handle(on_f.to_raw()),
                NanBox::handle(on_r.to_raw()),
                false,
            );
            return;
        }
        // A thenable (a non-promise object with a callable `then`) is adopted by
        // calling `then(resolve, reject)`.
        if let Some(vh) = value.as_handle().map(Handle::from_raw)
            && let Some(then) = self.realm.get_property(vh, "then")
            && then
                .as_handle()
                .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
        {
            let on_f = self.realm.new_bound_native(N_RESOLVE, handle);
            let on_r = self.realm.new_bound_native(N_REJECT, handle);
            let args = [NanBox::handle(on_f.to_raw()), NanBox::handle(on_r.to_raw())];
            // A throw from `then` rejects the promise.
            if let Err(ExecError::Throw(e)) = self.call_with_this(then, value, &args) {
                self.settle(handle, e, false);
            }
            return;
        }
        self.settle(handle, value, true);
    }

    /// Registers `then` reactions on `handle`, returning a new dependent promise.
    pub(crate) fn promise_then(&mut self, handle: Handle, on_f: NanBox, on_r: NanBox) -> NanBox {
        let result = self.register_then(handle, on_f, on_r, false);
        NanBox::handle(result.to_raw())
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
    /// The current settled state of `value`: `Some(Ok(v))` if fulfilled (a
    /// non-promise counts as fulfilled with itself), `Some(Err(e))` if rejected,
    /// `None` if it is a still-pending promise.
    pub(crate) fn settled_state(&self, value: NanBox) -> Option<Result<NanBox, NanBox>> {
        use crate::cell::PromiseStatus::{Fulfilled, Pending, Rejected};
        let Some(state) = value
            .as_handle()
            .and_then(|raw| self.realm.promise_state(Handle::from_raw(raw)))
        else {
            return Some(Ok(value));
        };
        let s = state.borrow();
        match s.status {
            Fulfilled => Some(Ok(s.value)),
            Rejected => Some(Err(s.value)),
            Pending => None,
        }
    }

    pub(crate) fn await_value(&mut self, value: NanBox) -> Result<NanBox, ExecError> {
        use crate::cell::PromiseStatus::{Fulfilled, Pending, Rejected};
        let Some(state) = value
            .as_handle()
            .and_then(|raw| self.realm.promise_state(Handle::from_raw(raw)))
        else {
            return Ok(value); // not a promise
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
