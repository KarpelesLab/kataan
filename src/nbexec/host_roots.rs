//! Root scopes: how a built-in that steps a JS iterator lets the code it
//! re-enters (a generator body, a user `next`) collect garbage.
//!
//! The protocol is described in `crate::realm::host_roots`. In short, a
//! built-in opens a [`RootScope`] over every value its frame holds across the
//! steps — the iterator record, the callback, the values gathered so far —
//! and steps through [`RootScope::step`], which lets the nested VM run
//! collect when the VM context below the built-in had published all of its
//! frames. Everything else the built-in re-enters (a getter, a callback
//! called the ordinary way) stays uncollectable, as before.
//!
//! What a scope roots on its own: the running `this` and `new.target` (the
//! values the host frames between the VM and the built-in saved before
//! replacing them are the enclosing scope's, or the outermost ambient ones
//! the interpreter roots itself) and the values passed to
//! [`Interp::root_scope`] — the built-in's arguments, typically, plus its
//! iterator and `next` method. Values gathered while stepping go in with
//! [`RootScope::hold`] and come back out with [`RootScope::take_held`].
//!
//! A scope must cover every value the built-in uses after a step that it did
//! not read afresh: a missed one is freed under it. The collector does not
//! move objects, so rooting a copy is enough, and results stored into
//! objects go through the realm's ordinary (write-barriered) stores.

use super::{ExecError, Interp, N_ASYNC_GEN_NEXT, N_GEN_NEXT, vm_to_exec};
use crate::heap::Handle;
use crate::nanbox::NanBox;
use alloc::vec::Vec;

/// An open root scope over an [`Interp`]: dereferences to it, and closes
/// (pops what it rooted) when dropped, on every path.
pub(crate) struct RootScope<'a> {
    interp: &'a mut Interp,
    /// Where this scope's values start on the realm's host root stack.
    mark: usize,
    /// The enclosing scope's crossing and relay, restored on close.
    prev_scope: u32,
    prev_relay: Option<(u32, u32)>,
}

impl Interp {
    /// Opens a root scope over `this`, `new.target` and `vals` (see the
    /// module docs).
    pub(crate) fn root_scope(&mut self, vals: &[NanBox]) -> RootScope<'_> {
        let (this, nt, call) = (self.this_val, self.new_target, self.call_serial);
        let hr = &mut self.realm.host_roots;
        let mark = hr.vals.len();
        let (prev_scope, prev_relay) = (hr.scope, hr.relay);
        hr.scope = hr.activation(call);
        hr.vals.push(this);
        hr.vals.push(nt);
        hr.vals.extend_from_slice(vals);
        RootScope {
            interp: self,
            mark,
            prev_scope,
            prev_relay,
        }
    }
}

impl RootScope<'_> {
    /// Roots `v` until the scope closes.
    #[inline]
    pub(crate) fn hold(&mut self, v: NanBox) {
        self.interp.realm.host_roots.vals.push(v);
    }

    /// Where the next [`hold`](Self::hold) goes: the start of a run of held
    /// values for [`take_held`](Self::take_held).
    #[inline]
    pub(crate) fn held_mark(&self) -> usize {
        self.interp.realm.host_roots.vals.len()
    }

    /// Replaces the value held at `pos` (a [`held_mark`](Self::held_mark)
    /// taken before holding it): a rooted slot for a value the caller keeps
    /// updating, such as an accumulator.
    #[inline]
    pub(crate) fn set_held(&mut self, pos: usize, v: NanBox) {
        self.interp.realm.host_roots.vals[pos] = v;
    }

    /// The values held from `mark` on, in order (no longer rooted).
    pub(crate) fn take_held(&mut self, mark: usize) -> Vec<NanBox> {
        let vals = &mut self.interp.realm.host_roots.vals;
        debug_assert!(mark >= self.mark && mark <= vals.len());
        vals.split_off(mark.max(self.mark).min(vals.len()))
    }

    /// `IteratorStep` + `IteratorValue` of `it` with its `next` method
    /// (`Interp::iter_step`), letting the code the step runs collect when it
    /// is a VM generator's body or a VM `next` function. The caller has
    /// rooted `it` and `next` in this scope, and holds nothing else unrooted
    /// across the call.
    pub(crate) fn step(&mut self, it: Handle, next: NanBox) -> Result<Option<NanBox>, ExecError> {
        let i = &mut *self.interp;
        if let Some(nh) = next.as_handle().map(Handle::from_raw)
            && i.created_realms.is_empty()
        {
            // A VM generator with the intrinsic `next`: resumed directly (what
            // the `%GeneratorPrototype%.next` dispatch would reach), with
            // nothing between the arm and the resumption.
            if i.realm.native_at(nh) == Some(N_GEN_NEXT)
                && i.realm.get_property(it, crate::nbvm::VM_GEN).is_some()
                && let Some(table) = i.vm_table.clone()
            {
                if crate::stack::exhausted() {
                    return Err(i.stack_exhausted());
                }
                i.realm.host_roots.arm();
                let r = crate::nbvm::host_generator_step(i, &table, NanBox::handle(it.to_raw()));
                i.realm.host_roots.armed = 0;
                return match r.map_err(vm_to_exec)? {
                    Ok(v) => Ok(v),
                    Err(res) => i.iter_result_value(res),
                };
            }
            // A VM `next`: `iter_step` calls it straight through
            // `call_with_this` (no other re-entry first).
            if i.realm.is_vm_function(nh) {
                i.realm.host_roots.arm();
                let r = i.iter_step(it, next);
                i.realm.host_roots.armed = 0;
                return r;
            }
        }
        // Anything else (a built-in iterator, an iterator helper, a bound or
        // proxied `next`): what the step calls first may open an active scope
        // of its own, this frame holding nothing unrooted (`relay`).
        let hr = &mut i.realm.host_roots;
        if !hr.live() {
            return i.iter_step(it, next);
        }
        let prev = hr.relay.replace((hr.scope, hr.serial));
        let r = i.iter_step(it, next);
        i.realm.host_roots.relay = prev;
        r
    }
}

impl RootScope<'_> {
    /// `Call(next, it, args)` of an iterator's `next` method: the iterator
    /// result (an async iterator's promise of one), letting the code the
    /// call runs collect when it is a VM generator's or VM async generator's
    /// body behind the intrinsic `next` (resumed directly, as that `next`
    /// would resume it), or a VM `next` function. The caller has rooted
    /// `it`, `next` and `args` in this scope, and holds nothing else
    /// unrooted across the call.
    pub(crate) fn next_result(
        &mut self,
        it: Handle,
        next: NanBox,
        args: &[NanBox],
    ) -> Result<NanBox, ExecError> {
        let i = &mut *self.interp;
        let itv = NanBox::handle(it.to_raw());
        if i.realm.host_roots.live()
            && i.created_realms.is_empty()
            && let Some(nh) = next.as_handle().map(Handle::from_raw)
            && let Some(id) = i.realm.native_at(nh)
            && (id == N_GEN_NEXT || id == N_ASYNC_GEN_NEXT)
            && let Some(table) = i.vm_table.clone()
        {
            let sent = args.first().copied().unwrap_or_else(NanBox::undefined);
            let sync = id == N_GEN_NEXT;
            let slot = if sync {
                crate::nbvm::VM_GEN
            } else {
                crate::nbvm::VM_AGEN
            };
            if i.realm.get_property(it, slot).is_some() {
                if crate::stack::exhausted() {
                    return Err(i.stack_exhausted());
                }
                // (What the intrinsic `next` does, with nothing between the
                // arm and the resumption.)
                i.realm.host_roots.arm();
                let r = if sync {
                    crate::nbvm::resume_vm_generator(i, &table, itv, 0, sent)
                } else {
                    crate::nbvm::vm_agen_request(i, &table, itv, 0, sent)
                };
                i.realm.host_roots.armed = 0;
                return r.map_err(vm_to_exec);
            }
        }
        i.call_cb(next, itv, args)
    }

    /// `Call(next, this)` of an iterator's `next` method whose result the
    /// caller returns as is (a wrapper's forwarding `next`), letting a VM
    /// `next` collect: `call_with_this` calls it straight (no other
    /// re-entry first). The caller has rooted `next` and `this` here.
    pub(crate) fn call_next(&mut self, next: NanBox, this: NanBox) -> Result<NanBox, ExecError> {
        self.interp.call_cb(next, this, &[])
    }
}

impl Interp {
    /// Opens a root scope of the collecting event loop's own (see
    /// `crate::realm::host_roots::LOOP`) over `vals`, active, so the handler
    /// the loop calls through [`call_cb`](Self::call_cb) may collect (a VM
    /// function) or open an active scope of its own (a built-in's
    /// continuation). Only the collecting loop's own job (`Interp::job_gc`)
    /// may open one; the caller roots in `vals` everything it uses after the
    /// call.
    pub(crate) fn root_scope_loop(&mut self, vals: &[NanBox]) -> RootScope<'_> {
        let sc = self.root_scope(vals);
        let hr = &mut sc.interp.realm.host_roots;
        hr.scope = hr.crossings | crate::realm::host_roots::LOOP;
        sc
    }

    /// `Call(f, this, args)` of a built-in's callback, made while every value
    /// the built-in uses after the call is rooted in its open root scope
    /// (its arguments, `this`, what it gathered — **including any value it
    /// passes in `args` and reads again afterwards**: the callee may
    /// overwrite its parameters before it collects). When that scope is
    /// active, a VM function `f` may collect (`call_with_this` enters it
    /// straight, no other re-entry first), and any other callee is relayed
    /// to, so a built-in it is may open an active scope of its own. Outside
    /// an active scope it is a plain `call_with_this`.
    #[inline(always)]
    pub(crate) fn call_cb(
        &mut self,
        f: NanBox,
        this: NanBox,
        args: &[NanBox],
    ) -> Result<NanBox, ExecError> {
        // (No scope open at all: the plain call, with nothing in between.)
        if self.realm.host_roots.scope == 0 {
            return self.call_with_this(f, this, args);
        }
        self.call_cb_scoped(f, this, args)
    }

    /// Whether [`call_cb`](Self::call_cb) of `f` made now would be a VM call
    /// that may collect (an active scope, one realm, a VM function): a
    /// built-in calling the same `f` many times from the same scope (whose
    /// state does not change between the calls, but for realms being made)
    /// asks once, then calls [`call_vm_cb`](Self::call_vm_cb) while
    /// `created_realms` stays empty.
    pub(crate) fn cb_is_vm(&self, f: NanBox) -> bool {
        self.realm.host_roots.live()
            && self.created_realms.is_empty()
            && self.vm_table.is_some()
            && f.as_handle()
                .is_some_and(|h| self.realm.is_vm_function(Handle::from_raw(h)))
    }

    /// [`call_cb`](Self::call_cb) of `f` once [`cb_is_vm`](Self::cb_is_vm)
    /// said it is a VM call that may collect (and no realm was made since):
    /// `call_with_this`'s VM call, numbered, collection on.
    pub(crate) fn call_vm_cb(
        &mut self,
        f: NanBox,
        this: NanBox,
        args: &[NanBox],
    ) -> Result<NanBox, ExecError> {
        if crate::stack::exhausted() {
            return Err(self.stack_exhausted());
        }
        let Some(table) = self.vm_table.clone() else {
            return self.call_with_this(f, this, args);
        };
        let outer = self.call_serial;
        let hr = &mut self.realm.host_roots;
        hr.serial = hr.serial.wrapping_add(1);
        self.call_serial = hr.serial;
        let r =
            crate::nbvm::call_vm_function_in(self, &table, f, this, args, true).map_err(vm_to_exec);
        self.call_serial = outer;
        r
    }

    /// [`call_cb`](Self::call_cb) with some root scope open.
    fn call_cb_scoped(
        &mut self,
        f: NanBox,
        this: NanBox,
        args: &[NanBox],
    ) -> Result<NanBox, ExecError> {
        let hr = &self.realm.host_roots;
        if !hr.live() || !self.created_realms.is_empty() {
            return self.call_with_this(f, this, args);
        }
        if self.vm_table.is_some()
            && let Some(h) = f.as_handle()
            && self.realm.is_vm_function(Handle::from_raw(h))
        {
            // `call_with_this`'s call of a VM function (one realm: numbered,
            // the stack checked, then `call_vm_function`), entered with
            // collection on: an active scope's arm would be taken by it.
            return self.call_vm_cb(f, this, args);
        }
        self.call_cb_other(f, this, args)
    }

    /// [`call_cb`](Self::call_cb) of anything but a VM function, under a
    /// live scope.
    #[inline(never)]
    fn call_cb_other(
        &mut self,
        f: NanBox,
        this: NanBox,
        args: &[NanBox],
    ) -> Result<NanBox, ExecError> {
        // A bound function (`f.bind(t, ...)`): what its `[[Call]]` does
        // (`call_with_this` forwards to the target with the bound `this` and
        // arguments first), the target's call made the same way. The bound
        // function holds the target, `this` and arguments, and the callee's
        // frame what it is passed.
        if let Some(h) = f.as_handle().map(Handle::from_raw)
            && self.realm.native_at(h).is_none()
            && self.realm.bound_native_at(h).is_none()
            && self.realm.proxy_at(h).is_none()
            && let Some(target) = self.realm.get_property(h, super::BOUND_TARGET)
        {
            let bthis = self
                .realm
                .get_property(h, super::BOUND_THIS)
                .unwrap_or(NanBox::undefined());
            let mut all = self
                .realm
                .get_property(h, super::BOUND_ARGS)
                .and_then(|a| a.as_handle())
                .map(Handle::from_raw)
                .and_then(|a| self.realm.array_elements(a).map(<[_]>::to_vec))
                .unwrap_or_default();
            all.extend_from_slice(args);
            if crate::stack::exhausted() {
                return Err(self.stack_exhausted());
            }
            return self.call_cb(target, bthis, &all);
        }
        let hr = &mut self.realm.host_roots;
        let prev = hr.relay.replace((hr.scope, hr.serial));
        let r = self.call_with_this(f, this, args);
        self.realm.host_roots.relay = prev;
        r
    }

    /// Roots `v` in the innermost open root scope until it closes (for code
    /// running inside a scope it did not open itself, such as the body of a
    /// built-in its caller wrapped in one).
    #[inline]
    pub(crate) fn root_hold(&mut self, v: NanBox) {
        self.realm.host_roots.vals.push(v);
    }

    /// A rooted slot in the innermost open root scope, holding `v` (see
    /// [`root_hold`](Self::root_hold)), for a value the caller keeps
    /// replacing ([`root_set`](Self::root_set)): an accumulator.
    #[inline]
    pub(crate) fn root_slot(&mut self, v: NanBox) -> usize {
        let vals = &mut self.realm.host_roots.vals;
        vals.push(v);
        vals.len() - 1
    }

    /// Replaces the value of the slot `pos` ([`root_slot`](Self::root_slot)).
    #[inline]
    pub(crate) fn root_set(&mut self, pos: usize, v: NanBox) {
        self.realm.host_roots.vals[pos] = v;
    }

    /// Drops the slot `pos` ([`root_slot`](Self::root_slot)) and everything
    /// rooted after it, for a caller whose frame (and those it called) no
    /// longer holds them.
    #[inline]
    pub(crate) fn root_truncate(&mut self, pos: usize) {
        self.realm.host_roots.vals.truncate(pos);
    }
}

impl core::ops::Deref for RootScope<'_> {
    type Target = Interp;
    #[inline]
    fn deref(&self) -> &Interp {
        self.interp
    }
}

impl core::ops::DerefMut for RootScope<'_> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Interp {
        self.interp
    }
}

impl Drop for RootScope<'_> {
    fn drop(&mut self) {
        let hr = &mut self.interp.realm.host_roots;
        hr.vals.truncate(self.mark);
        hr.scope = self.prev_scope;
        hr.armed = 0;
        hr.relay = self.prev_relay;
    }
}
