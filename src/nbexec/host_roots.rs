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

use super::{ExecError, Interp, N_GEN_NEXT, vm_to_exec};
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
        if hr.scope == 0 || hr.scope != hr.crossings {
            return i.iter_step(it, next);
        }
        let prev = hr.relay.replace((hr.crossings, hr.serial));
        let r = i.iter_step(it, next);
        i.realm.host_roots.relay = prev;
        r
    }
}

impl RootScope<'_> {
    /// `Call(next, this)` of an iterator's `next` method whose result the
    /// caller returns as is (a wrapper's forwarding `next`), letting a VM
    /// `next` collect: `call_with_this` calls it straight (no other
    /// re-entry first). The caller has rooted `next` and `this` here.
    pub(crate) fn call_next(&mut self, next: NanBox, this: NanBox) -> Result<NanBox, ExecError> {
        let i = &mut *self.interp;
        let armed = i.created_realms.is_empty()
            && next
                .as_handle()
                .is_some_and(|n| i.realm.is_vm_function(Handle::from_raw(n)))
            && i.realm.host_roots.arm();
        let r = i.call_with_this(next, this, &[]);
        if armed {
            i.realm.host_roots.armed = 0;
        }
        r
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
