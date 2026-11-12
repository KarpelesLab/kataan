//! Access to a suspended VM frame's cell ([`VmFrame`]), and the
//! iterator-result object, for the generator and async resume paths
//! (`nbvm::gen_fast`, `nbvm::drive_async`).
//!
//! A suspended generator or async function keeps its frame in one
//! engine-internal cell (never user-visible). Resuming moves the register
//! file out of the cell and suspending moves it back, instead of copying it
//! into a fresh array per `yield` / `await`.

use super::Realm;
use crate::cell::{Cell, VmFrame};
use crate::heap::Handle;
use crate::nanbox::NanBox;
use alloc::rc::Rc;
use alloc::vec::Vec;

impl Realm {
    /// Allocates the suspended-frame cell holding `frame`.
    pub(crate) fn new_vm_frame(&mut self, frame: VmFrame) -> Handle {
        self.heap.alloc(Cell::Frame(alloc::boxed::Box::new(frame)))
    }

    /// The frame cell a generator object `h` names in its hidden slot `key`
    /// (`VM_GEN` / `VM_AGEN`), read through the inline cache `cache`: on a
    /// hit, one shape compare and one slot read. `None` when `h` is no such
    /// generator.
    #[inline]
    pub(crate) fn generator_frame(
        &self,
        h: Handle,
        key: &str,
        cache: &mut crate::ic::PropertyCache,
    ) -> Option<Handle> {
        let v = match self.heap.get(h)? {
            Cell::Object(o) => match o.shaped_shape().and_then(|s| cache.hit_slot(s)) {
                Some(slot) => o.slot(slot),
                None => o.cached_get(key, cache),
            },
            _ => None,
        };
        let v = match v {
            Some(v) => v,
            None => self.get_property(h, key)?,
        };
        v.as_handle().map(Handle::from_raw)
    }

    /// The suspended frame at `h` (`None` for any other cell).
    pub(crate) fn vm_frame(&self, h: Handle) -> Option<&VmFrame> {
        match self.heap.get(h)? {
            Cell::Frame(f) => Some(f),
            _ => None,
        }
    }

    /// The suspended frame at `h`, mutably, through the generational write
    /// barrier (the mutable borrow itself; see `Heap::get_mut`). A caller
    /// that stores references into it calls
    /// [`vm_frame_stored`](Self::vm_frame_stored) afterwards.
    pub(crate) fn vm_frame_mut(&mut self, h: Handle) -> Option<&mut VmFrame> {
        match self.heap.get_mut(h)? {
            Cell::Frame(f) => Some(f),
            _ => None,
        }
    }

    /// After references were stored into the frame at `h` (its register
    /// file moving back in at a suspension): an incremental marking cycle's
    /// barrier shades them. (The generational barrier was the mutable
    /// borrow.)
    #[inline]
    pub(crate) fn vm_frame_stored(&mut self, h: Handle) {
        if self.incremental.is_some() {
            self.vm_frame_shade(h);
        }
    }

    #[cold]
    fn vm_frame_shade(&mut self, h: Handle) {
        let vals: Vec<NanBox> = match self.heap.get(h) {
            Some(Cell::Frame(f)) => f
                .regs
                .iter()
                .chain([&f.aux, &f.resume_fns.0, &f.resume_fns.1, &f.callee])
                .copied()
                .filter(|v| v.as_handle().is_some())
                .collect(),
            _ => return,
        };
        for v in vals {
            self.write_barrier(h, v);
        }
    }

    /// Removes the first element of the engine-internal array at `h` in
    /// place (an async generator's request queue): no user code can see
    /// it, so this is a plain shift of its elements.
    pub(crate) fn internal_array_shift(&mut self, h: Handle) -> Option<NanBox> {
        self.heap.get_mut(h)?.as_array_mut()?.shift_front()
    }

    /// The own data property `key` of the ordinary object at `h` through the
    /// inline cache `cache` (`None` for any other cell, or no such key).
    pub(crate) fn object_get_cached(
        &self,
        h: Handle,
        key: &str,
        cache: &mut crate::ic::PropertyCache,
    ) -> Option<NanBox> {
        self.heap.get(h)?.as_object()?.cached_get(key, cache)
    }

    /// `Get(promise, "constructor")` when it runs no code: `h` is a promise
    /// with no own properties whose `[[Prototype]]` is the intrinsic
    /// `%Promise.prototype%`, and that object's own `constructor` is a data
    /// property — its value. `None` otherwise (the full `[[Get]]`).
    /// `cache` is the caller's inline cache for that `constructor` read.
    pub(crate) fn plain_promise_constructor(
        &self,
        h: Handle,
        cache: &mut crate::ic::PropertyCache,
    ) -> Option<NanBox> {
        if !matches!(self.heap.get(h), Some(Cell::Promise(_)))
            || self.aux_props.get(&h.to_raw()).is_some()
        {
            return None;
        }
        let proto = self.promise_proto_intrinsic?;
        if self.object_proto(h) != Some(proto) {
            return None;
        }
        self.object_get_cached(proto, "constructor", cache)
    }

    /// The `[[Prototype]]` of the generator object `h` when it is an
    /// ordinary shaped object owning nothing but its one hidden state slot
    /// (so every named lookup on it continues at that prototype). An own
    /// accessor (kept beside the layout) is an own property too.
    #[inline]
    pub(crate) fn generator_owns_only_state(&self, h: Handle) -> Option<Handle> {
        let o = self.heap.get(h)?.as_object()?;
        if o.jit_flags() & crate::object::JIT_FLAG_EXOTIC != 0
            || o.shaped_shape()?.len() != 1
            || !o.has_no_accessors()
            || self.is_host_exotic(h)
        {
            return None;
        }
        o.proto()
    }

    /// [`generator_owns_only_state`](Self::generator_owns_only_state) and
    /// [`generator_frame`](Self::generator_frame) of the same object in one
    /// heap lookup: its `[[Prototype]]` and the frame cell its one own
    /// property, the hidden slot `key`, names. `None` when either would be.
    #[inline]
    pub(crate) fn generator_proto_and_frame(
        &self,
        h: Handle,
        key: &str,
        cache: &mut crate::ic::PropertyCache,
    ) -> Option<(Handle, Handle)> {
        let o = self.heap.get(h)?.as_object()?;
        let shape = o.shaped_shape()?;
        if o.jit_flags() & crate::object::JIT_FLAG_EXOTIC != 0
            || shape.len() != 1
            || !o.has_no_accessors()
            || self.is_host_exotic(h)
        {
            return None;
        }
        let v = match cache.hit_slot(shape) {
            Some(slot) => o.slot(slot),
            None => o.cached_get(key, cache),
        }?;
        Some((o.proto()?, Handle::from_raw(v.as_handle()?)))
    }

    /// The `[[Prototype]]` of the promise `h` when it owns no property (so
    /// a named lookup on it continues there).
    pub(crate) fn plain_promise_lookup_start(&self, h: Handle) -> Option<Handle> {
        if !matches!(self.heap.get(h), Some(Cell::Promise(_)))
            || self.aux_props.get(&h.to_raw()).is_some()
            || self.is_host_exotic(h)
        {
            return None;
        }
        self.object_proto(h)
    }

    /// `Get(promise, "then")` when it runs no code, as
    /// [`plain_promise_constructor`](Self::plain_promise_constructor) proves
    /// `constructor`'s: `h` is a promise with no own properties whose
    /// `[[Prototype]]` is the intrinsic `%Promise.prototype%`, which owns
    /// `then` as a data property — its value. `None` otherwise.
    pub(crate) fn plain_promise_then(
        &self,
        h: Handle,
        cache: &mut crate::ic::PropertyCache,
    ) -> Option<NanBox> {
        if !matches!(self.heap.get(h), Some(Cell::Promise(_)))
            || self.aux_props.get(&h.to_raw()).is_some()
        {
            return None;
        }
        let proto = self.promise_proto_intrinsic?;
        if self.object_proto(h) != Some(proto) {
            return None;
        }
        self.object_get_cached(proto, "then", cache)
    }

    /// Whether `Get(h, "then")` is `undefined` without running any code,
    /// for an iterator-result object: `h` is an ordinary object whose own
    /// properties are exactly the data properties `value` and `done` (the
    /// layout [`new_iter_result`](Self::new_iter_result) makes) and whose
    /// `[[Prototype]]` is the default object prototype, and no object on
    /// that prototype's chain has a `then` property. The chain is proved
    /// once per prototype epoch: each object on it is watched
    /// ([`JIT_FLAG_WATCHED`](crate::object::JIT_FLAG_WATCHED)), so a `then`
    /// defined anywhere on it, or a `[[Prototype]]` changed, retires the
    /// proof. `false` means "do the full `[[Get]]`".
    pub(crate) fn iter_result_thenless(&mut self, h: Handle) -> bool {
        let Some(shape) = self.iter_result_shape.as_ref() else {
            return false;
        };
        let Some(o) = self.heap.get(h).and_then(Cell::as_object) else {
            return false;
        };
        if !o
            .shaped_shape()
            .is_some_and(|s| alloc::rc::Rc::ptr_eq(s, shape))
            || !o.has_no_accessors()
            || o.jit_flags() & crate::object::JIT_FLAG_EXOTIC != 0
            || o.proto().is_none()
            || o.proto() != self.default_object_proto
        {
            return false;
        }
        if self.no_then_stamp == crate::object::proto_epoch() {
            return true;
        }
        let mut cur = self.default_object_proto;
        while let Some(c) = cur {
            match self.heap.get(c).and_then(Cell::as_object) {
                Some(p)
                    if p.jit_flags() & crate::object::JIT_FLAG_EXOTIC == 0
                        && !p.has_own_key("then") =>
                {
                    cur = p.proto();
                }
                _ => {
                    self.no_then_stamp = 0;
                    return false;
                }
            }
        }
        let mut cur = self.default_object_proto;
        while let Some(c) = cur
            && let Some(p) = self.heap.get_mut(c).and_then(Cell::as_object_mut)
        {
            p.watch();
            cur = p.proto();
        }
        self.no_then_stamp = crate::object::proto_epoch();
        true
    }

    /// A fresh generator object: `[[Prototype]]` `proto`, owning only the
    /// non-enumerable state slot `key` (`VM_GEN`, or `VM_AGEN` for an async
    /// generator: `is_async`) that holds `frame` — the object a fresh
    /// object's [`set_hidden_property`](Self::set_hidden_property) of that
    /// one key makes, built in its final layout at once.
    pub(crate) fn new_generator_object(
        &mut self,
        proto: Option<Handle>,
        key: &str,
        is_async: bool,
        frame: NanBox,
    ) -> Handle {
        if 1 < self.limits.object_dictionary_threshold {
            let i = usize::from(is_async);
            let shape = match &self.gen_shapes[i] {
                Some(s) if s.last_key() == Some(key) => Rc::clone(s),
                _ => {
                    let s = self.root_shape.transition(key);
                    self.gen_shapes[i] = Some(Rc::clone(&s));
                    s
                }
            };
            if let Some(mut obj) =
                crate::object::Object::with_data_hidden_prefix(shape, &[frame], 1)
            {
                obj.set_proto(proto);
                let h = self.heap.alloc(Cell::Object(obj));
                if let (Some(p), Some(marker)) = (proto, self.incremental.as_mut()) {
                    marker.mark_grey(p);
                }
                self.write_barrier(h, frame);
                return h;
            }
        }
        let g = self.new_object_with_proto(proto);
        self.set_hidden_property(g, key, frame);
        g
    }

    /// The own `prototype` data property of the VM closure `h` when reading
    /// it runs no code and makes nothing: `h` owes no prototype (it was
    /// made already) and its auxiliary object holds no accessor; read
    /// through the inline cache `cache`. `None` for anything else (the
    /// general `[[Get]]`).
    pub(crate) fn vm_fn_prototype_cached(
        &self,
        h: Handle,
        cache: &mut crate::ic::PropertyCache,
    ) -> Option<NanBox> {
        if !matches!(
            self.heap.get(h)?,
            Cell::VmFunction {
                owes_proto: false,
                ..
            }
        ) {
            return None;
        }
        let aux = self
            .heap
            .get(*self.aux_props.get(&h.to_raw())?)?
            .as_object()?;
        if !aux.has_no_accessors() {
            return None;
        }
        aux.cached_get("prototype", cache)
    }

    /// CreateIterResultObject(`value`, `done`): a fresh ordinary object with
    /// the data properties `value` and `done`, in that order — built in its
    /// final layout at once.
    pub fn new_iter_result(&mut self, value: NanBox, done: bool) -> Handle {
        // The usual case, the layout made: `new_object_in_shape`'s steps for
        // exactly these two slots (the second never holds a reference).
        if let Some(shape) = &self.iter_result_shape
            && 2 < self.limits.object_dictionary_threshold
            && let Some(mut obj) = crate::object::Object::with_data(
                alloc::rc::Rc::clone(shape),
                &[value, NanBox::boolean(done)],
            )
        {
            let proto = self.default_object_proto;
            obj.set_proto(proto);
            let h = self.heap.alloc(Cell::Object(obj));
            if let (Some(p), Some(marker)) = (proto, self.incremental.as_mut()) {
                marker.mark_grey(p);
            }
            self.write_barrier(h, value);
            return h;
        }
        let root = alloc::rc::Rc::clone(&self.root_shape);
        let shape = match &self.iter_result_shape {
            Some(s) => alloc::rc::Rc::clone(s),
            None => {
                let s = root.transition("value").transition("done");
                self.iter_result_shape = Some(alloc::rc::Rc::clone(&s));
                s
            }
        };
        match self.new_object_in_shape(&root, &shape, &[value, NanBox::boolean(done)]) {
            Some(h) => h,
            None => {
                let o = self.new_object();
                self.set_property(o, "value", value);
                self.set_property(o, "done", NanBox::boolean(done));
                o
            }
        }
    }
}
