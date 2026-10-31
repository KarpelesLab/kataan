//! In-place access to a suspended VM frame's state arrays, and the
//! iterator-result object, for the generator resume path (`nbvm::gen_fast`).
//!
//! A suspended generator keeps its frame in engine-internal arrays (never
//! user-visible: they hang off a hidden slot). Resuming moves the register
//! file out of its array and suspending moves it back, instead of copying it
//! into a fresh array per `yield`.

use super::Realm;
use crate::cell::Cell;
use crate::heap::Handle;
use crate::nanbox::NanBox;
use alloc::vec::Vec;

impl Realm {
    /// Moves the elements out of the engine-internal array at `h`, leaving
    /// it empty; `None` when `h` is not an array.
    pub(crate) fn internal_array_take(&mut self, h: Handle) -> Option<Vec<NanBox>> {
        let a = self.heap.get_mut(h)?.as_array_mut()?;
        Some(core::mem::take(a.vec_mut()))
    }

    /// Makes `elems` the elements of the engine-internal array at `h` (as
    /// [`internal_array_take`](Self::internal_array_take) left it), through
    /// the write barrier. `false`, having changed nothing, when `h` is not an
    /// array.
    pub(crate) fn internal_array_put(&mut self, h: Handle, elems: Vec<NanBox>) -> bool {
        if !matches!(self.heap.get(h), Some(Cell::Array(_))) {
            return false;
        }
        // (The barrier of a young array records nothing, so only an old one
        // or an incremental cycle's shading needs the walk.)
        if self.incremental.is_some() || self.heap.age(h).is_some_and(|a| a >= crate::gc::OLD_AGE) {
            for &x in &elems {
                if x.as_handle().is_some() {
                    self.write_barrier(h, x);
                }
            }
        }
        if let Some(a) = self.heap.get_mut(h).and_then(Cell::as_array_mut) {
            *a.vec_mut() = elems;
        }
        true
    }

    /// Overwrites element `i` of the engine-internal array at `h` with the
    /// Number `n` (a no-op past its end). Numbers need no write barrier.
    pub(crate) fn internal_array_set_number(&mut self, h: Handle, i: usize, n: f64) {
        if let Some(slot) = self
            .heap
            .get_mut(h)
            .and_then(Cell::as_array_mut)
            .and_then(|a| a.as_mut_slice().get_mut(i))
        {
            *slot = NanBox::number(n);
        }
    }

    /// Overwrites elements `i..` of the engine-internal array at `h` with
    /// the Numbers `ns` (those past its end are dropped). Numbers need no
    /// write barrier.
    pub(crate) fn internal_array_set_numbers_from(&mut self, h: Handle, i: usize, ns: &[f64]) {
        if let Some(a) = self.heap.get_mut(h).and_then(Cell::as_array_mut) {
            for (slot, &n) in a.as_mut_slice().iter_mut().skip(i).zip(ns) {
                *slot = NanBox::number(n);
            }
        }
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

    /// Replaces the elements of the engine-internal array at `h` with the
    /// Numbers `ns`.
    pub(crate) fn internal_array_set_numbers(&mut self, h: Handle, ns: impl Iterator<Item = f64>) {
        if let Some(a) = self.heap.get_mut(h).and_then(Cell::as_array_mut) {
            let v = a.vec_mut();
            v.clear();
            v.extend(ns.map(NanBox::number));
        }
    }

    /// `Get(promise, "constructor")` when it runs no code: `h` is a promise
    /// with no own properties whose `[[Prototype]]` is the intrinsic
    /// `%Promise.prototype%`, and that object's own `constructor` is a data
    /// property — its value. `None` otherwise (the full `[[Get]]`).
    pub(crate) fn plain_promise_constructor(&self, h: Handle) -> Option<NanBox> {
        if !matches!(self.heap.get(h), Some(Cell::Promise(_)))
            || self.aux_props.get(&h.to_raw()).is_some()
        {
            return None;
        }
        let proto = self.promise_proto_intrinsic?;
        if self.native_protos.get(&h.to_raw()).copied() != Some(proto) {
            return None;
        }
        self.heap.get(proto)?.as_object()?.get("constructor")
    }

    /// CreateIterResultObject(`value`, `done`): a fresh ordinary object with
    /// the data properties `value` and `done`, in that order — built in its
    /// final layout at once.
    pub fn new_iter_result(&mut self, value: NanBox, done: bool) -> Handle {
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
