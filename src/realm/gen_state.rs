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
use alloc::vec::Vec;

impl Realm {
    /// Allocates the suspended-frame cell holding `frame`.
    pub(crate) fn new_vm_frame(&mut self, frame: VmFrame) -> Handle {
        self.heap.alloc(Cell::Frame(alloc::boxed::Box::new(frame)))
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
