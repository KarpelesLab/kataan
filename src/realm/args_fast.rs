//! `arguments` objects made in one step.
//!
//! `CreateUnmappedArgumentsObject` / `CreateMappedArgumentsObject` (ECMA-262
//! 10.4.4.6-7) define the same handful of properties on every object they
//! make: the indices (enumerable data), `length` and `@@iterator` (data,
//! non-enumerable), and `callee` — the function itself (data,
//! non-enumerable) on a mapped object, the realm's `%ThrowTypeError%` as
//! both accessor halves (non-enumerable, non-configurable) on an unmapped
//! one. Built one property at a time that is a shape transition, a mark
//! and a key formatting per property; here the layout is made once per
//! argument count and kind, and each object is allocated with its slots
//! already filled.
//!
//! The layout puts the non-enumerable data keys first (`length`,
//! `@@iterator`, then `callee` on a mapped object, then the host's arguments
//! marker), so their marks are the object's hidden prefix, and the indices
//! after them. Key order is unobservable here: `[[OwnPropertyKeys]]` lists
//! integer indices first in ascending order, then the string keys in
//! creation order (`length` before `callee`), then the symbols.
//!
//! The parameter map of a mapped object is not part of the object: the host
//! records it separately (`Op::MapArguments`), as it does for an object made
//! the slow way.

use super::{Cell, Handle, NanBox, Realm};
use crate::shape::Shape;
use alloc::rc::Rc;

/// The most arguments an object made here holds (a longer call takes the
/// host's general path).
const ARGS_FAST_MAX: usize = 16;

/// The index keys `0..ARGS_FAST_MAX`.
const INDEX_KEYS: [&str; ARGS_FAST_MAX] = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15",
];

impl Realm {
    /// A fresh `arguments` object for a call with `args` to `callee`: mapped
    /// (`callee` an own data property) or unmapped (`callee` the poisoned
    /// accessor) — what the host's `make_arguments_object` leaves, before any
    /// parameter map. `None`, having allocated nothing, when this heap holds
    /// more than one realm (the intrinsics to use then depend on the
    /// callee's realm), the intrinsics are not installed yet, or there are
    /// more than [`ARGS_FAST_MAX`] arguments.
    pub fn new_arguments_object(
        &mut self,
        args: &[NanBox],
        callee: NanBox,
        mapped: bool,
    ) -> Option<Handle> {
        let argc = args.len();
        if argc > ARGS_FAST_MAX || self.vm_array_iter.values_fns.len() != 1 {
            return None;
        }
        let values_fn = NanBox::handle(self.vm_array_iter.values_fns[0].to_raw());
        let thrower = if mapped {
            None
        } else {
            Some(NanBox::handle(self.throw_type_error_intrinsic?.to_raw()))
        };
        let shape = self.arguments_shape(argc, mapped)?;
        let mut vals = [NanBox::undefined(); ARGS_FAST_MAX + 4];
        let hidden: u32 = if mapped { 4 } else { 3 };
        vals[0] = NanBox::number(argc as f64);
        vals[1] = values_fn;
        if mapped {
            vals[2] = callee;
        }
        vals[hidden as usize - 1] = NanBox::boolean(true);
        let n = hidden as usize + argc;
        vals[hidden as usize..n].copy_from_slice(args);
        let vals = &vals[..n];
        let mut obj = crate::object::Object::with_data_hidden_prefix(shape, vals, hidden)?;
        if let Some(t) = thrower {
            obj.define_accessor("callee", t, t);
            obj.set_hidden("callee");
            obj.set_non_configurable("callee");
        }
        // As `new_object` would: the default `[[Prototype]]`, shaded for an
        // incremental cycle, then each stored value through the barrier.
        let proto = self.default_object_proto;
        obj.set_proto(proto);
        let h = self.heap.alloc(Cell::Object(obj));
        if let (Some(p), Some(m)) = (proto, self.incremental.as_mut()) {
            m.mark_grey(p);
        }
        for v in vals.iter().chain(thrower.as_ref()) {
            self.write_barrier(h, *v);
        }
        Some(h)
    }

    /// `args[key]` — `key` an index Number, or `None` for `length` — on an
    /// `arguments` object [`new_arguments_object`](Self::new_arguments_object)
    /// made that still has its layout: then the key, when it names an index
    /// below the argument count or `length`, is an own data property (a
    /// delete, a new key or dictionary mode changes the layout; a redefinition
    /// as an accessor is checked for), so `[[Get]]` is its slot. `None` when
    /// that cannot be told this way (the caller takes the general path).
    #[must_use]
    pub fn arguments_own_get(&self, handle: Handle, key: Option<f64>) -> Option<NanBox> {
        let (o, argc, hidden) = self.arguments_layout(handle)?;
        let (slot, name) = match key {
            None => (0, "length"),
            Some(n) => {
                if !(0.0..ARGS_FAST_MAX as f64).contains(&n) || n != (n as usize) as f64 {
                    return None;
                }
                let i = n as usize;
                if i >= argc {
                    return None;
                }
                (hidden + i, INDEX_KEYS[i])
            }
        };
        if o.accessor(name).is_some() {
            return None;
        }
        o.slot(slot as u32)
    }

    /// `CreateListFromArrayLike(args)` for an `arguments` object made by
    /// [`new_arguments_object`](Self::new_arguments_object) that still has its
    /// layout, its `length` still the argument count and no index or `length`
    /// redefined as an accessor: the arguments, read from their slots (no
    /// `[[Get]]` here can run code or reach the prototype). `None` otherwise.
    #[must_use]
    pub fn arguments_values(&self, handle: Handle) -> Option<alloc::vec::Vec<NanBox>> {
        let (o, argc, hidden) = self.arguments_layout(handle)?;
        if o.accessor("length").is_some() || o.slot(0)?.as_number() != Some(argc as f64) {
            return None;
        }
        (0..argc)
            .map(|i| {
                if o.accessor(INDEX_KEYS[i]).is_some() {
                    return None;
                }
                o.slot((hidden + i) as u32)
            })
            .collect()
    }

    /// The object at `handle`, its argument count and the slot of its first
    /// index, when it is an `arguments` object with one of the kept layouts
    /// and no host hook intercepts its reads.
    fn arguments_layout(&self, handle: Handle) -> Option<(&crate::object::Object, usize, usize)> {
        if self.args_shapes.is_empty() {
            return None;
        }
        let Some(Cell::Object(o)) = self.heap.get(handle) else {
            return None;
        };
        let shape = o.shaped_shape()?;
        let len = shape.len() as usize;
        // The mapped layout has 4 leading keys, the unmapped one 3.
        let (argc, hidden) = match (len.checked_sub(4), len.checked_sub(3)) {
            (Some(a), _) if self.is_arguments_shape(shape, a, true) => (a, 4),
            (_, Some(a)) if self.is_arguments_shape(shape, a, false) => (a, 3),
            _ => return None,
        };
        if self.is_host_exotic(handle) {
            return None;
        }
        Some((o, argc, hidden))
    }

    /// Whether `shape` is the kept `arguments` layout for `argc` arguments.
    fn is_arguments_shape(&self, shape: &Rc<Shape>, argc: usize, mapped: bool) -> bool {
        matches!(
            self.args_shapes.get(2 * argc + usize::from(mapped)),
            Some(Some(s)) if Rc::ptr_eq(s, shape)
        )
    }

    /// The layout of an `arguments` object of `argc` arguments (see the
    /// module docs), made once and kept.
    fn arguments_shape(&mut self, argc: usize, mapped: bool) -> Option<Rc<Shape>> {
        let i = 2 * argc + usize::from(mapped);
        if let Some(Some(s)) = self.args_shapes.get(i) {
            return Some(Rc::clone(s));
        }
        let iter_key = self.vm_array_iter.iter_key.clone()?;
        let mut s = self.root_shape.add("length").add(&iter_key);
        if mapped {
            s = s.add("callee");
        }
        s = s.add(crate::nbexec::ARGS_MARKER);
        for k in &INDEX_KEYS[..argc] {
            s = s.add(k);
        }
        if self.args_shapes.len() <= i {
            self.args_shapes.resize(i + 1, None);
        }
        self.args_shapes[i] = Some(Rc::clone(&s));
        Some(s)
    }
}
