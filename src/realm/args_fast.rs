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
//! The parameter map of a bytecode-VM function's mapped object
//! (`Op::MapArguments`) is a hidden property of the object, [`ARGS_CELLS`]:
//! an array of the parameters' binding cells by index. While the object
//! keeps its layout, an index read, an index write and
//! `CreateListFromArrayLike` go straight to the slots and cells here; the
//! host reads the same map for everything else, and unmaps an index (clears
//! its cell) on a delete or a redefinition as an accessor or non-writable.

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

    /// `args[key]` (`key` an index Number, or `None` for `length`) on an
    /// `arguments` object [`new_arguments_object`](Self::new_arguments_object)
    /// made that still has its layout: then the key, when it names an index
    /// below the argument count or `length`, is an own data property (a
    /// delete, a new key or dictionary mode changes the layout; a redefinition
    /// as an accessor is checked for), so `[[Get]]` is its slot, or, for an
    /// index still mapped to a parameter (see
    /// [`attach_argument_cells`](Self::attach_argument_cells)), that
    /// parameter's cell. `None` when that cannot be told this way (the caller
    /// takes the general path).
    #[must_use]
    pub fn arguments_own_get(&self, handle: Handle, key: Option<f64>) -> Option<NanBox> {
        let l = self.arguments_layout(handle)?;
        let (slot, name) = match key {
            None => (0, "length"),
            Some(n) => {
                let i = arg_index(n, l.argc)?;
                if let Some(c) = l.cell(i) {
                    // A mapped index is a data property: a redefinition as an
                    // accessor or as non-writable unmaps it.
                    return self.cell_get(c);
                }
                (l.hidden + i, INDEX_KEYS[i])
            }
        };
        if l.obj.accessor(name).is_some() {
            return None;
        }
        l.obj.slot(slot as u32)
    }

    /// `args[n] = v` (a `[[Set]]` whose receiver is `args`) on an `arguments`
    /// object whose parameter cells are attached (see
    /// [`attach_argument_cells`](Self::attach_argument_cells)) and that still
    /// has its layout, for an index below the argument count that is a
    /// writable data property: writes the own property and, while the index
    /// is mapped, the parameter it aliases. `false`, having written nothing,
    /// otherwise (the caller takes the general path).
    pub fn arguments_own_set(&mut self, handle: Handle, n: f64, v: NanBox) -> bool {
        let Some(l) = self.arguments_layout(handle) else {
            return false;
        };
        let Some(i) = arg_index(n, l.argc) else {
            return false;
        };
        let name = INDEX_KEYS[i];
        if l.cells.is_none()
            || l.obj.is_frozen()
            || l.obj.is_readonly(name)
            || l.obj.accessor(name).is_some()
        {
            return false;
        }
        let cell = l.cell(i);
        if let Some(Cell::Object(o)) = self.heap.get_mut(handle) {
            o.set(name, v);
        }
        self.write_barrier(handle, v);
        if let Some(c) = cell {
            self.set_element(c, 0, v);
        }
        true
    }

    /// `CreateListFromArrayLike(args)` for an `arguments` object made by
    /// [`new_arguments_object`](Self::new_arguments_object) that still has its
    /// layout, its `length` still the argument count and no index or `length`
    /// redefined as an accessor: the arguments, read from their slots, or from
    /// the parameter cells of mapped indices (no `[[Get]]` here can run code
    /// or reach the prototype). `None` otherwise.
    #[must_use]
    pub fn arguments_values(&self, handle: Handle) -> Option<alloc::vec::Vec<NanBox>> {
        let l = self.arguments_layout(handle)?;
        let o = l.obj;
        if o.accessor("length").is_some() || o.slot(0)?.as_number() != Some(l.argc as f64) {
            return None;
        }
        (0..l.argc)
            .map(|i| {
                if let Some(c) = l.cell(i) {
                    return self.cell_get(c);
                }
                if o.accessor(INDEX_KEYS[i]).is_some() {
                    return None;
                }
                o.slot((l.hidden + i) as u32)
            })
            .collect()
    }

    /// CreateMappedArgumentsObject's `[[ParameterMap]]` for a bytecode-VM
    /// function: `cells` pairs each parameter index with that parameter's
    /// binding cell; an index is mapped when an argument was passed for it.
    /// The map is kept on the object itself, as the hidden [`ARGS_CELLS`]
    /// property holding an array of the cells by index (`undefined` for an
    /// unmapped one), so it lives exactly as long as the object and the
    /// collector traces it as it does any property. Returns whether any index
    /// is mapped (nothing is attached otherwise: the object then is the
    /// ordinary object it behaves as).
    pub fn attach_argument_cells(&mut self, handle: Handle, cells: &[(usize, NanBox)]) -> bool {
        let argc = self
            .get_property(handle, "length")
            .and_then(|l| l.as_number())
            .unwrap_or(0.0) as usize;
        let mut map: alloc::vec::Vec<NanBox> = alloc::vec::Vec::new();
        for &(i, c) in cells {
            if i < argc && c.as_handle().is_some() {
                if map.len() <= i {
                    map.resize(i + 1, NanBox::undefined());
                }
                map[i] = c;
            }
        }
        if map.is_empty() {
            return false;
        }
        // The layout with the map attached, kept (so the transition the
        // property add takes finds it again) while the mapped one is.
        if let Some(argc) = self.arguments_layout(handle).map(|l| l.argc) {
            self.arguments_cells_shape(argc);
        }
        let arr = self.new_array(map);
        self.set_hidden_property(handle, ARGS_CELLS, NanBox::handle(arr.to_raw()));
        self.args_cells_attached = true;
        true
    }

    /// Marks the object at `handle` as an `arguments` object whose
    /// parameter map the host keeps (see
    /// [`has_argument_map_marker`](Self::has_argument_map_marker)).
    pub fn mark_host_argument_map(&mut self, handle: Handle) {
        self.set_hidden_property(handle, ARGS_CELLS, NanBox::boolean(true));
        self.args_cells_attached = true;
    }

    /// Whether the object at `handle` has VM parameter cells attached (see
    /// [`attach_argument_cells`](Self::attach_argument_cells)), or the
    /// marker of a parameter map the host keeps.
    #[must_use]
    pub fn has_argument_map_marker(&self, handle: Handle) -> bool {
        self.args_cells_attached && self.get_property(handle, ARGS_CELLS).is_some()
    }

    /// The parameter cell index `i` of the mapped `arguments` object at
    /// `handle` aliases (see
    /// [`attach_argument_cells`](Self::attach_argument_cells)), while it is
    /// mapped.
    #[must_use]
    pub fn arguments_cell(&self, handle: Handle, i: usize) -> Option<Handle> {
        let map = self.get_property(handle, ARGS_CELLS)?.as_handle()?;
        let e = self.array_elements(Handle::from_raw(map))?;
        e.get(i)?.as_handle().map(Handle::from_raw)
    }

    /// Unmaps index `i` of the mapped `arguments` object at `handle` (the
    /// spec's `map.[[Delete]](P)`).
    pub fn arguments_unmap(&mut self, handle: Handle, i: usize) {
        if self.arguments_cell(handle, i).is_some()
            && let Some(map) = self
                .get_property(handle, ARGS_CELLS)
                .and_then(NanBox::as_handle)
        {
            self.set_element(Handle::from_raw(map), i, NanBox::undefined());
        }
    }

    /// The object at `handle` as an `arguments` object with one of the kept
    /// layouts that no host hook intercepts: its argument count, the slot of
    /// its first index and, when its parameter cells are attached, those.
    fn arguments_layout(&self, handle: Handle) -> Option<ArgsLayout<'_>> {
        if self.args_shapes.is_empty() {
            return None;
        }
        let Some(Cell::Object(o)) = self.heap.get(handle) else {
            return None;
        };
        let shape = o.shaped_shape()?;
        let len = shape.len() as usize;
        // The mapped layout has 4 leading keys, the unmapped one 3; the mapped
        // one with its cells attached has one more key after the indices.
        let (argc, hidden, kind) = if let Some(a) = len.checked_sub(5)
            && self.is_arguments_shape(shape, a, 2)
        {
            (a, 4, 2)
        } else if let Some(a) = len.checked_sub(4)
            && self.is_arguments_shape(shape, a, 1)
        {
            (a, 4, 1)
        } else if let Some(a) = len.checked_sub(3)
            && self.is_arguments_shape(shape, a, 0)
        {
            (a, 3, 0)
        } else {
            return None;
        };
        if self.is_host_exotic(handle) {
            return None;
        }
        let cells = if kind == 2 {
            // (An array this module made; a `true` there instead marks the
            // host's own parameter map, which only the host reads.)
            let m = o.slot((hidden + argc) as u32)?.as_handle()?;
            Some(self.array_elements(Handle::from_raw(m))?)
        } else {
            None
        };
        Some(ArgsLayout {
            obj: o,
            argc,
            hidden,
            cells,
        })
    }

    /// Whether `shape` is the kept `arguments` layout of `kind` (see
    /// [`ARGS_LAYOUT_KINDS`]) for `argc` arguments.
    fn is_arguments_shape(&self, shape: &Rc<Shape>, argc: usize, kind: usize) -> bool {
        matches!(
            self.args_shapes.get(ARGS_LAYOUT_KINDS * argc + kind),
            Some(Some(s)) if Rc::ptr_eq(s, shape)
        )
    }

    /// The layout of an `arguments` object of `argc` arguments (see the
    /// module docs), made once and kept.
    fn arguments_shape(&mut self, argc: usize, mapped: bool) -> Option<Rc<Shape>> {
        let i = ARGS_LAYOUT_KINDS * argc + usize::from(mapped);
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
        self.keep_args_shape(i, &s);
        Some(s)
    }

    /// The mapped layout of `argc` arguments with the parameter cells
    /// attached, made once and kept.
    fn arguments_cells_shape(&mut self, argc: usize) {
        let i = ARGS_LAYOUT_KINDS * argc + 2;
        if matches!(self.args_shapes.get(i), Some(Some(_))) {
            return;
        }
        if let Some(s) = self.arguments_shape(argc, true) {
            let s = s.add(ARGS_CELLS);
            self.keep_args_shape(i, &s);
        }
    }

    fn keep_args_shape(&mut self, i: usize, s: &Rc<Shape>) {
        if self.args_shapes.len() <= i {
            self.args_shapes.resize(i + 1, None);
        }
        self.args_shapes[i] = Some(Rc::clone(s));
    }
}

/// The kept layouts per argument count: unmapped, mapped, and mapped with
/// the parameter cells attached.
const ARGS_LAYOUT_KINDS: usize = 3;

/// The hidden property of a mapped `arguments` object holding its parameter
/// map (see [`Realm::attach_argument_cells`]).
pub const ARGS_CELLS: &str = "\0argsmapped";

/// An `arguments` object in one of the kept layouts (see
/// `Realm::arguments_layout`).
struct ArgsLayout<'a> {
    obj: &'a crate::object::Object,
    argc: usize,
    /// The slot of index 0.
    hidden: usize,
    /// The parameter cells by index, when attached.
    cells: Option<&'a [NanBox]>,
}

impl ArgsLayout<'_> {
    /// The parameter cell index `i` is mapped to, if any.
    fn cell(&self, i: usize) -> Option<Handle> {
        self.cells?.get(i)?.as_handle().map(Handle::from_raw)
    }
}

/// `n` as an argument index below `argc`.
fn arg_index(n: f64, argc: usize) -> Option<usize> {
    if !(0.0..ARGS_FAST_MAX as f64).contains(&n) || n != (n as usize) as f64 {
        return None;
    }
    let i = n as usize;
    (i < argc).then_some(i)
}
