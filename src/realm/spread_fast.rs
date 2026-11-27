//! The own enumerable data properties of a plain object, read in one pass.
//!
//! Object spread (`{ ...src, k: v }`) and object rest run
//! `CopyDataProperties`, whose general path (`Interp::copy_data_properties`)
//! asks `[[GetOwnProperty]]` for each key by building a fresh descriptor
//! object (four property writes) only to read its `enumerable` back, and then
//! reads the value through the full member read: about 16,000 instructions
//! to spread a three-property object, so an immutable-update reducer
//! (`state = { ...state, n: state.n + 1 }`) spent nearly all its time there.
//!
//! For an ordinary object cell with no accessor properties and no internal
//! (`\0`-prefixed) keys — those mark wrappers, `arguments` objects and other
//! exotic behaviour, and hold the symbol-keyed properties — every own key is a
//! plain data property: its `[[GetOwnProperty]]` runs no code, its
//! `[[Enumerable]]` is the slot's own flag and its value is the slot's value.
//! [`Realm::plain_data_entries`] reads exactly those, in `[[OwnPropertyKeys]]`
//! order (integer keys ascending, then insertion order); anything else answers
//! `None` and the caller takes the general path.
//!
//! Copying key by key is still O(n^2) for an n-key source: each add looks
//! its key up in the target's growing layout. When the target is empty (an
//! object literal that starts with the spread, or a rest pattern's object)
//! and the source's own properties are all enumerable data properties under
//! copyable keys (names or Symbols), [`Realm::spread_plain`] gives the target
//! the source's layout and values in one step (a dictionary-mode source: a
//! copy of its dictionary, sharing the keys), leaving out a rest pattern's
//! excluded keys. The target's key order is then the source's, which
//! `[[OwnPropertyKeys]]` re-sorts the same way for both. `Object.values` of
//! such an object reads its slots the same way
//! ([`Realm::plain_enum_values`]).

use super::{Cell, Handle, NanBox, Realm};
use crate::shape::Shape;
use alloc::rc::Rc;
use alloc::string::String;
use alloc::vec::Vec;

/// A remembered merge of a shaped source into a shaped target (see
/// [`Realm::merge_shaped_planned`]): the two layouts (held, so their
/// addresses are not reused), the layout the merge leaves, and per source
/// slot the result slot it lands in.
pub(crate) struct MergePlan {
    target: Rc<Shape>,
    source: Rc<Shape>,
    result: Rc<Shape>,
    map: Rc<[u32]>,
}

/// How many merge plans a realm remembers (a power of two).
const MERGE_PLANS: usize = 16;

/// A remembered object rest of a shaped source (see
/// [`Realm::rest_planned`]): the source layout (held, so its address is not
/// reused), the excluded keys, the layout of the copy (the source's minus
/// those keys) and the source slots it keeps, in order.
pub(crate) struct RestPlan {
    source: Rc<Shape>,
    excluded: alloc::boxed::Box<[String]>,
    result: Rc<Shape>,
    keep: Rc<[u32]>,
}

/// How many rest plans a realm remembers (a power of two).
const REST_PLANS: usize = 8;

impl Realm {
    /// `CopyDataProperties(target, src, excluded)` when it is a wholesale
    /// copy (see the module docs): `target` an empty ordinary object (an
    /// object literal's or a rest pattern's own fresh object) and `src` an
    /// ordinary object whose own properties are all enumerable data
    /// properties. `target` then takes `src`'s layout (or a copy of its
    /// dictionary) and values in one step, instead of re-adding the keys one
    /// at a time: each add looks its key up in the growing layout first, so
    /// a key-by-key copy of an n-key object was O(n^2). Performed, returning
    /// `true`; `false`, having changed nothing, for any other pair.
    pub(crate) fn spread_plain(
        &mut self,
        target: Handle,
        src: Handle,
        excluded: &[String],
    ) -> bool {
        if self.is_host_exotic(src) || self.is_function_proto_intrinsic(src) {
            return false;
        }
        if excluded.is_empty() && self.merge_shaped_planned(target, src) {
            return true;
        }
        let copy = match self.rest_planned(src, excluded) {
            Some(c) => c,
            None => {
                let Some(Cell::Object(s)) = self.heap.get(src) else {
                    return false;
                };
                let Some(copy) = s.plain_copy(excluded) else {
                    return false;
                };
                if !excluded.is_empty() {
                    self.rest_plan_record(src, excluded, &copy);
                }
                copy
            }
        };
        if self.is_host_exotic(target) {
            return false;
        }
        let threshold = self.limits.object_dictionary_threshold;
        let barrier = self.barrier_needed(target);
        let Some(Cell::Object(t)) = self.heap.get_mut(target) else {
            return false;
        };
        if copy.is_empty() {
            return true;
        }
        let mut vals: Vec<NanBox> = Vec::new();
        if barrier {
            copy.for_each_value(|v| {
                if v.as_handle().is_some() {
                    vals.push(v);
                }
            });
        }
        match t.adopt_plain(copy, threshold) {
            Ok(()) => {
                for v in vals {
                    self.write_barrier(target, v);
                }
                true
            }
            // A target that already has keys (`{ k: v, ...src }`,
            // `{ ...a, ...b }`): merged in one pass. (Spread defines, so the
            // target's prototype chain does not matter.)
            Err(copy) => {
                !self.is_function_proto_intrinsic(target)
                    && self.merge_plain_into(target, copy).is_ok()
            }
        }
    }

    /// `Object::plain_copy(excluded)` of the shaped object `src` (an object
    /// rest's) by a remembered [`RestPlan`] for its layout and those keys:
    /// the copy's layout and the kept slots' values, without walking the
    /// keys or building the layout again. `None` when no plan matches (or
    /// `src` is not a plain shaped object), for the caller's own copy.
    fn rest_planned(&self, src: Handle, excluded: &[String]) -> Option<crate::object::PlainCopy> {
        if excluded.is_empty() || self.rest_plans.is_empty() {
            return None;
        }
        let Some(Cell::Object(s)) = self.heap.get(src) else {
            return None;
        };
        let shape = s.plain_shaped_shape()?;
        let p = self.rest_plans.get(rest_way(shape, excluded))?.as_ref()?;
        if !Rc::ptr_eq(&p.source, shape) || *p.excluded != *excluded {
            return None;
        }
        let values = p
            .keep
            .iter()
            .map(|&i| s.slot(i))
            .collect::<Option<Vec<_>>>()?;
        Some(crate::object::PlainCopy::shaped(
            Rc::clone(&p.result),
            values,
        ))
    }

    /// Remembers the rest `copy` of the shaped object `src` without the keys
    /// `excluded` as a [`RestPlan`] (when both are shaped and the copy is the
    /// source's layout minus those keys, as `Object::plain_copy` makes it).
    fn rest_plan_record(
        &mut self,
        src: Handle,
        excluded: &[String],
        copy: &crate::object::PlainCopy,
    ) {
        let Some(result) = copy.shape() else {
            return;
        };
        let Some(Cell::Object(s)) = self.heap.get(src) else {
            return;
        };
        let Some(shape) = s.plain_shaped_shape() else {
            return;
        };
        let keep: Vec<u32> = shape
            .keys()
            .iter()
            .enumerate()
            .filter(|(_, k)| !excluded.iter().any(|e| e == *k))
            .map(|(i, _)| i as u32)
            .collect();
        if keep.len() != copy.len() {
            return;
        }
        let way = rest_way(shape, excluded);
        let plan = RestPlan {
            source: Rc::clone(shape),
            excluded: excluded.into(),
            result: Rc::clone(result),
            keep: keep.into(),
        };
        if self.rest_plans.is_empty() {
            self.rest_plans.resize_with(REST_PLANS, || None);
        }
        self.rest_plans[way] = Some(plan);
    }

    /// [`Object::merge_plain`](crate::object::Object::merge_plain) of the
    /// shaped object `src` into the non-empty shaped object `target`, by a
    /// plan remembered per pair of layouts: which slot of the result each
    /// source slot lands in, and the result's layout (so the keys are looked
    /// up once per pair, not once per merge). Performed, returning `true`;
    /// `false`, having changed nothing, for any other pair (the caller's
    /// general merge then decides). The caller has ruled out a host-exotic
    /// source and the `Function.prototype` intrinsic.
    pub(crate) fn merge_shaped_planned(&mut self, target: Handle, src: Handle) -> bool {
        if target == src || self.is_host_exotic(target) || self.is_function_proto_intrinsic(target)
        {
            return false;
        }
        let Some(Cell::Object(s)) = self.heap.get(src) else {
            return false;
        };
        if s.is_prim_wrapper() {
            return false;
        }
        let Some((sshape, values)) = s.plain_shaped_values() else {
            return false;
        };
        if values.is_empty() {
            return false;
        }
        let sshape = Rc::clone(sshape);
        let Some(Cell::Object(t)) = self.heap.get(target) else {
            return false;
        };
        if t.is_empty() || t.is_prim_wrapper() || !t.merge_target_ok() {
            return false;
        }
        let Some(tshape) = t.shaped_shape() else {
            return false;
        };
        let way = ((Rc::as_ptr(tshape) as usize >> 4)
            ^ (Rc::as_ptr(&sshape) as usize >> 6)
            ^ (Rc::as_ptr(&sshape) as usize >> 12))
            & (MERGE_PLANS - 1);
        let plan = match self.merge_plans.get(way) {
            Some(Some(p)) if Rc::ptr_eq(&p.target, tshape) && Rc::ptr_eq(&p.source, &sshape) => {
                (Rc::clone(&p.result), Rc::clone(&p.map))
            }
            _ => {
                let tshape = Rc::clone(tshape);
                let Some(p) = self.merge_plan(&tshape, &sshape) else {
                    return false;
                };
                if self.merge_plans.is_empty() {
                    self.merge_plans.resize_with(MERGE_PLANS, || None);
                }
                let out = (Rc::clone(&p.result), Rc::clone(&p.map));
                self.merge_plans[way] = Some(p);
                out
            }
        };
        let Some(Cell::Object(t)) = self.heap.get_mut(target) else {
            return false;
        };
        t.merge_planned(&plan.0, &plan.1, &values);
        if self.barrier_needed(target) {
            for v in values {
                if v.as_handle().is_some() {
                    self.write_barrier(target, v);
                }
            }
        }
        true
    }

    /// A [`MergePlan`] for merging objects of layout `source` into objects
    /// of layout `target`, when every source key is copyable (a property
    /// name or a Symbol) and the result stays below the dictionary
    /// threshold.
    fn merge_plan(&self, target: &Rc<Shape>, source: &Rc<Shape>) -> Option<MergePlan> {
        let mut result = Rc::clone(target);
        let mut map = Vec::with_capacity(source.len() as usize);
        for k in source.keys() {
            if k.as_bytes().first() == Some(&0) && !k.starts_with(crate::shape::SYMBOL_KEY_PREFIX) {
                return None;
            }
            match result.lookup(k) {
                Some(slot) => map.push(slot),
                None => {
                    map.push(result.len());
                    result = result.add(k);
                }
            }
        }
        if result.len() as usize >= self.limits.object_dictionary_threshold {
            return None;
        }
        Some(MergePlan {
            target: Rc::clone(target),
            source: Rc::clone(source),
            result,
            map: map.into(),
        })
    }

    /// [`Object::merge_plain`](crate::object::Object::merge_plain) of `copy`
    /// into the object at `target`, with the write barrier for each value
    /// stored: `false`, having changed nothing, when `target` is not a plain
    /// ordinary object (see there), a primitive wrapper or host-exotic.
    pub(crate) fn merge_plain_into(
        &mut self,
        target: Handle,
        copy: crate::object::PlainCopy,
    ) -> Result<(), crate::object::PlainCopy> {
        if self.is_host_exotic(target) {
            return Err(copy);
        }
        let threshold = self.limits.object_dictionary_threshold;
        let mut vals: Vec<NanBox> = Vec::new();
        if self.barrier_needed(target) {
            copy.for_each_value(|v| {
                if v.as_handle().is_some() {
                    vals.push(v);
                }
            });
        }
        let Some(Cell::Object(t)) = self.heap.get_mut(target) else {
            return Err(copy);
        };
        if t.is_prim_wrapper() {
            return Err(copy);
        }
        t.merge_plain(copy, threshold)?;
        for v in vals {
            self.write_barrier(target, v);
        }
        Ok(())
    }

    /// `Object.values(h)` when `h` is an ordinary object no step of whose
    /// `EnumerableOwnProperties` can run code (see
    /// [`Object::plain_enum_values`](crate::object::Object::plain_enum_values)).
    #[must_use]
    pub(crate) fn plain_enum_values(&self, h: Handle) -> Option<Vec<NanBox>> {
        if self.is_function_proto_intrinsic(h) {
            return None;
        }
        match self.heap.get(h)? {
            Cell::Object(o) => o.plain_enum_values(),
            _ => None,
        }
    }

    /// The own enumerable `(key, value)` pairs of `h`, in `[[OwnPropertyKeys]]`
    /// order, when `h` is an ordinary object whose own properties are all
    /// plain string-keyed data properties (see the module docs).
    pub(crate) fn plain_data_entries(&self, h: Handle) -> Option<Vec<(String, NanBox)>> {
        let Some(Cell::Object(o)) = self.heap.get(h) else {
            return None;
        };
        if !o.accessor_keys().is_empty() {
            return None;
        }
        let keys = o.ordered_keys();
        let mut out = Vec::with_capacity(keys.len());
        for k in keys {
            if k.as_bytes().first() == Some(&0) {
                return None;
            }
            if o.is_hidden(k) {
                continue;
            }
            out.push((String::from(k), o.get(k)?));
        }
        Some(out)
    }
}

/// A rest plan's way in the realm's table: by the source layout's address
/// and the first excluded key's length.
fn rest_way(shape: &Rc<Shape>, excluded: &[String]) -> usize {
    ((Rc::as_ptr(shape) as usize >> 4) ^ excluded.first().map_or(0, String::len)) & (REST_PLANS - 1)
}
