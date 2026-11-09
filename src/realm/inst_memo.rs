//! The `instanceof` right-hand side memo: `x instanceof F` for a VM closure
//! `F` with the default `[[HasInstance]]` reads `F.prototype` (an own data
//! property of `F`'s auxiliary object) on every evaluation — a by-name
//! lookup, after a by-name test that `F` owns no `@@hasInstance`. A few
//! recently asked closures are remembered here with their auxiliary object,
//! its shape and the `prototype` slot: while the auxiliary object keeps that
//! shape and has no accessors, its own keys are the ones seen then (so still
//! no `@@hasInstance`, and `prototype` a data property in that slot), and
//! the slot is read afresh (a write to the property does not change the
//! shape). `F`'s own `[[Prototype]]` is checked each time.

use super::{Cell, Handle, NanBox, Realm};
use crate::shape::Shape;
use alloc::rc::Rc;

/// How many closures [`InstMemo`] remembers (direct-mapped by handle).
const WAYS: usize = 4;

/// One remembered closure (see the module docs).
#[derive(Default)]
struct Way {
    /// The closure's handle bits (`0`: empty).
    f: u64,
    /// Its auxiliary object's handle bits.
    aux: u64,
    /// The auxiliary object's shape then (kept alive, so its address
    /// identifies it).
    shape: Option<Rc<Shape>>,
    /// The `prototype` slot under that shape.
    slot: u32,
}

/// See the module docs.
#[derive(Default)]
pub(crate) struct InstMemo {
    ways: [Way; WAYS],
}

impl Realm {
    /// For `x instanceof f`: when `f` is a VM closure whose
    /// `[[HasInstance]]` is the default (`inherits_default_has_instance`),
    /// its own `prototype` value (`None` if it has none), with an owed
    /// `prototype` materialized first; `None` for any other `f`.
    pub(crate) fn instance_of_target(&mut self, f: Handle) -> Option<Option<NanBox>> {
        let way = (f.to_raw() as usize) % WAYS;
        if let Some(v) = self.inst_memo_hit(f, way) {
            return Some(Some(v));
        }
        if !self.inherits_default_has_instance(f) {
            return None;
        }
        self.materialize_fn_proto(f);
        let p = self.get_property(f, "prototype");
        self.inst_memo_fill(f, way);
        Some(p)
    }

    /// The remembered `prototype` of `f` in `way`, when the memo still
    /// proves it (see the module docs).
    #[inline]
    fn inst_memo_hit(&self, f: Handle, way: usize) -> Option<NanBox> {
        let w = &self.inst_memo.ways[way];
        if w.f != f.to_raw() || self.multi_realm || !self.is_vm_function(f) {
            return None;
        }
        let aux = *self.aux_props.get(&f.to_raw())?;
        if aux.to_raw() != w.aux
            || !self
                .object_proto(f)
                .is_some_and(|p| self.is_function_proto_intrinsic(p))
        {
            return None;
        }
        match self.heap.get(aux) {
            Some(Cell::Object(o))
                if o.has_no_accessors()
                    && o.shaped_shape()
                        .zip(w.shape.as_ref())
                        .is_some_and(|(s, m)| Rc::ptr_eq(s, m)) =>
            {
                o.slot(w.slot)
            }
            _ => None,
        }
    }

    /// After `f` was found to have the default `[[HasInstance]]` and its
    /// `prototype` materialized: remembers it in `way` when its auxiliary
    /// object is shaped, has no accessors and holds `prototype` as data.
    fn inst_memo_fill(&mut self, f: Handle, way: usize) {
        let Some(aux) = self.aux_props.get(&f.to_raw()).copied() else {
            return;
        };
        let found = match self.heap.get(aux) {
            Some(Cell::Object(o)) if o.has_no_accessors() => o
                .shaped_shape()
                .and_then(|s| Some((Rc::clone(s), s.lookup("prototype")?))),
            _ => None,
        };
        self.inst_memo.ways[way] = match found {
            Some((shape, slot)) => Way {
                f: f.to_raw(),
                aux: aux.to_raw(),
                shape: Some(shape),
                slot,
            },
            None => Way::default(),
        };
    }
}
