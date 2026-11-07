//! `Object.assign(target, source)` between plain objects, without the
//! generic per-key steps.
//!
//! The spec copies each own key of `source` with `[[GetOwnProperty]]` (to
//! skip non-enumerable ones), `Get` and `Set(target, key, value, true)`. The
//! general path (`Interp`'s `N_OBJECT_ASSIGN`) builds a descriptor object per
//! key just to read its `enumerable` back and runs the full member write,
//! about 30,000 instructions for a four-key object.
//!
//! When `source` is an ordinary object whose own properties are exactly its
//! shape's enumerable data keys ([`Object::keys_follow_shape`]), none an
//! engine-internal key but a Symbol, every `[[GetOwnProperty]]` and `Get` runs
//! no code and yields the slot's value. When `target` is an ordinary
//! extensible object holding only plain writable data properties
//! ([`Object::is_plain_shaped`]) and no object on its prototype chain owns
//! any of those keys, every `Set` is an ordinary write or add of a data
//! property that no setter, read-only property or proxy trap observes. The
//! copy is then exactly the stores themselves, and the order of the stores is
//! unobservable (the target's key order is the order the keys are added in,
//! the source's own order, which `[[OwnPropertyKeys]]` re-sorts the same way
//! for both). An empty target simply takes the source's layout and values.
//!
//! The chain proof is remembered per (source layout, target prototype) while
//! the prototype epoch it was shown at is current: every link is watched, so
//! any change to a link's keys or prototype moves the epoch.
//!
//! [`Object::keys_follow_shape`]: crate::object::Object::keys_follow_shape
//! [`Object::is_plain_shaped`]: crate::object::Object::is_plain_shaped

use super::{Cell, Handle, NanBox, Realm};
use crate::shape::Shape;
use alloc::rc::Rc;
use alloc::vec::Vec;

/// How many (source layout, target prototype) proofs are remembered.
const ASSIGN_WAYS: usize = 4;

/// A remembered proof that the prototype chain from `proto` owns none of the
/// keys of `shape` (whose keys are all copyable), valid at prototype epoch
/// `stamp`.
struct AssignProof {
    shape: Rc<Shape>,
    proto: u64,
    stamp: u64,
}

/// The proofs `Object.assign`'s plain path keeps from one call to the next.
#[derive(Default)]
pub(crate) struct AssignMemo {
    ways: [Option<AssignProof>; ASSIGN_WAYS],
    next: usize,
}

impl Realm {
    /// One source step of `Object.assign(target, …, src, …)` when it is a
    /// plain copy (see the module docs): performed, returning `true`.
    /// `false`, having changed nothing, for any other pair. The caller rules
    /// out a mapped `arguments` object on either side.
    pub(crate) fn assign_plain(
        &mut self,
        target: Handle,
        src: Handle,
        memo: &mut AssignMemo,
    ) -> bool {
        let Some(Cell::Object(s)) = self.heap.get(src) else {
            return false;
        };
        if !s.keys_follow_shape() || s.is_prim_wrapper() {
            return false;
        }
        let Some(shape) = s.shaped_shape() else {
            return false;
        };
        let n = shape.len() as usize;
        let mut vals: Vec<NanBox> = Vec::with_capacity(n);
        for i in 0..n as u32 {
            match s.slot(i) {
                Some(v) => vals.push(v),
                None => return false,
            }
        }
        let shape = Rc::clone(shape);
        let Some(Cell::Object(t)) = self.heap.get(target) else {
            return false;
        };
        if !t.is_plain_shaped() || t.is_prim_wrapper() {
            return false;
        }
        let proto = t.proto();
        let target_empty = t.is_empty();
        if self.is_host_exotic(src)
            || self.is_host_exotic(target)
            || self.is_function_proto_intrinsic(src)
            || self.is_function_proto_intrinsic(target)
        {
            return false;
        }
        if n == 0 {
            return true;
        }
        if !self.assign_chain_proven(&shape, proto, memo) {
            return false;
        }
        if target_empty && n < self.limits.object_dictionary_threshold {
            let Some(Cell::Object(t)) = self.heap.get_mut(target) else {
                return false;
            };
            if t.adopt_data(Rc::clone(&shape), &vals) {
                for v in &vals {
                    self.write_barrier(target, *v);
                }
                return true;
            }
        }
        for (k, v) in shape.keys().into_iter().zip(vals) {
            self.set_property(target, k, v);
        }
        true
    }

    /// Whether no object on the prototype chain from `proto` owns a key of
    /// `shape` and every key of `shape` is a copyable one (a property name or
    /// a Symbol, no other engine-internal key): from `memo` while its proof
    /// is current, else shown now (watching every link) and remembered.
    fn assign_chain_proven(
        &mut self,
        shape: &Rc<Shape>,
        proto: Option<Handle>,
        memo: &mut AssignMemo,
    ) -> bool {
        let raw = proto.map_or(u64::MAX, Handle::to_raw);
        let epoch = crate::object::proto_epoch();
        if memo
            .ways
            .iter()
            .flatten()
            .any(|p| p.stamp == epoch && p.proto == raw && Rc::ptr_eq(&p.shape, shape))
        {
            return true;
        }
        let keys = shape.keys();
        if keys.iter().any(|k| {
            crate::realm::is_internal_key(k) && !k.starts_with(crate::shape::SYMBOL_KEY_PREFIX)
        }) {
            return false;
        }
        let mut links: [(u64, Option<u64>); 8] = [(0, None); 8];
        let mut len = 0;
        let mut cur = proto;
        while let Some(h) = cur {
            if len == links.len() {
                return false;
            }
            let Some((aux, _, next)) = self.key_link(h) else {
                return false;
            };
            if keys.iter().any(|k| self.has_own(h, k)) {
                return false;
            }
            links[len] = (h.to_raw(), aux);
            len += 1;
            cur = next;
        }
        for &(r, aux) in &links[..len] {
            self.watch_link(r, aux);
        }
        if !links[..len]
            .iter()
            .all(|&(r, aux)| self.link_watched(r, aux))
        {
            return false;
        }
        let way = memo.next % ASSIGN_WAYS;
        memo.next = memo.next.wrapping_add(1);
        memo.ways[way] = Some(AssignProof {
            shape: Rc::clone(shape),
            proto: raw,
            stamp: crate::object::proto_epoch(),
        });
        true
    }
}
