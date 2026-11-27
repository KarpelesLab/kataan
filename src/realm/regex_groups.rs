//! A match's named-groups object (`groups`, `indices.groups`, a replacer's
//! last argument) built in one step once its layout is known.
//!
//! Each such object is a fresh null-prototype ordinary object whose keys are
//! the RegExp's group names in source order, each a plain writable,
//! enumerable, configurable data property — the same layout every time for
//! a given RegExp. The first object built for a RegExp is made key by key
//! (`CreateDataProperty` per name), and its layout is remembered here against
//! the compiled regex; later ones are allocated straight into that layout
//! with their slots filled, which leaves exactly the object the key-by-key
//! build would.

use super::{Handle, NanBox, Rc, Realm, Shape};

/// How many compiled regexes' group layouts are remembered (most recent
/// first).
const GROUPS_SHAPES_MAX: usize = 8;

/// A remembered group layout: the compiled regex (held, so its identity
/// cannot be reused), the realm's root shape it grew from, and the shape.
pub(crate) type GroupsShape = (Rc<crate::regex::Regex>, Rc<Shape>, Rc<Shape>);

impl Realm {
    /// The named-groups object of a match of `re`: `undefined` without named
    /// groups, else a null-prototype object mapping each name to `vals[i]`,
    /// `i` its group index (`vals` indexed by group, one past the last named
    /// group at least). Of duplicate names (in disjoint alternatives) the
    /// participating occurrence's value wins at the first occurrence's key
    /// position: `participated(i)` says whether group `i` did.
    pub(crate) fn named_groups_object(
        &mut self,
        re: &Rc<crate::regex::Regex>,
        vals: &[NanBox],
        participated: impl Fn(usize) -> bool,
    ) -> NanBox {
        let names = re.group_names();
        if names.is_empty() {
            return NanBox::undefined();
        }
        let val = |i: usize| vals.get(i).copied().unwrap_or(NanBox::undefined());
        // A remembered layout only exists for distinct names, so the values
        // go in name order.
        if let Some(pos) = self.groups_shapes.iter().position(|e| Rc::ptr_eq(&e.0, re))
            && names.len() <= 16
        {
            let mut buf = [NanBox::undefined(); 16];
            for (slot, (idx, _)) in buf.iter_mut().zip(names) {
                *slot = val(*idx);
            }
            let (_, root, shape) = &self.groups_shapes[pos];
            let (root, shape) = (Rc::clone(root), Rc::clone(shape));
            if let Some(g) =
                self.new_object_in_shape_proto(&root, &shape, &buf[..names.len()], None)
            {
                if pos != 0 {
                    self.groups_shapes[..=pos].rotate_right(1);
                }
                return NanBox::handle(g.to_raw());
            }
        }
        let g: Handle = self.new_object_with_proto(None);
        let mut distinct = true;
        for (idx, name) in names {
            let v = val(*idx);
            if self.get_property(g, name).is_none() {
                self.set_property(g, name, v);
            } else {
                distinct = false;
                if participated(*idx) {
                    self.set_property(g, name, v);
                }
            }
        }
        if distinct && names.len() <= 16 {
            let keys: alloc::vec::Vec<&str> = names.iter().map(|(_, n)| n.as_str()).collect();
            if let Some(shape) = self.plain_shape_with_keys(g, &keys, None) {
                if self.groups_shapes.len() >= GROUPS_SHAPES_MAX {
                    self.groups_shapes.pop();
                }
                let root = self.root_shape_rc();
                self.groups_shapes.insert(0, (Rc::clone(re), root, shape));
            }
        }
        NanBox::handle(g.to_raw())
    }
}
