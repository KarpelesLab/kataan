//! The object-model context that ties the foundation together (`ROADMAP.md`
//! §3).
//!
//! [`nanbox`](crate::nanbox), [`shape`](crate::shape), [`object`](crate::object),
//! [`heap`](crate::heap), [`atom`](crate::atom), and [`gc`](crate::gc) are
//! independent pieces; a running engine needs them bundled behind one handle. A
//! `Realm` owns:
//! - the managed [`Heap`] of objects,
//! - the shared **root [`Shape`]** every object's layout descends from (so
//!   identically-structured objects share hidden classes across the realm), and
//! - the realm's [`AtomTable`] for interned property keys.
//!
//! It exposes the operations a VM performs on the heap — allocate an object, get
//! and set properties (as [`NanBox`] values addressed by [`Handle`]), and run a
//! collection from a root set — so the eventual VM migration targets this one
//! API rather than wiring the pieces together itself.
//!
//! Pure, safe `alloc`-only Rust.
//!
//! [`Heap`]: crate::heap::Heap
//! [`Shape`]: crate::shape::Shape
//! [`AtomTable`]: crate::atom::AtomTable
//! [`NanBox`]: crate::nanbox::NanBox
//! [`Handle`]: crate::heap::Handle

use crate::atom::{Atom, AtomTable};
use crate::cell::Cell;
use crate::gc::{self, Stats};
use crate::heap::{Handle, Heap};
use crate::nanbox::NanBox;
use crate::object::Object;
use crate::rope::Rope;
use crate::shape::Shape;
use alloc::rc::Rc;
use alloc::vec::Vec;

/// An object-model context: the heap, the shared root shape, and the atom table.
pub struct Realm {
    heap: Heap<Cell>,
    root_shape: Rc<Shape>,
    atoms: AtomTable,
}

impl Default for Realm {
    fn default() -> Self {
        Self::new()
    }
}

impl Realm {
    /// Creates an empty realm.
    #[must_use]
    pub fn new() -> Self {
        Self {
            heap: Heap::new(),
            root_shape: Shape::root(),
            atoms: AtomTable::new(),
        }
    }

    /// Allocates a fresh empty object in the heap and returns its handle.
    pub fn new_object(&mut self) -> Handle {
        let obj = Object::new(Rc::clone(&self.root_shape));
        self.heap.alloc(Cell::Object(obj))
    }

    /// Allocates a string value in the heap and returns its handle.
    pub fn new_string(&mut self, s: &str) -> Handle {
        self.heap.alloc(Cell::Str(Rope::from(s)))
    }

    /// Allocates an array of `elements` in the heap and returns its handle.
    pub fn new_array(&mut self, elements: Vec<NanBox>) -> Handle {
        self.heap.alloc(Cell::Array(elements))
    }

    /// The string at `handle` as a `String`, or `None` if it is not a string
    /// (or the handle is stale).
    #[must_use]
    pub fn string_value(&self, handle: Handle) -> Option<alloc::string::String> {
        Some(self.heap.get(handle)?.as_str()?.materialize())
    }

    /// The array elements at `handle`, or `None` if it is not an array.
    #[must_use]
    pub fn array_elements(&self, handle: Handle) -> Option<&[NanBox]> {
        self.heap.get(handle)?.as_array()
    }

    /// The `typeof` string for the heap value at `handle` (`"string"`/`"object"`),
    /// or `None` if the handle is stale.
    #[must_use]
    pub fn type_of(&self, handle: Handle) -> Option<&'static str> {
        Some(self.heap.get(handle)?.type_of())
    }

    /// The value of own property `key` on the object at `handle`, or `None` if
    /// the property is absent, the cell is not an object, or the handle is stale.
    #[must_use]
    pub fn get_property(&self, handle: Handle, key: &str) -> Option<NanBox> {
        self.heap.get(handle)?.as_object()?.get(key)
    }

    /// Sets own property `key` to `value` on the object at `handle`. Returns
    /// `false` if the handle is stale or the cell is not an object.
    pub fn set_property(&mut self, handle: Handle, key: &str, value: NanBox) -> bool {
        match self.heap.get_mut(handle).and_then(Cell::as_object_mut) {
            Some(obj) => {
                obj.set(key, value);
                true
            }
            None => false,
        }
    }

    /// Interns `key`, so callers can hold a `Copy` [`Atom`] for hot property
    /// names.
    pub fn intern(&mut self, key: &str) -> Atom {
        self.atoms.intern(key)
    }

    /// The number of live objects in the heap.
    #[must_use]
    pub fn object_count(&self) -> usize {
        self.heap.len()
    }

    /// Whether `handle` still refers to a live object.
    #[must_use]
    pub fn is_live(&self, handle: Handle) -> bool {
        self.heap.is_live(handle)
    }

    /// Runs a garbage collection, keeping everything reachable from `roots` and
    /// freeing the rest (including cycles). Returns the collection statistics.
    pub fn collect(&mut self, roots: &[Handle]) -> Stats {
        gc::collect(&mut self.heap, roots)
    }

    // --- value operations (the VM's `+`, `ToString`, `===` over heap values) ---

    /// Whether `v` is a heap string.
    #[must_use]
    fn is_string(&self, v: NanBox) -> bool {
        v.as_handle()
            .and_then(|raw| self.heap.get(Handle::from_raw(raw)))
            .is_some_and(|c| c.as_str().is_some())
    }

    /// The rope view of `v` for concatenation: a string cell's own rope (shared,
    /// so concatenation stays O(1)), or a fresh leaf from its `ToString`.
    fn rope_of(&self, v: NanBox) -> Rope {
        if let Some(raw) = v.as_handle()
            && let Some(rope) = self.heap.get(Handle::from_raw(raw)).and_then(Cell::as_str)
        {
            return rope.clone();
        }
        Rope::from(self.to_display_string(v).as_str())
    }

    /// ECMAScript `ToString` for display: numbers/booleans/null/undefined render
    /// directly; a string yields its text; an array joins its elements with
    /// `","`; a plain object is `"[object Object]"`.
    #[must_use]
    pub fn to_display_string(&self, v: NanBox) -> alloc::string::String {
        use crate::nanbox::Unpacked;
        match v.unpack() {
            Unpacked::Undefined => "undefined".into(),
            Unpacked::Null => "null".into(),
            Unpacked::Bool(b) => {
                if b {
                    "true".into()
                } else {
                    "false".into()
                }
            }
            Unpacked::Number(n) => alloc::format!("{n}"),
            Unpacked::Handle(raw) => match self.heap.get(Handle::from_raw(raw)) {
                Some(Cell::Str(r)) => r.materialize(),
                Some(Cell::Array(elems)) => {
                    let parts: Vec<alloc::string::String> = elems
                        .iter()
                        .map(|e| {
                            // A hole-ish nullish element renders empty, per `Array#join`.
                            if matches!(e.unpack(), Unpacked::Undefined | Unpacked::Null) {
                                alloc::string::String::new()
                            } else {
                                self.to_display_string(*e)
                            }
                        })
                        .collect();
                    parts.join(",")
                }
                Some(Cell::Object(_)) => "[object Object]".into(),
                None => "undefined".into(), // stale handle
            },
        }
    }

    /// The ECMAScript `+` operator (the cases this model covers): number + number
    /// is numeric addition; if either side is a string it is string
    /// concatenation, producing a new heap string built by joining ropes (so a
    /// loop of `+` stays O(1) per step). Other combinations coerce to string for
    /// now (full `ToPrimitive` arrives with the boxed primitives).
    pub fn add(&mut self, a: NanBox, b: NanBox) -> NanBox {
        if let (Some(x), Some(y)) = (a.as_number(), b.as_number()) {
            return NanBox::number(x + y);
        }
        if self.is_string(a) || self.is_string(b) {
            let combined = self.rope_of(a).concat(&self.rope_of(b));
            let handle = self.heap.alloc(Cell::Str(combined));
            return NanBox::handle(handle.to_raw());
        }
        // No string and not both numbers: numeric fast path yields NaN.
        NanBox::number(f64::NAN)
    }

    /// ECMAScript `ToNumber` (the cases this model covers): numbers pass
    /// through; `true`/`false` → `1`/`0`; `null` → `0`; `undefined` → `NaN`; a
    /// string is parsed (blank → `0`, a numeric literal → its value, else
    /// `NaN`); objects/arrays → `NaN` (full `ToPrimitive` arrives later).
    #[must_use]
    pub fn to_number(&self, v: NanBox) -> f64 {
        use crate::nanbox::Unpacked;
        match v.unpack() {
            Unpacked::Number(n) => n,
            Unpacked::Bool(b) => {
                if b {
                    1.0
                } else {
                    0.0
                }
            }
            Unpacked::Null => 0.0,
            Unpacked::Undefined => f64::NAN,
            Unpacked::Handle(raw) => {
                match self.heap.get(Handle::from_raw(raw)).and_then(Cell::as_str) {
                    Some(rope) => {
                        let s = rope.materialize();
                        let t = s.trim();
                        if t.is_empty() {
                            0.0
                        } else {
                            t.parse::<f64>().unwrap_or(f64::NAN)
                        }
                    }
                    None => f64::NAN, // object/array/stale
                }
            }
        }
    }

    /// The ECMAScript abstract relational comparison `a < b`: if *both* operands
    /// are strings they compare lexicographically by code point; otherwise both
    /// are coerced with `ToNumber`. Returns `None` when the result is undefined
    /// (a `NaN` operand) — the caller turns that into `false`.
    #[must_use]
    fn compare(&self, a: NanBox, b: NanBox) -> Option<core::cmp::Ordering> {
        if self.is_string(a) && self.is_string(b) {
            let sa = self.to_display_string(a);
            let sb = self.to_display_string(b);
            return Some(sa.cmp(&sb));
        }
        let (x, y) = (self.to_number(a), self.to_number(b));
        x.partial_cmp(&y) // None on NaN
    }

    /// `a < b` (boolean).
    #[must_use]
    pub fn less_than(&self, a: NanBox, b: NanBox) -> NanBox {
        NanBox::boolean(self.compare(a, b) == Some(core::cmp::Ordering::Less))
    }

    /// `a <= b` (boolean).
    #[must_use]
    pub fn less_equal(&self, a: NanBox, b: NanBox) -> NanBox {
        NanBox::boolean(matches!(
            self.compare(a, b),
            Some(core::cmp::Ordering::Less | core::cmp::Ordering::Equal)
        ))
    }

    /// `a > b` (boolean).
    #[must_use]
    pub fn greater_than(&self, a: NanBox, b: NanBox) -> NanBox {
        NanBox::boolean(self.compare(a, b) == Some(core::cmp::Ordering::Greater))
    }

    /// `a >= b` (boolean).
    #[must_use]
    pub fn greater_equal(&self, a: NanBox, b: NanBox) -> NanBox {
        NanBox::boolean(matches!(
            self.compare(a, b),
            Some(core::cmp::Ordering::Greater | core::cmp::Ordering::Equal)
        ))
    }

    /// `a - b` (numeric, both coerced with `ToNumber`).
    #[must_use]
    pub fn sub(&self, a: NanBox, b: NanBox) -> NanBox {
        NanBox::number(self.to_number(a) - self.to_number(b))
    }

    /// `a * b` (numeric).
    #[must_use]
    pub fn mul(&self, a: NanBox, b: NanBox) -> NanBox {
        NanBox::number(self.to_number(a) * self.to_number(b))
    }

    /// `a / b` (numeric; division by zero yields ±∞ / `NaN` per IEEE-754).
    #[must_use]
    pub fn div(&self, a: NanBox, b: NanBox) -> NanBox {
        NanBox::number(self.to_number(a) / self.to_number(b))
    }

    /// `a % b` (the ECMAScript remainder, which follows the sign of the
    /// dividend — Rust's `%` on `f64` matches).
    #[must_use]
    pub fn rem(&self, a: NanBox, b: NanBox) -> NanBox {
        NanBox::number(self.to_number(a) % self.to_number(b))
    }

    /// `a ** b` (exponentiation). Needs `std` for the float `powf` intrinsic
    /// (the alloc-only core omits it, like the rest of the engine's float math).
    #[cfg(feature = "std")]
    #[must_use]
    pub fn pow(&self, a: NanBox, b: NanBox) -> NanBox {
        NanBox::number(self.to_number(a).powf(self.to_number(b)))
    }

    /// Unary `-a` (numeric negation).
    #[must_use]
    pub fn neg(&self, a: NanBox) -> NanBox {
        NanBox::number(-self.to_number(a))
    }

    /// ECMAScript abstract equality (`==`) — strict equality plus coercion:
    /// `null == undefined`; a boolean is compared as its number; a number and a
    /// string compare numerically; two references use [`strict_equals`] (string
    /// value / object identity). (Full `ToPrimitive` on objects arrives later.)
    ///
    /// [`strict_equals`]: Realm::strict_equals
    #[must_use]
    pub fn loose_equals(&self, a: NanBox, b: NanBox) -> bool {
        use crate::nanbox::Unpacked::{Bool, Handle as H, Null, Undefined};
        let (ua, ub) = (a.unpack(), b.unpack());
        let nullish = |u| matches!(u, Undefined | Null);
        // null/undefined are == to each other and to nothing else.
        if nullish(ua) || nullish(ub) {
            return nullish(ua) && nullish(ub);
        }
        // Two references: strings by value, objects by identity.
        if matches!(ua, H(_)) && matches!(ub, H(_)) {
            return self.strict_equals(a, b);
        }
        // Booleans compare as their numeric value.
        if matches!(ua, Bool(_)) {
            return self.loose_equals(NanBox::number(self.to_number(a)), b);
        }
        if matches!(ub, Bool(_)) {
            return self.loose_equals(a, NanBox::number(self.to_number(b)));
        }
        // Remaining: number vs number, or number vs string — compare numerically.
        self.to_number(a) == self.to_number(b)
    }

    /// ECMAScript strict equality (`===`) over heap values: primitives compare
    /// by value; two strings compare by *content* (strings are primitives, so
    /// distinct allocations of `"ab"` are equal); other references compare by
    /// identity.
    #[must_use]
    pub fn strict_equals(&self, a: NanBox, b: NanBox) -> bool {
        match (a.as_handle(), b.as_handle()) {
            (Some(ha), Some(hb)) => {
                if ha == hb {
                    return true; // same heap cell
                }
                let sa = self.heap.get(Handle::from_raw(ha)).and_then(Cell::as_str);
                let sb = self.heap.get(Handle::from_raw(hb)).and_then(Cell::as_str);
                match (sa, sb) {
                    (Some(ra), Some(rb)) => ra.materialize() == rb.materialize(),
                    _ => false, // distinct non-string references
                }
            }
            // At least one primitive: decided by the boxed value itself.
            _ => a.strict_equals(b),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocate_and_access_properties() {
        let mut realm = Realm::new();
        let obj = realm.new_object();
        realm.set_property(obj, "x", NanBox::number(1.0));
        realm.set_property(obj, "y", NanBox::number(2.0));
        assert_eq!(realm.get_property(obj, "x"), Some(NanBox::number(1.0)));
        assert_eq!(realm.get_property(obj, "y"), Some(NanBox::number(2.0)));
        assert_eq!(realm.get_property(obj, "z"), None);
        assert_eq!(realm.object_count(), 1);
    }

    #[test]
    fn objects_share_hidden_classes_across_the_realm() {
        let mut realm = Realm::new();
        let a = realm.new_object();
        let b = realm.new_object();
        realm.set_property(a, "p", NanBox::number(1.0));
        realm.set_property(b, "p", NanBox::number(9.0));
        // Same structure built from the realm's shared root → same shape.
        let sa = Rc::clone(realm.heap.get(a).unwrap().as_object().unwrap().shape());
        let sb = Rc::clone(realm.heap.get(b).unwrap().as_object().unwrap().shape());
        assert!(Rc::ptr_eq(&sa, &sb));
    }

    #[test]
    fn gc_reclaims_unreachable_objects_including_cycles() {
        let mut realm = Realm::new();
        // root -> a; b <-> c form an unreachable cycle.
        let root = realm.new_object();
        let a = realm.new_object();
        realm.set_property(root, "a", NanBox::handle(a.to_raw()));

        let b = realm.new_object();
        let c = realm.new_object();
        realm.set_property(b, "c", NanBox::handle(c.to_raw()));
        realm.set_property(c, "b", NanBox::handle(b.to_raw()));
        assert_eq!(realm.object_count(), 4);

        let stats = realm.collect(&[root]);
        assert_eq!(stats.marked, 2); // root + a
        assert_eq!(stats.swept, 2); // the b<->c cycle
        assert!(realm.is_live(root) && realm.is_live(a));
        assert!(!realm.is_live(b) && !realm.is_live(c));
        assert_eq!(realm.object_count(), 2);
    }

    #[test]
    fn stale_handle_access_is_safe() {
        let mut realm = Realm::new();
        let obj = realm.new_object();
        realm.set_property(obj, "x", NanBox::number(1.0));
        realm.collect(&[]); // frees obj (no roots)
        assert!(!realm.is_live(obj));
        assert_eq!(realm.get_property(obj, "x"), None);
        assert!(!realm.set_property(obj, "x", NanBox::number(2.0)));
    }

    #[test]
    fn interning_is_shared() {
        let mut realm = Realm::new();
        let a = realm.intern("length");
        let b = realm.intern("length");
        assert_eq!(a, b);
    }

    #[test]
    fn strings_and_arrays_are_heap_values() {
        let mut realm = Realm::new();
        let s = realm.new_string("hello");
        let arr = realm.new_array(alloc::vec![NanBox::number(1.0), NanBox::number(2.0)]);
        let obj = realm.new_object();
        assert_eq!(realm.string_value(s).as_deref(), Some("hello"));
        assert_eq!(realm.array_elements(arr).map(<[_]>::len), Some(2));
        assert_eq!(realm.type_of(s), Some("string"));
        assert_eq!(realm.type_of(arr), Some("object"));
        assert_eq!(realm.type_of(obj), Some("object"));
        // A property op on a string cell is rejected (not an object).
        assert!(!realm.set_property(s, "x", NanBox::number(1.0)));
        assert_eq!(realm.get_property(s, "x"), None);
    }

    #[test]
    fn add_numbers_and_concatenates_strings() {
        let mut realm = Realm::new();
        // number + number → numeric.
        let n = realm.add(NanBox::number(2.0), NanBox::number(3.0));
        assert_eq!(n.as_number(), Some(5.0));
        // string + number → concatenation (coerced).
        let hi = realm.new_string("count: ");
        let r = realm.add(NanBox::handle(hi.to_raw()), NanBox::number(42.0));
        assert_eq!(realm.to_display_string(r), "count: 42");
        // string + string.
        let a = realm.new_string("foo");
        let b = realm.new_string("bar");
        let ab = realm.add(NanBox::handle(a.to_raw()), NanBox::handle(b.to_raw()));
        assert_eq!(realm.to_display_string(ab), "foobar");
    }

    #[test]
    fn string_append_loop_builds_correctly() {
        // The `s += part` shape, now through ropes in the heap.
        let mut realm = Realm::new();
        let mut acc = NanBox::handle(realm.new_string("").to_raw());
        let mut expected = alloc::string::String::new();
        for i in 0..30 {
            let part = realm.new_string(&alloc::format!("{i},"));
            acc = realm.add(acc, NanBox::handle(part.to_raw()));
            expected.push_str(&alloc::format!("{i},"));
        }
        assert_eq!(realm.to_display_string(acc), expected);
    }

    #[test]
    fn to_display_string_covers_kinds() {
        let mut realm = Realm::new();
        assert_eq!(realm.to_display_string(NanBox::undefined()), "undefined");
        assert_eq!(realm.to_display_string(NanBox::null()), "null");
        assert_eq!(realm.to_display_string(NanBox::boolean(true)), "true");
        assert_eq!(realm.to_display_string(NanBox::number(3.5)), "3.5");
        let arr = realm.new_array(alloc::vec![
            NanBox::number(1.0),
            NanBox::number(2.0),
            NanBox::number(3.0),
        ]);
        assert_eq!(
            realm.to_display_string(NanBox::handle(arr.to_raw())),
            "1,2,3"
        );
        let obj = realm.new_object();
        assert_eq!(
            realm.to_display_string(NanBox::handle(obj.to_raw())),
            "[object Object]"
        );
    }

    #[test]
    fn strict_equals_strings_by_value_objects_by_identity() {
        let mut realm = Realm::new();
        // Primitives.
        assert!(realm.strict_equals(NanBox::number(1.0), NanBox::number(1.0)));
        assert!(!realm.strict_equals(NanBox::number(1.0), NanBox::null()));
        // Two distinct string allocations with equal content are ===.
        let a = realm.new_string("hello");
        let b = realm.new_string("hello");
        let c = realm.new_string("world");
        assert_ne!(a.to_raw(), b.to_raw()); // genuinely different cells
        assert!(realm.strict_equals(NanBox::handle(a.to_raw()), NanBox::handle(b.to_raw())));
        assert!(!realm.strict_equals(NanBox::handle(a.to_raw()), NanBox::handle(c.to_raw())));
        // Objects compare by identity.
        let o1 = realm.new_object();
        let o2 = realm.new_object();
        assert!(realm.strict_equals(NanBox::handle(o1.to_raw()), NanBox::handle(o1.to_raw())));
        assert!(!realm.strict_equals(NanBox::handle(o1.to_raw()), NanBox::handle(o2.to_raw())));
    }

    #[test]
    fn to_number_coerces() {
        let mut realm = Realm::new();
        assert_eq!(realm.to_number(NanBox::number(3.5)), 3.5);
        assert_eq!(realm.to_number(NanBox::boolean(true)), 1.0);
        assert_eq!(realm.to_number(NanBox::boolean(false)), 0.0);
        assert_eq!(realm.to_number(NanBox::null()), 0.0);
        assert!(realm.to_number(NanBox::undefined()).is_nan());
        let s = realm.new_string("  42  ");
        assert_eq!(realm.to_number(NanBox::handle(s.to_raw())), 42.0);
        let blank = realm.new_string("");
        assert_eq!(realm.to_number(NanBox::handle(blank.to_raw())), 0.0);
        let bad = realm.new_string("nope");
        assert!(realm.to_number(NanBox::handle(bad.to_raw())).is_nan());
    }

    #[test]
    fn relational_operators_on_numbers_and_strings() {
        let mut realm = Realm::new();
        let one = NanBox::number(1.0);
        let two = NanBox::number(2.0);
        assert_eq!(realm.less_than(one, two).as_boolean(), Some(true));
        assert_eq!(realm.less_than(two, one).as_boolean(), Some(false));
        assert_eq!(realm.greater_than(two, one).as_boolean(), Some(true));
        assert_eq!(realm.less_equal(two, two).as_boolean(), Some(true));
        assert_eq!(realm.greater_equal(two, two).as_boolean(), Some(true));
        // A NaN operand makes every comparison false.
        let nan = NanBox::number(f64::NAN);
        assert_eq!(realm.less_than(nan, one).as_boolean(), Some(false));
        assert_eq!(realm.greater_than(nan, one).as_boolean(), Some(false));
        // Two strings compare lexicographically: "10" < "9" (string order).
        let s10 = realm.new_string("10");
        let s9 = realm.new_string("9");
        assert_eq!(
            realm
                .less_than(NanBox::handle(s10.to_raw()), NanBox::handle(s9.to_raw()))
                .as_boolean(),
            Some(true)
        );
        // Mixed string/number coerces to numeric: 10 < "9" is false (9 < 10).
        assert_eq!(
            realm
                .less_than(NanBox::number(10.0), NanBox::handle(s9.to_raw()))
                .as_boolean(),
            Some(false)
        );
    }

    #[test]
    fn arithmetic_operators() {
        let realm = Realm::new();
        let n = NanBox::number;
        assert_eq!(realm.sub(n(5.0), n(3.0)).as_number(), Some(2.0));
        assert_eq!(realm.mul(n(4.0), n(2.5)).as_number(), Some(10.0));
        assert_eq!(realm.div(n(9.0), n(2.0)).as_number(), Some(4.5));
        assert_eq!(realm.rem(n(7.0), n(3.0)).as_number(), Some(1.0));
        assert_eq!(realm.rem(n(-7.0), n(3.0)).as_number(), Some(-1.0)); // sign of dividend
        assert_eq!(realm.neg(n(3.0)).as_number(), Some(-3.0));
        // Division by zero is ±Infinity.
        assert_eq!(realm.div(n(1.0), n(0.0)).as_number(), Some(f64::INFINITY));
        #[cfg(feature = "std")]
        assert_eq!(realm.pow(n(2.0), n(10.0)).as_number(), Some(1024.0));
    }

    #[test]
    fn abstract_equality_coerces() {
        let mut realm = Realm::new();
        let n = NanBox::number;
        // null == undefined, but not == 0.
        assert!(realm.loose_equals(NanBox::null(), NanBox::undefined()));
        assert!(!realm.loose_equals(NanBox::null(), n(0.0)));
        // number == string by numeric coercion.
        let s1 = NanBox::handle(realm.new_string("1").to_raw());
        assert!(realm.loose_equals(n(1.0), s1));
        // boolean coerces to number: true == 1, false == 0.
        assert!(realm.loose_equals(NanBox::boolean(true), n(1.0)));
        assert!(realm.loose_equals(NanBox::boolean(false), n(0.0)));
        assert!(!realm.loose_equals(NanBox::boolean(true), n(2.0)));
        // strings by value; objects by identity.
        let a = NanBox::handle(realm.new_string("x").to_raw());
        let b = NanBox::handle(realm.new_string("x").to_raw());
        assert!(realm.loose_equals(a, b));
        let o1 = NanBox::handle(realm.new_object().to_raw());
        assert!(!realm.loose_equals(o1, n(0.0))); // object != 0 (ToNumber→NaN)
    }

    #[test]
    fn gc_keeps_a_mixed_object_array_string_graph() {
        let mut realm = Realm::new();
        // obj.name -> string; obj.items -> array; array[0] -> obj (a cycle).
        let obj = realm.new_object();
        let name = realm.new_string("widget");
        let items = realm.new_array(alloc::vec![NanBox::handle(obj.to_raw())]);
        realm.set_property(obj, "name", NanBox::handle(name.to_raw()));
        realm.set_property(obj, "items", NanBox::handle(items.to_raw()));
        let _unreachable = realm.new_string("garbage");
        assert_eq!(realm.object_count(), 4);

        let stats = realm.collect(&[obj]);
        assert_eq!(stats.marked, 3); // obj, name, items (cycle obj<-items kept)
        assert_eq!(stats.swept, 1); // the unreachable string
        assert!(realm.is_live(obj) && realm.is_live(name) && realm.is_live(items));
        assert_eq!(realm.string_value(name).as_deref(), Some("widget"));
    }
}
