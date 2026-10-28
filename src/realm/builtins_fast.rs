//! Narrow, read-only views of the heap for the built-in library's fast paths
//! (`JSON.stringify` and friends): each answers one question exactly or says
//! "don't know", so a caller can skip the generic `[[Get]]` machinery only when
//! the answer is provably the same.

use super::{Cell, Handle, NanBox, Realm};
use crate::shape::Shape;
use alloc::rc::Rc;

impl Realm {
    /// The shape of the object at `handle` when its own properties are exactly
    /// that shape's data keys, all enumerable, in shape order (see
    /// [`Object::keys_follow_shape`](crate::object::Object::keys_follow_shape)),
    /// and no host hook intercepts its reads (it is not host-exotic). `None`
    /// for anything else.
    #[must_use]
    pub fn plain_keyed_shape(&self, handle: Handle) -> Option<&Rc<Shape>> {
        let Some(Cell::Object(o)) = self.heap.get(handle) else {
            return None;
        };
        if !o.keys_follow_shape() || self.is_host_exotic(handle) {
            return None;
        }
        o.shaped_shape()
    }

    /// `EnumerableOwnProperties(O, key)` for an object whose own properties are
    /// all plain enumerable data properties (see
    /// [`plain_keyed_shape`](Self::plain_keyed_shape)) and which is no primitive
    /// wrapper (a String wrapper owns its indices exotically): its string keys
    /// in property order. `None` for anything else.
    #[must_use]
    pub fn plain_enumerable_keys(
        &self,
        handle: Handle,
    ) -> Option<alloc::vec::Vec<alloc::string::String>> {
        let Some(Cell::Object(o)) = self.heap.get(handle) else {
            return None;
        };
        if o.is_prim_wrapper() {
            return None;
        }
        self.plain_keyed_shape(handle)?;
        self.object_keys(handle)
    }

    /// [`plain_enumerable_keys`](Self::plain_enumerable_keys) as shared ropes.
    /// For such an object the key list is a function of its shape alone, so
    /// the last shape's list is remembered: listing the keys of objects built
    /// alike (a loop over records) then allocates nothing but the result's
    /// string cells, which share these immutable ropes.
    #[must_use]
    pub fn plain_enumerable_key_ropes(&self, handle: Handle) -> Option<Rc<[crate::rope::Rope]>> {
        let Some(Cell::Object(o)) = self.heap.get(handle) else {
            return None;
        };
        if o.is_prim_wrapper() {
            return None;
        }
        let shape = self.plain_keyed_shape(handle)?;
        if let Some((s, keys)) = &*self.key_ropes.borrow()
            && Rc::ptr_eq(s, shape)
        {
            return Some(keys.clone());
        }
        let shape = shape.clone();
        let keys: Rc<[crate::rope::Rope]> = self
            .object_keys(handle)?
            .iter()
            .map(|k| crate::rope::Rope::leaf(k))
            .collect();
        *self.key_ropes.borrow_mut() = Some((shape, keys.clone()));
        Some(keys)
    }

    /// Slot `slot` of the object at `handle`, provided the object still has
    /// shape `shape` and still satisfies
    /// [`plain_keyed_shape`](Self::plain_keyed_shape)'s conditions — so the slot
    /// holds the data property the caller resolved under that shape.
    #[must_use]
    pub fn plain_slot_under(&self, handle: Handle, shape: &Rc<Shape>, slot: u32) -> Option<NanBox> {
        let Some(Cell::Object(o)) = self.heap.get(handle) else {
            return None;
        };
        if !o.keys_follow_shape() || !o.shaped_shape().is_some_and(|s| Rc::ptr_eq(s, shape)) {
            return None;
        }
        o.slot(slot)
    }

    /// The flags of the RegExp at `handle`, borrowed (no source rendering).
    #[must_use]
    pub fn regexp_flags(&self, handle: Handle) -> Option<&str> {
        match self.heap.get(handle)? {
            Cell::RegExp { flags, .. } => Some(flags),
            _ => None,
        }
    }

    /// `CreateDataProperty(handle, key, value)` for a named key on a fresh
    /// array (or other cell whose named properties live in an auxiliary
    /// object), through `cache` as for an object literal's own properties
    /// ([`init_data_property_cached`](Self::init_data_property_cached)), so a
    /// site that always adds the same keys reuses one shape path. Falls back to
    /// [`set_property`](Self::set_property).
    pub fn init_aux_data_cached(
        &mut self,
        handle: Handle,
        key: &str,
        value: NanBox,
        cache: &mut crate::ic::PropertyCache,
    ) {
        if self.aux_eligible(handle) {
            let aux = self.aux_object(handle);
            if self.init_data_property_cached(aux, key, value, cache) {
                return;
            }
        }
        self.set_property(handle, key, value);
    }

    /// For a string cell, the UTF-16 code unit at index `i` (`None` when `i` is
    /// `None` or out of range); `None` for any other cell.
    #[must_use]
    pub fn string_unit_at_index(&self, handle: Handle, i: Option<usize>) -> Option<Option<u16>> {
        self.heap.get(handle)?.as_str()?;
        Some(i.and_then(|i| self.string_unit_at(handle, i)))
    }

    /// Whether `handle` is a [`Cell::Array`].
    #[must_use]
    pub fn is_plain_array_cell(&self, handle: Handle) -> bool {
        matches!(self.heap.get(handle), Some(Cell::Array(_)))
    }

    /// The identity of one prototype-chain link for a [`ChainKeyCache`]: the
    /// key epoch that changes with the link's set of own named keys (an
    /// array's, a RegExp's or a Map/Set's live in its auxiliary object, which
    /// may come and go), and the next link. `None` for anything other than an
    /// ordinary object, an array, a RegExp or a collection, and for a
    /// host-exotic one.
    fn key_link(&self, h: Handle) -> Option<(Option<u64>, u32, Option<Handle>)> {
        if self.is_host_exotic(h) {
            return None;
        }
        match self.heap.get(h)? {
            Cell::Object(o) => Some((None, o.key_epoch(), o.proto())),
            Cell::Array(_) | Cell::RegExp { .. } | Cell::Collection { .. } | Cell::Native(_) => {
                let (aux, epoch) = match self.aux_props.get(&h.to_raw()) {
                    Some(a) => (
                        Some(a.to_raw()),
                        self.heap.get(*a)?.as_object()?.key_epoch(),
                    ),
                    None => (None, 0),
                };
                Some((aux, epoch, self.object_proto(h)))
            }
            _ => None,
        }
    }

    /// For a method call `recv.key(...)` on the heap value `h`: the object the
    /// ordinary `[[Get]]` of the named key `key` (not an index, not `length`,
    /// not `lastIndex`) continues at once `h` itself is known not to own it —
    /// its `[[Prototype]]`, possibly `None`. `None` when `h` does own `key`, or
    /// is anything other than a string primitive, an ordinary non-wrapper
    /// object, an array, a RegExp or a Map/Set (or is host-exotic), whose own
    /// properties this cannot rule out. `recv_shape` memoizes one receiver
    /// shape already known to lack `key` (a shape fixes its data keys, so an
    /// accessor-free object of that shape lacks it too).
    fn method_lookup_start(
        &self,
        h: Handle,
        key: &str,
        recv_shape: &mut Option<Rc<Shape>>,
    ) -> Option<Option<Handle>> {
        if self.is_host_exotic(h) {
            return None;
        }
        match self.heap.get(h)? {
            Cell::Object(o) => {
                if o.is_prim_wrapper() {
                    return None;
                }
                let shape = o.shaped_shape().filter(|_| o.has_no_accessors());
                if let (Some(s), Some(m)) = (shape, recv_shape.as_ref())
                    && Rc::ptr_eq(s, m)
                {
                    return Some(o.proto());
                }
                if o.has_own_key(key) {
                    return None;
                }
                if let Some(s) = shape {
                    *recv_shape = Some(Rc::clone(s));
                }
                Some(o.proto())
            }
            // A string primitive owns only its indices and `length`; one with an
            // auxiliary object is a function-like marker cell, not a plain string.
            Cell::Str(_) => {
                if self.aux_props.contains_key(&h.to_raw()) {
                    return None;
                }
                Some(self.object_proto(h))
            }
            // A VM closure synthesizes `name` and `length` (and the legacy
            // `caller`/`arguments` are the VM's); its other own properties live
            // in its auxiliary object, as a native function's do.
            Cell::VmFunction { .. } if matches!(key, "name" | "caller" | "arguments") => None,
            Cell::Array(_)
            | Cell::RegExp { .. }
            | Cell::Collection { .. }
            | Cell::Native(_)
            | Cell::VmFunction { .. } => {
                if let Some(a) = self.aux_props.get(&h.to_raw())
                    && self.heap.get(*a)?.as_object()?.has_own_key(key)
                {
                    return None;
                }
                Some(self.object_proto(h))
            }
            _ => None,
        }
    }

    /// The object holding the named properties of the chain link `h`: `h`
    /// itself for an ordinary object, else its auxiliary object.
    fn named_props(&self, h: Handle) -> Option<&crate::object::Object> {
        match self.heap.get(h)? {
            Cell::Object(o) => Some(o),
            Cell::Array(_) | Cell::RegExp { .. } | Cell::Collection { .. } | Cell::Native(_) => {
                self.heap
                    .get(*self.aux_props.get(&h.to_raw())?)?
                    .as_object()
            }
            _ => None,
        }
    }

    /// The current value of the method `recv[key]` (a named key: not an
    /// index, not `length`, not `lastIndex`) when `cache` still proves where
    /// its `[[Get]]` lands: `recv` does not own `key` (see
    /// `method_lookup_start`), and from its prototype on the chain is exactly
    /// the recorded objects with their recorded key epochs, the last holding
    /// `key` as a data property (read now). `None` is a miss.
    #[must_use]
    pub fn method_site_get(
        &self,
        recv: Handle,
        key: &str,
        cache: &mut MethodSiteCache,
    ) -> Option<NanBox> {
        if !cache.armed || cache.swaps != self.cell_swaps || cache.prim {
            return None;
        }
        let cur = if cache.own {
            Some(recv)
        } else {
            self.method_lookup_start(recv, key, &mut cache.recv_shape)?
        };
        self.site_links_get(cur, key, cache)
    }

    /// [`method_site_get`](Self::method_site_get) for a *primitive* receiver
    /// (a Number, a Boolean, …), whose lookup starts at its prototype
    /// intrinsic `start` ([`primitive_proto`](Self::primitive_proto)) — a
    /// primitive owns no named property. Only a site filled by
    /// [`method_site_fill_prim`](Self::method_site_fill_prim) hits.
    #[must_use]
    pub fn method_site_get_prim(
        &self,
        start: Handle,
        key: &str,
        cache: &mut MethodSiteCache,
    ) -> Option<NanBox> {
        if !cache.armed || cache.swaps != self.cell_swaps || !cache.prim {
            return None;
        }
        self.site_links_get(Some(start), key, cache)
    }

    /// The walk of [`method_site_get`](Self::method_site_get) from the first
    /// chain link `cur`.
    fn site_links_get(
        &self,
        mut cur: Option<Handle>,
        key: &str,
        cache: &MethodSiteCache,
    ) -> Option<NanBox> {
        let last = cache.links.len().checked_sub(1)?;
        for (i, &(raw, aux, epoch)) in cache.links.iter().enumerate() {
            let h = cur?;
            if h.to_raw() != raw {
                return None;
            }
            let (a, e, next) = self.key_link(h)?;
            if a != aux || e != epoch {
                return None;
            }
            if i == last {
                // Same key set as when armed: a data property, at the
                // recorded slot while the layout is the recorded one.
                let o = self.named_props(h)?;
                if let (Some(s), Some(m)) = (o.shaped_shape(), cache.holder_shape.as_ref())
                    && Rc::ptr_eq(s, m)
                {
                    return o.slot(cache.holder_slot);
                }
                return self.holder_data(h, key);
            }
            cur = next;
        }
        None
    }

    /// Arms `cache` for `recv[key]` (see [`method_site_get`](Self::method_site_get))
    /// and returns the method's value, when the walk from `recv`'s prototype
    /// meets only ordinary objects, arrays, RegExps and collections (none
    /// host-exotic, at most eight deep) before one owning `key` as a data
    /// property. Leaves it unarmed (and returns `None`) otherwise.
    pub fn method_site_fill(
        &self,
        recv: Handle,
        key: &str,
        cache: &mut MethodSiteCache,
    ) -> Option<NanBox> {
        cache.armed = false;
        cache.links.clear();
        cache.holder_shape = None;
        cache.own = false;
        cache.prim = false;
        cache.swaps = self.cell_swaps;
        let cur = match self.method_lookup_start(recv, key, &mut cache.recv_shape) {
            Some(start) => start,
            // A static (`String.fromCharCode`): the receiver itself, an
            // ordinary object or a native function, holds the method. It is
            // then the chain's one link, so a hit requires this very receiver.
            None if matches!(self.heap.get(recv), Some(Cell::Object(_) | Cell::Native(_)))
                && self.named_props(recv).is_some_and(|o| !o.is_prim_wrapper())
                && self.has_own(recv, key) =>
            {
                cache.own = true;
                Some(recv)
            }
            None => return None,
        };
        self.site_links_fill(cur, key, cache)
    }

    /// [`method_site_fill`](Self::method_site_fill) for a primitive receiver
    /// whose lookup starts at `start` (see
    /// [`method_site_get_prim`](Self::method_site_get_prim)).
    pub fn method_site_fill_prim(
        &self,
        start: Handle,
        key: &str,
        cache: &mut MethodSiteCache,
    ) -> Option<NanBox> {
        cache.armed = false;
        cache.links.clear();
        cache.holder_shape = None;
        cache.own = false;
        cache.prim = true;
        cache.swaps = self.cell_swaps;
        self.site_links_fill(Some(start), key, cache)
    }

    /// The walk of [`method_site_fill`](Self::method_site_fill) from the first
    /// chain link `cur`.
    fn site_links_fill(
        &self,
        mut cur: Option<Handle>,
        key: &str,
        cache: &mut MethodSiteCache,
    ) -> Option<NanBox> {
        for _ in 0..8 {
            let h = cur?;
            let (aux, epoch, next) = self.key_link(h)?;
            cache.links.push((h.to_raw(), aux, epoch));
            if self.has_own(h, key) {
                let v = self.holder_data(h, key)?;
                let o = self.named_props(h)?;
                if let Some(s) = o.shaped_shape()
                    && let Some(slot) = s.lookup(key)
                {
                    cache.holder_shape = Some(Rc::clone(s));
                    cache.holder_slot = slot;
                }
                cache.armed = true;
                return Some(v);
            }
            cur = next;
        }
        None
    }

    /// The `[[Get]]` of `cache`'s key from `start` when `cache` still proves
    /// its outcome: `Some(None)` when no object on the chain owns the key,
    /// `Some(Some(v))` when the recorded holder still owns it as a data
    /// property whose current value is `v`. `None` is a miss. The live chain
    /// must still be exactly the recorded objects, each with its recorded key
    /// epoch (so none has gained or lost an own key, or turned one into an
    /// accessor), in which case the ordinary `[[Get]]` walk visits the same
    /// objects and stops at the same place.
    #[must_use]
    pub fn chain_key_cached(&self, start: Handle, cache: &ChainKeyCache) -> Option<Option<NanBox>> {
        if !cache.armed || cache.swaps != self.cell_swaps {
            return None;
        }
        let mut cur = Some(start);
        let last = cache.links.len().checked_sub(1);
        for (i, &(raw, aux, epoch)) in cache.links.iter().enumerate() {
            let h = cur?;
            if h.to_raw() != raw {
                return None;
            }
            match self.key_link(h) {
                Some((a, e, next)) if a == aux && e == epoch => {
                    if cache.holder && Some(i) == last {
                        return self.holder_data(h, &cache.key).map(Some);
                    }
                    cur = next;
                }
                _ => return None,
            }
        }
        cur.is_none().then_some(None)
    }

    /// The own data property `key` of a [`ChainKeyCache`] holder: an ordinary
    /// object's slot, or the named property in an array's or RegExp's
    /// auxiliary object. `None` when it is an accessor or absent.
    fn holder_data(&self, h: Handle, key: &str) -> Option<NanBox> {
        let o = match self.heap.get(h)? {
            Cell::Object(o) => o,
            Cell::Array(_) | Cell::RegExp { .. } | Cell::Collection { .. } | Cell::Native(_) => {
                self.heap
                    .get(*self.aux_props.get(&h.to_raw())?)?
                    .as_object()?
            }
            _ => return None,
        };
        if o.accessor(key).is_some() {
            return None;
        }
        o.get(key)
    }

    /// The ordinary `[[Get]]` of the named key `key` (not an index or
    /// `length`) from `start`, answered from the realm's `slot`-th shared
    /// `ChainKeyCache` when it still proves the outcome (see
    /// [`chain_key_cached`](Self::chain_key_cached)): `Some(None)` absent,
    /// `Some(Some(v))` a data property. `None` on a miss, after re-arming the
    /// cache; the caller then does the lookup itself.
    #[must_use]
    pub fn shared_chain_lookup(
        &self,
        slot: usize,
        start: Handle,
        key: &str,
    ) -> Option<Option<NanBox>> {
        let mut caches = self.chain_lookups.borrow_mut();
        if caches.len() <= slot {
            caches.resize_with(slot + 1, ChainKeyCache::default);
        }
        let cache = &mut caches[slot];
        if &*cache.key == key
            && let Some(hit) = self.chain_key_cached(start, cache)
        {
            return Some(hit);
        }
        // A different key re-arms from scratch (and gets its own fill budget).
        if &*cache.key != key {
            *cache = ChainKeyCache::default();
        }
        self.chain_key_fill(start, key, cache);
        None
    }

    /// Whether `cache` proves that no object on the chain from `start` owns
    /// its key (see [`chain_key_cached`](Self::chain_key_cached)).
    #[must_use]
    pub fn chain_absent_cached(&self, start: Handle, cache: &ChainKeyCache) -> bool {
        matches!(self.chain_key_cached(start, cache), Some(None))
    }

    /// Arms `cache` for the named key `key` (not an index, not `length`, not
    /// `lastIndex`) from `start` when every object on the chain up to the
    /// first that owns `key` is an ordinary object, an array or a RegExp (none
    /// host-exotic, at most eight deep), and that owner, if any, is an ordinary
    /// object holding `key` as a data property. Leaves it unarmed otherwise.
    /// Re-arming stops after a few fills, so a site whose chain keeps changing
    /// stops paying for it.
    pub fn chain_key_fill(&self, start: Handle, key: &str, cache: &mut ChainKeyCache) {
        cache.armed = false;
        if cache.fills >= 16 {
            return;
        }
        cache.fills += 1;
        cache.links.clear();
        cache.swaps = self.cell_swaps;
        cache.holder = false;
        if &*cache.key != key {
            cache.key = alloc::boxed::Box::from(key);
        }
        let mut cur = Some(start);
        for _ in 0..8 {
            let Some(h) = cur else {
                cache.armed = true;
                return;
            };
            let Some((aux, epoch, next)) = self.key_link(h) else {
                return;
            };
            if self.has_own(h, key) {
                if self.holder_data(h, key).is_none() {
                    return;
                }
                cache.links.push((h.to_raw(), aux, epoch));
                cache.holder = true;
                cache.armed = true;
                return;
            }
            cache.links.push((h.to_raw(), aux, epoch));
            cur = next;
        }
    }
}

/// A memo of one named key's `[[Get]]` outcome along one prototype chain —
/// absent everywhere, or a data property of a particular holder — validated by
/// object identity (a handle carries its slot's generation, so a recycled slot
/// never matches) and each object's key epoch
/// ([`Object::key_epoch`](crate::object::Object::key_epoch), bumped on every
/// change to its set of own keys). Unlike [`crate::ic::ChainCache`] it does not
/// need shaped objects, so it also covers dictionary-mode, array and RegExp
/// links. The holder's value is re-read on every hit.
#[derive(Default)]
pub struct ChainKeyCache {
    key: alloc::boxed::Box<str>,
    /// `(handle, auxiliary object handle, key epoch)` per link.
    links: alloc::vec::Vec<(u64, Option<u64>, u32)>,
    /// Whether the last link owns the key (else no link does).
    holder: bool,
    armed: bool,
    fills: u16,
    swaps: u32,
}

/// A method-call site's memo of where `recv[key]` resolves (see
/// [`Realm::method_site_get`]): like [`ChainKeyCache`] from the receiver's
/// prototype on, plus the holder's slot under its recorded shape and one
/// receiver shape known not to own the key, so neither end repeats a name
/// lookup on a hit.
#[derive(Default)]
pub struct MethodSiteCache {
    /// `(handle, auxiliary object handle, key epoch)` per link, the holder last.
    links: alloc::vec::Vec<(u64, Option<u64>, u32)>,
    holder_shape: Option<Rc<Shape>>,
    holder_slot: u32,
    recv_shape: Option<Rc<Shape>>,
    /// Whether the receiver itself is the holder (the only link).
    own: bool,
    /// Whether the site was filled for a primitive receiver (the walk starts
    /// at its prototype intrinsic).
    prim: bool,
    armed: bool,
    swaps: u32,
}

/// One remembered compiled regex program (see `Realm::regex_programs`).
#[cfg(feature = "regex")]
pub(crate) struct RegexProgram {
    source: alloc::boxed::Box<[u8]>,
    flags: alloc::boxed::Box<str>,
    program: Rc<crate::regex::Regex>,
}

/// How many compiled programs a realm remembers.
#[cfg(feature = "regex")]
const REGEX_PROGRAMS: usize = 32;

#[cfg(feature = "regex")]
impl Realm {
    /// The remembered program for `(source, flags)`, if any (moved to the
    /// most-recently-used end).
    pub(super) fn regex_program_lookup(
        &self,
        source: &[u8],
        flags: &str,
    ) -> Option<Rc<crate::regex::Regex>> {
        let mut progs = self.regex_programs.borrow_mut();
        let i = progs
            .iter()
            .rposition(|p| &*p.source == source && &*p.flags == flags)?;
        let p = progs.remove(i);
        let program = Rc::clone(&p.program);
        progs.push(p);
        Some(program)
    }

    /// Remembers `program` as the compilation of `(source, flags)`, dropping
    /// the least recently used entry when full.
    pub(super) fn regex_program_remember(
        &self,
        source: &[u8],
        flags: &str,
        program: &Rc<crate::regex::Regex>,
    ) {
        let mut progs = self.regex_programs.borrow_mut();
        if progs.len() >= REGEX_PROGRAMS {
            progs.remove(0);
        }
        progs.push(RegexProgram {
            source: source.into(),
            flags: flags.into(),
            program: Rc::clone(program),
        });
    }
}

/// The decimal digits of `Number::toString(n)` when `n` is an integer of
/// magnitude below 2^53 (`-0` reads as `0`): a buffer and the start of the
/// text in it. `None` for any other number.
fn small_integer_digits(n: f64) -> Option<([u8; 21], usize)> {
    const LIMIT: f64 = 9_007_199_254_740_992.0;
    if !(n > -LIMIT && n < LIMIT) {
        return None;
    }
    let i = n as i64;
    if i as f64 != n {
        return None;
    }
    let mut buf = [0u8; 21];
    let mut pos = buf.len();
    let mut u = i.unsigned_abs();
    loop {
        pos -= 1;
        buf[pos] = b'0' + (u % 10) as u8;
        u /= 10;
        if u == 0 {
            break;
        }
    }
    if i < 0 {
        pos -= 1;
        buf[pos] = b'-';
    }
    Some((buf, pos))
}

/// Appends `Number::toString(n)` to `out` when `n` is an integer of magnitude
/// below 2^53 (written as plain decimal digits, `-0` as `0`) and returns
/// `true`; returns `false`, having written nothing, for any other number.
pub(crate) fn push_small_integer(out: &mut alloc::string::String, n: f64) -> bool {
    let Some((buf, pos)) = small_integer_digits(n) else {
        return false;
    };
    // Digits are ASCII, so the slice is valid UTF-8; one reserve-and-copy.
    out.push_str(core::str::from_utf8(&buf[pos..]).unwrap_or_default());
    true
}

/// Appends `Number::toString(n)` to the WTF-8 buffer `out` — the digits
/// directly for a small integer, else the general rendering.
pub(crate) fn push_number_bytes(out: &mut alloc::vec::Vec<u8>, n: f64) {
    if let Some((buf, pos)) = small_integer_digits(n) {
        out.extend_from_slice(&buf[pos..]);
    } else {
        out.extend_from_slice(super::js_number_string(n).as_bytes());
    }
}

/// How many small non-negative integers keep a shared string rope.
const NUMBER_ROPES: usize = 256;

impl Realm {
    /// `ToString(n)` as a rope. The integers `0..256` share one rope each
    /// (a rope is immutable, so sharing is invisible), which spares
    /// `` `${i % 10}` `` and `"" + digit` an allocation per evaluation.
    pub(super) fn number_rope(&self, n: f64) -> crate::rope::Rope {
        if n >= 0.0 && n < NUMBER_ROPES as f64 && n == (n as usize) as f64 {
            let i = n as usize;
            let mut cache = self.number_ropes.borrow_mut();
            if cache.is_empty() {
                cache.resize(NUMBER_ROPES, None);
            }
            return cache[i]
                .get_or_insert_with(|| crate::rope::Rope::from(super::js_number_string(n).as_str()))
                .clone();
        }
        crate::rope::Rope::from(super::js_number_string(n).as_str())
    }

    /// A new string cell holding `ToString(v)` for a primitive `v` (a number
    /// shares its rope through `number_rope`).
    pub fn new_primitive_string(&mut self, v: NanBox) -> Handle {
        let rope = match v.as_number() {
            Some(n) => self.number_rope(n),
            None => crate::rope::Rope::from(self.to_display_string(v).as_str()),
        };
        self.heap.alloc(Cell::Str(rope))
    }

    /// `a + b` for a string `a` held as one short leaf and a number `b` (or the
    /// mirror case when `number_first`): the joined text written straight into
    /// one new leaf, with no intermediate string for the number. `None` when
    /// the operands are not of that form.
    pub(super) fn add_leaf_number(
        &self,
        s: NanBox,
        n: NanBox,
        number_first: bool,
    ) -> Option<crate::rope::Rope> {
        let n = n.as_number()?;
        let rope = self.heap.get(Handle::from_raw(s.as_handle()?))?.as_str()?;
        let bytes = rope.as_leaf_bytes()?;
        if bytes.len() > 40 || bytes.is_empty() {
            return None;
        }
        let (buf, pos) = small_integer_digits(n)?;
        let digits = &buf[pos..];
        let mut out = alloc::vec::Vec::with_capacity(bytes.len() + digits.len());
        if number_first {
            out.extend_from_slice(digits);
            out.extend_from_slice(bytes);
        } else {
            out.extend_from_slice(bytes);
            out.extend_from_slice(digits);
        }
        Some(crate::rope::Rope::from_wtf8(out))
    }
}
