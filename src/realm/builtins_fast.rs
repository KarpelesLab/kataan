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
    /// key epoch that changes with the link's set of own named keys (an array's
    /// or a RegExp's live in its auxiliary object, which may come and go), and
    /// the next link. `None` for anything other than an ordinary object, an
    /// array or a RegExp, and for a host-exotic one.
    fn key_link(&self, h: Handle) -> Option<(Option<u64>, u32, Option<Handle>)> {
        if self.is_host_exotic(h) {
            return None;
        }
        match self.heap.get(h)? {
            Cell::Object(o) => Some((None, o.key_epoch(), o.proto())),
            Cell::Array(_) | Cell::RegExp { .. } => {
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
            Cell::Array(_) | Cell::RegExp { .. } => self
                .heap
                .get(*self.aux_props.get(&h.to_raw())?)?
                .as_object()?,
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
    for &b in &buf[pos..] {
        out.push(char::from(b));
    }
    true
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
