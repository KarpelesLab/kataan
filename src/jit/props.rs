//! The generic tier's inline property paths' access to the heap
//! (`ROADMAP.md` §2.1).
//!
//! The inline property get/set paths address an object through the heap's slot
//! array. Rather than calling a helper for the array's base and length on every
//! access, compiled code reads them where they live — the `Realm` the context's
//! `realm` box points at, then the heap's slot `Vec` — so they are always
//! current: a helper that allocated (and so maybe reallocated the array) since
//! the last access is no hazard, because nothing is cached across it.
//!
//! The receiver qualification the helper did is folded into the object
//! itself: an object whose reads or writes must reach the host carries
//! [`crate::object::Object::jit_flags`] bits, tested inline, and a primitive
//! wrapper's shape carries its brand bit. Every offset here is probed on live
//! values (see [`PropsLayout::new`]), never assumed.

use super::call_seq::{cc, r};
use super::{JitLayout, Label, X64Assembler};
use crate::shape::Shape;
use alloc::rc::Rc;

/// `(bits & HANDLE_TAG) == HANDLE_TAG` iff `bits` is a heap handle.
const HANDLE_TAG: u64 = 0xfffc_0000_0000_0000;

/// The byte offsets the inline property paths add to
/// [`super::JitLayout`] (all derived at run time).
#[derive(Clone, Copy, Debug)]
pub(crate) struct PropsLayout {
    /// Slot-relative offset of the object's flags byte (the `JIT_FLAG_*` and
    /// integrity bits).
    pub off_jit_flags: i32,
    /// Slot-relative offset of the object's raw `proto` word.
    pub off_proto: i32,
    /// Slot-relative offset of the object's `key_epoch: u32`.
    pub off_epoch: i32,
    /// Slot-relative offset of the object's `extra` pointer (null for none).
    pub off_extra: i32,
    /// `ObjectExtra`-relative offset of the accessor list's length word.
    pub extra_acc_len: i32,
    /// `ObjectExtra`-relative offset of the accessor list's data pointer.
    pub extra_acc_ptr: i32,
    /// The size of one accessor entry.
    pub acc_stride: i32,
    /// Entry-relative offset of an accessor's getter.
    pub acc_getter: i32,
    /// Shape-word-relative offset of the shape's brand byte.
    pub shape_brands: i32,
    /// `Realm`-relative offset of the heap slot array's data pointer.
    pub heap_ptr: i32,
    /// `Realm`-relative offset of the heap slot array's length.
    pub heap_len: i32,
    /// `Realm`-relative offset of the "incremental marking is running" flag.
    pub marking: i32,
    /// Slot-relative offset of the slot's `age` (collections survived).
    pub off_slot_age: i32,
    /// The `Cell::VmFunction` discriminant byte.
    pub vmfn_disc: u8,
    /// The `Cell::Box` discriminant byte.
    pub box_disc: u8,
    /// Slot-relative offset of a binding cell's value.
    pub off_box_val: i32,
    /// The `Cell::Native` discriminant byte.
    pub native_disc: u8,
    /// Slot-relative offset of a built-in native function's `u16` id.
    pub off_native_id: i32,
    /// Slot-relative offset of a VM function's `func: u32`.
    pub off_vmfn_func: i32,
    /// Slot-relative offset of a VM function's captures `Vec` data pointer
    /// (its captures when `off_vmfn_caps_inline_len` holds `ARRAY_ON_HEAP`).
    pub off_vmfn_caps_ptr: i32,
    /// Slot-relative offset of that `Vec`'s length.
    pub off_vmfn_caps_len: i32,
    /// Slot-relative offset of a VM function's inline capture count byte
    /// (`ARRAY_ON_HEAP`: the captures are in the `Vec`).
    pub off_vmfn_caps_inline_len: i32,
    /// Slot-relative offset of a VM function's inline captures.
    pub off_vmfn_caps_inline: i32,
    /// Slot-relative offset of a VM function's captures `Vec` (its words).
    pub off_vmfn_caps_vec: i32,
    /// Shape-word-relative offset of the shape's `parent` word.
    pub shape_parent: i32,
    /// Shape-word-relative offset of the shape `Rc`'s strong count.
    pub rc_strong: i32,
    /// `Realm`-relative offset of `object_dictionary_threshold`.
    pub dict_threshold: i32,
    /// `Realm`-relative offset of `max_array_len`.
    pub max_array_len: i32,
    /// `Realm`-relative offsets of the heap's free-list data pointer and
    /// length, its live count and its allocations-since-collection count
    /// (inline object allocation, `p_alloc_object`).
    pub free_ptr: i32,
    /// See `free_ptr`.
    pub free_len: i32,
    /// See `free_ptr`.
    pub heap_live: i32,
    /// See `free_ptr`.
    pub heap_allocs: i32,
    /// `Realm`-relative offset of `jit_array_proto_default` (`bool`): inline
    /// array allocation requires it set (`p_alloc_array`).
    pub array_proto_default: i32,
    /// `Realm`-relative offset of the root shape's `Rc` word.
    pub root_shape: i32,
    /// `Realm`-relative offset of the default object prototype's raw `proto`
    /// word (`Op::NewObject`'s `[[Prototype]]`).
    pub default_proto: i32,
}

impl PropsLayout {
    /// The offsets, given the slot-relative base of an object (`obj_base`) and
    /// the object probe.
    pub(crate) fn new(
        slot: &crate::heap::SlotLayout,
        cell_base: usize,
        obj_base: usize,
        obj: &crate::object::ObjectLayout,
    ) -> Self {
        let i = |v: usize| i32::try_from(v).expect("jit layout offset fits in i32");
        let (heap_ptr, heap_len) = crate::realm::Realm::jit_heap_offsets();
        let (vmfn_disc, func_off, caps_off) = crate::cell::jit_vm_function_layout();
        let (caps_vec, caps_inline_len, caps_inline) = crate::cell::jit_captures_offsets();
        let (native_disc, native_id_off) = crate::cell::jit_native_layout();
        let (box_disc, box_val_off) = crate::cell::jit_box_layout();
        let (shape_parent, rc_strong) = shape_word_offsets();
        let (alloc, root_shape, default_proto) = crate::realm::Realm::jit_alloc_offsets();
        Self {
            default_proto: i(default_proto),
            free_ptr: i(alloc.free_ptr),
            free_len: i(alloc.free_len),
            heap_live: i(alloc.live),
            heap_allocs: i(alloc.allocs_since_gc),
            array_proto_default: i(crate::realm::Realm::jit_array_proto_default_offset()),
            root_shape: i(root_shape),
            shape_parent,
            rc_strong,
            dict_threshold: i(core::mem::offset_of!(crate::realm::Realm, limits)
                + core::mem::offset_of!(crate::limits::Limits, object_dictionary_threshold)),
            max_array_len: i(core::mem::offset_of!(crate::realm::Realm, limits)
                + core::mem::offset_of!(crate::limits::Limits, max_array_len)),
            marking: i(crate::realm::Realm::jit_marking_offset()),
            off_slot_age: i(slot.age_off),
            vmfn_disc,
            native_disc,
            off_native_id: i(cell_base + native_id_off),
            box_disc,
            off_box_val: i(cell_base + box_val_off),
            off_vmfn_func: i(cell_base + func_off),
            off_vmfn_caps_ptr: i(cell_base + caps_off + caps_vec + obj.vec_ptr_off),
            off_vmfn_caps_len: i(cell_base + caps_off + caps_vec + obj.vec_len_off),
            off_vmfn_caps_inline_len: i(cell_base + caps_off + caps_inline_len),
            off_vmfn_caps_inline: i(cell_base + caps_off + caps_inline),
            off_vmfn_caps_vec: i(cell_base + caps_off + caps_vec),
            off_jit_flags: i(obj_base + obj.jit_flags_off),
            off_proto: i(obj_base + obj.proto_off),
            off_epoch: i(obj_base + obj.epoch_off),
            off_extra: i(obj_base + obj.extra_off),
            extra_acc_len: i(obj.extra_acc_len_off),
            extra_acc_ptr: i(obj.extra_acc_ptr_off),
            acc_stride: i(obj.acc_stride),
            acc_getter: i(obj.acc_getter_off),
            shape_brands: i32::try_from(obj.shape_brands_off)
                .expect("jit layout offset fits in i32"),
            heap_ptr: i(heap_ptr),
            heap_len: i(heap_len),
        }
    }
}

/// The offsets, from a shape `Rc`'s stored word (what an object's shape
/// field and a cache hold), of the shape's `parent` word and of the `Rc`'s
/// strong count — probed on a live shape: the count is the word of the
/// `Rc`'s box that follows `Rc::strong_count` as clones come and go.
#[allow(unsafe_code)]
fn shape_word_offsets() -> (i32, i32) {
    let i = |v: isize| i32::try_from(v).expect("jit layout offset fits in i32");
    let root = Shape::root();
    let child = root.add("k");
    // SAFETY: an `Rc` is one pointer word.
    let word = unsafe { core::ptr::read(core::ptr::addr_of!(child).cast::<usize>()) };
    let parent = i((Rc::as_ptr(&child) as usize + Shape::parent_offset())
        .wrapping_sub(word)
        .cast_signed());
    // SAFETY: the parent word of the live `child`, at the offset just derived.
    let parent_word = unsafe { *((word as isize + parent as isize) as *const usize) };
    assert_eq!(
        parent_word,
        // SAFETY: as above, for `root`.
        unsafe { core::ptr::read(core::ptr::addr_of!(root).cast::<usize>()) }
    );
    // The strong count: the box word (before the value) that tracks clones.
    let value_off = Rc::as_ptr(&child) as usize - word;
    let mut strong = None;
    for off in (0..value_off).step_by(core::mem::size_of::<usize>()) {
        // SAFETY: a word of the live `Rc` box header, before its value.
        let read = || unsafe { *((word + off) as *const usize) };
        let before = read();
        let extra = Rc::clone(&child);
        let after = read();
        drop(extra);
        if before == Rc::strong_count(&child) && after == before + 1 && read() == before {
            strong = Some(off);
            break;
        }
    }
    let strong = strong.expect("could not find the Rc strong count");
    (parent, i(strong.cast_signed()))
}

/// The word an object's shape field (and a cache) holds for `shape`: what
/// compiled code compares a receiver's shape with.
#[allow(unsafe_code)]
pub(crate) fn shape_word(shape: &Rc<Shape>) -> u64 {
    // SAFETY: an `Rc` is one pointer word (see `shape_word_offsets`).
    unsafe { core::ptr::read(core::ptr::addr_of!(*shape).cast::<usize>()) as u64 }
}

/// [`crate::object::JIT_FLAG_EXOTIC`], as compiled code tests it.
pub(crate) const FLAG_EXOTIC: u8 = crate::object::JIT_FLAG_EXOTIC;

/// [`crate::object::JIT_FLAG_WATCHED`], as compiled code tests it.
pub(crate) const FLAG_WATCHED: u8 = crate::object::JIT_FLAG_WATCHED;

/// Whether every object of a recorded chain is (still) watched, so that a
/// validation of it may stamp its entry.
fn levels_watched(realm: &crate::realm::Realm, levels: &[ChainLevel]) -> bool {
    levels.iter().all(|l| {
        realm
            .jit_object(crate::heap::Handle::from_raw(l.handle))
            .is_some_and(|o| o.jit_flags() & FLAG_WATCHED != 0)
    })
}

/// The flags-byte bits that send an in-place property store to the helper:
/// a frozen object, and every generic-JIT flag (exotic, global, tracked
/// prototype, a non-writable property). Not [`FLAG_WATCHED`]: a watched
/// object's existing data properties are plain stores (a value is no part of
/// the layout its watchers validate).
pub(crate) const FLAGS_NO_INLINE_STORE: u8 = crate::object::OBJ_FLAG_FROZEN
    | crate::object::JIT_FLAG_EXOTIC
    | crate::object::JIT_FLAG_NO_INLINE_SET
    | crate::object::JIT_FLAG_READONLY
    | crate::object::JIT_FLAG_GLOBAL;

/// The flags-byte bits that rule out adding a property (beyond those an
/// in-place store's guards test): sealed or non-extensible (a frozen object
/// has both).
const FLAGS_NO_ADD: u8 = crate::object::OBJ_FLAG_SEALED | crate::object::OBJ_FLAG_NON_EXTENSIBLE;

impl X64Assembler {
    /// `test byte [base+disp32], imm8` (`base` neither `rsp` nor `r12`).
    pub(crate) fn p_test_byte_imm(&mut self, base: u8, disp: i32, imm: u8) {
        debug_assert!(base & 7 != 4, "no SIB form");
        if base >= 8 {
            self.code.push(0x41);
        }
        self.code.push(0xf6);
        self.code.push(0x80 | (base & 7));
        self.code.extend_from_slice(&disp.to_le_bytes());
        self.code.push(imm);
    }

    /// The heap slot array's base into `rax` and its length into `rdx`, read
    /// from the realm (`[r15 + realm]` is the context's `Box<Realm>` pointer).
    /// Clobbers `r11`. Exactly what the old `jit_arena` call returned for a
    /// qualifying receiver.
    pub(crate) fn p_load_arena(&mut self, realm_off: i32, l: &PropsLayout) {
        self.g_load(r::R11, r::R15, realm_off);
        self.g_load(r::RAX, r::R11, l.heap_ptr);
        self.g_load(r::RDX, r::R11, l.heap_len);
    }
}

/// The deepest prototype level a [`MethodSite`] caches a method at.
pub(crate) const CHAIN_MAX: usize = 4;

/// The receiver layouts a [`MethodSite`] remembers (a polymorphic site —
/// `task.run()` over several task classes — keeps each of them).
pub(crate) const METHOD_WAYS: usize = 4;

/// How many times a [`MethodSite`] is (re)filled before it stops trying (a
/// site whose receivers keep changing layout).
const FILL_LIMIT: u32 = 32;

/// One prototype-chain object of a [`MethodEntry`]: which object it is (its
/// raw handle), and its shape and key epoch when the entry was filled.
#[repr(C)]
#[derive(Default, Clone)]
struct ChainLevel {
    handle: u64,
    shape: Option<Rc<Shape>>,
    epoch: u32,
}

/// One way of a [`MethodSite`]: where a lookup of the site's key found the
/// method — on a receiver of shape `recv_shape` (one with no accessors at
/// all), at `depth` objects up its prototype chain (0: the receiver itself,
/// else `levels[depth - 1]`), in slot `slot` of that holder — or, for a
/// property read (`acc != 0`), the holder's getter, entry `acc - 1` of its
/// accessor list (whose order its key epoch fixes). Empty while
/// `recv_shape` is `None` (no shape word is null).
///
/// `stamp` is the prototype epoch (`crate::object::proto_epoch`) at which the
/// chain was last validated — every level then the recorded object, watched
/// (`JIT_FLAG_WATCHED`), with the recorded shape and key epoch — and
/// `holder` the raw handle of the object holding the property (for `depth >
/// 0`). While the epoch is still `stamp`, no watched object changed layout,
/// so a receiver of `recv_shape` whose `[[Prototype]]` is still
/// `levels[0].handle` has the recorded chain without re-walking it.
#[repr(C)]
#[derive(Default, Clone)]
struct MethodEntry {
    recv_shape: Option<Rc<Shape>>,
    depth: u32,
    slot: u32,
    acc: u32,
    stamp: u64,
    holder: u64,
    /// The address of the holder's value (its slot, or its getter for
    /// `acc != 0`) as of `stamp`, or `0`: while the epoch is still `stamp`
    /// the holder is the recorded, watched, unchanged object, and its cell
    /// has not moved (see `Object::jit_slot_ptr`).
    value: u64,
    levels: [ChainLevel; CHAIN_MAX],
}

/// A property site's prototype-chain caches. For a method call, read by
/// compiled code (see `X64Assembler::p_method_lookup`): up to
/// `METHOD_WAYS` entries (`MethodEntry`), replaced round-robin. For a
/// property write, the add cache (`AddEntry`), read by its helper.
///
/// A hit re-validates the entry's whole chain: the receiver's shape, then for
/// each level that the previous object's `[[Prototype]]` is still the
/// recorded object, and that its shape and key epoch are unchanged. A shape
/// fixes an object's data keys (so the key is absent below the holder and in
/// the holder's slot); an unchanged key epoch means no accessor was added or
/// removed since the fill checked there is none of that name. Host-exotic
/// objects and primitive wrappers are never recorded, and dictionary-mode
/// objects never match a shape. The value is read from the holder's slot live.
#[repr(C)]
pub struct MethodSite {
    entries: [MethodEntry; METHOD_WAYS],
    /// The site's key (UTF-8), for the fill helper.
    key_ptr: *const u8,
    /// The key's length.
    key_len: usize,
    /// The site's `PropertyCache`, for the fill helper.
    pub(crate) cache: *mut core::ffi::c_void,
    /// The way the next new entry replaces.
    next: usize,
    fills: u32,
    /// The site's add cache, when it is a property write.
    add: AddEntry,
}

impl MethodSite {
    /// An empty site for `key` with the property cache `cache`.
    pub(crate) fn new(key: &str, cache: *mut core::ffi::c_void) -> Self {
        Self {
            entries: Default::default(),
            key_ptr: key.as_ptr(),
            key_len: key.len(),
            cache,
            next: 0,
            fills: 0,
            add: AddEntry::default(),
        }
    }

    /// The site's key.
    pub(crate) fn key(&self) -> &'static str {
        // SAFETY: `key_ptr`/`key_len` are the owning `JitProto`'s boxed key
        // string, which outlives every use of the site (the helpers that read
        // it run while the compiled code holding the site runs).
        #[allow(unsafe_code)]
        unsafe {
            core::str::from_utf8_unchecked(core::slice::from_raw_parts(self.key_ptr, self.key_len))
        }
    }

    /// Records a lookup of the site's key on the object at `recv` (whose
    /// method the caller found to be a VM closure): the whole chain is walked
    /// and recorded only when every object on it is one the compiled check
    /// can validate (see the type docs). The entry replaces the way holding
    /// the same receiver layout and prototype, else the next way round.
    pub(crate) fn fill(&mut self, realm: &mut crate::realm::Realm, recv: crate::heap::Handle) {
        if !self.may_fill() || self.key().starts_with('\0') {
            return;
        }
        let Some(mut entry) = Self::entry_for(realm, recv, self.key(), false) else {
            self.count_fill(false);
            return;
        };
        if !self.count_fill(self.holds_layout(&entry)) {
            return;
        }
        watch_levels(realm, &entry.levels[..entry.depth as usize]);
        entry.stamp = crate::object::proto_epoch();
        entry.value = holder_value_addr(realm, &entry);
        self.store(entry);
    }

    /// Whether a fill may still be tried: under the limit, or some entry's
    /// stamp is stale (its revalidation, a restamp, is always allowed).
    fn may_fill(&self) -> bool {
        self.fills < FILL_LIMIT || {
            let now = crate::object::proto_epoch();
            self.entries
                .iter()
                .any(|e| e.recv_shape.is_some() && e.depth != 0 && e.stamp != now)
        }
    }

    /// Counts a fill against the limit unless it only restamps an entry the
    /// site already holds (`restamp`); whether the fill may proceed.
    fn count_fill(&mut self, restamp: bool) -> bool {
        if restamp {
            return true;
        }
        if self.fills >= FILL_LIMIT {
            // A retired site: drop its stale entries (they can only miss),
            // so that `may_fill` stops trying.
            let now = crate::object::proto_epoch();
            for e in &mut self.entries {
                if e.depth != 0 && e.stamp != now {
                    *e = MethodEntry::default();
                }
            }
            return false;
        }
        self.fills += 1;
        true
    }

    /// Whether some way already records `entry`'s layout exactly (the same
    /// receiver shape, holder, slot and chain), so storing it only restamps.
    fn holds_layout(&self, entry: &MethodEntry) -> bool {
        let same_shape = |a: &Option<Rc<Shape>>, b: &Option<Rc<Shape>>| match (a, b) {
            (Some(a), Some(b)) => Rc::ptr_eq(a, b),
            _ => false,
        };
        self.entries.iter().any(|e| {
            same_shape(&e.recv_shape, &entry.recv_shape)
                && e.depth == entry.depth
                && e.slot == entry.slot
                && e.acc == entry.acc
                && e.holder == entry.holder
                && e.levels[..e.depth as usize]
                    .iter()
                    .zip(&entry.levels[..entry.depth as usize])
                    .all(|(a, b)| {
                        a.handle == b.handle && a.epoch == b.epoch && same_shape(&a.shape, &b.shape)
                    })
        })
    }

    /// Records a property read of the site's key on the object at `recv`, as
    /// [`fill`](Self::fill) does a method lookup, except that the property
    /// may be an inherited data property or an inherited VM-function getter
    /// (which is returned, for the read's call cache).
    pub(crate) fn fill_get(
        &mut self,
        realm: &mut crate::realm::Realm,
        recv: crate::heap::Handle,
    ) -> Option<crate::nanbox::NanBox> {
        let key = self.key();
        if !self.may_fill() || key.starts_with('\0') || key == "__proto__" {
            return None;
        }
        let Some(mut entry) = Self::entry_for(realm, recv, key, true) else {
            self.count_fill(false);
            return None;
        };
        if !self.count_fill(self.holds_layout(&entry)) {
            return None;
        }
        watch_levels(realm, &entry.levels[..entry.depth as usize]);
        entry.stamp = crate::object::proto_epoch();
        entry.value = holder_value_addr(realm, &entry);
        let getter = (entry.acc != 0)
            .then(|| {
                let h =
                    crate::heap::Handle::from_raw(entry.levels[entry.depth as usize - 1].handle);
                realm.jit_object(h)?.accessor(key).map(|(g, _)| g)
            })
            .flatten();
        self.store(entry);
        getter
    }

    /// Puts `entry` in the way holding the same receiver layout and
    /// prototype, else the next way round.
    fn store(&mut self, entry: MethodEntry) {
        let way = self
            .entries
            .iter()
            .position(|e| {
                e.recv_shape
                    .as_ref()
                    .zip(entry.recv_shape.as_ref())
                    .is_some_and(|(a, b)| Rc::ptr_eq(a, b))
                    && e.depth == entry.depth
                    && e.levels[0].handle == entry.levels[0].handle
            })
            .unwrap_or_else(|| {
                let w = self.next % METHOD_WAYS;
                self.next = (w + 1) % METHOD_WAYS;
                w
            });
        self.entries[way] = entry;
    }

    /// The entry a lookup of `key` on `recv` makes, if it can be cached.
    fn entry_for(
        realm: &crate::realm::Realm,
        recv: crate::heap::Handle,
        key: &str,
        getters: bool,
    ) -> Option<MethodEntry> {
        let o = realm.jit_chain_object(recv)?;
        let shape = o.shaped_shape().filter(|_| o.has_no_accessors())?;
        let mut entry = MethodEntry {
            recv_shape: Some(Rc::clone(shape)),
            ..MethodEntry::default()
        };
        if let Some(slot) = shape.lookup(key) {
            entry.slot = slot;
            return Some(entry);
        }
        let mut cur = o.proto();
        for (d, level) in (1u32..).zip(entry.levels.iter_mut()) {
            let h = cur?;
            let p = realm.jit_chain_object(h)?;
            let ps = p.shaped_shape()?;
            *level = ChainLevel {
                handle: h.to_raw(),
                shape: Some(Rc::clone(ps)),
                epoch: p.key_epoch(),
            };
            if let Some(i) = p.accessor_index(key) {
                // Only a read takes an accessor, and only a VM-function getter.
                let (getter, _) = p.accessor(key)?;
                if !getters
                    || !getter
                        .as_handle()
                        .is_some_and(|g| realm.is_vm_function(crate::heap::Handle::from_raw(g)))
                {
                    return None;
                }
                entry.depth = d;
                entry.holder = h.to_raw();
                entry.acc = u32::try_from(i + 1).ok()?;
                return Some(entry);
            }
            if let Some(slot) = ps.lookup(key) {
                entry.depth = d;
                entry.holder = h.to_raw();
                entry.slot = slot;
                return Some(entry);
            }
            cur = p.proto();
        }
        None
    }
}

/// [`MethodEntry::value`] for a freshly validated `entry`: the address of its
/// holder's slot or getter (`0` for a receiver's own property or an
/// unaddressable one).
fn holder_value_addr(realm: &crate::realm::Realm, entry: &MethodEntry) -> u64 {
    if entry.depth == 0 {
        return 0;
    }
    let Some(o) = realm.jit_object(crate::heap::Handle::from_raw(entry.holder)) else {
        return 0;
    };
    let ptr = if entry.acc != 0 {
        o.jit_getter_ptr(entry.acc as usize - 1)
    } else {
        o.jit_slot_ptr(entry.slot)
    };
    ptr.map_or(0, |p| p as u64)
}

/// Flags every object of a recorded chain `levels` watched (see
/// `crate::object::JIT_FLAG_WATCHED`), so that a change to any of them
/// retires the recording entry's stamp.
fn watch_levels(realm: &mut crate::realm::Realm, levels: &[ChainLevel]) {
    for l in levels {
        realm.jit_watch(crate::heap::Handle::from_raw(l.handle));
    }
}

impl X64Assembler {
    /// `mov r32, [base+disp32]` (zero-extending; `base` neither `rsp` nor `r12`).
    fn p_load32(&mut self, dst: u8, base: u8, disp: i32) {
        self.p_rm32(0x8b, dst, base, disp);
    }

    /// `movzx edx, byte [base+disp32]`.
    fn p_load_u8_rdx(&mut self, base: u8, disp: i32) {
        debug_assert!(base & 7 != 4, "no SIB form");
        if base >= 8 {
            self.code.push(0x41);
        }
        self.code.extend_from_slice(&[0x0f, 0xb6]);
        self.code.push(0x80 | (2 << 3) | (base & 7));
        self.code.extend_from_slice(&disp.to_le_bytes());
    }

    /// `movzx r32, word [base+disp32]`.
    fn p_load16(&mut self, dst: u8, base: u8, disp: i32) {
        debug_assert!(base & 7 != 4, "no SIB form");
        let rex = 0x40 | (((dst >> 3) & 1) << 2) | ((base >> 3) & 1);
        if rex != 0x40 {
            self.code.push(rex);
        }
        self.code.extend_from_slice(&[0x0f, 0xb7]);
        self.code.push(0x80 | ((dst & 7) << 3) | (base & 7));
        self.code.extend_from_slice(&disp.to_le_bytes());
    }

    /// `inc word [base+disp32]`.
    fn p_inc16_mem(&mut self, base: u8, disp: i32) {
        debug_assert!(base & 7 != 4, "no SIB form");
        self.code.push(0x66);
        if base >= 8 {
            self.code.push(0x41);
        }
        self.code.push(0xff);
        self.code.push(0x80 | (base & 7));
        self.code.extend_from_slice(&disp.to_le_bytes());
    }

    /// A 32-bit `op reg, [base+disp32]`.
    fn p_rm32(&mut self, opcode: u8, reg: u8, base: u8, disp: i32) {
        debug_assert!(base & 7 != 4, "no SIB form");
        let rex = 0x40 | (((reg >> 3) & 1) << 2) | ((base >> 3) & 1);
        if rex != 0x40 {
            self.code.push(rex);
        }
        self.code.push(opcode);
        self.code.push(0x80 | ((reg & 7) << 3) | (base & 7));
        self.code.extend_from_slice(&disp.to_le_bytes());
    }

    /// `cmp dword [base+disp32], imm8` (sign-extended).
    fn p_cmp32_imm8(&mut self, base: u8, disp: i32, imm: i8) {
        debug_assert!(base & 7 != 4, "no SIB form");
        if base >= 8 {
            self.code.push(0x41);
        }
        self.code.push(0x83);
        self.code.push(0x80 | (7 << 3) | (base & 7));
        self.code.extend_from_slice(&disp.to_le_bytes());
        self.code.push(imm.cast_unsigned());
    }

    /// `shl rdx, 32`.
    fn p_shl_rdx_32(&mut self) {
        self.code.extend_from_slice(&[0x48, 0xc1, 0xe2, 32]);
    }

    /// `or rax, rdx`.
    fn p_or_rax_rdx(&mut self) {
        self.code.extend_from_slice(&[0x48, 0x09, 0xd0]);
    }

    /// `mov eax, eax` (clears `rax`'s upper half).
    fn p_zext_eax(&mut self) {
        self.code.extend_from_slice(&[0x89, 0xc0]);
    }

    /// Jumps to `miss` unless `rax` holds a heap handle. Clobbers `rcx`, `r11`.
    fn p_handle_check(&mut self, miss: Label) {
        self.g_movabs(r::R11, HANDLE_TAG);
        self.g_mov(r::RCX, r::RAX);
        self.g_and(r::RCX, r::R11);
        self.g_cmp(r::RCX, r::R11);
        self.g_jcc(cc::NE, miss);
    }

    /// `rax = r10 + rax * slot_stride` (the slot of arena index `rax`, the
    /// arena base in `r10`). The usual 72-byte stride is two `lea`s — two
    /// cycles on the dependency chain of every handle decode, where
    /// `imul` + `add` take four.
    fn p_index_to_slot(&mut self, l: &JitLayout) {
        if l.slot_stride == 72 {
            self.code.extend_from_slice(&[0x48, 0x8d, 0x04, 0xc0]); // lea rax, [rax + rax*8]
            self.code.extend_from_slice(&[0x49, 0x8d, 0x04, 0xc2]); // lea rax, [r10 + rax*8]
        } else {
            self.imul_rax_imm(l.slot_stride);
            self.add_rax_r10();
        }
    }

    /// From the handle bits in `rax` (a `NanBox` handle or a raw handle: only
    /// the low 48 bits are read), `rcx = &Slot` of a live slot with that
    /// generation, given the arena base in `r10` and length in `rdx`; else to
    /// `miss`. Clobbers `rax`, `r8`.
    fn p_slot_of(&mut self, l: &JitLayout, miss: Label) {
        self.mov_r8_rax();
        self.shr_r8_imm(32); // r8w = generation
        self.p_zext_eax(); // rax = index
        self.cmp_rax_rdx();
        self.jae(miss);
        self.p_index_to_slot(l);
        self.mov_rcx_rax();
        self.cmp_byte_at_rcx_imm(l.off_slot_occ, l.slot_occupied_disc);
        self.jne(miss);
        self.cmp_word_at_rcx_r8w(l.off_slot_gen);
        self.jne(miss);
    }

    /// Jumps to `miss` unless `rcx` is the slot of an ordinary (non-exotic)
    /// shaped object.
    fn p_plain_shaped(&mut self, l: &JitLayout, miss: Label) {
        self.cmp_byte_at_rcx_imm(l.off_cell_tag, l.cell_object_disc);
        self.jne(miss);
        self.cmp_byte_at_rcx_imm(l.off_od_tag, l.obj_shaped_disc);
        self.jne(miss);
        self.p_test_byte_imm(r::RCX, l.props.off_jit_flags, FLAG_EXOTIC);
        self.jne(miss);
    }

    /// `rax` = the prototype epoch (a plain load of its counter).
    fn p_load_proto_epoch(&mut self, dst: u8) {
        self.g_movabs(dst, crate::object::proto_epoch_addr() as u64);
        self.g_load(dst, dst, 0);
    }

    /// The method lookup of a [`MethodSite`] (see its docs), on the receiver
    /// value at `[recv_base + recv_disp]`: on a hit, the method value in
    /// `rax`; otherwise to `miss`. Reads only; clobbers `rax`, `rcx`, `rdx`,
    /// `rsi`, `rdi`, `r8`–`r11`.
    pub(crate) fn p_method_lookup(
        &mut self,
        l: &JitLayout,
        realm_off: i32,
        (recv_base, recv_disp): (u8, i32),
        site: *const MethodSite,
        miss: Label,
        getter: Option<Label>,
    ) {
        self.p_load_arena(realm_off, &l.props);
        self.mov_r10_rax();
        self.g_load(r::RAX, recv_base, recv_disp);
        self.p_handle_check(miss);
        self.p_slot_of(l, miss);
        self.p_plain_shaped(l, miss);
        self.p_method_lookup_from(l, site, miss, getter);
    }

    /// [`p_method_lookup`](Self::p_method_lookup) for a receiver already
    /// walked: `rcx` its live heap slot, an ordinary shaped object without the
    /// host-exotic flag (`p_plain_shaped`'s facts), with the arena reloaded
    /// here for the prototype levels.
    pub(crate) fn p_method_lookup_walked(
        &mut self,
        l: &JitLayout,
        realm_off: i32,
        site: *const MethodSite,
        miss: Label,
        getter: Option<Label>,
    ) {
        self.p_load_arena(realm_off, &l.props);
        self.mov_r10_rax();
        self.p_method_lookup_from(l, site, miss, getter);
    }

    /// The rest of a method lookup, from the walked receiver's slot in `rcx`
    /// and the arena in `r10` / `rdx`.
    fn p_method_lookup_from(
        &mut self,
        l: &JitLayout,
        site: *const MethodSite,
        miss: Label,
        getter: Option<Label>,
    ) {
        use core::mem::{offset_of, size_of};
        let i = |v: usize| i32::try_from(v).expect("site offset fits in i32");
        let level = |d: usize, f: usize| {
            i(offset_of!(MethodEntry, levels) + d * size_of::<ChainLevel>() + f)
        };
        // No accessors on the receiver: no `extra`, or an empty accessor list.
        let no_acc = self.new_label();
        self.g_load(r::R9, r::RCX, l.props.off_extra);
        self.g_test(r::R9);
        self.g_jcc(cc::E, no_acc);
        self.g_load(r::R9, r::R9, l.props.extra_acc_len);
        self.g_test(r::R9);
        self.g_jcc(cc::NE, miss);
        self.bind(no_acc);
        // rsi = the receiver's slot, r11 = the way, edi = ways left.
        self.g_mov(r::RSI, r::RCX);
        self.g_movabs(
            r::R11,
            site as u64 + u64::try_from(offset_of!(MethodSite, entries)).expect("offset"),
        );
        self.g_movabs(r::RDI, METHOD_WAYS as u64);
        let way = self.new_label();
        let next = self.new_label();
        let holder = self.new_label();
        let depth = i(offset_of!(MethodEntry, depth));
        self.bind(way);
        self.g_load(r::RAX, r::RSI, l.off_shape);
        self.g_cmp_mem(r::RAX, r::R11, i(offset_of!(MethodEntry, recv_shape)));
        self.g_jcc(cc::NE, next);
        self.g_mov(r::RCX, r::RSI);
        self.p_cmp32_imm8(r::R11, depth, 0);
        self.g_jcc(cc::E, holder);
        // The chain as validated at the entry's stamp, while the prototype
        // epoch still is the stamp: only the receiver's (unwatched) link to
        // it needs checking, and the holder is the recorded one. A stale
        // stamp is a miss: the helper revalidates and restamps the entry
        // (`MethodSite::fill`), keeping the walk out of the compiled code.
        self.p_load_proto_epoch(r::R9);
        self.g_cmp_mem(r::R9, r::R11, i(offset_of!(MethodEntry, stamp)));
        self.g_jcc(cc::NE, next);
        self.g_load(r::RAX, r::RCX, l.props.off_proto);
        self.g_cmp_mem(r::RAX, r::R11, level(0, offset_of!(ChainLevel, handle)));
        self.g_jcc(cc::NE, next);
        // The holder's value at its address as of the stamp, if recorded.
        let walk = self.new_label();
        let done = self.new_label();
        self.g_load(r::R9, r::R11, i(offset_of!(MethodEntry, value)));
        self.g_test(r::R9);
        self.g_jcc(cc::E, walk);
        self.g_load(r::RAX, r::R9, 0);
        self.p_cmp32_imm8(r::R11, i(offset_of!(MethodEntry, acc)), 0);
        self.g_jcc(cc::E, done);
        self.jmp(getter.unwrap_or(next));
        self.bind(walk);
        self.g_load(r::RAX, r::R11, i(offset_of!(MethodEntry, holder)));
        self.p_slot_of(l, next);
        // `rcx` = the holder's slot: its getter, or its `slot`-th value.
        self.bind(holder);
        let acc = i(offset_of!(MethodEntry, acc));
        let accessor = self.new_label();
        self.p_cmp32_imm8(r::R11, acc, 0);
        self.g_jcc(cc::NE, accessor);
        self.p_load32(r::RAX, r::R11, i(offset_of!(MethodEntry, slot)));
        self.p_obj_slot_addr(l, r::R9, next);
        self.g_load(r::RAX, r::R9, 0);
        self.jmp(done);
        // The getter: entry `acc - 1` of the holder's accessor list.
        self.bind(accessor);
        if let Some(getter) = getter {
            self.p_load32(r::RAX, r::R11, acc);
            self.p_dec_eax();
            self.g_load(r::R9, r::RCX, l.props.off_extra);
            self.g_test(r::R9);
            self.g_jcc(cc::E, next);
            self.g_cmp_mem(r::RAX, r::R9, l.props.extra_acc_len);
            self.g_jcc(cc::AE, next);
            self.imul_rax_imm(l.props.acc_stride);
            self.g_load(r::R9, r::R9, l.props.extra_acc_ptr);
            self.p_add_rax_r9();
            self.g_load(r::RAX, r::RAX, l.props.acc_getter);
            self.jmp(getter);
        } else {
            self.jmp(next);
        }
        self.bind(next);
        self.g_lea(r::R11, r::R11, i(size_of::<MethodEntry>()));
        self.p_dec_edi();
        self.g_jcc(cc::NE, way);
        self.jmp(miss);
        self.bind(done);
    }

    /// With a slot index (zero-extended) in `rax` and an object's heap slot in
    /// `rcx`: `dst` = the address of that property value — inline in the
    /// object for the first [`crate::object::INLINE_SLOTS`], else in its spill
    /// array — or to `miss` when the index is not below the object's slot
    /// count. Clobbers only `dst` (neither `rax` nor `rcx`).
    pub(crate) fn p_obj_slot_addr(&mut self, l: &JitLayout, dst: u8, miss: Label) {
        const INLINE: i32 = crate::object::INLINE_SLOTS as i32;
        debug_assert!(dst != r::RAX && dst != r::RCX);
        self.p_load16(dst, r::RCX, l.off_slots_len);
        self.g_cmp(r::RAX, dst);
        self.g_jcc(cc::AE, miss);
        let spill = self.new_label();
        let done = self.new_label();
        self.p_cmp32_imm(r::RAX, INLINE);
        self.g_jcc(cc::AE, spill);
        self.p_lea_index8(dst, r::RCX, r::RAX, l.off_inline);
        self.jmp(done);
        self.bind(spill);
        // A spilled slot means the object has its `extra` part.
        self.g_load(dst, r::RCX, l.props.off_extra);
        self.g_load(dst, dst, l.extra_spill_ptr);
        self.p_lea_index8(dst, dst, r::RAX, -8 * INLINE);
        self.bind(done);
    }

    /// The inline hit of `Op::InitProp` through its transition cache at
    /// `cache` (a `PropertyCache`): exactly
    /// `Object::init_data_cached`'s cached add, for the object whose heap slot
    /// is `rcx` and the value at
    /// `[rbx + val_disp]`. It qualifies when the object is an ordinary shaped
    /// one with no `jit_flags`, extensible, neither frozen nor sealed, with no
    /// `extra` part (no accessors, attribute marks or key-order log), whose
    /// shape is the cached shape's parent, with room for the value inline and
    /// below the dictionary threshold, and when the store needs nothing from
    /// the write barrier. Then: the value goes into the next inline slot, the
    /// object adopts the cached shape (moving one strong count from the old
    /// shape to the new; the cached shape holds its parent, so the old one
    /// stays alive), its key epoch and the cache's hit count are bumped.
    /// Otherwise to `miss`, having changed nothing. Clobbers `rax`, `r8`,
    /// `r9`, `r11`.
    pub(crate) fn p_init_prop_inline(
        &mut self,
        l: &JitLayout,
        realm_off: i32,
        cache: u64,
        val_disp: i32,
        miss: Label,
    ) {
        self.p_add_guards(l, realm_off, miss);
        self.p_init_prop_commit(l, realm_off, cache, val_disp, miss);
    }

    /// [`Self::p_init_prop_inline`] on a **fresh literal** (the object in
    /// `[rbx + obj_disp]`, see `recv_cache::fresh_inits`): a live ordinary
    /// object no other code has seen, so neither the walk's handle,
    /// liveness and family checks nor the add guards only such code could
    /// break (flags, frozen, sealed, extensible) are needed; what the
    /// literal's own earlier definitions may have changed (a helper's add:
    /// shaped storage, the `extra` part, inline room, the dictionary
    /// threshold) is still checked. Leaves `rcx` at the object's heap slot.
    pub(crate) fn p_init_prop_fresh(
        &mut self,
        l: &JitLayout,
        realm_off: i32,
        (obj_disp, val_disp): (i32, i32),
        cache: u64,
        miss: Label,
    ) {
        self.p_load_arena(realm_off, &l.props);
        self.mov_r10_rax();
        self.g_load(r::RAX, r::RBX, obj_disp);
        self.p_zext_eax(); // rax = handle index
        self.p_index_to_slot(l);
        self.mov_rcx_rax(); // rcx = &Slot
        self.cmp_byte_at_rcx_imm(l.off_od_tag, l.obj_shaped_disc);
        self.jne(miss);
        self.p_add_room(l, realm_off, miss);
        self.p_init_prop_commit(l, realm_off, cache, val_disp, miss);
    }

    /// The transition-cache half of an inline literal-property add, after
    /// the guards (the object's heap slot in `rcx`).
    fn p_init_prop_commit(
        &mut self,
        l: &JitLayout,
        realm_off: i32,
        cache: u64,
        val_disp: i32,
        miss: Label,
    ) {
        let cache_shape = l.cache_shape_off;
        // The cached shape, whose parent must be the object's shape.
        self.g_movabs(r::R11, cache);
        self.g_load(r::R9, r::R11, cache_shape);
        self.g_test(r::R9);
        self.g_jcc(cc::E, miss);
        self.g_load(r::R8, r::R9, l.props.shape_parent);
        self.g_cmp_mem(r::R8, r::RCX, l.off_shape);
        self.g_jcc(cc::NE, miss);
        self.g_load(r::RAX, r::RBX, val_disp);
        self.p_barrier_check(l, realm_off, miss);
        // Commit, and count the hit.
        self.g_movabs(r::R11, cache);
        self.p_add_commit(l, cache_shape, val_disp);
        self.p_inc32_mem(r::R11, l.cache_hits_off);
    }

    /// For the array whose heap slot is `rcx`: to `miss` unless it has no
    /// side-table state (`crate::cell::ArrayElems`) — its `length` is its
    /// element count and its in-bounds elements are plain writable data
    /// properties — and, for a store (`set`), its length is within
    /// `max_array_len`. What `Realm::jit_array_len_is_dense` (and, for a store,
    /// `Realm::jit_array_unrestricted`) decide, from the cell alone. Clobbers
    /// `r8`, `r11`.
    pub(crate) fn p_array_plain(&mut self, l: &JitLayout, realm_off: i32, set: bool, miss: Label) {
        self.cmp_byte_at_rcx_imm(l.off_arr_side, 0);
        self.jne(miss);
        if set {
            self.g_load(r::R11, r::R15, realm_off);
            self.p_arr_len(l, r::R8);
            self.g_cmp_mem(r::R8, r::R11, l.props.max_array_len);
            self.g_jcc(cc::A, miss);
        }
    }

    /// `movzx r32, byte [base+disp32]`.
    fn p_load8(&mut self, dst: u8, base: u8, disp: i32) {
        debug_assert!(base & 7 != 4, "no SIB form");
        let rex = 0x40 | (((dst >> 3) & 1) << 2) | ((base >> 3) & 1);
        if rex != 0x40 {
            self.code.push(rex);
        }
        self.code.extend_from_slice(&[0x0f, 0xb6]);
        self.code.push(0x80 | ((dst & 7) << 3) | (base & 7));
        self.code.extend_from_slice(&disp.to_le_bytes());
    }

    /// For the array whose heap slot is `rcx`: `reg` = its element count —
    /// the inline count, or the `Vec`'s length when the elements are there
    /// (`crate::cell::ArrayElems`). Clobbers only `reg` (and the flags).
    pub(crate) fn p_arr_len(&mut self, l: &JitLayout, reg: u8) {
        self.p_load8(reg, r::RCX, l.off_arr_inline_len);
        self.g_cmp_imm(reg, i32::from(crate::cell::ARRAY_ON_HEAP));
        self.g_cmove_mem(reg, r::RCX, l.off_arr_len);
    }

    /// For the array whose heap slot is `rcx`: `r8` = its element count and
    /// `r11` = the address of its first element — in the cell, or the `Vec`'s
    /// buffer (`crate::cell::ArrayElems`). Branch-free. Clobbers only `r8`
    /// and `r11` (and the flags).
    pub(crate) fn p_arr_len_ptr(&mut self, l: &JitLayout) {
        self.p_load8(r::R8, r::RCX, l.off_arr_inline_len);
        self.g_lea(r::R11, r::RCX, l.off_arr_inline);
        self.g_cmp_imm(r::R8, i32::from(crate::cell::ARRAY_ON_HEAP));
        self.g_cmove_mem(r::R8, r::RCX, l.off_arr_len);
        self.g_cmove_mem(r::R11, r::RCX, l.off_arr_ptr);
    }

    /// `cmove reg, [base+disp32]` (64-bit; the load happens either way).
    fn g_cmove_mem(&mut self, reg: u8, base: u8, disp: i32) {
        debug_assert!(base & 7 != 4, "no SIB form");
        let rex = 0x48 | (((reg >> 3) & 1) << 2) | ((base >> 3) & 1);
        self.code.extend_from_slice(&[rex, 0x0f, 0x44]);
        self.code.push(0x80 | ((reg & 7) << 3) | (base & 7));
        self.code.extend_from_slice(&disp.to_le_bytes());
    }

    /// The common guards of an inline property add, on the object whose heap
    /// slot is `rcx`: an ordinary shaped object with no `jit_flags`,
    /// extensible, neither frozen nor sealed (its whole flags byte zero), with
    /// no `extra` part (no accessors, attribute marks or key-order log), with
    /// room for one more value inline and below the dictionary threshold.
    /// Else to `miss`. Clobbers `rax`, `r8`, `r9`.
    fn p_add_guards(&mut self, l: &JitLayout, realm_off: i32, miss: Label) {
        self.cmp_byte_at_rcx_imm(l.off_cell_tag, l.cell_object_disc);
        self.jne(miss);
        self.cmp_byte_at_rcx_imm(l.off_od_tag, l.obj_shaped_disc);
        self.jne(miss);
        self.cmp_byte_at_rcx_imm(l.props.off_jit_flags, 0);
        self.jne(miss);
        self.p_add_room(l, realm_off, miss);
    }

    /// The add guards an object's own data-property definitions can change
    /// (on the shaped object whose heap slot is `rcx`): no `extra` part, room
    /// for one more value inline, below the dictionary threshold. Else to
    /// `miss`. Clobbers `rax`, `r8`, `r9`.
    fn p_add_room(&mut self, l: &JitLayout, realm_off: i32, miss: Label) {
        const INLINE: i32 = crate::object::INLINE_SLOTS as i32;
        self.g_load(r::R9, r::RCX, l.props.off_extra);
        self.g_test(r::R9);
        self.g_jcc(cc::NE, miss);
        self.p_load16(r::RAX, r::RCX, l.off_slots_len);
        self.p_cmp32_imm(r::RAX, INLINE);
        self.g_jcc(cc::AE, miss);
        self.g_load(r::R8, r::R15, realm_off);
        self.g_cmp_mem(r::RAX, r::R8, l.props.dict_threshold);
        self.g_jcc(cc::AE, miss);
    }

    /// The commit of an inline property add to the object whose heap slot is
    /// `rcx`, taking it to the shape whose `Rc` word is at `[r11 + to_disp]`
    /// (a child of its current shape, which that child keeps alive) with the
    /// value at `[rbx + val_disp]` in its next inline slot: the value, the
    /// count, the shape (one strong count moved from the old shape to the
    /// new) and the key epoch. Clobbers `rax`, `r8`, `r9`.
    fn p_add_commit(&mut self, l: &JitLayout, to_disp: i32, val_disp: i32) {
        self.p_load16(r::RAX, r::RCX, l.off_slots_len);
        self.p_lea_index8(r::R9, r::RCX, r::RAX, l.off_inline);
        self.g_load(r::R8, r::RBX, val_disp);
        self.g_store(r::R9, 0, r::R8);
        self.p_inc16_mem(r::RCX, l.off_slots_len);
        self.g_load(r::R9, r::R11, to_disp);
        self.g_inc_mem(r::R9, l.props.rc_strong);
        self.g_load(r::R8, r::RCX, l.off_shape);
        self.g_dec_mem(r::R8, l.props.rc_strong);
        self.g_store(r::RCX, l.off_shape, r::R9);
        self.p_inc32_mem(r::RCX, l.props.off_epoch);
    }

    /// To `miss` unless `site`'s add cache is armed (a cheap test before
    /// [`Self::p_add_prop_inline`]'s receiver walk). Clobbers `r11`.
    pub(crate) fn p_add_armed(&mut self, site: *const MethodSite, miss: Label) {
        let from = i32::try_from(
            core::mem::offset_of!(MethodSite, add) + core::mem::offset_of!(AddEntry, from),
        )
        .expect("site offset fits in i32");
        self.g_movabs(r::R11, site as u64);
        self.g_load(r::R11, r::R11, from);
        self.g_test(r::R11);
        self.g_jcc(cc::E, miss);
    }

    /// The inline hit of a [`MethodSite`]'s add cache ([`AddEntry`]; what
    /// `MethodSite::try_add` and then `Realm::jit_add_transition` do), for the
    /// object whose heap slot is `rcx` (arena base in `r10`, length in `rdx`)
    /// and the value at `[rbx + val_disp]`: the object passes
    /// [`Self::p_add_guards`] and has the entry's `from` shape; each of the
    /// entry's `depth` prototype levels is still the recorded object, ordinary
    /// and non-exotic, with the recorded shape and key epoch; the chain ends
    /// there; and the store needs nothing from the write barrier. Then the
    /// add is made ([`Self::p_add_commit`]); otherwise to `miss`, having
    /// changed nothing. Clobbers `rax`, `rcx`, `rsi`, `r8`, `r9`, `r11`.
    ///
    /// `walked`: the object is already known to be an ordinary shaped,
    /// non-exotic object, neither frozen nor carrying `jit_flags` (an
    /// in-place store's guards passed before its shape missed), so only the
    /// rest of the add guards are checked.
    pub(crate) fn p_add_prop_inline(
        &mut self,
        l: &JitLayout,
        realm_off: i32,
        site: *const MethodSite,
        val_disp: i32,
        miss: Label,
        walked: bool,
    ) {
        use core::mem::{offset_of, size_of};
        let i = |v: usize| i32::try_from(v).expect("site offset fits in i32");
        let add = offset_of!(MethodSite, add);
        let level = |d: usize, f: usize| {
            i(add + offset_of!(AddEntry, levels) + d * size_of::<ChainLevel>() + f)
        };
        let from = i(add + offset_of!(AddEntry, from));
        let to = i(add + offset_of!(AddEntry, to));
        let depth = i(add + offset_of!(AddEntry, depth));
        self.g_movabs(r::R11, site as u64);
        if walked {
            self.p_test_byte_imm(r::RCX, l.props.off_jit_flags, FLAGS_NO_ADD);
            self.jne(miss);
            self.p_add_room(l, realm_off, miss);
        } else {
            self.p_add_guards(l, realm_off, miss);
        }
        self.g_load(r::R9, r::R11, from);
        self.g_test(r::R9);
        self.g_jcc(cc::E, miss);
        self.g_cmp_mem(r::R9, r::RCX, l.off_shape);
        self.g_jcc(cc::NE, miss);
        // (`to` is set whenever `from` is: `fill_add` stores both.)
        // The prototype chain, receiver's `[[Prototype]]` first.
        self.g_mov(r::RSI, r::RCX);
        let end = self.new_label();
        let chain_ok = self.new_label();
        let stamp = i(add + offset_of!(AddEntry, stamp));
        // While the prototype epoch is the entry's stamp, the chain is the
        // validated one: only the receiver's link to it needs checking. A
        // stale stamp is a miss (the helper's `try_add` revalidates and
        // restamps).
        self.p_cmp32_imm8(r::R11, depth, 0);
        self.g_jcc(cc::E, end);
        self.p_load_proto_epoch(r::R9);
        self.g_cmp_mem(r::R9, r::R11, stamp);
        self.g_jcc(cc::NE, miss);
        self.g_load(r::RAX, r::RCX, l.props.off_proto);
        self.g_cmp_mem(r::RAX, r::R11, level(0, offset_of!(ChainLevel, handle)));
        self.g_jcc(cc::NE, miss);
        self.jmp(chain_ok);
        // No prototype at all.
        self.bind(end);
        self.g_load(r::RAX, r::RCX, l.props.off_proto);
        self.g_movabs(r::R8, crate::object::NO_PROTO);
        self.g_cmp(r::RAX, r::R8);
        self.g_jcc(cc::NE, miss);
        self.bind(chain_ok);
        self.g_mov(r::RCX, r::RSI);
        self.g_load(r::RAX, r::RBX, val_disp);
        self.p_barrier_check(l, realm_off, miss);
        self.g_movabs(r::R11, site as u64);
        self.p_add_commit(l, to, val_disp);
    }

    /// `inc dword [base+disp32]` (`base` neither `rsp` nor `r12`).
    fn p_inc32_mem(&mut self, base: u8, disp: i32) {
        debug_assert!(base & 7 != 4, "no SIB form");
        if base >= 8 {
            self.code.push(0x41);
        }
        self.code.push(0xff);
        self.code.push(0x80 | (base & 7));
        self.code.extend_from_slice(&disp.to_le_bytes());
    }

    /// `lea dst, [base + index*8 + disp32]` (`index` not `rsp`).
    pub(crate) fn p_lea_index8(&mut self, dst: u8, base: u8, index: u8, disp: i32) {
        debug_assert!(index != 4, "rsp is no index");
        let rex = 0x48 | (((dst >> 3) & 1) << 2) | (((index >> 3) & 1) << 1) | ((base >> 3) & 1);
        self.code.push(rex);
        self.code.push(0x8d);
        self.code.push(0x80 | ((dst & 7) << 3) | 4);
        self.code.push(0xc0 | ((index & 7) << 3) | (base & 7));
        self.code.extend_from_slice(&disp.to_le_bytes());
    }

    /// `mov dst, [base + index*8 + disp32]` (`index` not `rsp`).
    pub(crate) fn p_load_index8(&mut self, dst: u8, base: u8, index: u8, disp: i32) {
        debug_assert!(index != 4, "rsp is no index");
        let rex = 0x48 | (((dst >> 3) & 1) << 2) | (((index >> 3) & 1) << 1) | ((base >> 3) & 1);
        self.code.push(rex);
        self.code.push(0x8b);
        self.code.push(0x80 | ((dst & 7) << 3) | 4);
        self.code.push(0xc0 | ((index & 7) << 3) | (base & 7));
        self.code.extend_from_slice(&disp.to_le_bytes());
    }

    /// [`p_obj_slot_addr`](Self::p_obj_slot_addr)'s read, split for a hot
    /// path: with a slot index (zero-extended) in `rax` and an object's heap
    /// slot in `rcx`, `rax` = an inline property value; an index past the
    /// object's slot count goes to `miss`, one in the spill array to `spill`
    /// (whose code, [`p_obj_slot_load_spill`](Self::p_obj_slot_load_spill),
    /// can sit out of line). Clobbers `r11`.
    pub(crate) fn p_obj_slot_load_inline(&mut self, l: &JitLayout, miss: Label, spill: Label) {
        const INLINE: i32 = crate::object::INLINE_SLOTS as i32;
        self.p_load16(r::R11, r::RCX, l.off_slots_len);
        self.g_cmp(r::RAX, r::R11);
        self.g_jcc(cc::AE, miss);
        self.p_cmp32_imm(r::RAX, INLINE);
        self.g_jcc(cc::AE, spill);
        self.p_load_index8(r::RAX, r::RCX, r::RAX, l.off_inline);
    }

    /// The spill-array half of
    /// [`p_obj_slot_load_inline`](Self::p_obj_slot_load_inline): `rax` = the
    /// value, then to `back`. Clobbers `r11`.
    pub(crate) fn p_obj_slot_load_spill(&mut self, l: &JitLayout, back: Label) {
        const INLINE: i32 = crate::object::INLINE_SLOTS as i32;
        // A spilled slot means the object has its `extra` part.
        self.g_load(r::R11, r::RCX, l.props.off_extra);
        self.g_load(r::R11, r::R11, l.extra_spill_ptr);
        self.p_load_index8(r::RAX, r::R11, r::RAX, -8 * INLINE);
        self.jmp(back);
    }

    /// `rax` = slot `slot` of the object whose heap slot is `rcx`, a slot its
    /// shape (compared by the caller) is known to have: an inline value, or
    /// a spilled one (the object then has its `extra` part, whose spill
    /// array is as long as the shape requires). Clobbers `r11`.
    pub(crate) fn p_obj_fixed_slot_load(&mut self, l: &JitLayout, slot: u32) {
        match Self::fixed_slot(l, slot) {
            (None, disp) => self.g_load(r::RAX, r::RCX, disp),
            (Some(spill_ptr), disp) => {
                self.g_load(r::R11, r::RCX, l.props.off_extra);
                self.g_load(r::R11, r::R11, spill_ptr);
                self.g_load(r::RAX, r::R11, disp);
            }
        }
    }

    /// The store twin of
    /// [`p_obj_fixed_slot_load`](Self::p_obj_fixed_slot_load): slot `slot`
    /// `= rdx`. Clobbers `r11`.
    pub(crate) fn p_obj_fixed_slot_store(&mut self, l: &JitLayout, slot: u32) {
        match Self::fixed_slot(l, slot) {
            (None, disp) => self.g_store(r::RCX, disp, r::RDX),
            (Some(spill_ptr), disp) => {
                self.g_load(r::R11, r::RCX, l.props.off_extra);
                self.g_load(r::R11, r::R11, spill_ptr);
                self.g_store(r::R11, disp, r::RDX);
            }
        }
    }

    /// Where slot `slot` lives: `(None, d)` at `rcx + d`, or `(Some(p), d)`
    /// at `d` into the spill array whose pointer is at `p` into `extra`.
    fn fixed_slot(l: &JitLayout, slot: u32) -> (Option<i32>, i32) {
        const INLINE: u32 = crate::object::INLINE_SLOTS as u32;
        // A shaped object has at most `u16::MAX` slots, so these fit.
        let at = |i: u32| i32::try_from(i).expect("slot index fits in i32") * 8;
        if slot < INLINE {
            (None, l.off_inline + at(slot))
        } else {
            (Some(l.extra_spill_ptr), at(slot - INLINE))
        }
    }

    /// `mov [base + index*8 + disp32], src` (`index` not `rsp`).
    pub(crate) fn p_store_index8(&mut self, base: u8, index: u8, disp: i32, src: u8) {
        debug_assert!(index != 4, "rsp is no index");
        let rex = 0x48 | (((src >> 3) & 1) << 2) | (((index >> 3) & 1) << 1) | ((base >> 3) & 1);
        self.code.push(rex);
        self.code.push(0x89);
        self.code.push(0x80 | ((src & 7) << 3) | 4);
        self.code.push(0xc0 | ((index & 7) << 3) | (base & 7));
        self.code.extend_from_slice(&disp.to_le_bytes());
    }

    /// The store twin of
    /// [`p_obj_slot_load_inline`](Self::p_obj_slot_load_inline): with a slot
    /// index (zero-extended) in `rax`, an object's heap slot in `rcx` and the
    /// value in `rdx`, an inline property `= rdx`; an index past the object's
    /// slot count goes to `miss`, one in the spill array to `spill`
    /// ([`p_obj_slot_store_spill`](Self::p_obj_slot_store_spill)).
    pub(crate) fn p_obj_slot_store_inline(&mut self, l: &JitLayout, miss: Label, spill: Label) {
        const INLINE: i32 = crate::object::INLINE_SLOTS as i32;
        // The slot count is a u16 (the key epoch follows it): a 16-bit read.
        self.p_load16(r::R11, r::RCX, l.off_slots_len);
        self.g_cmp(r::RAX, r::R11);
        self.g_jcc(cc::AE, miss);
        self.p_cmp32_imm(r::RAX, INLINE);
        self.g_jcc(cc::AE, spill);
        self.p_store_index8(r::RCX, r::RAX, l.off_inline, r::RDX);
    }

    /// The spill-array half of
    /// [`p_obj_slot_store_inline`](Self::p_obj_slot_store_inline), then to
    /// `back`. Clobbers `r11`.
    pub(crate) fn p_obj_slot_store_spill(&mut self, l: &JitLayout, back: Label) {
        const INLINE: i32 = crate::object::INLINE_SLOTS as i32;
        // A spilled slot means the object has its `extra` part.
        self.g_load(r::R11, r::RCX, l.props.off_extra);
        self.g_load(r::R11, r::R11, l.extra_spill_ptr);
        self.p_store_index8(r::R11, r::RAX, -8 * INLINE, r::RDX);
        self.jmp(back);
    }

    /// `cmp r32, imm32`.
    pub(crate) fn p_cmp32_imm(&mut self, reg: u8, imm: i32) {
        if reg >= 8 {
            self.code.push(0x41);
        }
        self.code.push(0x81);
        self.code.push(0xc0 | (7 << 3) | (reg & 7));
        self.code.extend_from_slice(&imm.to_le_bytes());
    }

    /// `dec eax` (zero-extending).
    fn p_dec_eax(&mut self) {
        self.code.extend_from_slice(&[0xff, 0xc8]);
    }

    /// `add rax, r9`.
    fn p_add_rax_r9(&mut self) {
        self.code.extend_from_slice(&[0x4c, 0x01, 0xc8]);
    }

    /// To `miss` unless the value in `rax` is the built-in native function
    /// `id` (a live `Cell::Native` of that id). Clobbers `rax`, `rcx`, `rdx`,
    /// `r8`, `r10`, `r11`.
    pub(crate) fn p_is_native(&mut self, l: &JitLayout, realm_off: i32, id: u16, miss: Label) {
        self.p_handle_check(miss);
        self.g_load(r::R11, r::R15, realm_off);
        self.g_load(r::R10, r::R11, l.props.heap_ptr);
        self.g_load(r::RDX, r::R11, l.props.heap_len);
        self.p_slot_of(l, miss);
        self.cmp_byte_at_rcx_imm(l.off_cell_tag, l.props.native_disc);
        self.jne(miss);
        // cmp word [rcx+disp32], imm16
        self.code.extend_from_slice(&[0x66, 0x81, 0xb9]);
        self.code
            .extend_from_slice(&l.props.off_native_id.to_le_bytes());
        self.code.extend_from_slice(&id.to_le_bytes());
        self.jne(miss);
    }

    /// `dec edi`.
    fn p_dec_edi(&mut self) {
        self.code.extend_from_slice(&[0xff, 0xcf]);
    }

    /// Whether storing the value in `rax` into the object or array whose slot
    /// is `rcx` needs nothing from the write barrier (`Realm::write_barrier`):
    /// a non-handle value, or no incremental marking running and either a
    /// young container or a live old target (so no old-to-young edge to
    /// record). Otherwise to `miss`, whose helper store runs the barrier.
    /// Loads the arena itself when it needs it; preserves `rcx`, clobbers
    /// `rax`, `rdx`, `r8`–`r11`.
    pub(crate) fn p_barrier_check(&mut self, l: &JitLayout, realm_off: i32, miss: Label) {
        let done = self.new_label();
        let old = crate::gc::OLD_AGE;
        self.g_movabs(r::R11, HANDLE_TAG);
        self.g_mov(r::R9, r::RAX);
        self.g_and(r::R9, r::R11);
        self.g_cmp(r::R9, r::R11);
        self.g_jcc(cc::NE, done);
        self.g_load(r::R11, r::R15, realm_off);
        self.g_cmp_byte_imm(r::R11, l.props.marking, 0);
        self.g_jcc(cc::NE, miss);
        self.g_cmp_byte_imm(r::RCX, l.props.off_slot_age, old);
        self.g_jcc(cc::B, done);
        // The arena, for the target's slot (`r11` is still the realm).
        self.g_load(r::R10, r::R11, l.props.heap_ptr);
        self.g_load(r::RDX, r::R11, l.props.heap_len);
        self.g_mov(r::R9, r::RCX);
        self.p_slot_of(l, miss);
        self.g_cmp_byte_imm(r::RCX, l.props.off_slot_age, old);
        self.g_jcc(cc::B, miss);
        self.g_mov(r::RCX, r::R9);
        self.bind(done);
    }

    /// The truthiness of the value in `rax`, inline where it is not the
    /// helper's to decide: `null`/`undefined` to `falsy`; a VM function, or a
    /// shaped ordinary object without the `[[IsHTMLDDA]]` brand, to `truthy`.
    /// Anything else falls through (to the helper), with `rax` clobbered as
    /// are `rcx`, `rdx`, `r8`–`r11`.
    pub(crate) fn p_truthy_heap(&mut self, realm_off: i32, truthy: Label, falsy: Label) {
        use crate::nanbox::NanBox;
        let l = super::compute_jit_layout();
        let other = self.new_label();
        self.g_movabs(r::R11, NanBox::null().to_bits());
        self.g_cmp(r::RAX, r::R11);
        self.g_jcc(cc::E, falsy);
        self.g_movabs(r::R11, NanBox::undefined().to_bits());
        self.g_cmp(r::RAX, r::R11);
        self.g_jcc(cc::E, falsy);
        self.g_mov(r::R9, r::RAX);
        self.p_handle_check(other);
        self.p_load_arena(realm_off, &l.props);
        self.mov_r10_rax();
        self.g_mov(r::RAX, r::R9);
        self.p_slot_of(l, other);
        self.cmp_byte_at_rcx_imm(l.off_cell_tag, l.props.vmfn_disc);
        self.g_jcc(cc::E, truthy);
        self.cmp_byte_at_rcx_imm(l.off_cell_tag, l.cell_object_disc);
        self.jne(other);
        self.cmp_byte_at_rcx_imm(l.off_od_tag, l.obj_shaped_disc);
        self.jne(other);
        self.g_load(r::R11, r::RCX, l.off_shape);
        self.p_test_byte_imm(
            r::R11,
            l.props.shape_brands,
            crate::shape::brand_bit(crate::shape::BRAND_KEYS[1]),
        );
        self.g_jcc(cc::E, truthy);
        self.bind(other);
    }

    /// `mov byte|word|dword [rcx + disp32], imm` (`bytes` 1, 2 or 4).
    fn p_store_rcx_imm(&mut self, bytes: u8, disp: i32, imm: u32) {
        match bytes {
            1 => self.code.push(0xc6),
            2 => self.code.extend_from_slice(&[0x66, 0xc7]),
            _ => self.code.push(0xc7),
        }
        self.code.push(0x81); // mod=10, reg=0, rm=rcx
        self.code.extend_from_slice(&disp.to_le_bytes());
        match bytes {
            1 => self.code.push(imm as u8),
            2 => self.code.extend_from_slice(&(imm as u16).to_le_bytes()),
            _ => self.code.extend_from_slice(&imm.to_le_bytes()),
        }
    }

    /// Inline [`Realm::jit_new_object_with_proto`](crate::realm::Realm) for
    /// the raw handle (or `NO_PROTO`) in `rsi`, when the heap has a free slot
    /// and no incremental marking is running (which would need the
    /// prototype shaded): exactly `Heap::alloc`'s free-list path over
    /// `Object::new(root shape)` with that `[[Prototype]]` — pop the free
    /// list, count the allocation, write the object (one strong count taken
    /// on the root shape) and the slot's age and occupancy, keeping its
    /// generation. Then `rax` = the object's handle bits; otherwise to
    /// `miss`, having changed nothing. Keeps `rsi`; clobbers `rax`, `rcx`,
    /// `rdx`, `r9`, `r11`.
    pub(crate) fn p_alloc_object(&mut self, l: &JitLayout, realm_off: i32, miss: Label) {
        const INLINE: i32 = crate::object::INLINE_SLOTS as i32;
        let p = &l.props;
        self.g_load(r::R9, r::R15, realm_off);
        self.g_cmp_byte_imm(r::R9, p.marking, 0);
        self.g_jcc(cc::NE, miss);
        self.g_load(r::RAX, r::R9, p.free_len);
        self.g_test(r::RAX);
        self.g_jcc(cc::E, miss);
        self.code.extend_from_slice(&[0x48, 0xff, 0xc8]); // dec rax
        self.g_load(r::R11, r::R9, p.free_ptr);
        self.code.extend_from_slice(&[0x41, 0x8b, 0x14, 0x83]); // mov edx, [r11 + rax*4]
        self.g_store(r::R9, p.free_len, r::RAX);
        self.g_inc_mem(r::R9, p.heap_live);
        self.g_inc_mem(r::R9, p.heap_allocs);
        // rcx = the slot.
        self.g_mov(r::RAX, r::RDX);
        self.imul_rax_imm(l.slot_stride);
        self.g_add_mem(r::RAX, r::R9, p.heap_ptr);
        self.g_mov(r::RCX, r::RAX);
        // `Cell::Object(Object::new(root))`, field by field.
        self.p_store_rcx_imm(1, l.off_cell_tag, u32::from(l.cell_object_disc));
        self.p_store_rcx_imm(1, l.off_od_tag, u32::from(l.obj_shaped_disc));
        self.p_store_rcx_imm(1, p.off_jit_flags, 0);
        self.p_store_rcx_imm(2, l.off_slots_len, 0);
        self.p_store_rcx_imm(4, p.off_epoch, 0);
        self.g_load(r::R11, r::R9, p.root_shape);
        self.g_inc_mem(r::R11, p.rc_strong);
        self.g_store(r::RCX, l.off_shape, r::R11);
        self.g_store(r::RCX, p.off_proto, r::RSI);
        self.g_store_imm32(r::RCX, p.off_extra, 0);
        self.g_movabs(r::R11, crate::nanbox::NanBox::undefined().to_bits());
        for i in 0..INLINE {
            self.g_store(r::RCX, l.off_inline + 8 * i, r::R11);
        }
        // The slot: its generation stays; age 0, occupied.
        self.p_load16(r::RAX, r::RCX, l.off_slot_gen);
        self.p_store_rcx_imm(1, p.off_slot_age, 0);
        self.p_store_rcx_imm(1, l.off_slot_occ, u32::from(l.slot_occupied_disc));
        // rax = NanBox::handle(generation << 32 | index).
        self.shl_rax_imm(32);
        self.p_or_rax_rdx();
        self.g_movabs(r::R11, crate::nanbox::NanBox::handle(0).to_bits());
        self.code.extend_from_slice(&[0x4c, 0x09, 0xd8]); // or rax, r11
    }

    /// Inline [`Realm::new_array_from_slice`](crate::realm::Realm) of the
    /// register values at the frame displacements `elems` (at most
    /// [`ARRAY_INLINE_CAP`](crate::cell::ARRAY_INLINE_CAP) of them), when the
    /// heap has a free slot, no incremental marking is running and a new array
    /// needs no explicit `[[Prototype]]` record (`jit_array_proto_default`):
    /// exactly `Heap::alloc`'s free-list path over `ArrayElems::from_slice` —
    /// pop the free list, count the allocation, write the array (an empty
    /// `Vec`, no side state, the elements inline, the unused inline slots
    /// `undefined`) and the slot's age and occupancy, keeping its generation.
    /// Then `rax` = the array's handle bits; otherwise to `miss`, having
    /// changed nothing. Clobbers `rax`, `rcx`, `rdx`, `r9`, `r11`.
    pub(crate) fn p_alloc_array(
        &mut self,
        l: &JitLayout,
        realm_off: i32,
        elems: &[i32],
        miss: Label,
    ) {
        use crate::cell::ARRAY_INLINE_CAP;
        assert!(elems.len() <= ARRAY_INLINE_CAP, "an inline array literal");
        let p = &l.props;
        self.g_load(r::R9, r::R15, realm_off);
        self.g_cmp_byte_imm(r::R9, p.marking, 0);
        self.g_jcc(cc::NE, miss);
        self.g_cmp_byte_imm(r::R9, p.array_proto_default, 0);
        self.g_jcc(cc::E, miss);
        self.g_load(r::RAX, r::R9, p.free_len);
        self.g_test(r::RAX);
        self.g_jcc(cc::E, miss);
        self.code.extend_from_slice(&[0x48, 0xff, 0xc8]); // dec rax
        self.g_load(r::R11, r::R9, p.free_ptr);
        self.code.extend_from_slice(&[0x41, 0x8b, 0x14, 0x83]); // mov edx, [r11 + rax*4]
        self.g_store(r::R9, p.free_len, r::RAX);
        self.g_inc_mem(r::R9, p.heap_live);
        self.g_inc_mem(r::R9, p.heap_allocs);
        // rcx = the slot.
        self.g_mov(r::RAX, r::RDX);
        self.imul_rax_imm(l.slot_stride);
        self.g_add_mem(r::RAX, r::R9, p.heap_ptr);
        self.g_mov(r::RCX, r::RAX);
        // `Cell::Array(ArrayElems::from_slice(elems))`, field by field.
        self.p_store_rcx_imm(1, l.off_cell_tag, u32::from(l.cell_array_disc));
        for (k, &w) in l.arr_empty_vec.iter().enumerate() {
            self.g_movabs(r::R11, w);
            self.g_store(r::RCX, l.off_arr_vec + 8 * k as i32, r::R11);
        }
        self.p_store_rcx_imm(1, l.off_arr_side, 0);
        self.p_store_rcx_imm(1, l.off_arr_inline_len, elems.len() as u32);
        // Promoted registers' current values reach the frame first.
        self.spill_before_call();
        for (i, &d) in elems.iter().enumerate() {
            self.load_rax(d);
            self.g_store(r::RCX, l.off_arr_inline + 8 * i as i32, r::RAX);
        }
        if elems.len() < ARRAY_INLINE_CAP {
            self.g_movabs(r::R11, crate::nanbox::NanBox::undefined().to_bits());
            for i in elems.len()..ARRAY_INLINE_CAP {
                self.g_store(r::RCX, l.off_arr_inline + 8 * i as i32, r::R11);
            }
        }
        // The slot: its generation stays; age 0, occupied.
        self.p_load16(r::RAX, r::RCX, l.off_slot_gen);
        self.p_store_rcx_imm(1, p.off_slot_age, 0);
        self.p_store_rcx_imm(1, l.off_slot_occ, u32::from(l.slot_occupied_disc));
        // rax = NanBox::handle(generation << 32 | index).
        self.shl_rax_imm(32);
        self.p_or_rax_rdx();
        self.g_movabs(r::R11, crate::nanbox::NanBox::handle(0).to_bits());
        self.code.extend_from_slice(&[0x4c, 0x09, 0xd8]); // or rax, r11
    }

    /// Inline [`Realm::new_box`](crate::realm::Realm::new_box) when the heap
    /// has a free slot and no incremental marking is running: exactly
    /// `Heap::alloc`'s free-list path over `Cell::Box(undefined)` (as
    /// [`Self::p_alloc_object`] for an object). Then `rax` = the box's handle
    /// bits; otherwise to `miss`, having changed nothing. Clobbers `rax`,
    /// `rcx`, `rdx`, `r9`, `r11`.
    pub(crate) fn p_alloc_box(&mut self, l: &JitLayout, realm_off: i32, miss: Label) {
        let p = &l.props;
        self.g_load(r::R9, r::R15, realm_off);
        self.g_cmp_byte_imm(r::R9, p.marking, 0);
        self.g_jcc(cc::NE, miss);
        self.g_load(r::RAX, r::R9, p.free_len);
        self.g_test(r::RAX);
        self.g_jcc(cc::E, miss);
        self.code.extend_from_slice(&[0x48, 0xff, 0xc8]); // dec rax
        self.g_load(r::R11, r::R9, p.free_ptr);
        self.code.extend_from_slice(&[0x41, 0x8b, 0x14, 0x83]); // mov edx, [r11 + rax*4]
        self.g_store(r::R9, p.free_len, r::RAX);
        self.g_inc_mem(r::R9, p.heap_live);
        self.g_inc_mem(r::R9, p.heap_allocs);
        // rcx = the slot.
        self.g_mov(r::RAX, r::RDX);
        self.imul_rax_imm(l.slot_stride);
        self.g_add_mem(r::RAX, r::R9, p.heap_ptr);
        self.g_mov(r::RCX, r::RAX);
        // `Cell::Box(undefined)`.
        self.p_store_rcx_imm(1, l.off_cell_tag, u32::from(p.box_disc));
        self.g_movabs(r::R11, crate::nanbox::NanBox::undefined().to_bits());
        self.g_store(r::RCX, p.off_box_val, r::R11);
        // The slot: its generation stays; age 0, occupied.
        self.p_load16(r::RAX, r::RCX, l.off_slot_gen);
        self.p_store_rcx_imm(1, p.off_slot_age, 0);
        self.p_store_rcx_imm(1, l.off_slot_occ, u32::from(l.slot_occupied_disc));
        // rax = NanBox::handle(generation << 32 | index).
        self.shl_rax_imm(32);
        self.p_or_rax_rdx();
        self.g_movabs(r::R11, crate::nanbox::NanBox::handle(0).to_bits());
        self.code.extend_from_slice(&[0x4c, 0x09, 0xd8]); // or rax, r11
    }

    /// Inline [`Realm::new_vm_function_from`](crate::realm::Realm) of VM
    /// function `func` over the register values at the frame displacements
    /// `caps` (at most [`CAPTURES_INLINE_CAP`](crate::cell::CAPTURES_INLINE_CAP)
    /// of them), when the heap has a free slot and no incremental marking is
    /// running: exactly `Heap::alloc`'s free-list path over
    /// `Cell::VmFunction { func, captures: Captures::from_slice(..) }` (as
    /// [`Self::p_alloc_object`] for an object) — the captures inline, the
    /// unused inline slots `undefined`, an empty `Vec`. Then `rax` = the
    /// closure's handle bits; otherwise to `miss`, having changed nothing.
    /// (The caller checks that the closure needs nothing else: no
    /// `[[Prototype]]` record, no realm tag.) Clobbers `rax`, `rcx`, `rdx`,
    /// `r9`, `r11`.
    pub(crate) fn p_alloc_closure(
        &mut self,
        l: &JitLayout,
        realm_off: i32,
        func: u32,
        caps: &[i32],
        miss: Label,
    ) {
        use crate::cell::CAPTURES_INLINE_CAP;
        assert!(caps.len() <= CAPTURES_INLINE_CAP, "inline captures");
        let p = &l.props;
        self.g_load(r::R9, r::R15, realm_off);
        self.g_cmp_byte_imm(r::R9, p.marking, 0);
        self.g_jcc(cc::NE, miss);
        self.g_load(r::RAX, r::R9, p.free_len);
        self.g_test(r::RAX);
        self.g_jcc(cc::E, miss);
        self.code.extend_from_slice(&[0x48, 0xff, 0xc8]); // dec rax
        self.g_load(r::R11, r::R9, p.free_ptr);
        self.code.extend_from_slice(&[0x41, 0x8b, 0x14, 0x83]); // mov edx, [r11 + rax*4]
        self.g_store(r::R9, p.free_len, r::RAX);
        self.g_inc_mem(r::R9, p.heap_live);
        self.g_inc_mem(r::R9, p.heap_allocs);
        // rcx = the slot.
        self.g_mov(r::RAX, r::RDX);
        self.imul_rax_imm(l.slot_stride);
        self.g_add_mem(r::RAX, r::R9, p.heap_ptr);
        self.g_mov(r::RCX, r::RAX);
        // `Cell::VmFunction`, field by field.
        self.p_store_rcx_imm(1, l.off_cell_tag, u32::from(p.vmfn_disc));
        self.p_store_rcx_imm(4, p.off_vmfn_func, func);
        for (k, &w) in l.arr_empty_vec.iter().enumerate() {
            self.g_movabs(r::R11, w);
            self.g_store(r::RCX, p.off_vmfn_caps_vec + 8 * k as i32, r::R11);
        }
        self.p_store_rcx_imm(1, p.off_vmfn_caps_inline_len, caps.len() as u32);
        // Promoted registers' current values reach the frame first.
        self.spill_before_call();
        for (i, &d) in caps.iter().enumerate() {
            self.load_rax(d);
            self.g_store(r::RCX, p.off_vmfn_caps_inline + 8 * i as i32, r::RAX);
        }
        if caps.len() < CAPTURES_INLINE_CAP {
            self.g_movabs(r::R11, crate::nanbox::NanBox::undefined().to_bits());
            for i in caps.len()..CAPTURES_INLINE_CAP {
                self.g_store(r::RCX, p.off_vmfn_caps_inline + 8 * i as i32, r::R11);
            }
        }
        // The slot: its generation stays; age 0, occupied.
        self.p_load16(r::RAX, r::RCX, l.off_slot_gen);
        self.p_store_rcx_imm(1, p.off_slot_age, 0);
        self.p_store_rcx_imm(1, l.off_slot_occ, u32::from(l.slot_occupied_disc));
        // rax = NanBox::handle(generation << 32 | index).
        self.shl_rax_imm(32);
        self.p_or_rax_rdx();
        self.g_movabs(r::R11, crate::nanbox::NanBox::handle(0).to_bits());
        self.code.extend_from_slice(&[0x4c, 0x09, 0xd8]); // or rax, r11
    }

    /// The call sequence's callee identification, inline (what
    /// `jit_callee_info` computes): for the value in `rax`, `rax` = its
    /// `function id | capture count << 32` and `rdx` = its captures when it
    /// is a VM closure, else `rax = u64::MAX`. Clobbers `rcx`, `r8`–`r11`.
    pub(crate) fn p_callee_info(&mut self, l: &JitLayout, realm_off: i32) {
        let none = self.new_label();
        let done = self.new_label();
        self.g_mov(r::R9, r::RAX);
        self.p_load_arena(realm_off, &l.props);
        self.mov_r10_rax();
        self.g_mov(r::RAX, r::R9);
        self.p_handle_check(none);
        self.p_slot_of(l, none);
        self.cmp_byte_at_rcx_imm(l.off_cell_tag, l.props.vmfn_disc);
        self.jne(none);
        self.p_load32(r::RAX, r::RCX, l.props.off_vmfn_func);
        // The captures: inline in the cell, or in the `Vec`.
        let on_heap = self.new_label();
        self.p_load_u8_rdx(r::RCX, l.props.off_vmfn_caps_inline_len);
        self.g_cmp_imm(r::RDX, i32::from(crate::cell::ARRAY_ON_HEAP));
        self.g_jcc(cc::E, on_heap);
        self.p_shl_rdx_32();
        self.p_or_rax_rdx();
        self.g_lea(r::RDX, r::RCX, l.props.off_vmfn_caps_inline);
        self.jmp(done);
        self.bind(on_heap);
        self.g_load(r::RDX, r::RCX, l.props.off_vmfn_caps_len);
        self.p_shl_rdx_32();
        self.p_or_rax_rdx();
        self.g_load(r::RDX, r::RCX, l.props.off_vmfn_caps_ptr);
        self.jmp(done);
        self.bind(none);
        self.g_movabs(r::RAX, u64::MAX);
        self.bind(done);
    }
}

/// A `new` site's constructor cache, kept by `jit_construct_info`: the
/// constructor it last prepared, and what that took — the constructor's
/// auxiliary property object (holding `prototype` and the constructor marks)
/// with its shape, the slot of `prototype` there and the value read from it.
///
/// A hit needs the same constructor value (a VM function cell, so the same
/// function and captures), no cell swap since (so the same auxiliary object),
/// that object still of the recorded shape (so the same data keys: the
/// constructor mark present, no instance-field initializer, `prototype` in the
/// same slot) and that slot still holding the recorded object.
pub struct ConstructCache {
    ctor: u64,
    swaps: u32,
    slot: u32,
    aux: u64,
    aux_shape: Option<Rc<Shape>>,
    proto: u64,
}

impl core::fmt::Debug for ConstructCache {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ConstructCache")
            .field("ctor", &self.ctor)
            .field("warm", &self.aux_shape.is_some())
            .finish_non_exhaustive()
    }
}

impl Default for ConstructCache {
    fn default() -> Self {
        Self::new()
    }
}

impl ConstructCache {
    /// An empty cache.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            ctor: 0,
            swaps: 0,
            slot: 0,
            aux: 0,
            aux_shape: None,
            proto: 0,
        }
    }

    /// The `prototype` object for a `new ctor` when the cache holds `ctor`
    /// unchanged (see the type docs); `None` otherwise.
    pub(crate) fn hit(
        &self,
        realm: &crate::realm::Realm,
        ctor: u64,
    ) -> Option<crate::heap::Handle> {
        let shape = self.aux_shape.as_ref()?;
        if ctor != self.ctor || realm.jit_cell_swaps() != self.swaps {
            return None;
        }
        let aux = realm.jit_object(crate::heap::Handle::from_raw(self.aux))?;
        if !aux.shaped_shape().is_some_and(|s| Rc::ptr_eq(s, shape))
            || aux.slot(self.slot).map(crate::nanbox::NanBox::to_bits) != Some(self.proto)
        {
            return None;
        }
        crate::nanbox::NanBox::from_bits(self.proto)
            .as_handle()
            .map(crate::heap::Handle::from_raw)
    }

    /// Records a `new ctor` that found the object `proto` as `ctor.prototype`
    /// (the constructor having passed every other check).
    pub(crate) fn fill(
        &mut self,
        realm: &crate::realm::Realm,
        ctor: u64,
        proto: crate::nanbox::NanBox,
    ) {
        self.aux_shape = None;
        let Some(fh) = crate::nanbox::NanBox::from_bits(ctor)
            .as_handle()
            .map(crate::heap::Handle::from_raw)
        else {
            return;
        };
        let Some(aux) = realm.jit_aux_of(fh) else {
            return;
        };
        let Some(shape) = realm.jit_object(aux).and_then(|o| o.shaped_shape()) else {
            return;
        };
        let Some(slot) = shape.lookup("prototype") else {
            return;
        };
        *self = Self {
            ctor,
            swaps: realm.jit_cell_swaps(),
            slot,
            aux: aux.to_raw(),
            aux_shape: Some(Rc::clone(shape)),
            proto: proto.to_bits(),
        };
    }
}

/// A property-write site's add cache (part of a [`MethodSite`]): the last
/// write there that added the key as a new data property — on an object of
/// shape `from`, making it `to`, along a prototype chain of `depth` objects
/// (`levels`) that ends there and owns no property of that name.
///
/// A hit re-validates the whole chain as a [`MethodEntry`] does, and that it
/// still ends after `depth` objects: then no object on it has the key (data:
/// by shape; accessor: by key epoch), so no inherited setter, read-only
/// property or proxy can intercept the write, and the receiver (of shape
/// `from`, so without the key) gets a fresh writable, enumerable,
/// configurable data property — the child shape `to`.
///
/// `stamp` is as [`MethodEntry`]'s: the prototype epoch at which the chain was
/// last validated (every level watched), so that while it is current only
/// the receiver's link to `levels[0]` needs checking.
#[derive(Default)]
struct AddEntry {
    from: Option<Rc<Shape>>,
    to: Option<Rc<Shape>>,
    depth: u32,
    stamp: u64,
    levels: [ChainLevel; CHAIN_MAX],
}

impl MethodSite {
    /// The add cache's hit: when writing the site's key to `recv` is an add
    /// it recorded (see [`AddEntry`]), make it — exactly as the interpreter's
    /// write does — and return `true`; otherwise change nothing.
    pub(crate) fn try_add(
        &mut self,
        realm: &mut crate::realm::Realm,
        recv: u64,
        value: crate::nanbox::NanBox,
    ) -> bool {
        let e = &self.add;
        let (Some(from), Some(to)) = (e.from.as_ref(), e.to.as_ref()) else {
            return false;
        };
        let Some(h) = crate::nanbox::NanBox::from_bits(recv)
            .as_handle()
            .map(crate::heap::Handle::from_raw)
        else {
            return false;
        };
        let Some(o) = realm.jit_object(h) else {
            return false;
        };
        if o.jit_flags() != 0
            || !o.shaped_shape().is_some_and(|s| Rc::ptr_eq(s, from))
            || !o.jit_add_ok()
        {
            return false;
        }
        let mut cur = o.proto();
        for level in &e.levels[..e.depth as usize] {
            let Some(p) = cur.filter(|p| p.to_raw() == level.handle) else {
                return false;
            };
            let Some(po) = realm.jit_object(p) else {
                return false;
            };
            if po.jit_flags() & FLAG_EXOTIC != 0
                || po.key_epoch() != level.epoch
                || !po
                    .shaped_shape()
                    .zip(level.shape.as_ref())
                    .is_some_and(|(a, b)| Rc::ptr_eq(a, b))
            {
                return false;
            }
            cur = po.proto();
        }
        if cur.is_some() {
            return false;
        }
        // The chain holds as recorded: restamp the entry (when every level
        // is still watched) so compiled code takes it inline again.
        let restamp = levels_watched(realm, &e.levels[..e.depth as usize]);
        let to = Rc::clone(to);
        if restamp {
            self.add.stamp = crate::object::proto_epoch();
        }
        realm.jit_add_transition(h, to, value)
    }

    /// Records a write of the site's key to the object at `recv` that took it
    /// from shape `before` to its current shape, when that was an add the
    /// [`AddEntry`] check can prove again (else leaves the cache as it was).
    pub(crate) fn fill_add(
        &mut self,
        realm: &mut crate::realm::Realm,
        recv: crate::heap::Handle,
        before: &Rc<Shape>,
    ) {
        if self.fills >= FILL_LIMIT {
            return;
        }
        let key = self.key();
        // Internal (`\0`-prefixed) keys have rules of their own.
        if key.starts_with('\0') {
            return;
        }
        let Some(o) = realm.jit_chain_object(recv) else {
            return;
        };
        let Some(to) = o.shaped_shape() else {
            return;
        };
        if o.jit_flags() != 0
            || !o.jit_add_ok()
            || !to.parent().is_some_and(|p| Rc::ptr_eq(p, before))
            || to.lookup(key) != Some(before.len())
        {
            return;
        }
        self.fills += 1;
        let mut entry = AddEntry {
            from: Some(Rc::clone(before)),
            to: Some(Rc::clone(to)),
            ..AddEntry::default()
        };
        let mut cur = o.proto();
        for (d, level) in (0u32..).zip(entry.levels.iter_mut()) {
            let Some(h) = cur else {
                entry.depth = d;
                self.store_add(realm, entry);
                return;
            };
            let Some(p) = realm.jit_chain_object(h) else {
                return;
            };
            let Some(ps) = p.shaped_shape() else {
                return;
            };
            if p.accessor(key).is_some() || ps.lookup(key).is_some() {
                return;
            }
            *level = ChainLevel {
                handle: h.to_raw(),
                shape: Some(Rc::clone(ps)),
                epoch: p.key_epoch(),
            };
            cur = p.proto();
        }
        if cur.is_none() {
            entry.depth = CHAIN_MAX as u32;
            self.store_add(realm, entry);
        }
    }

    /// Installs `entry` as the add cache, its chain watched and stamped.
    fn store_add(&mut self, realm: &mut crate::realm::Realm, mut entry: AddEntry) {
        watch_levels(realm, &entry.levels[..entry.depth as usize]);
        entry.stamp = crate::object::proto_epoch();
        self.add = entry;
    }
}

#[cfg(test)]
mod tests {
    use crate::heap::Handle;
    use crate::nanbox::NanBox;
    use crate::realm::Realm;

    /// Every offset compiled code reads through [`super::PropsLayout`] agrees
    /// with a safe read of the same field.
    #[test]
    #[allow(unsafe_code)]
    fn props_layout_matches_safe_reads() {
        let jl = super::super::compute_jit_layout();
        let l = &jl.props;
        let mut realm = Realm::new();
        let p = realm.new_object();
        let o = realm.new_object();
        realm.set_object_proto(o, Some(p));
        realm.set_property(o, "x", NanBox::number(1.0));
        let w = realm.new_object();
        realm.set_property(w, crate::shape::PRIM_WRAP_KEY, NanBox::number(0.0));
        let f = realm.new_vm_function(7, alloc::vec![NanBox::undefined(), NanBox::number(2.0)]);
        realm.add_jit_flags(p, crate::object::JIT_FLAG_NO_INLINE_SET);
        realm.define_accessor(p, "g", NanBox::undefined(), NanBox::undefined());
        let arr = realm.new_array(alloc::vec![NanBox::number(1.0)]);
        let farr = realm.new_array(alloc::vec![NanBox::number(1.0)]);
        realm.freeze_object(farr);
        let (base, len) = realm.jit_arena_slots();
        let rp = core::ptr::addr_of!(realm).cast::<u8>();
        // SAFETY: reads at probed offsets inside the live realm, its heap's slot
        // array and the objects/shapes they point to (as compiled code does).
        unsafe {
            let at = |off: i32| rp.add(usize::try_from(off).unwrap());
            assert_eq!(*at(l.heap_ptr).cast::<*const u8>(), base);
            assert_eq!(*at(l.heap_len).cast::<usize>(), len);
            let slot = |h: Handle| {
                let i = usize::try_from(h.to_raw() & 0xffff_ffff).unwrap();
                base.add(i * usize::try_from(jl.slot_stride).unwrap())
            };
            let field = |h: Handle, off: i32| slot(h).add(usize::try_from(off).unwrap());
            let oo = realm.jit_object(o).unwrap();
            assert_eq!(*field(o, l.off_proto).cast::<u64>(), p.to_raw());
            assert_eq!(
                *field(p, l.off_proto).cast::<u64>(),
                realm
                    .jit_object(p)
                    .unwrap()
                    .proto()
                    .map_or(crate::object::NO_PROTO, Handle::to_raw)
            );
            assert_eq!(*field(o, l.off_epoch).cast::<u32>(), oo.key_epoch());
            assert_eq!(*field(o, l.off_jit_flags), 0);
            assert_eq!(
                *field(p, l.off_jit_flags),
                crate::object::JIT_FLAG_NO_INLINE_SET
            );
            assert!((*field(o, l.off_extra).cast::<*const u8>()).is_null());
            let ex = *field(p, l.off_extra).cast::<*const u8>();
            assert!(!ex.is_null());
            assert_eq!(
                *ex.add(usize::try_from(l.extra_acc_len).unwrap())
                    .cast::<usize>(),
                1
            );
            let brands = |h: Handle| {
                let word = *field(h, jl.off_shape).cast::<*const u8>();
                *word.offset(isize::try_from(l.shape_brands).unwrap()) & 1
            };
            assert_eq!(brands(o), 0);
            assert_eq!(brands(w), 1);
            // The shape word's parent link and strong count.
            let shape_word = |h: Handle| *field(h, jl.off_shape).cast::<*const u8>();
            let oshape = std::rc::Rc::clone(oo.shape());
            // What a compiled-in layout compares the field with.
            assert_eq!(shape_word(o) as u64, super::shape_word(&oshape));
            let parent = *shape_word(o)
                .offset(isize::try_from(l.shape_parent).unwrap())
                .cast::<*const u8>();
            assert_eq!(
                parent,
                core::ptr::read(core::ptr::addr_of!(*oshape.parent().unwrap()).cast::<*const u8>())
            );
            let strong = *shape_word(o)
                .offset(isize::try_from(l.rc_strong).unwrap())
                .cast::<usize>();
            assert_eq!(strong, std::rc::Rc::strong_count(&oshape));
            assert_eq!(*field(arr, jl.off_arr_side), 0);
            assert_eq!(*field(farr, jl.off_arr_side), 1);
            assert_eq!(
                *rp.add(usize::try_from(l.max_array_len).unwrap())
                    .cast::<usize>(),
                realm.limits.max_array_len
            );
            assert_eq!(*field(o, jl.off_slots_len).cast::<u16>(), 1);
            assert_eq!(
                *field(o, jl.off_inline).cast::<u64>(),
                NanBox::number(1.0).to_bits()
            );
            assert_eq!(*field(f, jl.off_cell_tag), l.vmfn_disc);
            let bx = realm.new_box();
            assert!(realm.box_set(bx, NanBox::number(6.5)));
            assert_eq!(*field(bx, jl.off_cell_tag), l.box_disc);
            assert_eq!(
                *field(bx, l.off_box_val).cast::<u64>(),
                NanBox::number(6.5).to_bits()
            );
            let nf = realm.new_native(0x4321);
            assert_eq!(*field(nf, jl.off_cell_tag), l.native_disc);
            assert_eq!(*field(nf, l.off_native_id).cast::<u16>(), 0x4321);
            let (fid, caps) = realm.vm_function(f).unwrap();
            assert_eq!(*field(f, l.off_vmfn_func).cast::<u32>(), fid);
            if caps.len() <= crate::cell::CAPTURES_INLINE_CAP {
                assert_eq!(
                    usize::from(*field(f, l.off_vmfn_caps_inline_len)),
                    caps.len()
                );
                assert_eq!(
                    field(f, l.off_vmfn_caps_inline).cast::<NanBox>(),
                    caps.as_ptr()
                );
            } else {
                assert_eq!(
                    *field(f, l.off_vmfn_caps_inline_len),
                    crate::cell::ARRAY_ON_HEAP
                );
                assert_eq!(*field(f, l.off_vmfn_caps_len).cast::<usize>(), caps.len());
                assert_eq!(
                    *field(f, l.off_vmfn_caps_ptr).cast::<*const NanBox>(),
                    caps.as_ptr()
                );
            }
        }
    }
}
