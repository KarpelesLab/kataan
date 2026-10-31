//! Compiled code's **global-object mirror** for a write to a global `var`
//! (or function) binding.
//!
//! A global `var` is a declarative binding of the global scope that the global
//! object also owns as a property, so the interpreter's `StoreGlobal` fast path
//! (`vm_store_global_fast`) assigns the binding and then mirrors the write onto
//! the property (`Realm::set_property`, a by-name search of the global
//! object's dictionary). Compiled code does the same with two plain stores:
//! `jit_helper_global_ref` validates the binding's address as for any global
//! write and, when the global object owns the name, also the address of that
//! property's value (`Realm::jit_data_ptr`: a writable data property of a
//! dictionary-mode, unfrozen global object, where `set_property` stores),
//! which it leaves in the op's [`MirrorCache`]. Both addresses hold under the
//! global binding epoch: a key added to or removed from the global object
//! (which moves its dictionary's values) bumps it, and so does any change to
//! whether a write to an existing property stores (`Object::freeze`,
//! `set_readonly`, `define_accessor`: `Object::global_attrs_changed`).
//!
//! A store of a heap handle into the (old) global object needs the write
//! barrier: compiled code stores a non-handle itself and hands a handle to
//! [`jit_helper_global_mirror`] (or, in a promoted loop, to the interpreter).
//!
//! Which `StoreGlobal`s get the mirror code is decided per op when its body is
//! compiled ([`MirrorCache::on`], [`plan_body`]): those the interpreter has not seen write a
//! name the global object lacks (`StoreGlobalCache::owned`). For any other op
//! the helper keeps refusing a name the global object owns, so its code (a
//! plain binding store) runs the op through the interpreter instead.

use core::cell::Cell;

/// Per `Op::StoreGlobal`: compiled code's global-object mirror state (see the
/// module docs).
///
/// The ops of one body that write the same name share one validated target:
/// the body's global binding cache gives them one binding address, which any
/// of them may have validated. So each op's state is its body's first such
/// op's (`canon`, set when the body is compiled). A clone is unplanned (its
/// `canon` would point into another body's bytecode).
#[derive(Debug, Default)]
pub(crate) struct MirrorCache {
    /// Whether the op's compiled code mirrors the write (set when its body is
    /// compiled, on the body's own copy of the bytecode).
    on: Cell<bool>,
    /// The address of the `MirrorCache` holding this op's target: the body's
    /// first op mirroring a write of the same name (`0`: unplanned).
    canon: Cell<usize>,
    /// The address of the global object's property value the write also goes
    /// to, `0` when it does not own the name — as last validated (under the
    /// current global binding epoch whenever compiled code holds the binding
    /// address it was validated with).
    ptr: Cell<u64>,
}

impl Clone for MirrorCache {
    fn clone(&self) -> Self {
        Self::default()
    }
}

#[cfg_attr(
    not(all(feature = "jit", target_os = "linux", target_arch = "x86_64")),
    allow(dead_code)
)]
impl MirrorCache {
    /// Whether the op's compiled code mirrors the write.
    pub(crate) fn on(&self) -> bool {
        self.on.get() && self.canon.get() != 0
    }

    /// The address compiled code reads the mirror target from (its canonical
    /// op's).
    pub(crate) fn ptr_addr(&self) -> usize {
        self.canon.get() + core::mem::offset_of!(MirrorCache, ptr)
    }

    /// The state this op's target lives in.
    #[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
    fn canonical(&self) -> &MirrorCache {
        let c = self.canon.get();
        if c == 0 {
            return self;
        }
        // SAFETY: `canon` is set only by `plan_body`, to a `MirrorCache` of an
        // op of the same boxed bytecode slice as this one (a clone resets
        // it), which lives exactly as long as this op does.
        #[allow(unsafe_code)]
        unsafe {
            &*(c as *const MirrorCache)
        }
    }

    /// Records a validated target (`0`: none).
    #[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
    pub(crate) fn set(&self, ptr: u64) {
        self.canonical().ptr.set(ptr);
    }
}

/// Plans a body's own bytecode copy (`bytecode`, which must not move
/// afterwards — a boxed slice): which `StoreGlobal`s mirror their write
/// ([`MirrorCache::on`]: every write of a name one of whose writes the
/// interpreter has not seen go to a global object without the name) and,
/// per name, the op whose state they share.
#[cfg_attr(
    not(all(feature = "jit", target_os = "linux", target_arch = "x86_64")),
    allow(dead_code)
)]
pub(crate) fn plan_body(bytecode: &[super::Op]) {
    // A name mirrors in all of the body's writes of it or in none: they may
    // share one validated binding address.
    let mut first: alloc::collections::BTreeMap<&str, usize> = alloc::collections::BTreeMap::new();
    for op in bytecode {
        if let super::Op::StoreGlobal { name, cache, .. } = op
            && cache.owned.get() != 1
        {
            first
                .entry(name.as_str())
                .or_insert(core::ptr::from_ref(&cache.jit) as usize);
        }
    }
    for op in bytecode {
        if let super::Op::StoreGlobal { name, cache, .. } = op {
            let m = &cache.jit;
            let canon = first.get(name.as_str()).copied().unwrap_or(0);
            m.on.set(canon != 0);
            m.canon.set(canon);
            m.ptr.set(0);
        }
    }
}

/// Mirrors a global `var` write of the heap handle `v` onto the global
/// object (the store of a handle compiled code leaves to Rust, for the write
/// barrier): the op's validated mirror target gets `v`. Returns 0, or 1
/// (having done nothing) when the caller is to run the op through the
/// interpreter instead.
///
/// # Safety
/// `ctx` must be the live `Ctx` the compiled body was entered with, `op` a
/// pointer into the bytecode the calling `JitProto` owns, and the op's mirror
/// target validated under the current global binding epoch.
#[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
pub(crate) extern "C" fn jit_helper_global_mirror(
    ctx: *mut core::ffi::c_void,
    op: u64,
    v: u64,
) -> u64 {
    // SAFETY: see the function contract — a live, otherwise-unborrowed `Ctx`.
    #[allow(unsafe_code)]
    let ctx = unsafe { &mut *(ctx as *mut super::Ctx) };
    // SAFETY: `op` points into the `JitProto`'s owned bytecode, alive while its
    // code runs.
    #[allow(unsafe_code)]
    let op = unsafe { &*(op as usize as *const super::Op) };
    let super::Op::StoreGlobal { cache, .. } = op else {
        return 0;
    };
    let m = cache.jit.canonical();
    let ptr = m.ptr.get();
    if ptr == 0 {
        return 0;
    }
    // The global object's handle as it is now: a compacting collection
    // renumbers handles (it moves the object's cell, not the dictionary value
    // the address points at), so a handle kept from the validation may be
    // stale, and a barrier on a stale handle records nothing.
    let Some(g) = ctx
        .host
        .as_deref()
        .and_then(|h| h.global_scope_if_current())
        .and_then(|(_, g)| g.as_handle())
    else {
        // (Not expected: the address was validated with the global scope
        // running.) The caller runs the op through the interpreter.
        return 1;
    };
    let v = crate::nanbox::NanBox::from_bits(v);
    // SAFETY: the address is the global object's property value as
    // `Realm::jit_data_ptr` returned it under the current global binding
    // epoch, which every change that could move or retire it bumps.
    #[allow(unsafe_code)]
    unsafe {
        *(ptr as usize as *mut crate::nanbox::NanBox) = v;
    }
    ctx.realm.jit_write_barrier(super::Handle::from_raw(g), v);
    0
}

/// Before `proto` is compiled: asks the global object, for each of its
/// `StoreGlobal`s the interpreter has not run yet, whether it owns the name
/// (filling `StoreGlobalCache::owned`, the interpreter's own cache of that
/// answer), so that the mirror plan ([`MirrorCache::on`]) also knows about
/// code that has not run — top-level code compiled at its first loop, say.
#[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
pub(super) fn prefill(ctx: &super::Ctx, proto: &super::FnProto) {
    let Some((_, global)) = ctx
        .host
        .as_deref()
        .and_then(|h| h.global_scope_if_current())
    else {
        return;
    };
    let Some(g) = global.as_handle().map(super::Handle::from_raw) else {
        return;
    };
    for op in &proto.ops {
        if let super::Op::StoreGlobal { name, cache, .. } = op
            && cache.owned.get() == 0
            && !name.starts_with('\u{1}')
        {
            super::global_owns_cached(&ctx.realm, g, name, cache);
        }
    }
}
