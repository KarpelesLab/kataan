//! Object literals with a constant key set (`{a: x, b: [y], c: {…}}`).
//!
//! Such a literal compiles to one [`Op::NewObjectLit`](super::Op::NewObjectLit)
//! instead of a `NewObject` and one `InitProp` per member (for the members
//! naming its first `OBJ_LIT_MAX_KEYS` distinct keys — the object's inline
//! slots and near spill; any later member stays an `InitProp`): their values are evaluated
//! first, in source order, and the object is then made in one step.
//! Nothing can observe the object before it is complete (no key is computed,
//! no member is a method, an accessor, a spread, a function to be named or a
//! `__proto__:`), so making it after the values is the same as making it
//! before them.
//!
//! The site remembers, in its [`LitShapeCache`], the shape the literal's
//! object ended up with the first time it was built member by member, when
//! that object came out as a plain shaped one (every key a plain data slot,
//! the `i`th key in the `i`th slot, nothing in its `extra` part, no flags,
//! the default `[[Prototype]]`). Later evaluations allocate the object in
//! that shape with its slots already filled ([`Realm::new_object_in_shape`]):
//! exactly the object the member-by-member build leaves, since a shape is
//! immutable and the build is a pure function of the realm's root shape and
//! the keys. The cache also records that root, so a site run in another realm
//! builds member by member (and re-arms).
//!
//! [`Realm::new_object_in_shape`]: crate::realm::Realm::new_object_in_shape

use super::{Ctx, NanBox, Reg, VmError};
use crate::shape::Shape;
use alloc::boxed::Box;
use alloc::rc::Rc;
use alloc::string::String;
use core::cell::Cell;

/// The most distinct keys an [`super::Op::NewObjectLit`] defines: the
/// inline slots of an object and its near spill (a longer literal's later
/// members are `Op::InitProp`s on the object it makes).
pub(crate) const OBJ_LIT_MAX_KEYS: usize = crate::object::INLINE_SLOTS + crate::object::SPILL_NEAR;

/// A [`super::Op::NewObjectLit`] site's memo: the realm root shape its
/// object's shape grows from, and that shape. Both empty, or both set.
///
/// `repr(C)` with the two words first: compiled code reads them at fixed
/// offsets (`ROOT_OFF`, `SHAPE_OFF`) as
/// the `Rc` pointers (`0` for `None`) they are.
#[repr(C)]
#[derive(Default)]
pub struct LitShapeCache {
    root: Cell<Option<Rc<Shape>>>,
    shape: Cell<Option<Rc<Shape>>>,
}

impl LitShapeCache {
    /// The byte offset of the root shape's `Rc` pointer.
    #[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
    pub(crate) const ROOT_OFF: i32 = 0;
    /// The byte offset of the literal's shape's `Rc` pointer.
    #[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
    pub(crate) const SHAPE_OFF: i32 = core::mem::size_of::<usize>() as i32;

    /// `(root, shape)` when armed (clones; the cache keeps its own).
    pub(crate) fn get(&self) -> Option<(Rc<Shape>, Rc<Shape>)> {
        self.with(|r, s| (Rc::clone(r), Rc::clone(s)))
    }

    /// `f(root, shape)` when armed, without touching the counts. (`f` must
    /// not reach this cache.)
    pub(crate) fn with<R>(&self, f: impl FnOnce(&Rc<Shape>, &Rc<Shape>) -> R) -> Option<R> {
        let root = self.root.take();
        let shape = self.shape.take();
        let out = match (&root, &shape) {
            (Some(r), Some(s)) => Some(f(r, s)),
            _ => None,
        };
        self.root.set(root);
        self.shape.set(shape);
        out
    }

    /// Arms the cache with `shape`, grown from the realm root `root`.
    pub(crate) fn arm(&self, root: Rc<Shape>, shape: Rc<Shape>) {
        self.root.set(Some(root));
        self.shape.set(Some(shape));
    }
}

const _: () = assert!(core::mem::size_of::<Option<Rc<Shape>>>() == core::mem::size_of::<usize>());

impl Clone for LitShapeCache {
    fn clone(&self) -> Self {
        let c = Self::default();
        if let Some((r, s)) = self.get() {
            c.arm(r, s);
        }
        c
    }
}

impl core::fmt::Debug for LitShapeCache {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("LitShapeCache")
            .field("armed", &self.get().is_some())
            .finish()
    }
}

/// The operands of [`super::Op::NewObjectLit`]: the literal's distinct keys
/// in first-occurrence order, each with the register of its (last) value,
/// and the site's shape memo.
#[derive(Clone, Debug)]
pub struct ObjLit {
    /// The keys, distinct, in property-creation order.
    pub keys: Box<[String]>,
    /// `srcs[i]`: the register holding `keys[i]`'s value.
    pub srcs: Box<[Reg]>,
    /// The site's memo of the object's final shape.
    pub cache: LitShapeCache,
}

/// `Op::NewObjectLit`: the literal's object, from the values in `regs`
/// (shared with the generic-JIT helper). Through the site's memo when it is
/// armed for this realm; else member by member (each exactly an
/// `Op::InitProp`), arming the memo when the result qualifies.
pub(super) fn vm_new_object_lit(
    ctx: &mut Ctx,
    lit: &ObjLit,
    regs: &[NanBox],
) -> Result<NanBox, VmError> {
    // The values, in key order (on the stack for a small literal).
    let mut buf = [NanBox::undefined(); OBJ_LIT_MAX_KEYS];
    let heap_vals: alloc::vec::Vec<NanBox>;
    let vals: &[NanBox] = if lit.srcs.len() <= buf.len() {
        for (d, &r) in buf.iter_mut().zip(&*lit.srcs) {
            *d = regs[r as usize];
        }
        &buf[..lit.srcs.len()]
    } else {
        heap_vals = lit.srcs.iter().map(|&r| regs[r as usize]).collect();
        &heap_vals
    };
    if let Some(h) = lit
        .cache
        .with(|root, shape| ctx.realm.new_object_in_shape(root, shape, vals))
        .flatten()
    {
        return Ok(NanBox::handle(h.to_raw()));
    }
    let h = ctx.realm.new_object();
    let o = NanBox::handle(h.to_raw());
    let mut plain = true;
    let mut ic = crate::ic::PropertyCache::default();
    for (k, &v) in lit.keys.iter().zip(vals) {
        if ctx.realm.init_data_property_cached(h, k, v, &mut ic) {
            continue;
        }
        plain = false;
        // `vm_init_prop`'s own fallback: the host's [[DefineOwnProperty]]
        // (on a fresh ordinary object it runs no JavaScript and so no
        // collection, which could not see the object yet).
        let kv = ctx.realm.key_value(k);
        match super::with_host(ctx, |x| x.define_data(o, kv, v, 7)) {
            Some(Ok(())) => {}
            Some(Err(e)) => return Err(VmError::from(e)),
            None => return Err(VmError::Unsupported),
        }
    }
    if plain && let Some(shape) = ctx.realm.plain_literal_shape(h, &lit.keys) {
        lit.cache.arm(ctx.realm.root_shape_rc(), shape);
    }
    Ok(o)
}
