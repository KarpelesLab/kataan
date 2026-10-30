//! The free-variable sets of a compile session's function nodes, computed once.
//!
//! The compiler asks for the free variables of a function (its captures, and
//! whether it uses `eval`/`with`/`this`/`arguments` through a nested
//! function) when it compiles that function *and* whenever it scans an
//! enclosing function's body, and each such scan descends into every nested
//! function. Unremembered, compiling `n` nested functions scans the innermost
//! one `n` times: quadratic in the nesting depth (10,000 levels took ~10 s).
//!
//! A *session* brackets the compilation of one borrowed root AST (a script,
//! a module, eval code, a `Function` body). It starts by scanning the whole
//! root once, bottom-up, recording the result of [`super::free_of_function`]
//! for every function node it reaches, keyed by the node's addresses (its
//! parameter list and body). After that the session
//! only reads: a function the scan did not record — one built later, such as a
//! synthesized body — is computed as before and not recorded.
//!
//! This is sound because every recorded node belongs to the root AST, which
//! the session's caller borrows immutably for the session's whole lifetime:
//! none of those nodes can be freed or changed while the session lives, so no
//! other node can take one of their addresses. Sessions nest (a session's
//! lookups also consult the enclosing ones, whose roots are borrowed too).
//!
//! The memo is per thread; a `no_std` build has no thread-local storage and
//! goes without it (each scan computes, as before).

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::string::String;

/// A function node's identity: the addresses and lengths of its parameter
/// list and body. (Two nodes with an empty parameter list *and* an empty body
/// may share a key, and an empty set of free variables.)
pub(super) type Key = (usize, usize, usize, usize);

/// The key of the function node with `params` and `body`.
pub(super) fn key(params: &[crate::ast::Param], body: &[crate::ast::Stmt]) -> Key {
    (
        params.as_ptr() as usize,
        params.len(),
        body.as_ptr() as usize,
        body.len(),
    )
}

#[cfg(feature = "std")]
mod imp {
    use super::{BTreeMap, BTreeSet, Key, String};
    use core::cell::RefCell;

    struct Session {
        /// Recording (the opening scan), or only reading.
        filling: bool,
        map: BTreeMap<Key, BTreeSet<String>>,
    }

    std::thread_local! {
        static SESSIONS: RefCell<alloc::vec::Vec<Session>> = const { RefCell::new(alloc::vec::Vec::new()) };
    }

    pub(in crate::nbvm) fn lookup(k: &Key) -> Option<BTreeSet<String>> {
        SESSIONS.with(|s| {
            s.borrow()
                .iter()
                .rev()
                .find_map(|sess| sess.map.get(k).cloned())
        })
    }

    pub(in crate::nbvm) fn record(k: Key, v: &BTreeSet<String>) {
        SESSIONS.with(|s| {
            if let Some(top) = s.borrow_mut().last_mut()
                && top.filling
            {
                top.map.insert(k, v.clone());
            }
        });
    }

    pub(in crate::nbvm) fn open() {
        SESSIONS.with(|s| {
            s.borrow_mut().push(Session {
                filling: true,
                map: BTreeMap::new(),
            });
        });
    }

    pub(in crate::nbvm) fn filled() {
        SESSIONS.with(|s| {
            if let Some(top) = s.borrow_mut().last_mut() {
                top.filling = false;
            }
        });
    }

    pub(in crate::nbvm) fn close() {
        SESSIONS.with(|s| {
            s.borrow_mut().pop();
        });
    }
}

#[cfg(not(feature = "std"))]
mod imp {
    use super::{BTreeSet, Key, String};
    pub(in crate::nbvm) fn lookup(_: &Key) -> Option<BTreeSet<String>> {
        None
    }
    pub(in crate::nbvm) fn record(_: Key, _: &BTreeSet<String>) {}
    pub(in crate::nbvm) fn open() {}
    pub(in crate::nbvm) fn filled() {}
    pub(in crate::nbvm) fn close() {}
}

pub(super) use imp::{lookup, record};

/// An open session over a root AST borrowed for `'a`; closing it (on drop)
/// forgets what it recorded. It cannot outlive the borrow of its root.
pub(super) struct Session<'a>(core::marker::PhantomData<&'a ()>);

impl Drop for Session<'_> {
    fn drop(&mut self) {
        imp::close();
    }
}

/// Opens a session over `root` and records its function nodes with `scan`,
/// which must walk only `root` (e.g. [`super::refs_stmt`] over its
/// statements).
pub(super) fn session<'a, T: ?Sized>(root: &'a T, scan: impl FnOnce(&'a T)) -> Session<'a> {
    imp::open();
    let guard = Session(core::marker::PhantomData);
    scan(root);
    imp::filled();
    guard
}
