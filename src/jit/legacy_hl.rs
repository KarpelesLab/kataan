//! Headerless calls of sloppy ordinary functions, while no code can read
//! their legacy `f.caller` / `f.arguments` (`crate::legacy_reach`).
//!
//! A sloppy ordinary body gets a direct entry like any other
//! (`NativeEntries::direct`), and `JitProto::call_target` hands it to a call
//! site only while [`crate::legacy_reach::reachable`] is off, noting its
//! address here ([`note_code`]). When the flag turns on,
//! [`invalidate`] walks every call-site array compiled on this thread (each
//! [`register`]ed when its body was compiled; held weakly, so a dropped
//! body's simply fails to upgrade) and drops each way entering one of those
//! addresses back to its header path (`CallTarget::NO_HL`). The flag never
//! turns off again, so nothing is noted after that.
//!
//! (An address noted here may later be reused by another body's code after
//! the first is dropped: dropping that body's headerless way too only costs
//! it its fast path.)

use super::{CallSite, CallTarget};
use alloc::rc::{Rc, Weak};
use alloc::vec::Vec;
use core::cell::{RefCell, UnsafeCell};

/// The call-site arrays compiled on this thread, and the direct entries of
/// sloppy bodies handed out to them.
struct Registry {
    sites: Vec<Weak<[UnsafeCell<CallSite>]>>,
    /// When `sites` reaches this length, its dropped entries are pruned.
    prune_at: usize,
    codes: Vec<u64>,
}

#[cfg(feature = "std")]
std::thread_local! {
    static REG: RefCell<Registry> = const {
        RefCell::new(Registry {
            sites: Vec::new(),
            prune_at: 64,
            codes: Vec::new(),
        })
    };
}

/// (Without `std` the flag is always on: nothing is ever noted.)
#[cfg(not(feature = "std"))]
pub(crate) fn invalidate() {}

/// Records a freshly compiled body's call sites (nothing once the flag is
/// on: no way of theirs will ever be headerless into a sloppy body).
pub(crate) fn register(calls: &Rc<[UnsafeCell<CallSite>]>) {
    #[cfg(not(feature = "std"))]
    let _ = calls;
    #[cfg(feature = "std")]
    if calls.is_empty() || crate::legacy_reach::reachable() {
        return;
    }
    #[cfg(feature = "std")]
    REG.with(|r| {
        let mut r = r.borrow_mut();
        if r.sites.len() >= r.prune_at {
            r.sites.retain(|w| w.strong_count() > 0);
            r.prune_at = (r.sites.len() * 2).max(64);
        }
        r.sites.push(Rc::downgrade(calls));
    });
}

/// Records a sloppy body's direct entry `code`, handed out to a call site.
pub(crate) fn note_code(code: u64) {
    #[cfg(not(feature = "std"))]
    let _ = code;
    #[cfg(feature = "std")]
    REG.with(|r| {
        let mut r = r.borrow_mut();
        if let Err(i) = r.codes.binary_search(&code) {
            r.codes.insert(i, code);
        }
    });
}

/// Drops every cached headerless way into a sloppy body (see the module
/// docs), and the registry with them.
#[cfg(feature = "std")]
pub(crate) fn invalidate() {
    let Registry { sites, codes, .. } = REG.with(|r| {
        core::mem::replace(
            &mut *r.borrow_mut(),
            Registry {
                sites: Vec::new(),
                prune_at: 64,
                codes: Vec::new(),
            },
        )
    });
    if codes.is_empty() {
        return;
    }
    for calls in sites.iter().filter_map(Weak::upgrade) {
        for c in calls.iter() {
            let site = c.get();
            for w in 0..CallSite::WAYS {
                // SAFETY: as `JitProto::retarget_calls`: the bodies that
                // read and write these caches are suspended below this call
                // (or not running), and the ways are written in place
                // through the cell's pointer. A call already past its
                // site's headerless test is popped by what it laid out, not
                // by the way.
                #[allow(unsafe_code)]
                unsafe {
                    let t = core::ptr::addr_of_mut!((*site).targets[w]);
                    if (*t).hl_code != 0 && codes.binary_search(&(*t).hl_code).is_ok() {
                        (*t).hl_code = 0;
                        (*t).hl_frame = CallTarget::NO_HL;
                    }
                }
            }
        }
    }
}
