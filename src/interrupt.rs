//! A cooperative interrupt flag — the engine's side of a host watchdog.
//!
//! A JavaScript program can run forever (`while (true) {}`), and an embedder
//! that has handed untrusted script to the engine needs a deadline it can
//! actually enforce. This is the mechanism: the host holds a handle, trips it
//! from a timer or another thread, and the engine notices at its next check
//! point and unwinds.
//!
//! # Why the flag is not a field on the interpreter
//!
//! `Interp` and `Realm` hold `Rc`s, so they are `!Send`/`!Sync` and a watchdog
//! thread can never touch them. The flag therefore lives in its own shared
//! allocation: the engine keeps one [`Interrupt`](crate::interrupt::Interrupt)
//! handle and the watchdog keeps a clone, and only the `AtomicBool` inside
//! crosses the thread boundary. An
//! `AtomicBool` *field* on `Interp` would be unreachable from the watchdog and
//! would not compile as `Send` anyway.
//!
//! # What it costs
//!
//! A `Relaxed` load and a predictable branch at each check point. `Relaxed` is
//! the right ordering: the flag carries no data, publishes nothing, and the
//! engine only needs to observe the write *eventually* — a deadline measured in
//! milliseconds does not care about a few microseconds of staleness. When no
//! interrupt is installed the check is a null test on an `Option`.
//!
//! # Where it is checked
//!
//! Loop back-edges are necessary but not sufficient. A program can also spin
//! without a back-edge — deep recursion, or one long-running builtin
//! (`"x".repeat(1e9)`, a large `sort`, a catastrophic regex). So the engine
//! checks at back-edges *and* at call entry, and the regex VM folds the flag
//! into its existing step budget. See `Interp::check_interrupt`.
//!
//! # Deliberately not catchable
//!
//! An interrupt unwinds as a non-throw abrupt completion, like
//! `ExecError::OptShortCircuit`. If it surfaced as a normal exception, then
//!
//! ```js
//! while (true) { try { } catch (e) { /* swallow */ } }
//! ```
//!
//! would defeat the watchdog entirely — which would make the whole mechanism a
//! suggestion rather than a deadline. The cost of that choice is that `finally`
//! blocks and `using` disposal do **not** run on an interrupt; a hard deadline
//! cannot promise to run arbitrary user cleanup, since that cleanup may itself
//! loop forever.

use alloc::sync::Arc;
use core::sync::atomic::{AtomicBool, Ordering};

/// A shared interrupt flag. Cheap to clone; every clone observes the same flag.
///
/// ```
/// # use kataan::interrupt::Interrupt;
/// let it = Interrupt::new();
/// let watchdog = it.clone();          // hand this to a timer thread
/// assert!(!it.is_tripped());
/// watchdog.trip();
/// assert!(it.is_tripped());
/// ```
#[derive(Clone, Default)]
pub struct Interrupt(Arc<AtomicBool>);

impl Interrupt {
    /// A fresh, untripped flag.
    #[must_use]
    pub fn new() -> Self {
        Self(Arc::new(AtomicBool::new(false)))
    }

    /// Requests that the engine stop at its next check point.
    ///
    /// Safe to call from any thread, including from a signal-handling context —
    /// it is a single relaxed store and allocates nothing.
    pub fn trip(&self) {
        self.0.store(true, Ordering::Relaxed);
    }

    /// Whether the flag is currently set.
    #[must_use]
    pub fn is_tripped(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }

    /// Clears the flag so the same handle can be reused for another run.
    ///
    /// The engine never clears it itself: an interrupt that unwound one
    /// execution must not silently arm-and-forget for the next one, so
    /// re-arming is the host's explicit decision.
    pub fn clear(&self) {
        self.0.store(false, Ordering::Relaxed);
    }
}

impl core::fmt::Debug for Interrupt {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("Interrupt")
            .field(&self.is_tripped())
            .finish()
    }
}
