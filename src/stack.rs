//! Native-stack headroom checks.
//!
//! The parser, the bytecode compiler, the validator and every built-in that
//! recurses over user data (`Array.prototype.join` over nested arrays,
//! `JSON.stringify`, `flat(Infinity)`, …) descend the *native* stack once per
//! nesting level. A fixed level count cannot bound that safely: a level costs
//! a few hundred bytes in an optimized build and several kilobytes in a debug
//! build, and the stack a thread has ranges from a few hundred KiB to the CLI's
//! 512 MiB. So instead of counting, these recursions ask [`exhausted`] whether
//! the stack pointer has crossed this thread's *limit address*, and fail with a
//! catchable error (a `SyntaxError` from the parser, a `RangeError` "Maximum
//! call stack size exceeded" from the runtime) before the guard page is hit.
//!
//! ## The limit
//! The stack grows downward on every target the engine supports, so the limit
//! is the lowest address recursion may reach. A thread that knows its stack
//! size declares it once at its top with [`set_thread_stack_size`] (the
//! `kataan` CLI and the Test262 runners do, for their 512 MiB threads); the
//! limit is then that many bytes below the call site, minus a reserve (see
//! [`reserve_for`]) for the frames that run between two checks and for the
//! work done after a check fails (building the error, unwinding, dropping).
//!
//! A thread that declares nothing gets [`DEFAULT_STACK_BUDGET`] below the
//! first check it performs: it assumes the 2 MiB a thread spawned by `std`
//! gets unless told otherwise (the main thread usually has 8 MiB), keeping
//! [`MIN_STACK_RESERVE`] of it as slack for the frames past the last check —
//! a debug build's interpreter frames are tens of kilobytes each. An
//! embedder running the engine on a *smaller* stack must declare it with
//! [`set_thread_stack_size`] (or the engine can still overflow it), and one on
//! a bigger stack should, to make the full depth available.
//!
//! The probe is the address of a local, which is safe Rust (no `unsafe`, no
//! foreign code); with `std` the limit is thread-local. Without `std` there is
//! no thread-local storage, so the limit is process-wide: a `no_std` embedder
//! that runs the engine on several threads must call [`set_stack_limit`]
//! appropriately (or keep to one thread).

#[cfg(not(feature = "std"))]
use core::sync::atomic::{AtomicUsize, Ordering};

/// Recursion budget of a thread that never declared its stack size, measured
/// down from the first check it performs: a `std` thread's default 2 MiB less
/// [`MIN_STACK_RESERVE`].
pub const DEFAULT_STACK_BUDGET: usize = 2 * 1024 * 1024 - MIN_STACK_RESERVE;

/// Smallest reserve kept below the budget of a declared stack.
pub const MIN_STACK_RESERVE: usize = 256 * 1024;

/// The reserve kept free below the limit of a `size`-byte stack: an eighth of
/// it, but at least [`MIN_STACK_RESERVE`].
#[must_use]
pub const fn reserve_for(size: usize) -> usize {
    let r = size / 8;
    if r < MIN_STACK_RESERVE {
        MIN_STACK_RESERVE
    } else {
        r
    }
}

/// An approximation of the current stack pointer: the address of a local.
#[inline(always)]
#[must_use]
pub fn current_sp() -> usize {
    let probe = 0u8;
    core::hint::black_box(&probe) as *const u8 as usize
}

#[cfg(feature = "std")]
std::thread_local! {
    /// This thread's limit address; 0 until the first check or declaration.
    static LIMIT: core::cell::Cell<usize> = const { core::cell::Cell::new(0) };
}

#[cfg(not(feature = "std"))]
static LIMIT: AtomicUsize = AtomicUsize::new(0);

#[cfg(feature = "std")]
#[inline]
fn load_limit() -> usize {
    LIMIT.with(core::cell::Cell::get)
}

#[cfg(not(feature = "std"))]
#[inline]
fn load_limit() -> usize {
    LIMIT.load(Ordering::Relaxed)
}

/// Sets this thread's limit address directly (the lowest stack address the
/// engine's recursions may reach). `0` resets it, so the next check re-derives
/// the default from [`DEFAULT_STACK_BUDGET`].
pub fn set_stack_limit(limit: usize) {
    #[cfg(feature = "std")]
    LIMIT.with(|l| l.set(limit));
    #[cfg(not(feature = "std"))]
    LIMIT.store(limit, Ordering::Relaxed);
}

/// Declares that the calling thread's stack is `size` bytes, of which the
/// caller sits near the top: the engine may then recurse to `size` minus
/// [`reserve_for`]`(size)` below this call. Call it first thing in a thread
/// spawned with an explicit `stack_size`.
pub fn set_thread_stack_size(size: usize) {
    let budget = size.saturating_sub(reserve_for(size));
    set_stack_limit(current_sp().saturating_sub(budget));
}

/// This thread's limit address, deriving the default on first use.
#[inline]
fn limit() -> usize {
    let l = load_limit();
    if l != 0 {
        return l;
    }
    init_default_limit()
}

#[cold]
#[inline(never)]
fn init_default_limit() -> usize {
    let l = current_sp().saturating_sub(DEFAULT_STACK_BUDGET).max(1);
    set_stack_limit(l);
    l
}

/// Whether the native stack is too deep to recurse further: the caller should
/// fail with a recoverable error instead of descending.
#[inline]
#[must_use]
pub fn exhausted() -> bool {
    current_sp() < limit()
}

/// Bytes of recursion budget left above this thread's limit (0 when
/// [`exhausted`]).
#[must_use]
pub fn remaining() -> usize {
    current_sp().saturating_sub(limit())
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    #[test]
    fn declared_stack_bounds_recursion() {
        fn dive(n: usize) -> usize {
            let pad = core::hint::black_box([0u8; 512]);
            if exhausted() {
                return n + usize::from(pad[0]);
            }
            dive(n + 1)
        }
        let h = std::thread::Builder::new()
            .stack_size(4 * 1024 * 1024)
            .spawn(|| {
                set_thread_stack_size(4 * 1024 * 1024);
                assert!(!exhausted());
                assert!(remaining() > 2 * 1024 * 1024);
                dive(0)
            })
            .unwrap();
        let levels = h.join().unwrap();
        // ~3.5 MiB of 512+-byte frames: thousands of levels, but bounded.
        assert!(levels > 1000, "{levels}");
    }
}
