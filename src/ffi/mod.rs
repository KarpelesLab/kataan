//! C ABI for `kataan` (the `ffi` feature).
//!
//! This is the only module permitted broad use of `unsafe` (the crate sets
//! `unsafe_code = "deny"`, not `forbid`, for exactly this purpose). It exposes
//! `extern "C"` entry points declared in `include/kataan.h`.
//!
//! ## Conventions (mirroring the sibling `purecrypto` C ABI)
//!
//! - Fallible functions return [`KtStatus`] (`0` = success, negative = error).
//! - Variable-length output uses the in/out length convention: pass a buffer
//!   and a `*out_len` holding its capacity; on return `*out_len` is the actual
//!   (or, on [`KtStatus::BufferTooSmall`], the required) length.
//! - Opaque handles are created and freed by the library; every `*_new` is
//!   paired with a `*_free`.
//! - Every entry point that can run engine code catches panics, so a Rust
//!   panic surfaces as [`KtStatus::Internal`] rather than unwinding across the
//!   boundary.
//!
//! Build a C library with, e.g.:
//! `cargo rustc --lib --release --features ffi --crate-type staticlib`
//! (or `--crate-type cdylib`).
//!
//! The surface here is the Phase-A seed (version + status codes + a
//! length-convention string copy); the runtime/context/value entry points
//! arrive with the VM in later phases (see `ROADMAP.md` §5).
#![allow(unsafe_code)]
#![allow(unreachable_pub)]

use core::ffi::{c_char, c_int};

/// Status codes returned across the C ABI. `0` is success; negatives are
/// errors. The numeric values are part of the ABI and must stay stable.
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KtStatus {
    /// The call succeeded.
    Ok = 0,
    /// A `NULL` pointer was passed where a valid pointer was required.
    NullPointer = -1,
    /// The supplied output buffer was too small; `*out_len` holds the
    /// required length.
    BufferTooSmall = -2,
    /// The input was not valid (e.g. not valid UTF-8, or a malformed script).
    InvalidInput = -3,
    /// An internal engine error or a caught Rust panic.
    Internal = -100,
}

/// Returns the engine version as a static, NUL-terminated C string. The
/// returned pointer is valid for the lifetime of the program and must not be
/// freed by the caller.
///
/// # Safety
///
/// Always safe to call; the returned pointer is to static storage.
#[unsafe(no_mangle)]
pub extern "C" fn kt_version() -> *const c_char {
    // A `static` NUL-terminated copy of CARGO_PKG_VERSION.
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}

/// Copies the engine version into `buf` (capacity `*len` bytes), writing the
/// number of bytes used (excluding any NUL) back into `*len`. Follows the
/// in/out length convention: call with `*len == 0` to query the required
/// length.
///
/// # Safety
///
/// `len` must be a valid pointer to a `usize`. If `*len > 0`, `buf` must point
/// to at least `*len` writable bytes. Passing `NULL` for `len` returns
/// [`KtStatus::NullPointer`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kt_version_copy(buf: *mut c_char, len: *mut usize) -> c_int {
    let status = (|| {
        if len.is_null() {
            return KtStatus::NullPointer;
        }
        let version = env!("CARGO_PKG_VERSION").as_bytes();
        // SAFETY: caller guarantees `len` points to a valid `usize`.
        let cap = unsafe { *len };
        // SAFETY: same.
        unsafe { *len = version.len() };
        if cap < version.len() {
            return KtStatus::BufferTooSmall;
        }
        if buf.is_null() {
            return KtStatus::NullPointer;
        }
        // SAFETY: `buf` has at least `cap >= version.len()` writable bytes.
        unsafe {
            core::ptr::copy_nonoverlapping(version.as_ptr(), buf as *mut u8, version.len());
        }
        KtStatus::Ok
    })();
    status as c_int
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_string_is_nul_terminated() {
        let ptr = kt_version();
        assert!(!ptr.is_null());
        // SAFETY: kt_version returns a valid static C string.
        let s = unsafe { core::ffi::CStr::from_ptr(ptr) };
        assert_eq!(s.to_str().unwrap(), crate::VERSION);
    }

    #[test]
    fn version_copy_length_query_and_copy() {
        let mut len: usize = 0;
        // Query length.
        let rc = unsafe { kt_version_copy(core::ptr::null_mut(), &mut len) };
        assert_eq!(rc, KtStatus::BufferTooSmall as i32);
        assert_eq!(len, crate::VERSION.len());

        // Copy into an adequately sized buffer.
        let mut buf = alloc::vec![0i8; len];
        let rc = unsafe { kt_version_copy(buf.as_mut_ptr(), &mut len) };
        assert_eq!(rc, KtStatus::Ok as i32);
        let bytes: alloc::vec::Vec<u8> = buf.iter().map(|&b| b as u8).collect();
        assert_eq!(core::str::from_utf8(&bytes).unwrap(), crate::VERSION);

        // NULL len pointer.
        let rc = unsafe { kt_version_copy(core::ptr::null_mut(), core::ptr::null_mut()) };
        assert_eq!(rc, KtStatus::NullPointer as i32);
    }
}
