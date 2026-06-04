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
//! arrive with the VM in later phases (see `ROADMAP.md` §6).
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

/// Evaluates a JavaScript source string and writes its result into `out`.
///
/// `source`/`source_len` are the UTF-8 script (not required to be
/// NUL-terminated). The completion value is rendered as a string and copied
/// into `out` following the in/out length convention (`*out_len` is the buffer
/// capacity on input and the produced length on output; call with `*out_len ==
/// 0` to query the required length).
///
/// Returns [`KtStatus::Ok`] on success. On a parse error or an uncaught throw,
/// returns [`KtStatus::InvalidInput`] and writes the error message into `out`
/// (so the caller can surface it). A caught Rust panic yields
/// [`KtStatus::Internal`].
///
/// # Safety
///
/// `out_len` must be a valid pointer to a `usize`. If `source_len > 0`,
/// `source` must point to at least `source_len` readable bytes. If the produced
/// output fits, `out` must point to at least `*out_len` writable bytes.
#[cfg(feature = "std")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kt_eval(
    source: *const c_char,
    source_len: usize,
    out: *mut c_char,
    out_len: *mut usize,
) -> c_int {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    let outcome = catch_unwind(AssertUnwindSafe(|| {
        if out_len.is_null() || (source.is_null() && source_len != 0) {
            return KtStatus::NullPointer;
        }
        // SAFETY: caller guarantees `source` covers `source_len` bytes.
        let bytes = unsafe { core::slice::from_raw_parts(source as *const u8, source_len) };
        let Ok(src) = core::str::from_utf8(bytes) else {
            return KtStatus::InvalidInput;
        };
        let (text, ok) = match eval_to_string(src) {
            Ok(value) => (value, true),
            Err(message) => (message, false),
        };
        // SAFETY: `out_len` is non-null (checked) and `out` honors the
        // length convention.
        match unsafe { copy_out(text.as_bytes(), out, out_len) } {
            KtStatus::Ok if !ok => KtStatus::InvalidInput,
            other => other,
        }
    }));
    match outcome {
        Ok(status) => status as c_int,
        Err(_) => KtStatus::Internal as c_int,
    }
}

/// Parses and runs `src`, returning the completion value's string on success or
/// the thrown value's string on an uncaught throw / parse error.
#[cfg(feature = "std")]
fn eval_to_string(src: &str) -> Result<alloc::string::String, alloc::string::String> {
    // The new-representation engine: the bytecode VM with a tree-walker fallback.
    crate::nbvm::execute(src).map(|(_output, completion)| completion)
}

/// Copies `data` into `out` per the in/out length convention.
///
/// # Safety
///
/// `out_len` must be a valid pointer to a `usize`; if `data` fits in the
/// reported capacity, `out` must point to at least that many writable bytes.
#[cfg(feature = "std")]
unsafe fn copy_out(data: &[u8], out: *mut c_char, out_len: *mut usize) -> KtStatus {
    // SAFETY: caller guarantees `out_len` is valid.
    let cap = unsafe { *out_len };
    unsafe { *out_len = data.len() };
    if cap < data.len() {
        return KtStatus::BufferTooSmall;
    }
    if out.is_null() {
        return KtStatus::NullPointer;
    }
    // SAFETY: `out` has at least `cap >= data.len()` writable bytes.
    unsafe {
        core::ptr::copy_nonoverlapping(data.as_ptr(), out as *mut u8, data.len());
    }
    KtStatus::Ok
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

    #[cfg(feature = "std")]
    fn eval_str(src: &str) -> (KtStatus, alloc::string::String) {
        let mut len: usize = 0;
        // Length query.
        let rc = unsafe {
            kt_eval(
                src.as_ptr() as *const c_char,
                src.len(),
                core::ptr::null_mut(),
                &mut len,
            )
        };
        let mut buf = alloc::vec![0i8; len];
        let rc2 = unsafe {
            kt_eval(
                src.as_ptr() as *const c_char,
                src.len(),
                buf.as_mut_ptr(),
                &mut len,
            )
        };
        // The query returns BufferTooSmall (or the final status if len was 0).
        let _ = rc;
        let bytes: alloc::vec::Vec<u8> = buf.iter().map(|&b| b as u8).collect();
        let text = alloc::string::String::from_utf8(bytes).unwrap();
        (
            if rc2 == KtStatus::Ok as i32 {
                KtStatus::Ok
            } else {
                KtStatus::InvalidInput
            },
            text,
        )
    }

    #[cfg(feature = "std")]
    #[test]
    fn eval_runs_javascript() {
        let (status, out) = eval_str("const f = (a, b) => a * b; f(6, 7)");
        assert_eq!(status, KtStatus::Ok);
        assert_eq!(out, "42");

        let (status, out) = eval_str("[1, 2, 3].map(x => x * x).join(',')");
        assert_eq!(status, KtStatus::Ok);
        assert_eq!(out, "1,4,9");
    }

    #[cfg(feature = "std")]
    #[test]
    fn eval_reports_throws_and_parse_errors() {
        let (status, out) = eval_str("throw new TypeError('boom')");
        assert_eq!(status, KtStatus::InvalidInput);
        assert_eq!(out, "TypeError: boom");

        let (status, _) = eval_str("const = =");
        assert_eq!(status, KtStatus::InvalidInput);
    }
}
