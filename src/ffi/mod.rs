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

/// Allocates `len` bytes inside the library's allocator and returns the
/// pointer, or `NULL` if `len` is `0` or the allocation fails.
///
/// A C caller normally owns its own buffers and never needs this. It exists for
/// hosts that cannot allocate *inside* the module's address space — notably
/// WebAssembly, where JavaScript can only reach linear memory and so must ask
/// the module for a region before writing a script into it.
///
/// Every pointer returned here must be released with [`kt_free`], passing the
/// same `len`.
///
/// # Safety
///
/// Always safe to call. The returned region is uninitialized.
#[unsafe(no_mangle)]
pub extern "C" fn kt_alloc(len: usize) -> *mut u8 {
    if len == 0 {
        return core::ptr::null_mut();
    }
    let Ok(layout) = std::alloc::Layout::from_size_align(len, 1) else {
        return core::ptr::null_mut();
    };
    // SAFETY: `layout` has a non-zero size (checked above).
    unsafe { std::alloc::alloc(layout) }
}

/// Releases a region obtained from [`kt_alloc`].
///
/// # Safety
///
/// `ptr` must be a pointer returned by [`kt_alloc`] that has not already been
/// freed, and `len` must be the length it was allocated with. A `NULL` `ptr` or
/// a zero `len` is a no-op.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kt_free(ptr: *mut u8, len: usize) {
    if ptr.is_null() || len == 0 {
        return;
    }
    let Ok(layout) = std::alloc::Layout::from_size_align(len, 1) else {
        return;
    };
    // SAFETY: the caller guarantees `ptr` came from `kt_alloc` with this `len`,
    // so the layout matches the one it was allocated with.
    unsafe { std::alloc::dealloc(ptr, layout) }
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
        // SAFETY: caller guarantees `source` covers `source_len` bytes; a null
        // pointer with zero length is the empty input (never dereferenced).
        let bytes: &[u8] = if source.is_null() {
            &[]
        } else {
            unsafe { core::slice::from_raw_parts(source as *const u8, source_len) }
        };
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

/// Renders `source` as the engine's **token stream** — one line per token,
/// `start..end kind "text"` — into `out` (in/out length convention).
///
/// This is the first stage of the pipeline `kt_eval` runs end to end, exposed so
/// a host can show *how* the engine sees a script rather than only what it
/// returns. It is the same dump the `kataan lex` subcommand prints.
///
/// Returns [`KtStatus::Ok`], or [`KtStatus::InvalidInput`] with the error
/// message in `out` if the source does not tokenize.
///
/// # Safety
///
/// Same contract as [`kt_eval`].
#[cfg(feature = "std")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kt_lex(
    source: *const c_char,
    source_len: usize,
    out: *mut c_char,
    out_len: *mut usize,
) -> c_int {
    // SAFETY: forwarded verbatim to the shared helper, which documents the
    // same contract this function does.
    unsafe {
        stage_dump(source, source_len, out, out_len, |src| {
            use alloc::string::ToString;
            use core::fmt::Write as _;
            let tokens = crate::lexer::Lexer::new(src)
                .tokenize()
                .map_err(|e| e.to_string())?;
            let mut dump = alloc::string::String::new();
            for tok in &tokens {
                let newline = if tok.newline_before { " <nl>" } else { "" };
                let _ = writeln!(
                    dump,
                    "{:>5}..{:<5} {:<26} {:?}{newline}",
                    tok.span.start,
                    tok.span.end,
                    alloc::format!("{:?}", tok.kind),
                    // The lexer works over WTF-8 *bytes* now, and this dump is a
                    // human-readable diagnostic, so render the lossy text form
                    // rather than a byte slice.
                    crate::wtf8::to_string_lossy(tok.text(src.as_bytes())),
                );
            }
            Ok(dump)
        })
    }
}

/// Renders `source` as the engine's **abstract syntax tree** into `out` (in/out
/// length convention) — the same dump the `kataan parse` subcommand prints.
///
/// Returns [`KtStatus::Ok`], or [`KtStatus::InvalidInput`] with the parse error
/// in `out`.
///
/// # Safety
///
/// Same contract as [`kt_eval`].
#[cfg(feature = "std")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kt_parse(
    source: *const c_char,
    source_len: usize,
    out: *mut c_char,
    out_len: *mut usize,
) -> c_int {
    // SAFETY: forwarded verbatim to the shared helper.
    unsafe {
        stage_dump(source, source_len, out, out_len, |src| {
            use alloc::string::ToString;
            crate::parser::Parser::parse_program(src)
                .map(|program| alloc::format!("{program:#?}"))
                .map_err(|e| e.to_string())
        })
    }
}

/// Shared body of the pipeline-stage dumps ([`kt_lex`], [`kt_parse`]): validate
/// the pointers, decode the source, run `render`, and copy the text out — with
/// an error message taking the same path as a successful dump so the caller can
/// display it either way.
///
/// # Safety
///
/// Same contract as [`kt_eval`].
#[cfg(feature = "std")]
unsafe fn stage_dump(
    source: *const c_char,
    source_len: usize,
    out: *mut c_char,
    out_len: *mut usize,
    render: impl FnOnce(&str) -> Result<alloc::string::String, alloc::string::String>,
) -> c_int {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    let outcome = catch_unwind(AssertUnwindSafe(|| {
        if out_len.is_null() || (source.is_null() && source_len != 0) {
            return KtStatus::NullPointer;
        }
        // SAFETY: caller guarantees `source` covers `source_len` bytes; a null
        // pointer with zero length is the empty input (never dereferenced).
        let bytes: &[u8] = if source.is_null() {
            &[]
        } else {
            unsafe { core::slice::from_raw_parts(source as *const u8, source_len) }
        };
        let Ok(src) = core::str::from_utf8(bytes) else {
            return KtStatus::InvalidInput;
        };
        let (text, ok) = match render(src) {
            Ok(text) => (text, true),
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

/// Evaluates `source`, capturing **both** what the script printed and its
/// completion value.
///
/// [`kt_eval`] reports only the completion value, which is enough for an
/// expression but loses everything a script wrote with `console.log`. This
/// variant writes both into `out`, separated by a single NUL byte:
///
/// ```text
/// <console output> 0x00 <completion value>
/// ```
///
/// so each half is itself a NUL-terminated C string and `*out_len` (the usual
/// in/out length convention) covers the pair. A script that printed nothing
/// yields an empty first half.
///
/// Returns [`KtStatus::Ok`] on success. On a parse error or an uncaught throw
/// the error message is written as the *completion* half (the console half
/// holds whatever the script managed to print first) and the status is
/// [`KtStatus::InvalidInput`]. A caught Rust panic yields
/// [`KtStatus::Internal`].
///
/// # Safety
///
/// Same contract as [`kt_eval`]: `out_len` must be a valid pointer to a
/// `usize`; `source` must cover `source_len` readable bytes when that is
/// non-zero; and `out` must be writable for `*out_len` bytes when the output
/// fits.
#[cfg(feature = "std")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kt_eval_capture(
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
        // SAFETY: caller guarantees `source` covers `source_len` bytes; a null
        // pointer with zero length is the empty input (never dereferenced).
        let bytes: &[u8] = if source.is_null() {
            &[]
        } else {
            unsafe { core::slice::from_raw_parts(source as *const u8, source_len) }
        };
        let Ok(src) = core::str::from_utf8(bytes) else {
            return KtStatus::InvalidInput;
        };
        let (printed, outcome) =
            crate::nbvm::execute_capturing(src, crate::limits::Limits::default());
        let (completion, ok) = match outcome {
            Ok(completion) => (completion, true),
            Err(message) => (message, false),
        };
        let mut joined = printed.into_bytes();
        joined.push(0);
        joined.extend_from_slice(completion.as_bytes());
        // SAFETY: `out_len` is non-null (checked) and `out` honors the
        // length convention.
        match unsafe { copy_out(&joined, out, out_len) } {
            KtStatus::Ok if !ok => KtStatus::InvalidInput,
            other => other,
        }
    }));
    match outcome {
        Ok(status) => status as c_int,
        Err(_) => KtStatus::Internal as c_int,
    }
}

/// Evaluates `source` with a pre-installed global `ArrayBuffer` named `buffer`,
/// built as an engine-**owned** copy of the caller's `data` region (A6, #11).
/// After the run, the buffer's (possibly script-mutated) bytes are written back
/// into `data` in place, so the caller observes JS-level writes through a view
/// over `buffer`. The completion value's string is written into `out` per the
/// in/out length convention.
///
/// Returns [`KtStatus::Ok`] on success; [`KtStatus::InvalidInput`] (with the
/// error message in `out`) on a parse error / uncaught throw; a caught panic
/// yields [`KtStatus::Internal`].
///
/// # Safety
///
/// `out_len` must be a valid pointer to a `usize`. If `source_len > 0`, `source`
/// must cover `source_len` readable bytes. If `data_len > 0`, `data` must point
/// to `data_len` readable/writable bytes (read in, written back). If the output
/// fits, `out` must point to `*out_len` writable bytes.
#[cfg(feature = "std")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kt_eval_with_buffer(
    source: *const c_char,
    source_len: usize,
    data: *mut u8,
    data_len: usize,
    out: *mut c_char,
    out_len: *mut usize,
) -> c_int {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        if out_len.is_null()
            || (source.is_null() && source_len != 0)
            || (data.is_null() && data_len != 0)
        {
            return KtStatus::NullPointer;
        }
        // SAFETY: caller guarantees `source` covers `source_len` bytes.
        let sbytes: &[u8] = if source.is_null() {
            &[]
        } else {
            unsafe { core::slice::from_raw_parts(source as *const u8, source_len) }
        };
        let Ok(src) = core::str::from_utf8(sbytes) else {
            return KtStatus::InvalidInput;
        };
        // SAFETY: caller guarantees `data` covers `data_len` readable/writable bytes.
        let region: &mut [u8] = if data.is_null() {
            &mut []
        } else {
            unsafe { core::slice::from_raw_parts_mut(data, data_len) }
        };
        let (text, ok) = match eval_with_owned_buffer(src, region) {
            Ok(value) => (value, true),
            Err(message) => (message, false),
        };
        // SAFETY: `out_len` is non-null; `out` honors the length convention.
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

/// Evaluates `source` with a pre-installed global `ArrayBuffer` named `buffer`
/// that wraps the caller's `data` region **zero-copy** (A6, #11): JS writes
/// through a view over `buffer` hit `data` in place and are observed by the
/// caller immediately after the call returns — no copy back. The region is *not*
/// freed by the engine; it must outlive the call. The completion value's string
/// is written into `out` per the in/out length convention.
///
/// Returns [`KtStatus::Ok`] on success; [`KtStatus::InvalidInput`] on a parse
/// error / uncaught throw; [`KtStatus::Internal`] on a caught panic.
///
/// # Safety
///
/// All of [`kt_eval_with_buffer`]'s contract, plus: `data`/`data_len` must remain
/// a valid, **uniquely-owned** mutable region for the entire duration of this
/// call (no other alias may be used while the engine holds it). The engine
/// reads and writes it in place and does not deallocate it.
#[cfg(feature = "std")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kt_eval_with_external_buffer(
    source: *const c_char,
    source_len: usize,
    data: *mut u8,
    data_len: usize,
    out: *mut c_char,
    out_len: *mut usize,
) -> c_int {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        if out_len.is_null()
            || (source.is_null() && source_len != 0)
            || (data.is_null() && data_len != 0)
        {
            return KtStatus::NullPointer;
        }
        // SAFETY: caller guarantees `source` covers `source_len` bytes.
        let sbytes: &[u8] = if source.is_null() {
            &[]
        } else {
            unsafe { core::slice::from_raw_parts(source as *const u8, source_len) }
        };
        let Ok(src) = core::str::from_utf8(sbytes) else {
            return KtStatus::InvalidInput;
        };
        // SAFETY: forwarded to the caller's `kt_eval_with_external_buffer`
        // contract — `data`/`data_len` is a valid, unique mutable region for the
        // call; the wrapped buffer is dropped before this returns.
        let (text, ok) = match unsafe { eval_with_external_buffer(src, data, data_len) } {
            Ok(value) => (value, true),
            Err(message) => (message, false),
        };
        // SAFETY: `out_len` is non-null; `out` honors the length convention.
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

/// Runs `src` with a pre-installed global `ArrayBuffer` named `buffer`, built as
/// an engine-**owned** copy of `data` (A6, #11). After the run, the buffer's
/// (possibly script-mutated) bytes are written back into `data` in place — so a
/// caller observes JS-level writes through a view over `buffer`, and the owned
/// store round-trips. Returns the completion string, or an error message.
#[cfg(feature = "std")]
fn eval_with_owned_buffer(
    src: &str,
    data: &mut [u8],
) -> Result<alloc::string::String, alloc::string::String> {
    use alloc::string::ToString;
    let program = crate::parser::Parser::parse_program(src).map_err(|e| e.to_string())?;
    let mut interp = crate::nbexec::Interp::new();
    let buf = interp.array_buffer_from_bytes(data);
    interp.declare_global("buffer", crate::nanbox::NanBox::handle(buf.to_raw()));
    let value = interp.run(&program).map_err(|e| alloc::format!("{e:?}"))?;
    // Copy the post-run store back into the caller's region (round-trip proof).
    if let Some(bytes_h) = interp.array_buffer_bytes_handle(buf)
        && let Some(post) = interp.realm().bytes_at(bytes_h)
    {
        let n = post.len().min(data.len());
        data[..n].copy_from_slice(&post[..n]);
    }
    Ok(interp.realm().to_display_string(value))
}

/// Runs `src` with a pre-installed global `ArrayBuffer` named `buffer` that wraps
/// the caller's `data` region **zero-copy** (A6, #11): JS writes through a view
/// over `buffer` hit `data` in place, observed by the caller after the call. The
/// region must stay valid for the duration of the call (it is not freed here).
/// Returns the completion string, or an error message.
///
/// # Safety
/// `data` must be a unique, valid mutable region for the whole call; no other
/// alias may be used while the engine holds it.
#[cfg(feature = "std")]
#[allow(unsafe_code)]
unsafe fn eval_with_external_buffer(
    src: &str,
    data: *mut u8,
    len: usize,
) -> Result<alloc::string::String, alloc::string::String> {
    use alloc::string::ToString;
    let program = crate::parser::Parser::parse_program(src).map_err(|e| e.to_string())?;
    let mut interp = crate::nbexec::Interp::new();
    // SAFETY: the caller's contract guarantees `data`/`len` is a valid, unique
    // mutable region for the duration of this call; `None` free => the engine
    // never deallocates it. The buffer (and any view) is dropped with `interp`
    // at the end of this function, before the call returns to the caller.
    #[allow(unsafe_code)]
    let buf = unsafe { interp.array_buffer_from_external(data, len, None) };
    interp.declare_global("buffer", crate::nanbox::NanBox::handle(buf.to_raw()));
    let value = interp.run(&program).map_err(|e| alloc::format!("{e:?}"))?;
    Ok(interp.realm().to_display_string(value))
}

/// Compiles `src` to a portable `.ktbc` bytecode artifact, or an error message.
#[cfg(feature = "std")]
fn compile_to_bytes(src: &str) -> Result<alloc::vec::Vec<u8>, alloc::string::String> {
    use alloc::string::ToString;
    let program = crate::parser::Parser::parse_program(src).map_err(|e| e.to_string())?;
    let protos = crate::nbvm::compile_program(&program).map_err(|e| alloc::format!("{e:?}"))?;
    Ok(crate::bytecode::serialize(&protos))
}

/// Verifies and runs a `.ktbc` artifact, returning the completion string or an
/// error message.
#[cfg(feature = "std")]
fn run_bytecode_to_string(bytes: &[u8]) -> Result<alloc::string::String, alloc::string::String> {
    let protos =
        crate::bytecode::deserialize_verified(bytes).map_err(|e| alloc::format!("{e:?}"))?;
    let mut realm = crate::realm::Realm::new();
    match crate::nbvm::run_program_capturing(&mut realm, &protos, 0, &[]) {
        Ok((value, _output)) => Ok(realm.to_display_string(value)),
        Err(e) => Err(alloc::format!("{e:?}")),
    }
}

/// Runs `src` and serializes the object graph of its completion value to a D′
/// snapshot, or an error message if the completion isn't a heap object.
#[cfg(feature = "std")]
fn snapshot_source(src: &str) -> Result<alloc::vec::Vec<u8>, alloc::string::String> {
    use alloc::string::ToString;
    let program = crate::parser::Parser::parse_program(src).map_err(|e| e.to_string())?;
    let mut interp = crate::nbexec::Interp::new();
    let value = interp.run(&program).map_err(|e| alloc::format!("{e:?}"))?;
    if value.as_handle().is_none() {
        return Err("completion value is not a heap object to snapshot".to_string());
    }
    Ok(interp.snapshot(&[value]))
}

/// Restores a D′ snapshot into a fresh interpreter and renders its first root
/// value to a string — the load → reload path for data graphs across the C ABI.
#[cfg(feature = "std")]
fn restore_to_string(bytes: &[u8]) -> Result<alloc::string::String, alloc::string::String> {
    let mut interp = crate::nbexec::Interp::new();
    let roots = interp
        .restore_snapshot(bytes)
        .map_err(|e| alloc::format!("{e:?}"))?;
    let root = roots
        .first()
        .copied()
        .unwrap_or(crate::nanbox::NanBox::undefined());
    Ok(interp.realm().to_display_string(root))
}

/// Compiles `source` to a `.ktbc` bytecode artifact written into `out`.
///
/// # Safety
///
/// Same contract as [`kt_eval`]: `out_len` must be valid; `source` must cover
/// `source_len` bytes; `out` must hold `*out_len` writable bytes when the result
/// fits.
#[cfg(feature = "std")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kt_compile(
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
        // SAFETY: caller guarantees `source` covers `source_len` bytes; a null
        // pointer with zero length is the empty input (never dereferenced).
        let bytes: &[u8] = if source.is_null() {
            &[]
        } else {
            unsafe { core::slice::from_raw_parts(source as *const u8, source_len) }
        };
        let Ok(src) = core::str::from_utf8(bytes) else {
            return KtStatus::InvalidInput;
        };
        let (data, ok) = match compile_to_bytes(src) {
            Ok(artifact) => (artifact, true),
            Err(message) => (message.into_bytes(), false),
        };
        // SAFETY: `out_len` is non-null; `out` honors the length convention.
        match unsafe { copy_out(&data, out, out_len) } {
            KtStatus::Ok if !ok => KtStatus::InvalidInput,
            other => other,
        }
    }));
    match outcome {
        Ok(status) => status as c_int,
        Err(_) => KtStatus::Internal as c_int,
    }
}

/// Verifies and runs a `.ktbc` artifact, writing its result string into `out`.
///
/// # Safety
///
/// Same contract as [`kt_eval`]: `out_len` must be valid; `bytecode` must cover
/// `bytecode_len` bytes; `out` must hold `*out_len` writable bytes when the
/// result fits.
#[cfg(feature = "std")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kt_load_bytecode(
    bytecode: *const c_char,
    bytecode_len: usize,
    out: *mut c_char,
    out_len: *mut usize,
) -> c_int {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        if out_len.is_null() || (bytecode.is_null() && bytecode_len != 0) {
            return KtStatus::NullPointer;
        }
        // SAFETY: caller guarantees `bytecode` covers `bytecode_len` bytes; a
        // null pointer with zero length is the empty input (never dereferenced).
        let bytes: &[u8] = if bytecode.is_null() {
            &[]
        } else {
            unsafe { core::slice::from_raw_parts(bytecode as *const u8, bytecode_len) }
        };
        let (text, ok) = match run_bytecode_to_string(bytes) {
            Ok(value) => (value, true),
            Err(message) => (message, false),
        };
        // SAFETY: `out_len` is non-null; `out` honors the length convention.
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

/// Runs `source` and writes a D′ snapshot of its completion value's object graph
/// into `out` — portable bytes that [`kt_restore`] reloads. The completion must be
/// a heap object (object/array/string/…); a primitive completion yields
/// [`KtStatus::InvalidInput`] with the message in `out`.
///
/// # Safety
///
/// Same contract as [`kt_eval`]: `out_len` must be valid; `source` must cover
/// `source_len` bytes; `out` must hold `*out_len` writable bytes when the result
/// fits.
#[cfg(feature = "std")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kt_snapshot(
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
        // SAFETY: caller guarantees `source` covers `source_len` bytes; a null
        // pointer with zero length is the empty input (never dereferenced).
        let bytes: &[u8] = if source.is_null() {
            &[]
        } else {
            unsafe { core::slice::from_raw_parts(source as *const u8, source_len) }
        };
        let Ok(src) = core::str::from_utf8(bytes) else {
            return KtStatus::InvalidInput;
        };
        let (data, ok) = match snapshot_source(src) {
            Ok(artifact) => (artifact, true),
            Err(message) => (message.into_bytes(), false),
        };
        // SAFETY: `out_len` is non-null; `out` honors the length convention.
        match unsafe { copy_out(&data, out, out_len) } {
            KtStatus::Ok if !ok => KtStatus::InvalidInput,
            other => other,
        }
    }));
    match outcome {
        Ok(status) => status as c_int,
        Err(_) => KtStatus::Internal as c_int,
    }
}

/// Restores a snapshot written by [`kt_snapshot`] into a fresh runtime and writes
/// its first root value's string rendering into `out` (the cross-process load →
/// reload path for data graphs). A malformed snapshot yields
/// [`KtStatus::InvalidInput`].
///
/// # Safety
///
/// Same contract as [`kt_load_bytecode`]: `out_len` must be valid; `snapshot` must
/// cover `snapshot_len` bytes; `out` must hold `*out_len` writable bytes when the
/// result fits.
#[cfg(feature = "std")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kt_restore(
    snapshot: *const c_char,
    snapshot_len: usize,
    out: *mut c_char,
    out_len: *mut usize,
) -> c_int {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        if out_len.is_null() || (snapshot.is_null() && snapshot_len != 0) {
            return KtStatus::NullPointer;
        }
        // SAFETY: caller guarantees `snapshot` covers `snapshot_len` bytes; a
        // null pointer with zero length is the empty input (never dereferenced).
        let bytes: &[u8] = if snapshot.is_null() {
            &[]
        } else {
            unsafe { core::slice::from_raw_parts(snapshot as *const u8, snapshot_len) }
        };
        let (text, ok) = match restore_to_string(bytes) {
            Ok(value) => (value, true),
            Err(message) => (message, false),
        };
        // SAFETY: `out_len` is non-null; `out` honors the length convention.
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

// --- Value layer -----------------------------------------------------------
//
// The ABI-stable JS value and its pure constructors/inspectors (touch no engine
// state, so they need no `unsafe`). The context/callback bridge that lets C code
// build heap values and register functions layers on top (later §4.0 work).

/// An ABI-stable JavaScript value handed across the C boundary: the raw
/// [`NanBox`](crate::nanbox::NanBox) encoding. Opaque to C callers, who construct
/// and inspect it only through the `kt_value_*` functions.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct KtValue(pub u64);

impl KtValue {
    #[inline]
    fn nan(self) -> crate::nanbox::NanBox {
        crate::nanbox::NanBox::from_bits(self.0)
    }
    #[inline]
    fn of(v: crate::nanbox::NanBox) -> Self {
        KtValue(v.to_bits())
    }
}

/// The JS `undefined` value.
#[unsafe(no_mangle)]
pub extern "C" fn kt_value_undefined() -> KtValue {
    KtValue::of(crate::nanbox::NanBox::undefined())
}

/// The JS `null` value.
#[unsafe(no_mangle)]
pub extern "C" fn kt_value_null() -> KtValue {
    KtValue::of(crate::nanbox::NanBox::null())
}

/// A JS Number.
#[unsafe(no_mangle)]
pub extern "C" fn kt_value_number(n: f64) -> KtValue {
    KtValue::of(crate::nanbox::NanBox::number(n))
}

/// A JS Boolean.
#[unsafe(no_mangle)]
pub extern "C" fn kt_value_boolean(b: bool) -> KtValue {
    KtValue::of(crate::nanbox::NanBox::boolean(b))
}

/// Whether `v` is a Number.
#[unsafe(no_mangle)]
pub extern "C" fn kt_value_is_number(v: KtValue) -> bool {
    v.nan().is_number()
}

/// The numeric value of `v` (`NaN` if `v` is not a Number — gate on
/// [`kt_value_is_number`]).
#[unsafe(no_mangle)]
pub extern "C" fn kt_value_as_number(v: KtValue) -> f64 {
    v.nan().as_number().unwrap_or(f64::NAN)
}

/// Whether `v` is a Boolean.
#[unsafe(no_mangle)]
pub extern "C" fn kt_value_is_boolean(v: KtValue) -> bool {
    v.nan().as_boolean().is_some()
}

/// The boolean value of `v` (`false` if `v` is not a Boolean).
#[unsafe(no_mangle)]
pub extern "C" fn kt_value_as_boolean(v: KtValue) -> bool {
    v.nan().as_boolean().unwrap_or(false)
}

/// Whether `v` is exactly `undefined`.
#[unsafe(no_mangle)]
pub extern "C" fn kt_value_is_undefined(v: KtValue) -> bool {
    v.0 == crate::nanbox::NanBox::undefined().to_bits()
}

/// Whether `v` is exactly `null`.
#[unsafe(no_mangle)]
pub extern "C" fn kt_value_is_null(v: KtValue) -> bool {
    v.0 == crate::nanbox::NanBox::null().to_bits()
}

/// Whether `v` is a heap reference (object/array/function/string/…) rather than
/// an inline primitive.
#[unsafe(no_mangle)]
pub extern "C" fn kt_value_is_object(v: KtValue) -> bool {
    v.nan().is_handle()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn value_layer_round_trips() {
        // Constructors + inspectors agree, and the u64 encoding survives a copy
        // (the ABI-stable representation a C host passes by value).
        let n = kt_value_number(42.5);
        assert!(kt_value_is_number(n));
        assert!(!kt_value_is_boolean(n));
        assert!(!kt_value_is_object(n));
        assert_eq!(kt_value_as_number(n), 42.5);
        assert_eq!(kt_value_as_number(KtValue(n.0)), 42.5);

        let b = kt_value_boolean(true);
        assert!(kt_value_is_boolean(b));
        assert!(kt_value_as_boolean(b));
        assert!(!kt_value_is_number(b));

        assert!(kt_value_is_undefined(kt_value_undefined()));
        assert!(kt_value_is_null(kt_value_null()));
        assert!(!kt_value_is_null(kt_value_undefined()));
        assert!(!kt_value_is_number(kt_value_undefined()));
    }

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

    /// Runs `src` once through `kt_eval_with_buffer` over `data` (mutated in
    /// place), returning `(status, completion)`. A single call with an ample
    /// `out` is used (no length-query pass), because the owned-buffer entry point
    /// mutates `data` — re-running it would not be idempotent.
    #[cfg(feature = "std")]
    fn eval_buf(src: &str, data: &mut [u8]) -> (KtStatus, alloc::string::String) {
        let mut buf = alloc::vec![0i8; 256];
        let mut len: usize = buf.len();
        let rc = unsafe {
            kt_eval_with_buffer(
                src.as_ptr() as *const c_char,
                src.len(),
                data.as_mut_ptr(),
                data.len(),
                buf.as_mut_ptr(),
                &mut len,
            )
        };
        let bytes: alloc::vec::Vec<u8> = buf[..len].iter().map(|&b| b as u8).collect();
        let text = alloc::string::String::from_utf8(bytes).unwrap();
        let status = if rc == KtStatus::Ok as i32 {
            KtStatus::Ok
        } else {
            KtStatus::InvalidInput
        };
        (status, text)
    }

    #[cfg(feature = "std")]
    #[test]
    fn eval_with_owned_buffer_round_trips() {
        // The script reads the seeded bytes (sum) and mutates one; the owned store
        // is written back into the caller's `data` (round-trip proof, A6).
        let mut data = [1u8, 2, 3, 4];
        let (status, out) = eval_buf(
            "const v = new Uint8Array(buffer); const s = v[0]+v[1]+v[2]+v[3]; v[0] = 200; s",
            &mut data,
        );
        assert_eq!(status, KtStatus::Ok);
        assert_eq!(out, "10");
        assert_eq!(data, [200u8, 2, 3, 4]);
    }

    #[cfg(feature = "std")]
    #[test]
    fn eval_with_external_buffer_is_zero_copy() {
        // The script writes through a view over the wrapped region; the caller's
        // own bytes change in place (no copy back is performed — zero-copy, A6).
        let mut data = [0u8; 8];
        data[0] = 9;
        let mut len: usize = 0;
        let src = "const v = new Uint8Array(buffer); const seen = v[0]; v[5] = 77; seen";
        unsafe {
            kt_eval_with_external_buffer(
                src.as_ptr() as *const c_char,
                src.len(),
                data.as_mut_ptr(),
                data.len(),
                core::ptr::null_mut(),
                &mut len,
            )
        };
        let mut buf = alloc::vec![0i8; len];
        let rc = unsafe {
            kt_eval_with_external_buffer(
                src.as_ptr() as *const c_char,
                src.len(),
                data.as_mut_ptr(),
                data.len(),
                buf.as_mut_ptr(),
                &mut len,
            )
        };
        assert_eq!(rc, KtStatus::Ok as i32);
        let bytes: alloc::vec::Vec<u8> = buf.iter().map(|&b| b as u8).collect();
        assert_eq!(alloc::string::String::from_utf8(bytes).unwrap(), "9");
        assert_eq!(data[5], 77); // the external region itself was mutated
    }

    #[cfg(feature = "std")]
    fn snapshot_bytes(src: &str) -> (KtStatus, alloc::vec::Vec<u8>) {
        let mut len: usize = 0;
        unsafe {
            kt_snapshot(
                src.as_ptr() as *const c_char,
                src.len(),
                core::ptr::null_mut(),
                &mut len,
            )
        };
        let mut buf = alloc::vec![0i8; len];
        let rc = unsafe {
            kt_snapshot(
                src.as_ptr() as *const c_char,
                src.len(),
                buf.as_mut_ptr(),
                &mut len,
            )
        };
        let bytes: alloc::vec::Vec<u8> = buf.iter().map(|&b| b as u8).collect();
        let status = if rc == KtStatus::Ok as i32 {
            KtStatus::Ok
        } else {
            KtStatus::InvalidInput
        };
        (status, bytes)
    }

    #[cfg(feature = "std")]
    fn restore_str(snapshot: &[u8]) -> (KtStatus, alloc::string::String) {
        let mut len: usize = 0;
        unsafe {
            kt_restore(
                snapshot.as_ptr() as *const c_char,
                snapshot.len(),
                core::ptr::null_mut(),
                &mut len,
            )
        };
        let mut buf = alloc::vec![0i8; len];
        let rc = unsafe {
            kt_restore(
                snapshot.as_ptr() as *const c_char,
                snapshot.len(),
                buf.as_mut_ptr(),
                &mut len,
            )
        };
        let bytes: alloc::vec::Vec<u8> = buf.iter().map(|&b| b as u8).collect();
        let status = if rc == KtStatus::Ok as i32 {
            KtStatus::Ok
        } else {
            KtStatus::InvalidInput
        };
        (status, alloc::string::String::from_utf8(bytes).unwrap())
    }

    #[cfg(feature = "std")]
    #[test]
    fn snapshot_and_restore_round_trip() {
        // A data graph snapshotted in one runtime restores and renders in another
        // through the C ABI alone.
        let (s1, bytes) = snapshot_bytes("[1, 2, 3]");
        assert_eq!(s1, KtStatus::Ok);
        assert!(!bytes.is_empty());
        let (s2, text) = restore_str(&bytes);
        assert_eq!(s2, KtStatus::Ok);
        assert_eq!(text, "1,2,3");

        // A primitive completion has no object graph to snapshot.
        let (s3, _) = snapshot_bytes("42");
        assert_eq!(s3, KtStatus::InvalidInput);

        // A malformed snapshot is rejected, not panicked on.
        let (s4, _) = restore_str(b"not a snapshot");
        assert_eq!(s4, KtStatus::InvalidInput);
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
