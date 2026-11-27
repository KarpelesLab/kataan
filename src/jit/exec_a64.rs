//! W^X executable memory on AArch64 (Linux and macOS), with the instruction
//! cache maintenance the architecture requires before freshly written code
//! may run.
//!
//! The mapping is the x86-64 one's: map read-write, copy the code, flip the
//! pages to read-execute, unmap on drop. Only how the three memory calls are
//! made differs:
//!
//! - **Linux**: raw system calls (`svc #0`, number in `x8`: `mmap` = 222,
//!   `mprotect` = 226, `munmap` = 215), no C library, like the x86-64 path.
//! - **macOS**: the system call interface is private there, so the calls
//!   go through `libSystem` (which every macOS process links anyway; Rust's
//!   own runtime calls the same functions). A process without the hardened
//!   runtime may flip a private anonymous mapping from RW to RX with
//!   `mprotect`, so no `MAP_JIT` mapping and no per-thread write toggling
//!   (`pthread_jit_write_protect_np`) is needed: every buffer is written once,
//!   before it is ever executable.
//!
//! Unlike x86-64, AArch64's instruction fetch is not coherent with data
//! writes: after copying the code, each data-cache line it spans is cleaned
//! to the point of unification (`dc cvau`), each instruction-cache line
//! invalidated (`ic ivau`), with barriers between (`dsb ish`) and a context
//! synchronization (`isb`) at the end. On Linux that is done here from the
//! cache geometry in `CTR_EL0` (readable at EL0 on Linux); on macOS by
//! `sys_icache_invalidate`, the platform's supported way.

/// `PROT_READ | PROT_WRITE` (same values on Linux and macOS).
const PROT_RW: usize = 0x1 | 0x2;
/// `PROT_READ | PROT_EXEC`.
const PROT_RX: usize = 0x1 | 0x4;

#[cfg(target_os = "linux")]
mod sys {
    //! The Linux AArch64 system calls.

    /// `MAP_PRIVATE | MAP_ANONYMOUS`.
    const MAP_PRIVATE_ANON: usize = 0x02 | 0x20;
    const SYS_MMAP: usize = 222;
    const SYS_MPROTECT: usize = 226;
    const SYS_MUNMAP: usize = 215;

    /// The granule the mapping is sized in (the kernel rounds up to its own
    /// page size, which may be 4, 16 or 64 KiB).
    pub(super) const PAGE: usize = 4096;

    /// Issues a Linux AArch64 system call with up to six arguments.
    ///
    /// SAFETY: the caller passes a valid system call number and arguments;
    /// this executes `svc #0` with the kernel's register convention (number
    /// in `x8`, arguments in `x0..x5`, result in `x0`; nothing else is
    /// clobbered).
    #[allow(unsafe_code)]
    unsafe fn syscall6(n: usize, a: [usize; 6]) -> isize {
        let ret: isize;
        // SAFETY: a single `svc #0` with the documented register use.
        unsafe {
            core::arch::asm!(
                "svc #0",
                in("x8") n,
                inlateout("x0") a[0] as isize => ret,
                in("x1") a[1],
                in("x2") a[2],
                in("x3") a[3],
                in("x4") a[4],
                in("x5") a[5],
                options(nostack),
            );
        }
        ret
    }

    /// `mmap(NULL, len, PROT_RW, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0)`, or
    /// `None`.
    pub(super) fn map_rw(len: usize) -> Option<*mut u8> {
        // SAFETY: an anonymous mapping; the kernel returns a new mapping or
        // a small negative errno.
        #[allow(unsafe_code)]
        let raw = unsafe {
            syscall6(
                SYS_MMAP,
                [0, len, super::PROT_RW, MAP_PRIVATE_ANON, usize::MAX, 0],
            )
        };
        (!(-4095..0).contains(&raw)).then_some(raw as *mut u8)
    }

    /// `mprotect(base, len, PROT_RX)`; whether it succeeded.
    pub(super) fn protect_rx(base: *mut u8, len: usize) -> bool {
        // SAFETY: `base`/`len` describe a mapping `map_rw` made.
        #[allow(unsafe_code)]
        let r = unsafe { syscall6(SYS_MPROTECT, [base as usize, len, super::PROT_RX, 0, 0, 0]) };
        r == 0
    }

    /// `munmap(base, len)`.
    pub(super) fn unmap(base: *mut u8, len: usize) {
        // SAFETY: unmapping a whole mapping `map_rw` made, which nothing
        // uses any more.
        #[allow(unsafe_code)]
        unsafe {
            syscall6(SYS_MUNMAP, [base as usize, len, 0, 0, 0, 0]);
        }
    }

    /// Makes the `len` bytes of code at `start` visible to instruction
    /// fetch (see the module docs): `dc cvau` / `dsb ish` / `ic ivau` /
    /// `dsb ish` / `isb`, by the line sizes `CTR_EL0` reports, skipping the
    /// steps it says the hardware makes unnecessary (`IDC`, `DIC`).
    pub(super) fn sync_icache(start: *const u8, len: usize) {
        let ctr: u64;
        // SAFETY: reading `CTR_EL0`, which Linux lets EL0 read.
        #[allow(unsafe_code)]
        unsafe {
            core::arch::asm!("mrs {}, ctr_el0", out(reg) ctr, options(nomem, nostack, preserves_flags));
        }
        let dline = 4usize << ((ctr >> 16) & 0xF);
        let iline = 4usize << (ctr & 0xF);
        let (begin, end) = (start as usize, start as usize + len);
        if ctr & (1 << 28) == 0 {
            let mut p = begin & !(dline - 1);
            while p < end {
                // SAFETY: a cache clean by address within the mapped code.
                #[allow(unsafe_code)]
                unsafe {
                    core::arch::asm!("dc cvau, {}", in(reg) p, options(nostack, preserves_flags));
                }
                p += dline;
            }
        }
        // SAFETY: a barrier.
        #[allow(unsafe_code)]
        unsafe {
            core::arch::asm!("dsb ish", options(nostack, preserves_flags));
        }
        if ctr & (1 << 29) == 0 {
            let mut p = begin & !(iline - 1);
            while p < end {
                // SAFETY: an instruction-cache invalidate by address within
                // the mapped code.
                #[allow(unsafe_code)]
                unsafe {
                    core::arch::asm!("ic ivau, {}", in(reg) p, options(nostack, preserves_flags));
                }
                p += iline;
            }
            // SAFETY: a barrier.
            #[allow(unsafe_code)]
            unsafe {
                core::arch::asm!("dsb ish", options(nostack, preserves_flags));
            }
        }
        // SAFETY: a context synchronization barrier.
        #[allow(unsafe_code)]
        unsafe {
            core::arch::asm!("isb", options(nostack, preserves_flags));
        }
    }
}

#[cfg(target_os = "macos")]
mod sys {
    //! The macOS memory calls, through `libSystem` (see the module docs).

    use core::ffi::{c_int, c_void};

    /// `MAP_PRIVATE | MAP_ANON` (macOS values).
    const MAP_PRIVATE_ANON: c_int = 0x0002 | 0x1000;

    /// Apple Silicon's page size (mapping lengths are rounded to it).
    pub(super) const PAGE: usize = 16384;

    #[allow(unsafe_code)]
    unsafe extern "C" {
        fn mmap(
            addr: *mut c_void,
            len: usize,
            prot: c_int,
            flags: c_int,
            fd: c_int,
            off: i64,
        ) -> *mut c_void;
        fn mprotect(addr: *mut c_void, len: usize, prot: c_int) -> c_int;
        fn munmap(addr: *mut c_void, len: usize) -> c_int;
        fn sys_icache_invalidate(start: *mut c_void, len: usize);
    }

    /// An anonymous private read-write mapping of `len` bytes, or `None`.
    pub(super) fn map_rw(len: usize) -> Option<*mut u8> {
        // SAFETY: an anonymous mapping (no file, no fixed address); returns
        // `MAP_FAILED` (-1) on failure.
        #[allow(unsafe_code)]
        let p = unsafe {
            mmap(
                core::ptr::null_mut(),
                len,
                super::PROT_RW as c_int,
                MAP_PRIVATE_ANON,
                -1,
                0,
            )
        };
        (p as isize != -1 && !p.is_null()).then_some(p.cast::<u8>())
    }

    /// Flips the mapping to read-execute; whether it succeeded.
    pub(super) fn protect_rx(base: *mut u8, len: usize) -> bool {
        // SAFETY: `base`/`len` describe a mapping `map_rw` made.
        #[allow(unsafe_code)]
        let r = unsafe { mprotect(base.cast::<c_void>(), len, super::PROT_RX as c_int) };
        r == 0
    }

    /// Unmaps a mapping `map_rw` made.
    pub(super) fn unmap(base: *mut u8, len: usize) {
        // SAFETY: unmapping a whole mapping `map_rw` made, which nothing
        // uses any more.
        #[allow(unsafe_code)]
        unsafe {
            munmap(base.cast::<c_void>(), len);
        }
    }

    /// Makes the code at `start` visible to instruction fetch.
    pub(super) fn sync_icache(start: *const u8, len: usize) {
        // SAFETY: the range is mapped code this buffer owns.
        #[allow(unsafe_code)]
        unsafe {
            sys_icache_invalidate(start.cast_mut().cast::<c_void>(), len);
        }
    }
}

/// An executable code region of its own pages: mapped RW, the code copied
/// in, flipped to RX (W^X), the instruction cache synchronized; unmapped on
/// drop. As on x86-64 the code starts at a rotating cache-line offset into
/// its first page (cache colouring), and is position independent.
pub(super) struct ExecBuffer {
    base: *mut u8,
    ptr: *mut u8,
    len: usize,
    /// The code's own length in bytes.
    code_len: usize,
}

/// The colour of the next buffer (see [`ExecBuffer`]).
static NEXT_COLOUR: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(0);

/// Cache lines between successive buffers' start offsets (odd, so the
/// offsets cycle through every line of a 4 KiB page).
const COLOUR_STEP_LINES: usize = 13;

impl ExecBuffer {
    /// Maps `code` into a fresh executable region, or `None` on failure.
    pub(super) fn new(code: &[u8]) -> Option<Self> {
        if code.is_empty() {
            return None;
        }
        let colour = NEXT_COLOUR.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        let offset = (colour * COLOUR_STEP_LINES * 64) % 4096;
        let len = (offset + code.len()).div_ceil(sys::PAGE) * sys::PAGE;
        let base = sys::map_rw(len)?;
        // SAFETY: `offset < 4096 <= len`, inside the mapping.
        #[allow(unsafe_code)]
        let ptr = unsafe { base.add(offset) };
        // SAFETY: `ptr` points to `len - offset` freshly mapped writable bytes
        // and `code.len() <= len - offset`; the regions do not overlap.
        #[allow(unsafe_code)]
        unsafe {
            core::ptr::copy_nonoverlapping(code.as_ptr(), ptr, code.len());
        }
        if !sys::protect_rx(base, len) {
            sys::unmap(base, len);
            return None;
        }
        sys::sync_icache(ptr, code.len());
        Some(Self {
            base,
            ptr,
            len,
            code_len: code.len(),
        })
    }

    /// The code's length in bytes.
    #[must_use]
    pub(super) fn code_len(&self) -> usize {
        self.code_len
    }

    /// The executable code pointer.
    #[must_use]
    pub(super) fn ptr(&self) -> *const u8 {
        self.ptr
    }
}

impl Drop for ExecBuffer {
    fn drop(&mut self) {
        sys::unmap(self.base, self.len);
    }
}
