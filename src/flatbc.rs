//! Flat, fixed-record bytecode executed **in place** over a byte buffer — the
//! true zero-copy reload path (`ROADMAP.md` §2.2): the interpreter runs directly
//! over the mapped bytes, decoding one fixed-size record at a time, without ever
//! deserializing into an owned `Vec<Op>`.
//!
//! Unlike [`crate::bytecode`] (the `KTBC` container, which `deserialize`s into
//! `FnProto`s), this format is laid out so a program can be `mmap`'d and run as
//! is. Each instruction is a 16-byte record, so the program counter indexes the
//! buffer arithmetically (`header + pc * 16`) — jumps need no offset table — and
//! the only allocation at run time is the mutable `i64` register file, exactly as
//! a real engine keeps a register/stack over immutable mapped code.
//!
//! Numeric (integer) subset, mirroring the JIT's register IR. Pure, safe
//! `alloc`-only Rust (the optional `mmap` run path uses the same audited raw
//! syscalls as the JIT).

use alloc::vec::Vec;

/// `KFLT` — the flat-bytecode magic.
const MAGIC: &[u8; 4] = b"KFLT";
/// Header bytes: magic(4) + n_regs(u16) + n_params(u16) + n_ops(u32).
const HEADER: usize = 12;
/// Fixed bytes per instruction record.
const RECORD: usize = 16;

/// A flat-bytecode instruction (the encode-side IR). Registers are `u16`; `imm`
/// doubles as a jump target / argument index where applicable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(missing_docs)] // field names mirror the IR
pub enum FlatOp {
    Arg { dst: u16, index: u16 },
    Const { dst: u16, imm: i64 },
    Add { dst: u16, a: u16, b: u16 },
    Sub { dst: u16, a: u16, b: u16 },
    Mul { dst: u16, a: u16, b: u16 },
    And { dst: u16, a: u16, b: u16 },
    Or { dst: u16, a: u16, b: u16 },
    Xor { dst: u16, a: u16, b: u16 },
    Move { dst: u16, src: u16 },
    Lt { dst: u16, a: u16, b: u16 },
    JumpIfFalse { cond: u16, target: u32 },
    Jump { target: u32 },
    Ret { src: u16 },
}

const T_ARG: u8 = 0;
const T_CONST: u8 = 1;
const T_ADD: u8 = 2;
const T_SUB: u8 = 3;
const T_MUL: u8 = 4;
const T_AND: u8 = 5;
const T_OR: u8 = 6;
const T_XOR: u8 = 7;
const T_MOVE: u8 = 8;
const T_LT: u8 = 9;
const T_JF: u8 = 10;
const T_JMP: u8 = 11;
const T_RET: u8 = 12;

/// Encodes a program into the flat layout: a 12-byte header followed by one
/// 16-byte record per op.
#[must_use]
pub fn encode(n_regs: u16, n_params: u16, ops: &[FlatOp]) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEADER + ops.len() * RECORD);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&n_regs.to_le_bytes());
    out.extend_from_slice(&n_params.to_le_bytes());
    out.extend_from_slice(&(ops.len() as u32).to_le_bytes());
    for op in ops {
        // tag, reserved, dst, a, b, imm — 1 + 1 + 2 + 2 + 2 + 8 = 16 bytes.
        let (tag, dst, a, b, imm): (u8, u16, u16, u16, i64) = match *op {
            FlatOp::Arg { dst, index } => (T_ARG, dst, 0, 0, i64::from(index)),
            FlatOp::Const { dst, imm } => (T_CONST, dst, 0, 0, imm),
            FlatOp::Add { dst, a, b } => (T_ADD, dst, a, b, 0),
            FlatOp::Sub { dst, a, b } => (T_SUB, dst, a, b, 0),
            FlatOp::Mul { dst, a, b } => (T_MUL, dst, a, b, 0),
            FlatOp::And { dst, a, b } => (T_AND, dst, a, b, 0),
            FlatOp::Or { dst, a, b } => (T_OR, dst, a, b, 0),
            FlatOp::Xor { dst, a, b } => (T_XOR, dst, a, b, 0),
            FlatOp::Move { dst, src } => (T_MOVE, dst, src, 0, 0),
            FlatOp::Lt { dst, a, b } => (T_LT, dst, a, b, 0),
            FlatOp::JumpIfFalse { cond, target } => (T_JF, 0, cond, 0, i64::from(target)),
            FlatOp::Jump { target } => (T_JMP, 0, 0, 0, i64::from(target)),
            FlatOp::Ret { src } => (T_RET, 0, src, 0, 0),
        };
        out.push(tag);
        out.push(0); // reserved
        out.extend_from_slice(&dst.to_le_bytes());
        out.extend_from_slice(&a.to_le_bytes());
        out.extend_from_slice(&b.to_le_bytes());
        out.extend_from_slice(&imm.to_le_bytes());
    }
    out
}

/// Why a flat program failed to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlatError {
    /// Bad magic or a truncated header/record.
    Malformed,
    /// A register index `>= n_regs`.
    BadRegister,
    /// A jump target `>= n_ops`.
    BadTarget,
    /// An unknown opcode tag.
    BadTag,
    /// Control fell off the end without a `Ret`.
    NoReturn,
}

/// Executes a flat program **directly over `bytes`** (e.g. an `mmap`'d file),
/// with `args` bound to registers `0..n_params`. The buffer is never copied or
/// deserialized; each record is decoded on demand and every index is
/// bounds-checked, so running untrusted mapped bytes is memory-safe.
///
/// # Errors
/// Returns [`FlatError`] for a malformed buffer or an out-of-range index.
pub fn run(bytes: &[u8], args: &[i64]) -> Result<i64, FlatError> {
    if bytes.len() < HEADER || &bytes[0..4] != MAGIC {
        return Err(FlatError::Malformed);
    }
    let n_regs = u16::from_le_bytes([bytes[4], bytes[5]]) as usize;
    let _n_params = u16::from_le_bytes([bytes[6], bytes[7]]) as usize;
    let n_ops = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
    if bytes.len() < HEADER + n_ops * RECORD {
        return Err(FlatError::Malformed);
    }
    // The one runtime allocation: the mutable register file over immutable code.
    let mut regs = alloc::vec![0i64; n_regs];
    let reg = |regs: &[i64], r: u16| regs.get(r as usize).copied().ok_or(FlatError::BadRegister);

    let mut pc = 0usize;
    while pc < n_ops {
        let off = HEADER + pc * RECORD;
        let rec = &bytes[off..off + RECORD];
        let tag = rec[0];
        let dst = u16::from_le_bytes([rec[2], rec[3]]);
        let a = u16::from_le_bytes([rec[4], rec[5]]);
        let b = u16::from_le_bytes([rec[6], rec[7]]);
        let imm = i64::from_le_bytes([
            rec[8], rec[9], rec[10], rec[11], rec[12], rec[13], rec[14], rec[15],
        ]);
        let set = |regs: &mut [i64], r: u16, v: i64| -> Result<(), FlatError> {
            *regs.get_mut(r as usize).ok_or(FlatError::BadRegister)? = v;
            Ok(())
        };
        match tag {
            T_ARG => {
                let v = args.get(imm as usize).copied().unwrap_or(0);
                set(&mut regs, dst, v)?;
                pc += 1;
            }
            T_CONST => {
                set(&mut regs, dst, imm)?;
                pc += 1;
            }
            T_ADD | T_SUB | T_MUL | T_AND | T_OR | T_XOR => {
                let (x, y) = (reg(&regs, a)?, reg(&regs, b)?);
                let v = match tag {
                    T_ADD => x.wrapping_add(y),
                    T_SUB => x.wrapping_sub(y),
                    T_MUL => x.wrapping_mul(y),
                    T_AND => x & y,
                    T_OR => x | y,
                    _ => x ^ y,
                };
                set(&mut regs, dst, v)?;
                pc += 1;
            }
            T_MOVE => {
                let v = reg(&regs, a)?;
                set(&mut regs, dst, v)?;
                pc += 1;
            }
            T_LT => {
                let v = i64::from(reg(&regs, a)? < reg(&regs, b)?);
                set(&mut regs, dst, v)?;
                pc += 1;
            }
            T_JF => {
                // The condition register is encoded in field `a`.
                let target = imm as usize;
                if target >= n_ops {
                    return Err(FlatError::BadTarget);
                }
                if reg(&regs, a)? == 0 {
                    pc = target;
                } else {
                    pc += 1;
                }
            }
            T_JMP => {
                let target = imm as usize;
                if target >= n_ops {
                    return Err(FlatError::BadTarget);
                }
                pc = target;
            }
            // The returned register is encoded in field `a`.
            T_RET => return reg(&regs, a),
            _ => return Err(FlatError::BadTag),
        }
    }
    Err(FlatError::NoReturn)
}

/// Maps `path` read-only and runs the flat program **over the mapped bytes**
/// (zero-copy: the file is the program). Linux/x86-64 only.
#[cfg(all(feature = "std", target_os = "linux", target_arch = "x86_64"))]
pub use mmap::run_file;

#[cfg(all(feature = "std", target_os = "linux", target_arch = "x86_64"))]
mod mmap {
    use super::{FlatError, run};
    use std::os::unix::io::AsRawFd;

    const PROT_READ: usize = 0x1;
    const MAP_PRIVATE: usize = 0x02;
    const SYS_MMAP: usize = 9;
    const SYS_MUNMAP: usize = 11;

    /// SAFETY: issues the `syscall` instruction with the System V syscall ABI.
    #[allow(unsafe_code)]
    unsafe fn syscall6(n: usize, a1: usize, a2: usize, a3: usize, a4: usize, a5: usize) -> isize {
        let ret: isize;
        // SAFETY: a single syscall with documented inputs/clobbers.
        unsafe {
            core::arch::asm!(
                "syscall",
                inlateout("rax") n as isize => ret,
                in("rdi") a1, in("rsi") a2, in("rdx") a3,
                in("r10") a4, in("r8") a5, in("r9") 0usize,
                out("rcx") _, out("r11") _,
                options(nostack, preserves_flags),
            );
        }
        ret
    }

    /// Runs the flat program at `path` directly from a memory mapping.
    ///
    /// # Errors
    /// `io::Error` on a failed open/`mmap`, or `InvalidData` wrapping the
    /// [`FlatError`] if the mapped bytes don't run.
    pub fn run_file(path: &str, args: &[i64]) -> std::io::Result<i64> {
        let file = std::fs::File::open(path)?;
        let len = file.metadata()?.len() as usize;
        if len == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "empty program",
            ));
        }
        let fd = file.as_raw_fd() as usize;
        // SAFETY: a standard read-only file mapping.
        #[allow(unsafe_code)]
        let raw = unsafe { syscall6(SYS_MMAP, 0, len, PROT_READ, MAP_PRIVATE, fd) };
        if (-4095..0).contains(&raw) {
            return Err(std::io::Error::last_os_error());
        }
        let ptr = raw as *const u8;
        // SAFETY: `ptr..ptr+len` is a valid read-only mapping of the whole file,
        // alive until the `munmap` below; `run` only reads it.
        #[allow(unsafe_code)]
        let result: Result<i64, FlatError> = {
            let bytes = unsafe { core::slice::from_raw_parts(ptr, len) };
            run(bytes, args)
        };
        // SAFETY: unmapping exactly the region we mapped.
        #[allow(unsafe_code)]
        unsafe {
            syscall6(SYS_MUNMAP, ptr as usize, len, 0, 0, 0);
        }
        result.map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, alloc::format!("{e:?}"))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn encode_run_arithmetic() {
        // r2 = (r0 + r1) * r0 ; ret r2
        let ops = vec![
            FlatOp::Arg { dst: 0, index: 0 },
            FlatOp::Arg { dst: 1, index: 1 },
            FlatOp::Add { dst: 2, a: 0, b: 1 },
            FlatOp::Mul { dst: 2, a: 2, b: 0 },
            FlatOp::Ret { src: 2 },
        ];
        let bytes = encode(3, 2, &ops);
        assert_eq!(&bytes[0..4], MAGIC);
        for (a, b) in [(3i64, 4), (10, -2), (0, 0), (-5, 5)] {
            assert_eq!(run(&bytes, &[a, b]).unwrap(), (a + b) * a, "({a},{b})");
        }
    }

    #[test]
    fn flat_loop_executes_in_place() {
        // sum(0..n): r1=0(acc); r2=0(i); loop: if !(i<n) goto end; acc+=i; i+=1; goto loop; end: ret acc
        // r0 = n (arg), r3 = 1 (const), r4 = scratch (i<n)
        let ops = vec![
            FlatOp::Arg { dst: 0, index: 0 },           // 0: n
            FlatOp::Const { dst: 1, imm: 0 },           // 1: acc = 0
            FlatOp::Const { dst: 2, imm: 0 },           // 2: i = 0
            FlatOp::Const { dst: 3, imm: 1 },           // 3: one = 1
            FlatOp::Lt { dst: 4, a: 2, b: 0 },          // 4: cond = i < n
            FlatOp::JumpIfFalse { cond: 4, target: 9 }, // 5: if !cond goto 9
            FlatOp::Add { dst: 1, a: 1, b: 2 },         // 6: acc += i
            FlatOp::Add { dst: 2, a: 2, b: 3 },         // 7: i += 1
            FlatOp::Jump { target: 4 },                 // 8: goto 4
            FlatOp::Ret { src: 1 },                     // 9: ret acc
        ];
        let bytes = encode(5, 1, &ops);
        for n in [0i64, 1, 5, 10, 100] {
            assert_eq!(run(&bytes, &[n]).unwrap(), n * (n - 1) / 2, "sum 0..{n}");
        }
    }

    #[test]
    fn rejects_malformed_and_out_of_range() {
        assert_eq!(run(b"XXXX", &[]), Err(FlatError::Malformed));
        // Truncated record.
        let mut bytes = encode(1, 0, &[FlatOp::Ret { src: 0 }]);
        bytes.truncate(bytes.len() - 1);
        assert_eq!(run(&bytes, &[]), Err(FlatError::Malformed));
        // Out-of-range register.
        let bad = encode(1, 0, &[FlatOp::Ret { src: 9 }]);
        assert_eq!(run(&bad, &[]), Err(FlatError::BadRegister));
        // Out-of-range jump target.
        let bad = encode(1, 0, &[FlatOp::Jump { target: 99 }]);
        assert_eq!(run(&bad, &[]), Err(FlatError::BadTarget));
    }

    #[cfg(all(feature = "std", target_os = "linux", target_arch = "x86_64"))]
    #[test]
    fn mmap_zero_copy_execution() {
        // Write a flat program to a file, then execute it directly from an mmap
        // — no deserialize, the mapped bytes are the program.
        let ops = vec![
            FlatOp::Arg { dst: 0, index: 0 },
            FlatOp::Const { dst: 1, imm: 7 },
            FlatOp::Mul { dst: 0, a: 0, b: 1 },
            FlatOp::Ret { src: 0 },
        ];
        let bytes = encode(2, 1, &ops);
        let path = std::env::temp_dir().join(alloc::format!(
            "kataan_flat_{}.kflt",
            std::process::id() as u64
        ));
        std::fs::write(&path, &bytes).unwrap();
        let r = run_file(path.to_str().unwrap(), &[6]);
        std::fs::remove_file(&path).ok();
        assert_eq!(
            r.unwrap(),
            42,
            "6 * 7 executed zero-copy from the mapped file"
        );
    }
}
