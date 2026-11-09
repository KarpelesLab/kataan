//! **Hoisted-view element ops of any value in the integer version**: a
//! speculative region's integer version reads and writes a hoisted typed
//! array view (`typed::ViewSlots`) by an int index inline, whatever the
//! element kind, so a loop over a `Float64Array` with int counters and
//! indices (`a[ri + j] -= f * a[rk + j]`), or one storing int values into a
//! `Uint8Array` (`next[row + x] = v`), runs no generic code there: the
//! generic emitter's typed path would have the int GPRs saved around it and
//! the index converted into its double location first.
//!
//! Each fast path touches only `rax`, `rdx`, `r11`, `xmm0` and `xmm1` (and
//! a read's destination), with the element at `[r11 + rax << size]`; any
//! miss (the register not the hoisted view, an index out of bounds, a kind
//! the path does not serve, a value it does not convert) falls into the
//! op's generic code (`int_skip`), which runs the op from the start, before
//! anything was written.

use super::super::{NumGen, RAX, RDX, disp};
use super::IK;
use crate::jit::X64Assembler;
use crate::jit::call_seq::{cc, r};
use crate::jit::typed::{
    self, FLOAT32, FLOAT64, INT8, INT16, INT32, UINT8, UINT8_CLAMPED, UINT16, UINT32,
};

/// x86 condition codes beyond `cc`'s.
const CC_O: u8 = 0x0;
const CC_NP: u8 = 0xb;
const CC_NS: u8 = 0x9;
const CC_LE: u8 = 0xe;

impl X64Assembler {
    /// `[pfx] REX op ModRM SIB` with the element operand `[r11 + rax << s]`
    /// (`reg` the ModRM register field, `w` REX.W).
    fn v_elem(&mut self, pfx: Option<u8>, w: bool, opcode: &[u8], reg: u8, s: u8) {
        if let Some(p) = pfx {
            self.code.push(p);
        }
        self.code
            .push(0x41 | (u8::from(w) << 3) | (((reg >> 3) & 1) << 2));
        self.code.extend_from_slice(opcode);
        self.code.push(((reg & 7) << 3) | 4); // mod=00, rm=SIB
        self.code.push((s << 6) | 0x03); // index rax, base r11
    }

    /// To `miss` unless the register at frame displacement `recv` holds
    /// hoisted view `v` and the index in `rax` is below its length; then
    /// `r11` = its first element's address. Clobbers `r11` (and `rax` only
    /// to give it back).
    fn v_index(&mut self, v: typed::ViewSlots, recv: i32, miss: crate::jit::Label) {
        if let Some(h) = v.held {
            self.g_mov(r::R11, r::RAX);
            self.load_rax(recv);
            self.g_cmp_mem(r::RAX, r::RBP, h);
            self.g_jcc(cc::NE, miss);
            self.g_mov(r::RAX, r::R11);
        }
        self.g_cmp_mem(r::RAX, r::RBP, v.len);
        self.g_jcc(cc::AE, miss); // past the length, or negative
        self.g_load(r::R11, r::RBP, v.base);
    }
}

impl NumGen {
    /// The inline fast path of `dst = obj[index]` in a speculative region's
    /// integer version, `index` an int register and `dst` none, `obj` a
    /// hoisted view: the element as a Number into `dst`'s double-version
    /// location, past the op's generic code (`int_skip`). See the module
    /// docs.
    pub(in super::super) fn ihoisted_view_get_any(
        &mut self,
        a: &mut X64Assembler,
        dst: u16,
        obj: u16,
        index: u16,
        v: typed::ViewSlots,
    ) -> Option<()> {
        let x = self.xmm_of(self.imode, dst).unwrap_or(0);
        let slow = a.new_label();
        let done = self.int_skip.take().unwrap_or_else(|| a.new_label());
        let got = a.new_label();
        self.index_rax(a, index)?;
        a.v_index(v, disp(obj), slow);
        for kind in [
            UINT8,
            INT32,
            FLOAT64,
            FLOAT32,
            UINT16,
            INT16,
            INT8,
            UINT32,
            UINT8_CLAMPED,
        ] {
            let next = a.new_label();
            a.g_cmp_byte_imm(r::RBP, v.kind, kind);
            a.g_jcc(cc::NE, next);
            let s = typed::size_log2(kind);
            match kind {
                INT8 => a.v_elem(None, false, &[0x0f, 0xbe], RAX, s), // movsx eax, byte
                UINT8 | UINT8_CLAMPED => a.v_elem(None, false, &[0x0f, 0xb6], RAX, s),
                INT16 => a.v_elem(None, false, &[0x0f, 0xbf], RAX, s), // movsx eax, word
                UINT16 => a.v_elem(None, false, &[0x0f, 0xb7], RAX, s), // movzx eax, word
                INT32 | UINT32 => a.v_elem(None, false, &[0x8b], RAX, s), // mov eax, dword
                FLOAT32 => a.v_elem(Some(0xf3), false, &[0x0f, 0x5a], x, s), // cvtss2sd
                _ => a.v_elem(Some(0xf2), false, &[0x0f, 0x10], x, s), // movsd
            }
            match kind {
                // (The zero-extended `u32`.)
                UINT32 => a.cvtsi2sd_x_gpr(x, RAX),
                FLOAT32 | FLOAT64 => {
                    // A NaN is the canonical one.
                    a.sse_rr(0x66, 0x2e, x, x); // ucomisd x, x
                    a.jcc(CC_NP, got);
                    a.movabs_rax(crate::nanbox::NanBox::number(f64::NAN).to_bits() as i64);
                    a.movq_x_gpr(x, RAX);
                }
                _ => a.cvtsi2sd32(x, RAX),
            }
            a.jmp(got);
            a.bind(next);
        }
        a.jmp(slow);
        a.bind(got);
        if self.xmm_of(self.imode, dst).is_none() {
            a.movsd_mem_x(disp(dst), 0);
        }
        a.jmp(done);
        a.bind(slow);
        self.int_skip = Some(done);
        Some(())
    }

    /// The inline fast path of `obj[index] = val` in a speculative region's
    /// integer version, `index` an int register, `obj` a hoisted view: the
    /// element written, past the op's generic code (`int_skip`), as
    /// `typed::t_set` converts it, when `val` is an int register, a promoted
    /// (so Number) one or a Number in the register file. See the module
    /// docs.
    pub(in super::super) fn ihoisted_view_set(
        &mut self,
        a: &mut X64Assembler,
        obj: u16,
        index: u16,
        val: u16,
        v: typed::ViewSlots,
    ) -> Option<()> {
        let int = self.is_int(val);
        let vx = if int {
            None
        } else {
            self.xmm_of(self.imode, val)
        };
        let slow = a.new_label();
        let done = self.int_skip.take().unwrap_or_else(|| a.new_label());
        // The value: an int32 in `edx`, or a Number in `xmm0` (one in the
        // register file checked to be one); then the index and the element
        // base (`v_index` keeps `rdx` and `xmm0`).
        match (int, vx) {
            (true, _) => match self.ik(val)? {
                IK::Gpr(g) => a.mov32_rr(RDX, g),
                IK::Imm(c) => a.mov32_imm(RDX, c),
                IK::Mem(d) => a.ld32(RDX, d),
            },
            (false, Some(x)) => a.movapd_xx(0, x),
            (false, None) => {
                a.load_rdx(disp(val));
                a.raw(&[0x48, 0x89, 0xd1]); // mov rcx, rdx
                a.movabs_r11(super::super::QNAN as i64);
                a.and_rcx_r11();
                a.cmp_rcx_r11();
                a.g_jcc(cc::E, slow);
                a.movq_x_gpr(0, RDX);
            }
        }
        self.index_rax(a, index)?;
        a.v_index(v, disp(obj), slow);
        let floats = a.new_label();
        let clamped = a.new_label();
        let (byte, dword) = (a.new_label(), a.new_label());
        a.g_cmp_byte_imm(r::RBP, v.kind, UINT32);
        a.g_jcc(cc::A, floats);
        a.g_cmp_byte_imm(r::RBP, v.kind, UINT8_CLAMPED);
        a.g_jcc(cc::E, clamped);
        if !int {
            // `ToInt8` .. `ToUint32`: the truncation's low bits, for a
            // Number below 2^63 in magnitude (else the "integer indefinite"
            // `i64::MIN`, the only value `cmp rdx, 1` overflows on).
            a.raw(&[0xf2, 0x48, 0x0f, 0x2c, 0xd0]); // cvttsd2si rdx, xmm0
            a.raw(&[0x48, 0x83, 0xfa, 0x01]); // cmp rdx, 1
            a.jcc(CC_O, slow);
        }
        a.g_cmp_byte_imm(r::RBP, v.kind, UINT8);
        a.g_jcc(cc::BE, byte);
        a.g_cmp_byte_imm(r::RBP, v.kind, INT32);
        a.g_jcc(cc::AE, dword);
        a.v_elem(Some(0x66), false, &[0x89], RDX, 1); // mov word [..], dx
        a.jmp(done);
        a.bind(dword);
        a.v_elem(None, false, &[0x89], RDX, 2); // mov dword [..], edx
        a.jmp(done);
        // `ToUint8Clamp`: a Number rounded half to even (`cvtsd2si` in the
        // default rounding mode) before the clamp; an int32 clamped.
        a.bind(clamped);
        if !int {
            a.raw(&[0xf2, 0x48, 0x0f, 0x2d, 0xd0]); // cvtsd2si rdx, xmm0
            a.raw(&[0x48, 0x83, 0xfa, 0x01]); // cmp rdx, 1
            a.jcc(CC_O, slow);
        } else {
            a.raw(&[0x48, 0x63, 0xd2]); // movsxd rdx, edx
        }
        let low = a.new_label();
        let high = a.new_label();
        a.raw(&[0x48, 0x85, 0xd2]); // test rdx, rdx
        a.jcc(CC_NS, low);
        a.raw(&[0x31, 0xd2]); // xor edx, edx
        a.bind(low);
        a.raw(&[0x48, 0x81, 0xfa, 0xff, 0, 0, 0]); // cmp rdx, 255
        a.jcc(CC_LE, high);
        a.raw(&[0xba, 0xff, 0, 0, 0]); // mov edx, 255
        a.bind(high);
        a.bind(byte);
        a.v_elem(None, false, &[0x88], RDX, 0); // mov byte [..], dl
        a.jmp(done);
        // The float kinds store any Number.
        a.bind(floats);
        if int {
            a.cvtsi2sd32(0, RDX);
        }
        let not64 = a.new_label();
        a.g_cmp_byte_imm(r::RBP, v.kind, FLOAT64);
        a.g_jcc(cc::NE, not64);
        a.v_elem(Some(0xf2), false, &[0x0f, 0x11], 0, 3); // movsd [..], xmm0
        a.jmp(done);
        a.bind(not64);
        a.g_cmp_byte_imm(r::RBP, v.kind, FLOAT32);
        a.g_jcc(cc::NE, slow);
        a.raw(&[0xf2, 0x0f, 0x5a, 0xc8]); // cvtsd2ss xmm1, xmm0
        a.v_elem(Some(0xf3), false, &[0x0f, 0x11], 1, 2); // movss [..], xmm1
        a.jmp(done);
        a.bind(slow);
        self.int_skip = Some(done);
        Some(())
    }
}
