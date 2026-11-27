//! The A64 encoder checked against the GNU assembler: each case's text was
//! assembled offline with `aarch64-unknown-linux-gnu-as` and the words it
//! produced are recorded next to it (`tools/gen_a64_table.py` regenerates
//! them), so this test needs no cross toolchain and runs on every host.

use super::*;

type Case = (&'static str, fn(&mut A64Assembler), &'static [u32]);

macro_rules! case {
    ($text:expr, $f:expr, $words:expr) => {
        ($text, $f as fn(&mut A64Assembler), &$words)
    };
}

#[rustfmt::skip]
fn cases() -> Vec<Case> {
    alloc::vec![
        // A64ENC-BEGIN
        case!("mov x1, x2", |a| a.mov(1, 2), [0xaa0203e1]),
        case!("mov w3, w30", |a| a.mov_w(3, 30), [0x2a1e03e3]),
        case!("mov x29, sp", |a| a.mov_sp(29, SP), [0x910003fd]),
        case!("mov sp, x29", |a| a.mov_sp(SP, 29), [0x910003bf]),
        case!("movz x5, #0x1234, lsl #16", |a| a.movz(5, 0x1234, 1), [0xd2a24685]),
        case!("movk x5, #0xbeef, lsl #48", |a| a.movk(5, 0xbeef, 3), [0xf2f7dde5]),
        case!("movn x7, #0", |a| a.movn(7, 0, 0), [0x92800007]),
        case!("movz x0, #0", |a| a.mov_imm(0, 0), [0xd2800000]),
        case!("movz x0, #0x7fff", |a| a.mov_imm(0, 0x7fff), [0xd28fffe0]),
        case!("movn x0, #0", |a| a.mov_imm(0, u64::MAX), [0x92800000]),
        case!("movn x0, #4", |a| a.mov_imm(0, (-5i64) as u64), [0x92800080]),
        case!("movz x9, #0x20, lsl #48", |a| a.mov_imm(9, 1u64 << 53), [0xd2e00409]),
        case!("movz x10, #0xffe0, lsl #48", |a| a.mov_imm(10, (-(1i64 << 53)) as u64), [0xd2fffc0a]),
        case!("movz x1, #0x5678; movk x1, #0x1234, lsl #16; movk x1, #0xdef0, lsl #32; movk x1, #0x9abc, lsl #48", |a| a.mov_imm(1, 0x9abc_def0_1234_5678), [0xd28acf01, 0xf2a24681, 0xf2dbde01, 0xf2f35781]),
        case!("movz x2, #0x1, lsl #16; movk x2, #0x1, lsl #48", |a| a.mov_imm(2, 0x0001_0000_0001_0000), [0xd2a00022, 0xf2e00022]),
        case!("movn x3, #0x1234, lsl #16", |a| a.mov_imm(3, !0x1234_0000u64), [0x92a24683]),
        case!("add x1, x2, x3", |a| a.add(1, 2, 3), [0x8b030041]),
        case!("adds x4, x5, x6", |a| a.adds(4, 5, 6), [0xab0600a4]),
        case!("sub x7, x8, x9", |a| a.sub(7, 8, 9), [0xcb090107]),
        case!("subs x10, x11, x12", |a| a.subs(10, 11, 12), [0xeb0c016a]),
        case!("add x1, x2, x3, lsl #3", |a| a.add_shifted(1, 2, 3, Shift::Lsl, 3), [0x8b030c41]),
        case!("add x1, x2, x3, asr #63", |a| a.add_shifted(1, 2, 3, Shift::Asr, 63), [0x8b83fc41]),
        case!("add x16, sp, x16, uxtx", |a| a.add_ext(IP0, SP, IP0), [0x8b3063f0]),
        case!("cmp x0, x1", |a| a.cmp(0, 1), [0xeb01001f]),
        case!("cmp x3, x2, asr #63", |a| a.cmp_shifted(3, 2, Shift::Asr, 63), [0xeb82fc7f]),
        case!("str x2, [sp, #-16]!", |a| a.push(2), [0xf81f0fe2]),
        case!("ldr x3, [sp], #16", |a| a.pop(3), [0xf84107e3]),
        case!("cmp w0, w1", |a| a.cmp_w(0, 1), [0x6b01001f]),
        case!("neg x2, x3", |a| a.neg(2, 3), [0xcb0303e2]),
        case!("negs x2, x3", |a| a.negs(2, 3), [0xeb0303e2]),
        case!("add x0, x1, #4095", |a| a.add_imm(0, 1, 4095), [0x913ffc20]),
        case!("add x0, x1, #0x1000", |a| a.add_imm(0, 1, 0x1000), [0x91400420]),
        case!("sub x0, x1, #16", |a| a.add_imm(0, 1, -16), [0xd1004020]),
        case!("sub sp, sp, #528", |a| a.sub_imm(SP, SP, 528), [0xd10843ff]),
        case!("add sp, sp, #528", |a| a.sub_imm(SP, SP, -528), [0x910843ff]),
        case!("adds x3, x4, #1", |a| a.adds_imm(3, 4, 1), [0xb1000483]),
        case!("cmp x5, #0", |a| a.cmp_imm(5, 0), [0xf10000bf]),
        case!("cmp x5, #42", |a| a.cmp_imm(5, 42), [0xf100a8bf]),
        case!("cmn x5, #42", |a| a.cmp_imm(5, -42), [0xb100a8bf]),
        case!("mul x1, x2, x3", |a| a.mul(1, 2, 3), [0x9b037c41]),
        case!("smulh x4, x5, x6", |a| a.smulh(4, 5, 6), [0x9b467ca4]),
        case!("msub x0, x1, x2, x3", |a| a.msub(0, 1, 2, 3), [0x9b028c20]),
        case!("sdiv x0, x1, x2", |a| a.sdiv(0, 1, 2), [0x9ac20c20]),
        case!("and x0, x1, x2", |a| a.and(0, 1, 2), [0x8a020020]),
        case!("orr x0, x1, x2", |a| a.orr(0, 1, 2), [0xaa020020]),
        case!("eor x0, x1, x2", |a| a.eor(0, 1, 2), [0xca020020]),
        case!("tst x3, x4", |a| a.tst(3, 4), [0xea04007f]),
        case!("and w0, w1, w2", |a| a.and_w(0, 1, 2), [0x0a020020]),
        case!("orr w0, w1, w2", |a| a.orr_w(0, 1, 2), [0x2a020020]),
        case!("eor w0, w1, w2", |a| a.eor_w(0, 1, 2), [0x4a020020]),
        case!("mvn w3, w4", |a| a.mvn_w(3, 4), [0x2a2403e3]),
        case!("mvn x3, x4", |a| a.mvn(3, 4), [0xaa2403e3]),
        case!("lsl w0, w1, w2", |a| a.lsl_w(0, 1, 2), [0x1ac22020]),
        case!("lsr w0, w1, w2", |a| a.lsr_w(0, 1, 2), [0x1ac22420]),
        case!("asr w0, w1, w2", |a| a.asr_w(0, 1, 2), [0x1ac22820]),
        case!("lsl x0, x1, x2", |a| a.lsl(0, 1, 2), [0x9ac22020]),
        case!("asr x0, x1, x2", |a| a.asr(0, 1, 2), [0x9ac22820]),
        case!("lsl x0, x1, #3", |a| a.lsl_imm(0, 1, 3), [0xd37df020]),
        case!("lsl x0, x1, #0", |a| a.lsl_imm(0, 1, 0), [0xd340fc20]),
        case!("lsr x0, x1, #47", |a| a.lsr_imm(0, 1, 47), [0xd36ffc20]),
        case!("asr x0, x1, #63", |a| a.asr_imm(0, 1, 63), [0x937ffc20]),
        case!("sxtw x0, w1", |a| a.sxtw(0, 1), [0x93407c20]),
        case!("cset x0, eq", |a| a.cset(0, Cond::Eq), [0x9a9f17e0]),
        case!("cset x1, lt", |a| a.cset(1, Cond::Lt), [0x9a9fa7e1]),
        case!("cset x2, mi", |a| a.cset(2, Cond::Mi), [0x9a9f57e2]),
        case!("cset x3, vs", |a| a.cset(3, Cond::Vs), [0x9a9f77e3]),
        case!("csel x0, x1, x2, gt", |a| a.csel(0, 1, 2, Cond::Gt), [0x9a82c020]),
        case!("ldr x0, [sp, #8]", |a| a.ldr(0, SP, 8), [0xf94007e0]),
        case!("ldr x0, [x29, #32760]", |a| a.ldr(0, 29, 32760), [0xf97fffa0]),
        case!("ldur x0, [x29, #-8]", |a| a.ldr(0, 29, -8), [0xf85f83a0]),
        case!("ldur x0, [x1, #3]", |a| a.ldr(0, 1, 3), [0xf8403020]),
        case!("movn x16, #0x1ff; add x16, x29, x16, uxtx; ldr x0, [x16]", |a| a.ldr(0, 29, -512), [0x92803ff0, 0x8b3063b0, 0xf9400200]),
        case!("movz x16, #0x8000; add x16, sp, x16, uxtx; ldr x0, [x16]", |a| a.ldr(0, SP, 32768), [0xd2900010, 0x8b3063f0, 0xf9400200]),
        case!("str x1, [sp, #16]", |a| a.str(1, SP, 16), [0xf9000be1]),
        case!("stur x1, [x29, #-256]", |a| a.str(1, 29, -256), [0xf81003a1]),
        case!("ldr w2, [x3, #12]", |a| a.ldr_w(2, 3, 12), [0xb9400c62]),
        case!("str w2, [x3, #4]", |a| a.str_w(2, 3, 4), [0xb9000462]),
        case!("ldrh w2, [x3, #6]", |a| a.ldrh(2, 3, 6), [0x79400c62]),
        case!("ldrb w2, [x3, #7]", |a| a.ldrb(2, 3, 7), [0x39401c62]),
        case!("strb w2, [x3, #1]", |a| a.strb(2, 3, 1), [0x39000462]),
        case!("ldr d0, [sp, #24]", |a| a.ldr_d(0, SP, 24), [0xfd400fe0]),
        case!("str d31, [x1, #8]", |a| a.str_d(31, 1, 8), [0xfd00043f]),
        case!("ldur d1, [x29, #-16]", |a| a.ldr_d(1, 29, -16), [0xfc5f03a1]),
        case!("ldr x0, [x1, x2, lsl #3]", |a| a.ldr_idx(0, 1, 2), [0xf8627820]),
        case!("str x0, [x1, x2, lsl #3]", |a| a.str_idx(0, 1, 2), [0xf8227820]),
        case!("stp x29, x30, [sp, #-16]!", |a| a.stp_pre(FP, LR, SP, -16), [0xa9bf7bfd]),
        case!("ldp x29, x30, [sp], #16", |a| a.ldp_post(FP, LR, SP, 16), [0xa8c17bfd]),
        case!("stp x19, x20, [sp, #16]", |a| a.stp(19, 20, SP, 16), [0xa90153f3]),
        case!("ldp x19, x20, [sp, #-512]", |a| a.ldp(19, 20, SP, -512), [0xa96053f3]),
        case!("br x16", |a| a.br(IP0), [0xd61f0200]),
        case!("blr x16", |a| a.blr(IP0), [0xd63f0200]),
        case!("ret", |a| a.ret(), [0xd65f03c0]),
        case!("nop", |a| a.nop(), [0xd503201f]),
        case!("brk #0x3e8", |a| a.brk(1000), [0xd4207d00]),
        case!("movz x16, #0x5678; movk x16, #0x1234, lsl #16; blr x16", |a| a.call_abs(0x1234_5678), [0xd28acf10, 0xf2a24690, 0xd63f0200]),
        case!("b .+8; nop", |a| { let l = a.new_label(); a.b(l); a.nop(); a.bind(l); }, [0x14000002, 0xd503201f]),
        case!("nop; b .-4", |a| { let l = a.new_label(); a.bind(l); a.nop(); a.b(l); }, [0xd503201f, 0x17ffffff]),
        case!("bl .+4", |a| { let l = a.new_label(); a.bl(l); a.bind(l); }, [0x94000001]),
        case!("b.vs .+12; nop; nop", |a| { let l = a.new_label(); a.b_cond(Cond::Vs, l); a.nop(); a.nop(); a.bind(l); }, [0x54000066, 0xd503201f, 0xd503201f]),
        case!("nop; b.le .-4", |a| { let l = a.new_label(); a.bind(l); a.nop(); a.b_cond(Cond::Le, l); }, [0xd503201f, 0x54ffffed]),
        case!("cbz x3, .+8; nop", |a| { let l = a.new_label(); a.cbz(3, l); a.nop(); a.bind(l); }, [0xb4000043, 0xd503201f]),
        case!("nop; cbnz x4, .-4", |a| { let l = a.new_label(); a.bind(l); a.nop(); a.cbnz(4, l); }, [0xd503201f, 0xb5ffffe4]),
        case!("tbz x5, #63, .+8; nop", |a| { let l = a.new_label(); a.tbz(5, 63, l); a.nop(); a.bind(l); }, [0xb6f80045, 0xd503201f]),
        case!("nop; tbnz x6, #3, .-4", |a| { let l = a.new_label(); a.bind(l); a.nop(); a.tbnz(6, 3, l); }, [0xd503201f, 0x371fffe6]),
        case!("fadd d0, d1, d2", |a| a.fadd(0, 1, 2), [0x1e622820]),
        case!("fsub d0, d1, d2", |a| a.fsub(0, 1, 2), [0x1e623820]),
        case!("fmul d0, d1, d2", |a| a.fmul(0, 1, 2), [0x1e620820]),
        case!("fdiv d0, d1, d2", |a| a.fdiv(0, 1, 2), [0x1e621820]),
        case!("fmax d3, d4, d5", |a| a.fmax(3, 4, 5), [0x1e654883]),
        case!("fmin d3, d4, d5", |a| a.fmin(3, 4, 5), [0x1e655883]),
        case!("fsqrt d6, d7", |a| a.fsqrt(6, 7), [0x1e61c0e6]),
        case!("fabs d6, d7", |a| a.fabs(6, 7), [0x1e60c0e6]),
        case!("fneg d6, d7", |a| a.fneg(6, 7), [0x1e6140e6]),
        case!("frintm d8, d9", |a| a.frintm(8, 9), [0x1e654128]),
        case!("frintp d8, d9", |a| a.frintp(8, 9), [0x1e64c128]),
        case!("frintz d8, d9", |a| a.frintz(8, 9), [0x1e65c128]),
        case!("fcmp d1, d2", |a| a.fcmp(1, 2), [0x1e622020]),
        case!("fcmp d1, #0.0", |a| a.fcmp_zero(1), [0x1e602028]),
        case!("fmov d1, d2", |a| a.fmov(1, 2), [0x1e604041]),
        case!("fmov d1, x2", |a| a.fmov_from_x(1, 2), [0x9e670041]),
        case!("fmov x1, d2", |a| a.fmov_to_x(1, 2), [0x9e660041]),
        case!("scvtf d1, x2", |a| a.scvtf(1, 2), [0x9e620041]),
        case!("fcvtzs x1, d2", |a| a.fcvtzs(1, 2), [0x9e780041]),
        // A64ENC-END
    ]
}

#[test]
fn encodings_match_the_gnu_assembler() {
    let mut bad = Vec::new();
    for (text, f, words) in cases() {
        let mut a = A64Assembler::new();
        f(&mut a);
        let got = a.finish().expect("encodes");
        let want: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        if got != want {
            let got: Vec<u32> = got
                .chunks(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            bad.push(alloc::format!("{text}: got {got:08x?} want {words:08x?}"));
        }
    }
    assert!(bad.is_empty(), "mis-encoded:\n{}", bad.join("\n"));
}

#[test]
fn out_of_range_operands_refuse_the_code() {
    let mut a = A64Assembler::new();
    a.add_imm(0, 1, 0x1001);
    assert!(a.finish().is_none(), "add #0x1001 does not encode");
    let mut a = A64Assembler::new();
    a.stp_pre(FP, LR, SP, -520);
    assert!(a.finish().is_none(), "stp offset out of range");
    let mut a = A64Assembler::new();
    let l = a.new_label();
    a.b(l);
    assert!(a.finish().is_none(), "unbound label");
    // A `tbz` reaches ±32 KiB only.
    let mut a = A64Assembler::new();
    let l = a.new_label();
    a.tbz(0, 0, l);
    for _ in 0..8192 {
        a.nop();
    }
    a.bind(l);
    assert!(a.finish().is_none(), "tbz out of reach");
}
