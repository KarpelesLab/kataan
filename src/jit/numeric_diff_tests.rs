//! Randomized differential tests of the Int and Float tiers' native code
//! against their interpreter oracles ([`eval_reg`], [`eval_float`]), on
//! whichever backend the host runs (x86-64, or AArch64 under the A64
//! lowering). Straight-line programs over small register files, with
//! operands chosen to hit the edges: overflow and the ±2^53 range for the
//! Int tier (a result there must leave with the `i64::MAX` sentinel), and
//! NaN, ±0 and ±infinity for the Float tier.

use super::*;

/// A tiny deterministic generator (xorshift64*).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

const INT_EDGES: [i64; 14] = [
    0,
    1,
    -1,
    2,
    7,
    -13,
    31,
    32,
    i32::MAX as i64,
    i32::MIN as i64,
    SAFE_INT_MAX,
    -SAFE_INT_MAX,
    1 << 40,
    -(1 << 30),
];

/// The Int tier's semantics with its leaves: `None` where the native code
/// returns the deopt sentinel.
fn eval_reg_checked(ops: &[RegOp], n_regs: usize, args: &[i64]) -> Option<i64> {
    let mut regs = alloc::vec![0i64; n_regs];
    let in_range = |v: i64| (-SAFE_INT_MAX..=SAFE_INT_MAX).contains(&v);
    for op in ops {
        match *op {
            RegOp::Bin { dst, a, b, op } => {
                let (x, y) = (regs[a as usize], regs[b as usize]);
                let v = match op {
                    BinOp2::Add => x.checked_add(y)?,
                    BinOp2::Sub => x.checked_sub(y)?,
                    BinOp2::Mul => x.checked_mul(y)?,
                    BinOp2::And => x & y,
                    BinOp2::Or => x | y,
                    BinOp2::Xor => x ^ y,
                };
                in_range(v).then_some(())?;
                regs[dst as usize] = v;
            }
            RegOp::Neg { dst, a } => {
                let v = regs[a as usize].checked_neg()?;
                in_range(v).then_some(())?;
                regs[dst as usize] = v;
            }
            RegOp::Mod { dst, a, b } => {
                let (x, y) = (regs[a as usize], regs[b as usize]);
                if y == 0 || x < 0 {
                    return None;
                }
                regs[dst as usize] = x % y;
            }
            RegOp::Ret { src } => return Some(regs[src as usize]),
            // Everything else cannot leave: one step of the plain oracle,
            // run on the current registers (seeded as constants).
            _ => {
                let mut prog: Vec<RegOp> = (0..n_regs)
                    .map(|i| RegOp::Const {
                        dst: i as u8,
                        imm: regs[i],
                    })
                    .collect();
                prog.push(*op);
                let before = prog.len();
                regs = (0..n_regs)
                    .map(|i| {
                        prog.truncate(before);
                        prog.push(RegOp::Ret { src: i as u8 });
                        eval_reg(&prog, n_regs, args)
                    })
                    .collect();
            }
        }
    }
    None
}

fn random_reg_program(rng: &mut Rng, n_regs: u8, n_args: u8) -> Vec<RegOp> {
    let mut ops = Vec::new();
    for i in 0..n_args {
        ops.push(RegOp::Arg { dst: i, index: i });
    }
    for r in n_args..n_regs {
        let imm = INT_EDGES[rng.below(INT_EDGES.len() as u64) as usize];
        ops.push(RegOp::Const { dst: r, imm });
    }
    let n = 4 + rng.below(12);
    let reg = |rng: &mut Rng| rng.below(u64::from(n_regs)) as u8;
    for _ in 0..n {
        let (dst, a, b) = (reg(rng), reg(rng), reg(rng));
        let bin = [
            BinOp2::Add,
            BinOp2::Sub,
            BinOp2::Mul,
            BinOp2::And,
            BinOp2::Or,
            BinOp2::Xor,
        ][rng.below(6) as usize];
        let bit = [BinOp2::And, BinOp2::Or, BinOp2::Xor][rng.below(3) as usize];
        let sh = [ShiftOp::Shl, ShiftOp::Sar, ShiftOp::Shr][rng.below(3) as usize];
        ops.push(match rng.below(11) {
            0 | 1 => RegOp::Bin { dst, a, b, op: bin },
            2 => RegOp::Move { dst, src: a },
            3 => RegOp::Lt { dst, a, b },
            4 => RegOp::Eqz { dst, a },
            5 => RegOp::Eq { dst, a, b },
            6 => RegOp::Neg { dst, a },
            7 => RegOp::BitNot32 { dst, a },
            8 => RegOp::Mod { dst, a, b },
            9 => RegOp::Bit32 { dst, a, b, op: bit },
            _ => RegOp::Shift32 { dst, a, b, op: sh },
        });
    }
    ops.push(RegOp::Ret { src: reg(rng) });
    ops
}

#[test]
fn int_tier_matches_its_oracle() {
    if !available() {
        return;
    }
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    for case in 0..1500 {
        let n_args = rng.below(4) as u8;
        let n_regs = n_args + 1 + rng.below(5) as u8;
        let ops = random_reg_program(&mut rng, n_regs, n_args);
        let f = JitFunction::compile_reg(n_regs as usize, n_args as usize, &ops)
            .unwrap_or_else(|| panic!("case {case} compiles: {ops:?}"));
        for _ in 0..4 {
            let args: Vec<i64> = (0..n_args)
                .map(|_| INT_EDGES[rng.below(INT_EDGES.len() as u64) as usize])
                .collect();
            let want = eval_reg_checked(&ops, n_regs as usize, &args).unwrap_or(i64::MAX);
            assert_eq!(
                f.call_args(&args),
                want,
                "case {case} args {args:?}: {ops:?}"
            );
        }
    }
}

#[test]
fn int_tier_loops_and_branches() {
    if !available() {
        return;
    }
    // sum = 0; i = 0; while (i < n) { sum += i * i; i += 1 } return sum
    let ops = [
        RegOp::Arg { dst: 0, index: 0 },
        RegOp::Const { dst: 1, imm: 0 },
        RegOp::Const { dst: 2, imm: 0 },
        RegOp::Const { dst: 3, imm: 1 },
        RegOp::Lt { dst: 4, a: 2, b: 0 },
        RegOp::JumpIfFalse {
            cond: 4,
            target: 10,
        },
        RegOp::Bin {
            dst: 5,
            a: 2,
            b: 2,
            op: BinOp2::Mul,
        },
        RegOp::Bin {
            dst: 1,
            a: 1,
            b: 5,
            op: BinOp2::Add,
        },
        RegOp::Bin {
            dst: 2,
            a: 2,
            b: 3,
            op: BinOp2::Add,
        },
        RegOp::Jump { target: 4 },
        RegOp::Ret { src: 1 },
    ];
    let f = JitFunction::compile_reg(6, 1, &ops).expect("compiles");
    for n in [0i64, 1, 2, 10, 1000, -4] {
        assert_eq!(f.call_args(&[n]), eval_reg(&ops, 6, &[n]), "n = {n}");
    }
}

const FLOAT_EDGES: [f64; 16] = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    0.5,
    -2.5,
    3.75,
    1e300,
    -1e-300,
    f64::INFINITY,
    f64::NEG_INFINITY,
    f64::NAN,
    7.0,
    -7.0,
    0.1,
    123_456.789,
];

fn random_float_program(rng: &mut Rng, n_regs: u8, n_args: u8) -> Vec<FloatOp> {
    let mut ops = Vec::new();
    for i in 0..n_args {
        ops.push(FloatOp::Arg { dst: i, index: i });
    }
    for r in n_args..n_regs {
        let imm = FLOAT_EDGES[rng.below(FLOAT_EDGES.len() as u64) as usize];
        ops.push(FloatOp::Const { dst: r, imm });
    }
    let n = 3 + rng.below(10);
    let reg = |rng: &mut Rng| rng.below(u64::from(n_regs)) as u8;
    for _ in 0..n {
        let (dst, a, b) = (reg(rng), reg(rng), reg(rng));
        let op = [FBinOp::Add, FBinOp::Sub, FBinOp::Mul, FBinOp::Div][rng.below(4) as usize];
        ops.push(match rng.below(15) {
            0 | 1 => FloatOp::Bin { dst, a, b, op },
            2 => FloatOp::Mod { dst, a, b },
            3 => FloatOp::Move { dst, src: a },
            4 => FloatOp::Lt { dst, a, b },
            5 => FloatOp::Neg { dst, a },
            6 => FloatOp::Sqrt { dst, a },
            7 => FloatOp::Abs { dst, a },
            8 => FloatOp::Max { dst, a, b },
            9 => FloatOp::Min { dst, a, b },
            10 => FloatOp::Floor { dst, a },
            11 => FloatOp::Ceil { dst, a },
            12 => FloatOp::Trunc { dst, a },
            13 => FloatOp::Eqz { dst, a },
            _ => FloatOp::Eq { dst, a, b },
        });
    }
    ops.push(FloatOp::Ret { src: reg(rng) });
    ops
}

/// Same value: identical bits, or both NaN (whose payload no JS program
/// can observe).
fn same(x: f64, y: f64) -> bool {
    x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan())
}

#[test]
fn float_tier_matches_its_oracle() {
    if !available() || !has_sse41() {
        return;
    }
    let mut rng = Rng(0xD1B5_4A32_D192_ED03);
    for case in 0..1500 {
        let n_args = rng.below(5) as u8;
        let n_regs = n_args + 1 + rng.below(4) as u8;
        let ops = random_float_program(&mut rng, n_regs, n_args);
        let f = JitFunction::compile_float(n_regs as usize, n_args as usize, &ops)
            .unwrap_or_else(|| panic!("case {case} compiles: {ops:?}"));
        for _ in 0..4 {
            let args: Vec<f64> = (0..n_args)
                .map(|_| FLOAT_EDGES[rng.below(FLOAT_EDGES.len() as u64) as usize])
                .collect();
            let want = eval_float(&ops, n_regs as usize, &args);
            let got = f.call_args_f64(&args);
            assert!(
                same(got, want),
                "case {case} args {args:?}: got {got:?} want {want:?}: {ops:?}"
            );
        }
    }
}
