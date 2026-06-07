//! A WebAssembly execution engine (Phase H) — the peer engine that *runs*
//! `.wasm` modules, distinct from [`crate::wasm`] (which *compiles* JS to WASM).
//!
//! This first increment is the decode-and-execute core: a binary decoder for the
//! `type`/`function`/`export`/`code` sections, and a stack interpreter for the
//! numeric instruction set (`i32`/`i64` const + integer arithmetic, comparisons,
//! `local.get`/`local.set`, `call`, structured `block`/`loop`/`if` with
//! `br`/`br_if`/`return`). Linear memory, tables, globals, and the float set —
//! the "non-numeric" surface — build on this.
//!
//! Pure, safe `alloc`-only Rust; no foreign code.

use alloc::vec::Vec;

/// A decode or execution error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WasmRtError(pub &'static str);

/// A WebAssembly value (the numeric core; reference types land with the host
/// interop bridge).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Val {
    /// 32-bit integer (sign-agnostic; ops pick the interpretation).
    I32(i32),
    /// 64-bit integer.
    I64(i64),
    /// 32-bit float.
    F32(f32),
    /// 64-bit float.
    F64(f64),
}

impl Val {
    /// The value type of this runtime value.
    fn val_type(self) -> ValType {
        match self {
            Val::I32(_) => ValType::I32,
            Val::I64(_) => ValType::I64,
            Val::F32(_) => ValType::F32,
            Val::F64(_) => ValType::F64,
        }
    }
    fn as_i32(self) -> Result<i32, WasmRtError> {
        match self {
            Val::I32(v) => Ok(v),
            _ => Err(WasmRtError("type mismatch: expected i32")),
        }
    }
    fn as_i64(self) -> Result<i64, WasmRtError> {
        match self {
            Val::I64(v) => Ok(v),
            _ => Err(WasmRtError("type mismatch: expected i64")),
        }
    }
    fn as_f32(self) -> Result<f32, WasmRtError> {
        match self {
            Val::F32(v) => Ok(v),
            _ => Err(WasmRtError("type mismatch: expected f32")),
        }
    }
    fn as_f64(self) -> Result<f64, WasmRtError> {
        match self {
            Val::F64(v) => Ok(v),
            _ => Err(WasmRtError("type mismatch: expected f64")),
        }
    }

    /// Marshals this WASM value to a JS value (a `NanBox` number) — the result
    /// side of the JS↔WASM boundary.
    #[must_use]
    pub fn to_nanbox(self) -> crate::nanbox::NanBox {
        crate::nanbox::NanBox::number(match self {
            Val::I32(v) => f64::from(v),
            Val::I64(v) => v as f64,
            Val::F32(v) => f64::from(v),
            Val::F64(v) => v,
        })
    }

    /// Marshals a JS value (`NanBox`) into a WASM value of type `ty` — the
    /// argument side of the boundary. Numbers and booleans convert; other JS
    /// types are rejected.
    #[must_use]
    pub fn from_nanbox(v: crate::nanbox::NanBox, ty: ValType) -> Option<Val> {
        use crate::nanbox::Unpacked;
        let n = match v.unpack() {
            Unpacked::Number(n) => n,
            Unpacked::Bool(b) => f64::from(u8::from(b)),
            _ => return None,
        };
        Some(match ty {
            ValType::I32 => Val::I32(n as i32),
            ValType::I64 => Val::I64(n as i64),
            ValType::F32 => Val::F32(n as f32),
            ValType::F64 => Val::F64(n),
        })
    }
}

/// A value type in a function signature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValType {
    /// `i32`
    I32,
    /// `i64`
    I64,
    /// `f32`
    F32,
    /// `f64`
    F64,
}

/// A function type `(params) -> (results)`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FuncType {
    /// Parameter types.
    pub params: Vec<ValType>,
    /// Result types.
    pub results: Vec<ValType>,
}

/// A decoded function body: extra local declarations + the raw instruction
/// bytes (executed by [`Module::call`]).
#[derive(Debug, Clone, Default)]
struct FuncBody {
    /// Locals beyond the parameters, in order.
    locals: Vec<ValType>,
    /// The instruction byte range (the body's code, sans the leading locals).
    code: Vec<u8>,
}

/// A decoded WebAssembly module ready to execute.
#[derive(Debug, Default)]
pub struct Module {
    types: Vec<FuncType>,
    /// `func_index -> type_index`.
    func_types: Vec<u32>,
    bodies: Vec<FuncBody>,
    /// `name -> func_index`.
    exports: Vec<(alloc::string::String, u32)>,
    /// `name -> global_index` for exported globals.
    global_exports: Vec<(alloc::string::String, u32)>,
    /// Linear memory minimum size, in 64 KiB pages (`None` = no memory).
    mem_min_pages: Option<u32>,
    /// Linear memory maximum size in pages, if the limits declared one. A
    /// `memory.grow` past this returns -1.
    mem_max_pages: Option<u32>,
    /// Data segments in declaration order: `(offset, bytes)`. `Some(off)` is an
    /// *active* segment applied at instantiation; `None` is a *passive* segment,
    /// copied on demand by `memory.init`. Both are indexed by `memory.init` /
    /// `data.drop`.
    data: Vec<(Option<u32>, Vec<u8>)>,
    /// Module globals: `(initial value, mutable)`.
    globals: Vec<(Val, bool)>,
    /// The function table (`funcref`): each slot is a function index, or `None`
    /// for an uninitialized slot (traps on `call_indirect`).
    table: Vec<Option<u32>>,
    /// Imported functions, occupying the low function indices `0..n_imported_funcs`
    /// (module name, field name, type index). The host supplies their behavior.
    func_imports: Vec<(alloc::string::String, alloc::string::String, u32)>,
    /// Imported globals, occupying the low global indices before module-defined
    /// globals (module name, field name, type, mutable). The host supplies values.
    global_imports: Vec<(alloc::string::String, alloc::string::String, ValType, bool)>,
    /// `true` if the module imports its linear memory from the host (rather than
    /// defining its own); the host supplies the backing bytes at instantiation.
    mem_imported: bool,
    /// The `start` function index, run automatically at instantiation, if any.
    start: Option<u32>,
}

/// A host function backing an `import` — receives the WASM call's arguments and
/// returns its results. This is how a JS (or Rust) function is callable from
/// inside a WASM module.
pub type HostFunc = alloc::boxed::Box<dyn Fn(&[Val]) -> Result<Vec<Val>, WasmRtError>>;

/// The import dispatcher threaded through execution: given an imported function
/// index and its arguments, produce the results. Threading it as a parameter
/// (rather than storing a closure) lets the dispatcher borrow the host engine
/// mutably — so a JS function can be invoked for an import — without aliasing the
/// instance state.
pub type ImportHost<'a> = &'a mut dyn FnMut(usize, &[Val]) -> Result<Vec<Val>, WasmRtError>;

/// One linear-memory page, in bytes (WebAssembly fixes this at 64 KiB).
const PAGE_SIZE: usize = 65536;

/// The mutable instance state threaded through execution: linear memory and the
/// current global values.
struct Store {
    mem: Vec<u8>,
    globals: Vec<Val>,
    /// Per data segment: `true` once `data.drop` has run (its bytes are released,
    /// so a later `memory.init` of a non-zero length traps).
    dropped: Vec<bool>,
    /// The memory's declared maximum in pages (the limits' upper bound), if any.
    mem_max_pages: Option<u32>,
}

/// LEB128 cursor over a byte slice.
struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }
    fn byte(&mut self) -> Result<u8, WasmRtError> {
        let b = *self
            .bytes
            .get(self.pos)
            .ok_or(WasmRtError("unexpected end of input"))?;
        self.pos += 1;
        Ok(b)
    }
    fn bytes(&mut self, n: usize) -> Result<&'a [u8], WasmRtError> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|e| *e <= self.bytes.len())
            .ok_or(WasmRtError("unexpected end of input"))?;
        let s = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(s)
    }
    /// Unsigned LEB128.
    fn u32(&mut self) -> Result<u32, WasmRtError> {
        let mut result: u64 = 0;
        let mut shift = 0;
        loop {
            let b = self.byte()?;
            result |= u64::from(b & 0x7f) << shift;
            if b & 0x80 == 0 {
                break;
            }
            shift += 7;
            if shift >= 64 {
                return Err(WasmRtError("LEB128 overflow"));
            }
        }
        u32::try_from(result).map_err(|_| WasmRtError("u32 out of range"))
    }
    /// Signed LEB128 (32-bit).
    fn i32(&mut self) -> Result<i32, WasmRtError> {
        Ok(self.i64()? as i32)
    }
    /// Signed LEB128 (64-bit).
    fn i64(&mut self) -> Result<i64, WasmRtError> {
        let mut result: i64 = 0;
        let mut shift = 0;
        loop {
            let b = self.byte()?;
            result |= i64::from(b & 0x7f) << shift;
            shift += 7;
            if b & 0x80 == 0 {
                if shift < 64 && b & 0x40 != 0 {
                    result |= -1i64 << shift;
                }
                break;
            }
            if shift >= 64 {
                return Err(WasmRtError("LEB128 overflow"));
            }
        }
        Ok(result)
    }
    fn done(&self) -> bool {
        self.pos >= self.bytes.len()
    }
    /// Consumes a `0xfc`-prefixed opcode's sub-opcode and its immediates (so a
    /// body scan steps over `memory.copy`/`fill`/`init`/`data.drop`/`trunc_sat`).
    fn skip_fc(&mut self) -> Result<(), WasmRtError> {
        match self.u32()? {
            0x08 => {
                self.u32()?; // memory.init: data index …
                self.byte()?; // … + reserved memory index
            }
            0x09 => {
                self.u32()?; // data.drop: data index
            }
            0x0a => {
                self.byte()?; // memory.copy: two reserved memory indices
                self.byte()?;
            }
            0x0b => {
                self.byte()?; // memory.fill: one reserved memory index
            }
            _ => {} // trunc_sat (0x00..=0x07): no further immediate
        }
        Ok(())
    }
}

fn val_type(b: u8) -> Result<ValType, WasmRtError> {
    match b {
        0x7f => Ok(ValType::I32),
        0x7e => Ok(ValType::I64),
        0x7d => Ok(ValType::F32),
        0x7c => Ok(ValType::F64),
        _ => Err(WasmRtError("unsupported value type")),
    }
}

/// `f32::trunc` for `no_std`.
fn f32_trunc(x: f32) -> f32 {
    let bits = x.to_bits();
    let exp = ((bits >> 23) & 0xff) as i64 - 127;
    if exp < 0 {
        f32::from_bits(bits & (1 << 31))
    } else if exp >= 23 {
        x
    } else {
        f32::from_bits(bits & !((1u32 << (23 - exp)) - 1))
    }
}
/// Round half-to-even, with the sign of zero preserved (WebAssembly `nearest`).
fn f64_nearest(x: f64) -> f64 {
    if x.is_nan() || x.is_infinite() || x == 0.0 {
        return x;
    }
    let t = f64_trunc(x);
    let diff = f64_abs(x - t);
    let away = if x > 0.0 { 1.0 } else { -1.0 };
    let r = if diff < 0.5 {
        t
    } else if diff > 0.5 {
        t + away
    } else if (t as i64) % 2 == 0 {
        t // already even
    } else {
        t + away
    };
    if r == 0.0 && x < 0.0 { -0.0 } else { r }
}
fn f32_nearest(x: f32) -> f32 {
    f64_nearest(f64::from(x)) as f32
}
fn f64_floor(x: f64) -> f64 {
    let t = f64_trunc(x);
    if t > x { t - 1.0 } else { t }
}
fn f64_ceil(x: f64) -> f64 {
    let t = f64_trunc(x);
    if t < x { t + 1.0 } else { t }
}
fn f32_floor(x: f32) -> f32 {
    let t = f32_trunc(x);
    if t > x { t - 1.0 } else { t }
}
fn f32_ceil(x: f32) -> f32 {
    let t = f32_trunc(x);
    if t < x { t + 1.0 } else { t }
}
fn f64_copysign(a: f64, b: f64) -> f64 {
    f64::from_bits((a.to_bits() & 0x7fff_ffff_ffff_ffff) | (b.to_bits() & (1 << 63)))
}
fn f32_copysign(a: f32, b: f32) -> f32 {
    f32::from_bits((a.to_bits() & 0x7fff_ffff) | (b.to_bits() & (1 << 31)))
}

/// `i32.trunc_f64_s` and friends: truncate `a` toward zero, trapping on NaN/∞ or
/// when the result is outside the target integer range (per the spec — Rust's
/// `as` saturates, which is *not* the WebAssembly behavior).
fn trunc_i32_s(a: f64) -> Result<i32, WasmRtError> {
    let t = f64_trunc(a);
    if a.is_nan() || !(-2_147_483_648.0..=2_147_483_647.0).contains(&t) {
        return Err(WasmRtError("integer overflow"));
    }
    Ok(t as i32)
}
fn trunc_i32_u(a: f64) -> Result<i32, WasmRtError> {
    let t = f64_trunc(a);
    if a.is_nan() || !(0.0..=4_294_967_295.0).contains(&t) {
        return Err(WasmRtError("integer overflow"));
    }
    Ok(t as u32 as i32)
}
fn trunc_i64_s(a: f64) -> Result<i64, WasmRtError> {
    let t = f64_trunc(a);
    // 2^63 is representable; 2^63-1 is not, so the upper bound is exclusive 2^63.
    if a.is_nan() || !(-9_223_372_036_854_775_808.0..9_223_372_036_854_775_808.0).contains(&t) {
        return Err(WasmRtError("integer overflow"));
    }
    Ok(t as i64)
}
fn trunc_i64_u(a: f64) -> Result<i64, WasmRtError> {
    let t = f64_trunc(a);
    if a.is_nan() || !(0.0..18_446_744_073_709_551_616.0).contains(&t) {
        return Err(WasmRtError("integer overflow"));
    }
    Ok(t as u64 as i64)
}

/// A structured-control frame for operand-stack type validation: the block's
/// input/output value types, the value-stack height at entry, and whether the
/// block is currently in an `unreachable` (polymorphic-stack) state.
struct Ctrl {
    is_loop: bool,
    ins: Vec<ValType>,
    outs: Vec<ValType>,
    height: usize,
    unreachable: bool,
}

/// The WebAssembly operand-stack type checker: a value-type stack (where `None`
/// is the polymorphic "unknown" pushed after `unreachable`) plus a control stack.
/// Implements the standard validation algorithm — branch-target arities, block
/// in/out matching, and unreachable polymorphism.
struct TypeChecker {
    vals: Vec<Option<ValType>>,
    ctrls: Vec<Ctrl>,
}

impl TypeChecker {
    fn new() -> Self {
        Self {
            vals: Vec::new(),
            ctrls: Vec::new(),
        }
    }
    fn push(&mut self, t: ValType) {
        self.vals.push(Some(t));
    }
    /// Pops one operand, honoring the current block's floor (below which a pop is
    /// an underflow if reachable, or the polymorphic `Unknown` if unreachable).
    fn pop(&mut self) -> Result<Option<ValType>, WasmRtError> {
        let frame = self
            .ctrls
            .last()
            .ok_or(WasmRtError("control stack empty"))?;
        if self.vals.len() == frame.height {
            if frame.unreachable {
                return Ok(None);
            }
            return Err(WasmRtError("operand stack underflow"));
        }
        Ok(self.vals.pop().unwrap())
    }
    /// Pops one operand requiring type `t` (an `Unknown` matches anything).
    fn pop_expect(&mut self, t: ValType) -> Result<(), WasmRtError> {
        match self.pop()? {
            Some(g) if g != t => Err(WasmRtError("operand type mismatch")),
            _ => Ok(()),
        }
    }
    fn pop_many(&mut self, ts: &[ValType]) -> Result<(), WasmRtError> {
        for t in ts.iter().rev() {
            self.pop_expect(*t)?;
        }
        Ok(())
    }
    fn push_many(&mut self, ts: &[ValType]) {
        for t in ts {
            self.push(*t);
        }
    }
    fn push_ctrl(&mut self, is_loop: bool, ins: Vec<ValType>, outs: Vec<ValType>) {
        let frame = Ctrl {
            is_loop,
            ins: ins.clone(),
            outs,
            height: self.vals.len(),
            unreachable: false,
        };
        self.ctrls.push(frame);
        self.push_many(&ins);
    }
    /// Ends the current block: its output types must be exactly on top; returns
    /// them so the caller can re-push them onto the enclosing block.
    fn pop_ctrl(&mut self) -> Result<Vec<ValType>, WasmRtError> {
        let outs = self
            .ctrls
            .last()
            .ok_or(WasmRtError("control stack empty"))?
            .outs
            .clone();
        self.pop_many(&outs)?;
        let frame = self.ctrls.pop().unwrap();
        if self.vals.len() != frame.height {
            return Err(WasmRtError("block leaves extra operands"));
        }
        Ok(outs)
    }
    /// The types a branch to label `n` (0 = innermost) transfers: a loop's inputs,
    /// any other block's outputs.
    fn label_types(&self, n: usize) -> Option<Vec<ValType>> {
        let len = self.ctrls.len();
        if n >= len {
            return None;
        }
        let frame = &self.ctrls[len - 1 - n];
        Some(if frame.is_loop {
            frame.ins.clone()
        } else {
            frame.outs.clone()
        })
    }
    /// Enters the `unreachable` state for the current block: the stack is reset to
    /// the block floor and subsequent pops yield the polymorphic `Unknown`.
    fn set_unreachable(&mut self) -> Result<(), WasmRtError> {
        let frame = self
            .ctrls
            .last_mut()
            .ok_or(WasmRtError("control stack empty"))?;
        self.vals.truncate(frame.height);
        frame.unreachable = true;
        Ok(())
    }
}

/// The zero value for a value type (local/global default).
fn zero_val(t: ValType) -> Val {
    match t {
        ValType::I32 => Val::I32(0),
        ValType::I64 => Val::I64(0),
        ValType::F32 => Val::F32(0.0),
        ValType::F64 => Val::F64(0.0),
    }
}

// Pure-`core` float helpers — `f64::abs`/`min`/`max`/`sqrt` live in `std`, but
// the engine is `alloc`-only, so they are reimplemented here over `core`.

fn f64_abs(x: f64) -> f64 {
    f64::from_bits(x.to_bits() & 0x7fff_ffff_ffff_ffff)
}

/// `f64::trunc` for `no_std` (round toward zero, by masking the fraction bits).
fn f64_trunc(x: f64) -> f64 {
    let bits = x.to_bits();
    let exp = ((bits >> 52) & 0x7ff) as i64 - 1023;
    if exp < 0 {
        // |x| < 1 → truncates to ±0 (keep the sign bit).
        f64::from_bits(bits & (1 << 63))
    } else if exp >= 52 {
        x // already an integer (or NaN/∞)
    } else {
        f64::from_bits(bits & !((1u64 << (52 - exp)) - 1))
    }
}
fn f32_abs(x: f32) -> f32 {
    f32::from_bits(x.to_bits() & 0x7fff_ffff)
}

/// WebAssembly `min`: NaN-propagating, with `-0 < +0`.
fn f64_min(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a == b {
        // ±0: the negative zero is the minimum.
        if a.is_sign_negative() { a } else { b }
    } else if a < b {
        a
    } else {
        b
    }
}
fn f64_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a == b {
        if a.is_sign_positive() { a } else { b }
    } else if a > b {
        a
    } else {
        b
    }
}
fn f32_min(a: f32, b: f32) -> f32 {
    f64_min(f64::from(a), f64::from(b)) as f32
}
fn f32_max(a: f32, b: f32) -> f32 {
    f64_max(f64::from(a), f64::from(b)) as f32
}

/// `sqrt` via Newton–Raphson with a bit-hack initial guess (no `std`/libm).
fn f64_sqrt(x: f64) -> f64 {
    if x.is_nan() || x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 || x.is_infinite() {
        return x; // sqrt(±0) = ±0, sqrt(+inf) = +inf
    }
    // Initial guess: halve the biased exponent (the classic bit approximation).
    let mut g = f64::from_bits((x.to_bits() >> 1) + (0x1ff8_0000_0000_0000));
    // A handful of Newton iterations converge to full f64 precision.
    for _ in 0..6 {
        g = 0.5 * (g + x / g);
    }
    g
}
fn f32_sqrt(x: f32) -> f32 {
    f64_sqrt(f64::from(x)) as f32
}

/// Reads a constant `i32.const N; end` initializer expression (data/element
/// segment offsets and `i32` globals).
fn read_const_i32_expr(s: &mut Reader) -> Result<i32, WasmRtError> {
    if s.byte()? != 0x41 {
        return Err(WasmRtError("expected i32.const in const expr"));
    }
    let v = s.i32()?;
    if s.byte()? != 0x0b {
        return Err(WasmRtError("expected end of const expr"));
    }
    Ok(v)
}

impl Module {
    /// Decodes a WebAssembly binary module.
    ///
    /// # Errors
    /// Returns `WasmRtError` on a bad magic/version or a malformed section.
    pub fn decode(bytes: &[u8]) -> Result<Module, WasmRtError> {
        let mut r = Reader::new(bytes);
        if r.bytes(4)? != b"\0asm" {
            return Err(WasmRtError("bad magic"));
        }
        if r.bytes(4)? != [0x01, 0, 0, 0] {
            return Err(WasmRtError("unsupported version"));
        }
        let mut m = Module::default();
        while !r.done() {
            let id = r.byte()?;
            let size = r.u32()? as usize;
            let body = r.bytes(size)?;
            let mut s = Reader::new(body);
            match id {
                0 => {} // custom section: ignore
                1 => Self::decode_types(&mut s, &mut m)?,
                2 => Self::decode_imports(&mut s, &mut m)?,
                3 => Self::decode_functions(&mut s, &mut m)?,
                4 => Self::decode_table(&mut s, &mut m)?,
                5 => Self::decode_memory(&mut s, &mut m)?,
                6 => Self::decode_globals(&mut s, &mut m)?,
                7 => Self::decode_exports(&mut s, &mut m)?,
                8 => m.start = Some(s.u32()?),
                9 => Self::decode_elements(&mut s, &mut m)?,
                10 => Self::decode_code(&mut s, &mut m)?,
                11 => Self::decode_data(&mut s, &mut m)?,
                // Other sections (import/start) are skipped for now.
                _ => {}
            }
        }
        m.validate()?;
        Ok(m)
    }

    /// Structural validation run after decoding: rejects modules whose
    /// cross-references are out of range (the cheap, always-checkable part of
    /// WebAssembly validation — enough that `assert_invalid`-style malformed
    /// modules are refused rather than failing only at run time).
    fn validate(&self) -> Result<(), WasmRtError> {
        let n_funcs = self.func_types.len(); // imports + defined
        let n_defined = n_funcs.saturating_sub(self.func_imports.len());
        // Every defined function must have exactly one code body.
        if self.bodies.len() != n_defined {
            return Err(WasmRtError("function and code section counts differ"));
        }
        // Every function's declared type index must exist.
        if self
            .func_types
            .iter()
            .any(|&t| t as usize >= self.types.len())
        {
            return Err(WasmRtError("function references an invalid type index"));
        }
        // Exports and the start function must reference real functions.
        if self.exports.iter().any(|(_, i)| *i as usize >= n_funcs) {
            return Err(WasmRtError("export references an invalid function index"));
        }
        if self.start.is_some_and(|s| s as usize >= n_funcs) {
            return Err(WasmRtError("start references an invalid function index"));
        }
        // Body-level index validation: every local/global/call/type reference in
        // each defined function must be in range.
        let n_imp = self.func_imports.len();
        let n_globals = self.global_imports.len() + self.globals.len();
        for (i, body) in self.bodies.iter().enumerate() {
            let ty = self
                .func_types
                .get(n_imp + i)
                .and_then(|t| self.types.get(*t as usize))
                .ok_or(WasmRtError("function has no type"))?;
            let n_locals = ty.params.len() + body.locals.len();
            self.validate_body(&body.code, n_locals, n_globals, n_funcs)?;
            self.validate_types(ty, body)?;
        }
        Ok(())
    }

    /// The value type of global `idx` (imported globals first, then defined).
    fn global_type(&self, idx: u32) -> Option<ValType> {
        let i = idx as usize;
        if i < self.global_imports.len() {
            Some(self.global_imports[i].2)
        } else {
            self.globals
                .get(i - self.global_imports.len())
                .map(|(v, _)| v.val_type())
        }
    }

    /// Operand-stack **type** validation for a function body, control-flow aware:
    /// a [`TypeChecker`] simulates the value-type stack through numeric, local,
    /// global, memory, const, drop/select, and call instructions *and* the full
    /// structured control flow (block/loop/if/else/end, br/br_if/br_table/return,
    /// unreachable) — branch-target arities, block in/out matching, and the
    /// polymorphic stack after `unreachable`. Rejects type mismatches, stack
    /// underflow, and wrong block/function result types. An instruction whose
    /// stack effect isn't modeled (or a multi-value block type) makes validation
    /// bail conservatively (accept), so a valid module is never rejected.
    fn validate_types(&self, ty: &FuncType, body: &FuncBody) -> Result<(), WasmRtError> {
        use ValType::{F32, F64, I32, I64};
        let mut locals: Vec<ValType> = ty.params.clone();
        locals.extend_from_slice(&body.locals);
        // (pops top-to-bottom, push) for a fixed-signature instruction.
        type Effect = (&'static [ValType], Option<ValType>);
        let simple: fn(u8) -> Option<Effect> = |op| {
            Some(match op {
                0x45 => (&[I32], Some(I32)),             // i32.eqz
                0x46..=0x4f => (&[I32, I32], Some(I32)), // i32 compares
                0x67..=0x69 => (&[I32], Some(I32)),      // clz/ctz/popcnt
                0x6a..=0x78 => (&[I32, I32], Some(I32)), // i32 binary
                0x50 => (&[I64], Some(I32)),             // i64.eqz
                0x51..=0x5a => (&[I64, I64], Some(I32)), // i64 compares
                0x79..=0x7b => (&[I64], Some(I64)),      // i64 clz/ctz/popcnt
                0x7c..=0x8a => (&[I64, I64], Some(I64)), // i64 binary
                0x5b..=0x60 => (&[F32, F32], Some(I32)), // f32 compares
                0x61..=0x66 => (&[F64, F64], Some(I32)), // f64 compares
                0x8b..=0x91 => (&[F32], Some(F32)),      // f32 unary (abs/neg/…/sqrt)
                0x92..=0x98 => (&[F32, F32], Some(F32)), // f32 binary
                0x99..=0x9f => (&[F64], Some(F64)),      // f64 unary
                0xa0..=0xa6 => (&[F64, F64], Some(F64)), // f64 binary (+ copysign)
                0xa7 => (&[I64], Some(I32)),             // i32.wrap_i64
                0xaa | 0xab => (&[F64], Some(I32)),      // i32.trunc_f64
                0xac | 0xad => (&[I32], Some(I64)),      // i64.extend_i32
                0xb0 | 0xb1 => (&[F64], Some(I64)),      // i64.trunc_f64
                0xa8 | 0xa9 => (&[F32], Some(I32)),      // i32.trunc_f32
                0xae | 0xaf => (&[F32], Some(I64)),      // i64.trunc_f32
                0xb2 | 0xb3 => (&[I32], Some(F32)),      // f32.convert_i32
                0xb4 | 0xb5 => (&[I64], Some(F32)),      // f32.convert_i64
                0xb6 => (&[F64], Some(F32)),             // f32.demote_f64
                0xb7 | 0xb8 => (&[I32], Some(F64)),      // f64.convert_i32
                0xb9 | 0xba => (&[I64], Some(F64)),      // f64.convert_i64_s/_u
                0xbb => (&[F32], Some(F64)),             // f64.promote_f32
                0xbc => (&[F32], Some(I32)),             // i32.reinterpret_f32
                0xbd => (&[F64], Some(I64)),             // i64.reinterpret_f64
                0xc0 | 0xc1 => (&[I32], Some(I32)),      // i32.extend8/16_s
                0xc2..=0xc4 => (&[I64], Some(I64)),      // i64.extend8/16/32_s
                0xbe => (&[I32], Some(F32)),             // f32.reinterpret_i32
                0xbf => (&[I64], Some(F64)),             // f64.reinterpret_i64
                _ => return None,
            })
        };
        // Parse a block type: empty (0x40) or a single result; `None` signals a
        // multi-value (type-index) block, which we don't model — bail to accept.
        type BlockType = Option<(Vec<ValType>, Vec<ValType>)>;
        let block_type = |r: &mut Reader| -> Result<BlockType, WasmRtError> {
            let b = r.byte()?;
            if b == 0x40 {
                Ok(Some((Vec::new(), Vec::new())))
            } else if let Ok(t) = val_type(b) {
                Ok(Some((Vec::new(), alloc::vec![t])))
            } else {
                Ok(None)
            }
        };

        let mut tc = TypeChecker::new();
        // The function body is an implicit block producing the function's results.
        tc.push_ctrl(false, Vec::new(), ty.results.clone());
        let mut r = Reader::new(&body.code);
        while !r.done() {
            let op = r.byte()?;
            match op {
                0x00 => tc.set_unreachable()?, // unreachable
                0x01 => {}                     // nop
                0x02 | 0x03 => {
                    // block / loop
                    let Some((ins, outs)) = block_type(&mut r)? else {
                        return Ok(());
                    };
                    tc.pop_many(&ins)?;
                    tc.push_ctrl(op == 0x03, ins, outs);
                }
                0x04 => {
                    // if: pop the i32 condition, then open a (then) block.
                    let Some((ins, outs)) = block_type(&mut r)? else {
                        return Ok(());
                    };
                    tc.pop_expect(I32)?;
                    tc.pop_many(&ins)?;
                    tc.push_ctrl(false, ins, outs);
                }
                0x05 => {
                    // else: close the then-arm, reopen with the same in/out.
                    let frame = tc.ctrls.last().ok_or(WasmRtError("else without if"))?;
                    let (ins, outs) = (frame.ins.clone(), frame.outs.clone());
                    tc.pop_many(&outs)?;
                    if tc.vals.len() != tc.ctrls.last().unwrap().height {
                        return Err(WasmRtError("then-branch leaves extra operands"));
                    }
                    tc.ctrls.pop();
                    tc.push_ctrl(false, ins, outs);
                }
                0x0b => {
                    // end: close the block; push its results to the parent (or, for
                    // the function block, finish).
                    let outs = tc.pop_ctrl()?;
                    if tc.ctrls.is_empty() {
                        return Ok(());
                    }
                    tc.push_many(&outs);
                }
                0x0c => {
                    // br
                    let n = r.u32()? as usize;
                    let lt = tc
                        .label_types(n)
                        .ok_or(WasmRtError("branch label out of range"))?;
                    tc.pop_many(&lt)?;
                    tc.set_unreachable()?;
                }
                0x0d => {
                    // br_if
                    let n = r.u32()? as usize;
                    let lt = tc
                        .label_types(n)
                        .ok_or(WasmRtError("branch label out of range"))?;
                    tc.pop_expect(I32)?;
                    tc.pop_many(&lt)?;
                    tc.push_many(&lt);
                }
                0x0e => {
                    // br_table: all labels must share the default's arity.
                    let count = r.u32()?;
                    let mut labels = Vec::with_capacity(count as usize + 1);
                    for _ in 0..=count {
                        labels.push(r.u32()? as usize);
                    }
                    tc.pop_expect(I32)?;
                    let default = *labels.last().unwrap();
                    let dlt = tc
                        .label_types(default)
                        .ok_or(WasmRtError("br_table default out of range"))?;
                    for &n in &labels {
                        let lt = tc
                            .label_types(n)
                            .ok_or(WasmRtError("br_table label out of range"))?;
                        if lt.len() != dlt.len() {
                            return Err(WasmRtError("br_table label arity mismatch"));
                        }
                    }
                    tc.pop_many(&dlt)?;
                    tc.set_unreachable()?;
                }
                0x0f => {
                    // return
                    tc.pop_many(&ty.results.clone())?;
                    tc.set_unreachable()?;
                }
                0x1a => {
                    tc.pop()?; // drop (any operand)
                }
                0x1b => {
                    tc.pop_expect(I32)?; // select condition
                    let a = tc.pop()?;
                    let b = tc.pop()?;
                    if let (Some(x), Some(y)) = (a, b)
                        && x != y
                    {
                        return Err(WasmRtError("select arms differ in type"));
                    }
                    tc.vals.push(a.or(b));
                }
                0x20 => {
                    let i = r.u32()? as usize;
                    tc.push(*locals.get(i).ok_or(WasmRtError("bad local"))?);
                }
                0x21 => {
                    let i = r.u32()? as usize;
                    tc.pop_expect(*locals.get(i).ok_or(WasmRtError("bad local"))?)?;
                }
                0x22 => {
                    let i = r.u32()? as usize;
                    let t = *locals.get(i).ok_or(WasmRtError("bad local"))?;
                    tc.pop_expect(t)?;
                    tc.push(t);
                }
                0x23 => {
                    let t = self
                        .global_type(r.u32()?)
                        .ok_or(WasmRtError("bad global"))?;
                    tc.push(t);
                }
                0x24 => {
                    let t = self
                        .global_type(r.u32()?)
                        .ok_or(WasmRtError("bad global"))?;
                    tc.pop_expect(t)?;
                }
                0x41 => {
                    r.i32()?;
                    tc.push(I32);
                }
                0x42 => {
                    r.i64()?;
                    tc.push(I64);
                }
                0x43 => {
                    r.bytes(4)?;
                    tc.push(F32);
                }
                0x44 => {
                    r.bytes(8)?;
                    tc.push(F64);
                }
                0x10 => {
                    let cty = self
                        .func_type(r.u32()?)
                        .ok_or(WasmRtError("bad call target"))?
                        .clone();
                    tc.pop_many(&cty.params)?;
                    tc.push_many(&cty.results);
                }
                0x11 => {
                    let t = r.u32()?;
                    r.u32()?; // table index
                    let cty = self
                        .types
                        .get(t as usize)
                        .ok_or(WasmRtError("bad type"))?
                        .clone();
                    tc.pop_expect(I32)?; // the table index operand
                    tc.pop_many(&cty.params)?;
                    tc.push_many(&cty.results);
                }
                // Memory loads: pop the i32 address, push the loaded type.
                0x28..=0x35 => {
                    r.u32()?;
                    r.u32()?;
                    tc.pop_expect(I32)?;
                    tc.push(match op {
                        0x29 | 0x30..=0x35 => I64,
                        0x2a => F32,
                        0x2b => F64,
                        _ => I32,
                    });
                }
                // Memory stores: pop the value then the i32 address.
                0x36..=0x3e => {
                    r.u32()?;
                    r.u32()?;
                    let vt = match op {
                        0x37 | 0x3c..=0x3e => I64,
                        0x38 => F32,
                        0x39 => F64,
                        _ => I32,
                    };
                    tc.pop_expect(vt)?;
                    tc.pop_expect(I32)?;
                }
                0x3f => {
                    r.byte()?;
                    tc.push(I32); // memory.size
                }
                0x40 => {
                    r.byte()?;
                    tc.pop_expect(I32)?;
                    tc.push(I32); // memory.grow
                }
                0xfc => {
                    let sub = r.u32()?;
                    match sub {
                        // trunc_sat: [Fxx] → Ixx (saturating, never traps).
                        0x00 | 0x01 => {
                            tc.pop_expect(F32)?;
                            tc.push(I32);
                        }
                        0x02 | 0x03 => {
                            tc.pop_expect(F64)?;
                            tc.push(I32);
                        }
                        0x04 | 0x05 => {
                            tc.pop_expect(F32)?;
                            tc.push(I64);
                        }
                        0x06 | 0x07 => {
                            tc.pop_expect(F64)?;
                            tc.push(I64);
                        }
                        0x0a => {
                            r.byte()?; // memory.copy: dst, src, n
                            r.byte()?;
                            tc.pop_expect(I32)?;
                            tc.pop_expect(I32)?;
                            tc.pop_expect(I32)?;
                        }
                        0x0b => {
                            r.byte()?; // memory.fill: dst, val, n
                            tc.pop_expect(I32)?;
                            tc.pop_expect(I32)?;
                            tc.pop_expect(I32)?;
                        }
                        _ => return Ok(()), // other 0xfc ops: accept conservatively
                    }
                }
                _ => {
                    if let Some((pops, push)) = simple(op) {
                        tc.pop_many(pops)?;
                        if let Some(t) = push {
                            tc.push(t);
                        }
                    } else {
                        return Ok(()); // unmodeled opcode: accept conservatively
                    }
                }
            }
        }
        Ok(())
    }

    /// Scans one function body, checking every index immediate is in range.
    fn validate_body(
        &self,
        code: &[u8],
        n_locals: usize,
        n_globals: usize,
        n_funcs: usize,
    ) -> Result<(), WasmRtError> {
        let mut r = Reader::new(code);
        while !r.done() {
            match r.byte()? {
                0x20..=0x22 => {
                    if r.u32()? as usize >= n_locals {
                        return Err(WasmRtError("local index out of range"));
                    }
                }
                0x23 | 0x24 => {
                    if r.u32()? as usize >= n_globals {
                        return Err(WasmRtError("global index out of range"));
                    }
                }
                0x10 => {
                    if r.u32()? as usize >= n_funcs {
                        return Err(WasmRtError("call target out of range"));
                    }
                }
                0x11 => {
                    let t = r.u32()?;
                    r.u32()?; // table index
                    if t as usize >= self.types.len() {
                        return Err(WasmRtError("call_indirect type out of range"));
                    }
                }
                // Skip the immediates of the remaining instructions (mirrors the
                // control-flow scanners) so indices aren't misread.
                0x02..=0x04 => {
                    r.byte()?;
                }
                0x0c | 0x0d => {
                    r.u32()?;
                }
                0x0e => {
                    let c = r.u32()?;
                    for _ in 0..=c {
                        r.u32()?;
                    }
                }
                0x28..=0x3e => {
                    r.u32()?;
                    r.u32()?;
                }
                0x3f | 0x40 => {
                    r.byte()?;
                }
                0x41 => {
                    r.i32()?;
                }
                0x42 => {
                    r.i64()?;
                }
                0x43 => {
                    r.bytes(4)?;
                }
                0x44 => {
                    r.bytes(8)?;
                }
                0xfc => {
                    r.skip_fc()?;
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn decode_types(s: &mut Reader, m: &mut Module) -> Result<(), WasmRtError> {
        let count = s.u32()?;
        for _ in 0..count {
            if s.byte()? != 0x60 {
                return Err(WasmRtError("expected functype"));
            }
            let np = s.u32()?;
            let mut params = Vec::with_capacity(np as usize);
            for _ in 0..np {
                params.push(val_type(s.byte()?)?);
            }
            let nr = s.u32()?;
            let mut results = Vec::with_capacity(nr as usize);
            for _ in 0..nr {
                results.push(val_type(s.byte()?)?);
            }
            m.types.push(FuncType { params, results });
        }
        Ok(())
    }

    fn decode_imports(s: &mut Reader, m: &mut Module) -> Result<(), WasmRtError> {
        let count = s.u32()?;
        for _ in 0..count {
            let mlen = s.u32()? as usize;
            let module = alloc::string::String::from_utf8(s.bytes(mlen)?.to_vec())
                .map_err(|_| WasmRtError("bad import module name"))?;
            let flen = s.u32()? as usize;
            let field = alloc::string::String::from_utf8(s.bytes(flen)?.to_vec())
                .map_err(|_| WasmRtError("bad import field name"))?;
            let kind = s.byte()?;
            match kind {
                0x00 => {
                    // A function import: its type index. It occupies the next
                    // function index, before any module-defined function, so push
                    // it onto `func_types` now (the import section precedes the
                    // function section).
                    let type_idx = s.u32()?;
                    m.func_types.push(type_idx);
                    m.func_imports.push((module, field, type_idx));
                }
                // table (0x01) / memory (0x02) / global (0x03) imports carry a
                // descriptor we skip for now (only function imports are wired).
                0x01 => {
                    s.byte()?; // elemtype
                    let flag = s.byte()?;
                    s.u32()?;
                    if flag == 1 {
                        s.u32()?;
                    }
                }
                0x02 => {
                    // An imported linear memory: the host supplies the bytes.
                    let flag = s.byte()?;
                    let min = s.u32()?;
                    if flag == 1 {
                        s.u32()?; // max
                    }
                    m.mem_imported = true;
                    m.mem_min_pages = Some(min);
                }
                0x03 => {
                    // A global import: it occupies the next global index, before
                    // any module-defined global.
                    let vt = val_type(s.byte()?)?;
                    let mutable = s.byte()? == 1;
                    m.global_imports.push((module, field, vt, mutable));
                }
                _ => return Err(WasmRtError("unsupported import kind")),
            }
        }
        Ok(())
    }

    fn decode_functions(s: &mut Reader, m: &mut Module) -> Result<(), WasmRtError> {
        let count = s.u32()?;
        for _ in 0..count {
            m.func_types.push(s.u32()?);
        }
        Ok(())
    }

    /// The number of imported functions (which occupy the low function indices).
    fn n_imported_funcs(&self) -> usize {
        self.func_imports.len()
    }

    /// The number of imported globals (which occupy the low global indices).
    fn n_imported_globals(&self) -> usize {
        self.global_imports.len()
    }

    /// Whether global `index` (imports + defined) is mutable.
    fn global_mutable(&self, index: usize) -> Option<bool> {
        let n = self.n_imported_globals();
        if index < n {
            self.global_imports.get(index).map(|(_, _, _, m)| *m)
        } else {
            self.globals.get(index - n).map(|(_, m)| *m)
        }
    }

    /// The signature of function `index` (imports included).
    #[must_use]
    pub fn func_type(&self, index: u32) -> Option<&FuncType> {
        self.func_types
            .get(index as usize)
            .and_then(|t| self.types.get(*t as usize))
    }

    /// The `(module, field)` names of the module's function imports, in the order
    /// the host must supply [`HostFunc`]s to [`Instance::with_imports`].
    #[must_use]
    pub fn import_names(&self) -> Vec<(&str, &str)> {
        self.func_imports
            .iter()
            .map(|(m, f, _)| (m.as_str(), f.as_str()))
            .collect()
    }

    /// The `(module, field, type)` of each imported global, in declaration order —
    /// the order a host supplies values to
    /// [`Instance::with_host_imports_and_globals`].
    #[must_use]
    pub fn global_import_names(&self) -> Vec<(&str, &str, ValType)> {
        self.global_imports
            .iter()
            .map(|(m, f, t, _)| (m.as_str(), f.as_str(), *t))
            .collect()
    }

    fn decode_memory(s: &mut Reader, m: &mut Module) -> Result<(), WasmRtError> {
        let count = s.u32()?;
        if count > 0 {
            // limits: flag (0 = min only, 1 = min+max), then min (and max).
            let flag = s.byte()?;
            let min = s.u32()?;
            if flag == 1 {
                m.mem_max_pages = Some(s.u32()?);
            }
            m.mem_min_pages = Some(min);
        }
        Ok(())
    }

    fn decode_table(s: &mut Reader, m: &mut Module) -> Result<(), WasmRtError> {
        let count = s.u32()?;
        for _ in 0..count {
            let _elemtype = s.byte()?; // 0x70 funcref (or 0x6f externref)
            let flag = s.byte()?;
            let min = s.u32()?;
            if flag == 1 {
                let _max = s.u32()?;
            }
            // One table per module (multi-table is post-MVP); size to `min`.
            m.table = alloc::vec![None; min as usize];
        }
        Ok(())
    }

    fn decode_elements(s: &mut Reader, m: &mut Module) -> Result<(), WasmRtError> {
        let count = s.u32()?;
        for _ in 0..count {
            let mode = s.u32()?;
            // Mode 0 = active, table 0, with an `i32.const off; end` offset.
            if mode != 0 {
                return Err(WasmRtError("unsupported element segment mode"));
            }
            let off = read_const_i32_expr(s)? as usize;
            let n = s.u32()? as usize;
            for i in 0..n {
                let func = s.u32()?;
                let slot = off + i;
                if slot >= m.table.len() {
                    return Err(WasmRtError("element segment out of bounds"));
                }
                m.table[slot] = Some(func);
            }
        }
        Ok(())
    }

    fn decode_globals(s: &mut Reader, m: &mut Module) -> Result<(), WasmRtError> {
        let count = s.u32()?;
        for _ in 0..count {
            let vt = val_type(s.byte()?)?;
            let mutable = s.byte()? == 1;
            // The init expr — a single typed const followed by `end`.
            let init = match vt {
                ValType::I32 => Val::I32(read_const_i32_expr(s)?),
                ValType::I64 => {
                    if s.byte()? != 0x42 {
                        return Err(WasmRtError("expected i64.const global init"));
                    }
                    let v = s.i64()?;
                    if s.byte()? != 0x0b {
                        return Err(WasmRtError("expected end of global init"));
                    }
                    Val::I64(v)
                }
                ValType::F32 => {
                    if s.byte()? != 0x43 {
                        return Err(WasmRtError("expected f32.const global init"));
                    }
                    let v = f32::from_le_bytes(s.bytes(4)?.try_into().unwrap());
                    if s.byte()? != 0x0b {
                        return Err(WasmRtError("expected end of global init"));
                    }
                    Val::F32(v)
                }
                ValType::F64 => {
                    if s.byte()? != 0x44 {
                        return Err(WasmRtError("expected f64.const global init"));
                    }
                    let v = f64::from_le_bytes(s.bytes(8)?.try_into().unwrap());
                    if s.byte()? != 0x0b {
                        return Err(WasmRtError("expected end of global init"));
                    }
                    Val::F64(v)
                }
            };
            m.globals.push((init, mutable));
        }
        Ok(())
    }

    fn decode_data(s: &mut Reader, m: &mut Module) -> Result<(), WasmRtError> {
        let count = s.u32()?;
        for _ in 0..count {
            let mode = s.u32()?;
            // 0 = active (memory 0, offset expr); 1 = passive (bytes only);
            // 2 = active with an explicit memory index (which must be 0 here).
            let off = match mode {
                0 => Some(read_const_i32_expr(s)? as u32),
                1 => None,
                2 => {
                    if s.u32()? != 0 {
                        return Err(WasmRtError("data segment: memory index must be 0"));
                    }
                    Some(read_const_i32_expr(s)? as u32)
                }
                _ => return Err(WasmRtError("unsupported data segment mode")),
            };
            let len = s.u32()? as usize;
            let bytes = s.bytes(len)?.to_vec();
            m.data.push((off, bytes));
        }
        Ok(())
    }

    fn decode_exports(s: &mut Reader, m: &mut Module) -> Result<(), WasmRtError> {
        let count = s.u32()?;
        for _ in 0..count {
            let nlen = s.u32()? as usize;
            let name = alloc::string::String::from_utf8(s.bytes(nlen)?.to_vec())
                .map_err(|_| WasmRtError("bad export name"))?;
            let kind = s.byte()?;
            let index = s.u32()?;
            match kind {
                0x00 => m.exports.push((name, index)), // function export
                0x03 => m.global_exports.push((name, index)), // global export
                _ => {} // memory/table exports: not surfaced here yet
            }
        }
        Ok(())
    }

    fn decode_code(s: &mut Reader, m: &mut Module) -> Result<(), WasmRtError> {
        let count = s.u32()?;
        for _ in 0..count {
            let body_size = s.u32()? as usize;
            let body = s.bytes(body_size)?;
            let mut b = Reader::new(body);
            let nlocal_runs = b.u32()?;
            let mut locals = Vec::new();
            for _ in 0..nlocal_runs {
                let n = b.u32()?;
                let t = val_type(b.byte()?)?;
                for _ in 0..n {
                    locals.push(t);
                }
            }
            m.bodies.push(FuncBody {
                locals,
                code: body[b.pos..].to_vec(),
            });
        }
        Ok(())
    }

    /// The names of all exported functions, in declaration order.
    #[must_use]
    pub fn export_names(&self) -> Vec<&str> {
        self.exports.iter().map(|(n, _)| n.as_str()).collect()
    }

    /// The `(name, global index)` of each exported global, in declaration order.
    #[must_use]
    pub fn global_exports(&self) -> Vec<(&str, u32)> {
        self.global_exports
            .iter()
            .map(|(n, i)| (n.as_str(), *i))
            .collect()
    }

    /// Whether global `index` (imports + defined) is declared mutable.
    #[must_use]
    pub fn global_is_mutable(&self, index: u32) -> bool {
        self.global_mutable(index as usize).unwrap_or(false)
    }

    /// The index of an exported function, or `None`.
    #[must_use]
    pub fn export(&self, name: &str) -> Option<u32> {
        self.exports
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, i)| *i)
    }

    /// Allocates and initializes this module's instance state (linear memory with
    /// data segments applied, and the initial global values).
    fn new_store(&self) -> Result<Store, WasmRtError> {
        let pages = self.mem_min_pages.unwrap_or(0) as usize;
        let mut mem = alloc::vec![0u8; pages * PAGE_SIZE];
        for (off, bytes) in &self.data {
            // Only active segments are applied here; passive ones wait for
            // `memory.init`.
            let Some(off) = off else { continue };
            let start = *off as usize;
            let end = start
                .checked_add(bytes.len())
                .filter(|e| *e <= mem.len())
                .ok_or(WasmRtError("data segment out of bounds"))?;
            mem[start..end].copy_from_slice(bytes);
        }
        // Global index space: imported globals (zero-initialized; the host may
        // override them) first, then module-defined globals.
        let mut globals: Vec<Val> = self
            .global_imports
            .iter()
            .map(|(_, _, t, _)| zero_val(*t))
            .collect();
        globals.extend(self.globals.iter().map(|(v, _)| *v));
        // Active segments are applied above and then immediately dropped (per spec);
        // passive segments stay live until an explicit `data.drop`.
        let dropped: Vec<bool> = self.data.iter().map(|(off, _)| off.is_some()).collect();
        Ok(Store {
            mem,
            globals,
            dropped,
            mem_max_pages: self.mem_max_pages,
        })
    }

    /// Calls function `index` with `args` over a fresh instance, dispatching any
    /// imported function through `host`.
    ///
    /// # Errors
    /// Returns `WasmRtError` on a type mismatch, a missing function, or an
    /// unsupported instruction.
    pub fn call(&self, index: u32, args: &[Val]) -> Result<Vec<Val>, WasmRtError> {
        let mut store = self.new_store()?;
        // No host: an imported call (which an import-free module has none of) errors.
        let mut none = |_i: usize, _a: &[Val]| Err(WasmRtError("missing host function for import"));
        self.call_with_store(index, args, &mut store, &mut none)
    }

    /// Like [`call`](Self::call) but over a caller-provided instance store and
    /// import dispatcher (both shared across nested `call`s).
    fn call_with_store(
        &self,
        index: u32,
        args: &[Val],
        store: &mut Store,
        host: ImportHost,
    ) -> Result<Vec<Val>, WasmRtError> {
        let ty = self
            .func_types
            .get(index as usize)
            .and_then(|t| self.types.get(*t as usize))
            .ok_or(WasmRtError("no such function"))?;
        if args.len() != ty.params.len() {
            return Err(WasmRtError("argument count mismatch"));
        }
        // A function import dispatches through the host callback.
        let n_imp = self.n_imported_funcs();
        if (index as usize) < n_imp {
            return host(index as usize, args);
        }
        let body = self
            .bodies
            .get(index as usize - n_imp)
            .ok_or(WasmRtError("no such function body"))?;
        // Locals = parameters, then zero-initialized declared locals.
        let mut locals: Vec<Val> = args.to_vec();
        for lt in &body.locals {
            locals.push(zero_val(*lt));
        }
        let mut stack: Vec<Val> = Vec::new();
        self.exec(&body.code, &mut locals, &mut stack, store, host)?;
        // Take the result count off the top of the stack.
        let n = ty.results.len();
        if stack.len() < n {
            return Err(WasmRtError("missing results"));
        }
        Ok(stack.split_off(stack.len() - n))
    }

    /// Executes an instruction stream, mutating `locals`/`stack`. Returns `Ok`
    /// on normal completion (`end`/`return`).
    #[allow(clippy::too_many_lines)]
    fn exec(
        &self,
        code: &[u8],
        locals: &mut [Val],
        stack: &mut Vec<Val>,
        store: &mut Store,
        host: ImportHost,
    ) -> Result<Flow, WasmRtError> {
        let mut r = Reader::new(code);
        macro_rules! pop {
            () => {
                stack.pop().ok_or(WasmRtError("stack underflow"))?
            };
        }
        // Computes a bounds-checked effective address for a memory access of
        // `size` bytes: `popped_base (u32) + offset`.
        macro_rules! mem_addr {
            ($r:expr, $size:expr) => {{
                let _align = $r.u32()?;
                let offset = $r.u32()?;
                let base = pop!().as_i32()? as u32 as u64;
                let addr = base + u64::from(offset);
                let end = addr + $size as u64;
                if end > store.mem.len() as u64 {
                    return Err(WasmRtError("out of bounds memory access"));
                }
                addr as usize
            }};
        }
        macro_rules! bin_i32 {
            ($f:expr) => {{
                let b = pop!().as_i32()?;
                let a = pop!().as_i32()?;
                stack.push(Val::I32($f(a, b)));
            }};
        }
        macro_rules! bin_i64 {
            ($f:expr) => {{
                let b = pop!().as_i64()?;
                let a = pop!().as_i64()?;
                stack.push(Val::I64($f(a, b)));
            }};
        }
        // i64 comparison → i32 boolean.
        macro_rules! cmp_i64 {
            ($f:expr) => {{
                let b = pop!().as_i64()?;
                let a = pop!().as_i64()?;
                stack.push(Val::I32(i32::from($f(a, b))));
            }};
        }
        macro_rules! bin_f64 {
            ($f:expr) => {{
                let b = pop!().as_f64()?;
                let a = pop!().as_f64()?;
                stack.push(Val::F64($f(a, b)));
            }};
        }
        // f64 comparison → i32 boolean.
        macro_rules! cmp_f64 {
            ($f:expr) => {{
                let b = pop!().as_f64()?;
                let a = pop!().as_f64()?;
                stack.push(Val::I32(i32::from($f(a, b))));
            }};
        }
        macro_rules! bin_f32 {
            ($f:expr) => {{
                let b = pop!().as_f32()?;
                let a = pop!().as_f32()?;
                stack.push(Val::F32($f(a, b)));
            }};
        }
        macro_rules! cmp_f32 {
            ($f:expr) => {{
                let b = pop!().as_f32()?;
                let a = pop!().as_f32()?;
                stack.push(Val::I32(i32::from($f(a, b))));
            }};
        }
        while !r.done() {
            let op = r.byte()?;
            match op {
                0x00 => return Err(WasmRtError("unreachable executed")), // unreachable (trap)
                0x01 => {}                                               // nop
                0x0b => return Ok(Flow::Normal),                         // end
                0x0f => return Ok(Flow::Return),                         // return
                0x1a => {
                    pop!(); // drop
                }
                // local.get / local.set / local.tee
                0x20 => {
                    let i = r.u32()? as usize;
                    stack.push(*locals.get(i).ok_or(WasmRtError("bad local"))?);
                }
                0x21 => {
                    let i = r.u32()? as usize;
                    let v = pop!();
                    *locals.get_mut(i).ok_or(WasmRtError("bad local"))? = v;
                }
                0x22 => {
                    let i = r.u32()? as usize;
                    let v = *stack.last().ok_or(WasmRtError("stack underflow"))?;
                    *locals.get_mut(i).ok_or(WasmRtError("bad local"))? = v;
                }
                // global.get / global.set
                0x23 => {
                    let i = r.u32()? as usize;
                    stack.push(*store.globals.get(i).ok_or(WasmRtError("bad global"))?);
                }
                0x24 => {
                    let i = r.u32()? as usize;
                    if !self.global_mutable(i).unwrap_or(false) {
                        return Err(WasmRtError("set of immutable/undefined global"));
                    }
                    let v = pop!();
                    *store.globals.get_mut(i).ok_or(WasmRtError("bad global"))? = v;
                }
                0x41 => stack.push(Val::I32(r.i32()?)), // i32.const
                0x42 => stack.push(Val::I64(r.i64()?)), // i64.const
                // call
                0x10 => {
                    let callee = r.u32()?;
                    let cty = self
                        .func_types
                        .get(callee as usize)
                        .and_then(|t| self.types.get(*t as usize))
                        .ok_or(WasmRtError("bad call target"))?;
                    let n = cty.params.len();
                    if stack.len() < n {
                        return Err(WasmRtError("call argument underflow"));
                    }
                    let cargs = stack.split_off(stack.len() - n);
                    let res = self.call_with_store(callee, &cargs, store, host)?;
                    stack.extend(res);
                }
                // call_indirect: typeidx, tableidx; pop the table index, look up
                // the function, check its signature, call it.
                0x11 => {
                    let type_idx = r.u32()? as usize;
                    let _table_idx = r.u32()?; // 0x00 (reserved in the MVP)
                    let idx = pop!().as_i32()? as usize;
                    let func = self
                        .table
                        .get(idx)
                        .copied()
                        .flatten()
                        .ok_or(WasmRtError("undefined element (call_indirect)"))?;
                    // The dynamic signature must match the static expected type.
                    let expected = self.types.get(type_idx);
                    let actual = self
                        .func_types
                        .get(func as usize)
                        .and_then(|t| self.types.get(*t as usize));
                    if expected.is_none() || expected != actual {
                        return Err(WasmRtError("indirect call type mismatch"));
                    }
                    let n = expected.unwrap().params.len();
                    if stack.len() < n {
                        return Err(WasmRtError("call argument underflow"));
                    }
                    let cargs = stack.split_off(stack.len() - n);
                    let res = self.call_with_store(func, &cargs, store, host)?;
                    stack.extend(res);
                }
                // --- linear memory ---
                0x28 => {
                    let a = mem_addr!(r, 4);
                    let v = i32::from_le_bytes(store.mem[a..a + 4].try_into().unwrap());
                    stack.push(Val::I32(v)); // i32.load
                }
                0x29 => {
                    let a = mem_addr!(r, 8);
                    let v = i64::from_le_bytes(store.mem[a..a + 8].try_into().unwrap());
                    stack.push(Val::I64(v)); // i64.load
                }
                0x2c => {
                    let a = mem_addr!(r, 1);
                    stack.push(Val::I32(i32::from(store.mem[a] as i8))); // i32.load8_s
                }
                0x2d => {
                    let a = mem_addr!(r, 1);
                    stack.push(Val::I32(i32::from(store.mem[a]))); // i32.load8_u
                }
                0x2e => {
                    let a = mem_addr!(r, 2);
                    let v = i16::from_le_bytes(store.mem[a..a + 2].try_into().unwrap());
                    stack.push(Val::I32(i32::from(v))); // i32.load16_s
                }
                0x2f => {
                    let a = mem_addr!(r, 2);
                    let v = u16::from_le_bytes(store.mem[a..a + 2].try_into().unwrap());
                    stack.push(Val::I32(i32::from(v))); // i32.load16_u
                }
                0x30 => {
                    let a = mem_addr!(r, 1);
                    stack.push(Val::I64(i64::from(store.mem[a] as i8))); // i64.load8_s
                }
                0x31 => {
                    let a = mem_addr!(r, 1);
                    stack.push(Val::I64(i64::from(store.mem[a]))); // i64.load8_u
                }
                0x32 => {
                    let a = mem_addr!(r, 2);
                    let v = i16::from_le_bytes(store.mem[a..a + 2].try_into().unwrap());
                    stack.push(Val::I64(i64::from(v))); // i64.load16_s
                }
                0x33 => {
                    let a = mem_addr!(r, 2);
                    let v = u16::from_le_bytes(store.mem[a..a + 2].try_into().unwrap());
                    stack.push(Val::I64(i64::from(v))); // i64.load16_u
                }
                0x34 => {
                    let a = mem_addr!(r, 4);
                    let v = i32::from_le_bytes(store.mem[a..a + 4].try_into().unwrap());
                    stack.push(Val::I64(i64::from(v))); // i64.load32_s
                }
                0x35 => {
                    let a = mem_addr!(r, 4);
                    let v = u32::from_le_bytes(store.mem[a..a + 4].try_into().unwrap());
                    stack.push(Val::I64(i64::from(v))); // i64.load32_u
                }
                0x36 => {
                    let v = {
                        let v = pop!().as_i32()?;
                        let a = mem_addr!(r, 4);
                        (a, v)
                    };
                    store.mem[v.0..v.0 + 4].copy_from_slice(&v.1.to_le_bytes()); // i32.store
                }
                0x37 => {
                    let (a, v) = {
                        let v = pop!().as_i64()?;
                        let a = mem_addr!(r, 8);
                        (a, v)
                    };
                    store.mem[a..a + 8].copy_from_slice(&v.to_le_bytes()); // i64.store
                }
                0x3a => {
                    let (a, v) = {
                        let v = pop!().as_i32()?;
                        let a = mem_addr!(r, 1);
                        (a, v)
                    };
                    store.mem[a] = v as u8; // i32.store8
                }
                0x3b => {
                    let (a, v) = {
                        let v = pop!().as_i32()?;
                        let a = mem_addr!(r, 2);
                        (a, v)
                    };
                    store.mem[a..a + 2].copy_from_slice(&(v as u16).to_le_bytes()); // i32.store16
                }
                0x3c => {
                    let (a, v) = {
                        let v = pop!().as_i64()?;
                        let a = mem_addr!(r, 1);
                        (a, v)
                    };
                    store.mem[a] = v as u8; // i64.store8
                }
                0x3d => {
                    let (a, v) = {
                        let v = pop!().as_i64()?;
                        let a = mem_addr!(r, 2);
                        (a, v)
                    };
                    store.mem[a..a + 2].copy_from_slice(&(v as u16).to_le_bytes()); // i64.store16
                }
                0x3e => {
                    let (a, v) = {
                        let v = pop!().as_i64()?;
                        let a = mem_addr!(r, 4);
                        (a, v)
                    };
                    store.mem[a..a + 4].copy_from_slice(&(v as u32).to_le_bytes()); // i64.store32
                }
                0x3f => {
                    let _reserved = r.byte()?;
                    stack.push(Val::I32((store.mem.len() / PAGE_SIZE) as i32)); // memory.size
                }
                0x40 => {
                    let _reserved = r.byte()?;
                    let delta = pop!().as_i32()? as u32 as usize;
                    let old = store.mem.len() / PAGE_SIZE;
                    let new_pages = old + delta;
                    // The hard ceiling is 2^16 pages (4 GiB); a declared maximum
                    // (the limits' upper bound) further caps growth. Either failure
                    // leaves memory untouched and returns -1.
                    let ceiling = store
                        .mem_max_pages
                        .map_or(0x1_0000, |m| (m as usize).min(0x1_0000));
                    if new_pages <= ceiling {
                        store.mem.resize(new_pages * PAGE_SIZE, 0);
                        stack.push(Val::I32(old as i32));
                    } else {
                        stack.push(Val::I32(-1));
                    }
                }
                // i32 arithmetic / bitwise
                0x45 => {
                    let a = pop!().as_i32()?;
                    stack.push(Val::I32(i32::from(a == 0)));
                } // i32.eqz
                0x46 => bin_i32!(|a, b| i32::from(a == b)),
                0x47 => bin_i32!(|a, b| i32::from(a != b)),
                0x48 => bin_i32!(|a, b| i32::from(a < b)), // lt_s
                0x49 => bin_i32!(|a: i32, b: i32| i32::from((a as u32) < b as u32)), // lt_u
                0x4a => bin_i32!(|a, b| i32::from(a > b)), // gt_s
                0x4b => bin_i32!(|a: i32, b: i32| i32::from((a as u32) > b as u32)), // gt_u
                0x4c => bin_i32!(|a, b| i32::from(a <= b)), // le_s
                0x4d => bin_i32!(|a: i32, b: i32| i32::from((a as u32) <= b as u32)), // le_u
                0x4e => bin_i32!(|a, b| i32::from(a >= b)), // ge_s
                0x4f => bin_i32!(|a: i32, b: i32| i32::from((a as u32) >= b as u32)), // ge_u
                0x67 => {
                    let a = pop!().as_i32()?;
                    stack.push(Val::I32(a.leading_zeros() as i32)); // clz
                }
                0x68 => {
                    let a = pop!().as_i32()?;
                    stack.push(Val::I32(a.trailing_zeros() as i32)); // ctz
                }
                0x69 => {
                    let a = pop!().as_i32()?;
                    stack.push(Val::I32(a.count_ones() as i32)); // popcnt
                }
                0x6a => bin_i32!(i32::wrapping_add),
                0x6b => bin_i32!(i32::wrapping_sub),
                0x6c => bin_i32!(i32::wrapping_mul),
                0x6d => {
                    let b = pop!().as_i32()?;
                    let a = pop!().as_i32()?;
                    if b == 0 {
                        return Err(WasmRtError("integer divide by zero"));
                    }
                    if a == i32::MIN && b == -1 {
                        return Err(WasmRtError("integer overflow")); // i32.div_s overflow
                    }
                    stack.push(Val::I32(a / b)); // div_s
                }
                0x6e => {
                    let b = pop!().as_i32()? as u32;
                    let a = pop!().as_i32()? as u32;
                    if b == 0 {
                        return Err(WasmRtError("integer divide by zero"));
                    }
                    stack.push(Val::I32((a / b) as i32)); // div_u
                }
                0x6f => {
                    let b = pop!().as_i32()?;
                    let a = pop!().as_i32()?;
                    if b == 0 {
                        return Err(WasmRtError("integer divide by zero"));
                    }
                    stack.push(Val::I32(a.wrapping_rem(b))); // rem_s
                }
                0x70 => {
                    let b = pop!().as_i32()? as u32;
                    let a = pop!().as_i32()? as u32;
                    if b == 0 {
                        return Err(WasmRtError("integer divide by zero"));
                    }
                    stack.push(Val::I32((a % b) as i32)); // rem_u
                }
                0x71 => bin_i32!(|a, b| a & b),
                0x72 => bin_i32!(|a, b| a | b),
                0x73 => bin_i32!(|a, b| a ^ b),
                0x74 => bin_i32!(|a: i32, b: i32| a.wrapping_shl(b as u32)),
                0x75 => bin_i32!(|a: i32, b: i32| a.wrapping_shr(b as u32)), // shr_s
                0x76 => bin_i32!(|a: i32, b: i32| ((a as u32).wrapping_shr(b as u32)) as i32), // shr_u
                0x77 => bin_i32!(|a: i32, b: i32| a.rotate_left(b as u32)),
                0x78 => bin_i32!(|a: i32, b: i32| a.rotate_right(b as u32)),
                // i64 comparisons (→ i32)
                0x50 => {
                    let a = pop!().as_i64()?;
                    stack.push(Val::I32(i32::from(a == 0))); // i64.eqz
                }
                0x51 => cmp_i64!(|a, b| a == b),
                0x52 => cmp_i64!(|a, b| a != b),
                0x53 => cmp_i64!(|a, b| a < b), // lt_s
                0x54 => cmp_i64!(|a: i64, b: i64| (a as u64) < b as u64), // lt_u
                0x55 => cmp_i64!(|a, b| a > b), // gt_s
                0x56 => cmp_i64!(|a: i64, b: i64| (a as u64) > b as u64), // gt_u
                0x57 => cmp_i64!(|a, b| a <= b), // le_s
                0x59 => cmp_i64!(|a, b| a >= b), // ge_s
                // i64 arithmetic / bitwise
                0x7c => bin_i64!(i64::wrapping_add),
                0x7d => bin_i64!(i64::wrapping_sub),
                0x7e => bin_i64!(i64::wrapping_mul),
                0x7f => {
                    let b = pop!().as_i64()?;
                    let a = pop!().as_i64()?;
                    if b == 0 {
                        return Err(WasmRtError("integer divide by zero"));
                    }
                    if a == i64::MIN && b == -1 {
                        return Err(WasmRtError("integer overflow")); // i64.div_s overflow
                    }
                    stack.push(Val::I64(a / b)); // i64.div_s
                }
                0x58 => cmp_i64!(|a: i64, b: i64| (a as u64) <= b as u64), // le_u
                0x5a => cmp_i64!(|a: i64, b: i64| (a as u64) >= b as u64), // ge_u
                0x79 => {
                    let a = pop!().as_i64()?;
                    stack.push(Val::I64(i64::from(a.leading_zeros()))); // i64.clz
                }
                0x7a => {
                    let a = pop!().as_i64()?;
                    stack.push(Val::I64(i64::from(a.trailing_zeros()))); // i64.ctz
                }
                0x7b => {
                    let a = pop!().as_i64()?;
                    stack.push(Val::I64(i64::from(a.count_ones()))); // i64.popcnt
                }
                0x80 => {
                    let b = pop!().as_i64()? as u64;
                    let a = pop!().as_i64()? as u64;
                    if b == 0 {
                        return Err(WasmRtError("integer divide by zero"));
                    }
                    stack.push(Val::I64((a / b) as i64)); // i64.div_u
                }
                0x81 => {
                    let b = pop!().as_i64()?;
                    let a = pop!().as_i64()?;
                    if b == 0 {
                        return Err(WasmRtError("integer divide by zero"));
                    }
                    stack.push(Val::I64(a.wrapping_rem(b))); // i64.rem_s
                }
                0x82 => {
                    let b = pop!().as_i64()? as u64;
                    let a = pop!().as_i64()? as u64;
                    if b == 0 {
                        return Err(WasmRtError("integer divide by zero"));
                    }
                    stack.push(Val::I64((a % b) as i64)); // i64.rem_u
                }
                0x83 => bin_i64!(|a, b| a & b),
                0x84 => bin_i64!(|a, b| a | b),
                0x85 => bin_i64!(|a, b| a ^ b),
                0x86 => bin_i64!(|a: i64, b: i64| a.wrapping_shl(b as u32)),
                0x87 => bin_i64!(|a: i64, b: i64| a.wrapping_shr(b as u32)), // shr_s
                0x88 => bin_i64!(|a: i64, b: i64| ((a as u64).wrapping_shr(b as u32)) as i64), // shr_u
                0x89 => bin_i64!(|a: i64, b: i64| a.rotate_left(b as u32)),
                0x8a => bin_i64!(|a: i64, b: i64| a.rotate_right(b as u32)),
                // conversions
                0xa7 => {
                    let a = pop!().as_i64()?;
                    stack.push(Val::I32(a as i32)); // i32.wrap_i64
                }
                0xaa => {
                    let a = pop!().as_f64()?;
                    stack.push(Val::I32(trunc_i32_s(a)?)); // i32.trunc_f64_s
                }
                0xab => {
                    let a = pop!().as_f64()?;
                    stack.push(Val::I32(trunc_i32_u(a)?)); // i32.trunc_f64_u
                }
                0xb0 => {
                    let a = pop!().as_f64()?;
                    stack.push(Val::I64(trunc_i64_s(a)?)); // i64.trunc_f64_s
                }
                0xb1 => {
                    let a = pop!().as_f64()?;
                    stack.push(Val::I64(trunc_i64_u(a)?)); // i64.trunc_f64_u
                }
                0xac => {
                    let a = pop!().as_i32()?;
                    stack.push(Val::I64(i64::from(a))); // i64.extend_i32_s
                }
                0xad => {
                    let a = pop!().as_i32()? as u32;
                    stack.push(Val::I64(i64::from(a))); // i64.extend_i32_u
                }
                0xb7 => {
                    let a = pop!().as_i32()?;
                    stack.push(Val::F64(f64::from(a))); // f64.convert_i32_s
                }
                0xb8 => {
                    let a = pop!().as_i32()? as u32;
                    stack.push(Val::F64(f64::from(a))); // f64.convert_i32_u
                }
                0xb9 => {
                    let a = pop!().as_i64()?;
                    stack.push(Val::F64(a as f64)); // f64.convert_i64_s
                }
                0xba => {
                    let a = pop!().as_i64()? as u64;
                    stack.push(Val::F64(a as f64)); // f64.convert_i64_u
                }
                0xbb => {
                    let a = pop!().as_f32()?;
                    stack.push(Val::F64(f64::from(a))); // f64.promote_f32
                }
                0xb6 => {
                    let a = pop!().as_f64()?;
                    stack.push(Val::F32(a as f32)); // f32.demote_f64
                }
                // f32 conversions from integers.
                0xb2 => {
                    let a = pop!().as_i32()?;
                    stack.push(Val::F32(a as f32)); // f32.convert_i32_s
                }
                0xb3 => {
                    let a = pop!().as_i32()? as u32;
                    stack.push(Val::F32(a as f32)); // f32.convert_i32_u
                }
                0xb4 => {
                    let a = pop!().as_i64()?;
                    stack.push(Val::F32(a as f32)); // f32.convert_i64_s
                }
                0xb5 => {
                    let a = pop!().as_i64()? as u64;
                    stack.push(Val::F32(a as f32)); // f32.convert_i64_u
                }
                // Integer truncations from f32 (trap on NaN/∞ and out-of-range).
                0xa8 => {
                    let a = f64::from(pop!().as_f32()?);
                    stack.push(Val::I32(trunc_i32_s(a)?)); // i32.trunc_f32_s
                }
                0xa9 => {
                    let a = f64::from(pop!().as_f32()?);
                    stack.push(Val::I32(trunc_i32_u(a)?)); // i32.trunc_f32_u
                }
                0xae => {
                    let a = f64::from(pop!().as_f32()?);
                    stack.push(Val::I64(trunc_i64_s(a)?)); // i64.trunc_f32_s
                }
                0xaf => {
                    let a = f64::from(pop!().as_f32()?);
                    stack.push(Val::I64(trunc_i64_u(a)?)); // i64.trunc_f32_u
                }
                // Bit-level reinterpretations (no value change, only the type).
                0xbc => {
                    let a = pop!().as_f32()?;
                    stack.push(Val::I32(a.to_bits() as i32)); // i32.reinterpret_f32
                }
                0xbd => {
                    let a = pop!().as_f64()?;
                    stack.push(Val::I64(a.to_bits() as i64)); // i64.reinterpret_f64
                }
                0xbe => {
                    let a = pop!().as_i32()?;
                    stack.push(Val::F32(f32::from_bits(a as u32))); // f32.reinterpret_i32
                }
                0xbf => {
                    let a = pop!().as_i64()?;
                    stack.push(Val::F64(f64::from_bits(a as u64))); // f64.reinterpret_i64
                }
                // Sign-extension operators (standardized into the core spec).
                0xc0 => {
                    let a = pop!().as_i32()?;
                    stack.push(Val::I32(i32::from(a as i8))); // i32.extend8_s
                }
                0xc1 => {
                    let a = pop!().as_i32()?;
                    stack.push(Val::I32(i32::from(a as i16))); // i32.extend16_s
                }
                0xc2 => {
                    let a = pop!().as_i64()?;
                    stack.push(Val::I64(i64::from(a as i8))); // i64.extend8_s
                }
                0xc3 => {
                    let a = pop!().as_i64()?;
                    stack.push(Val::I64(i64::from(a as i16))); // i64.extend16_s
                }
                0xc4 => {
                    let a = pop!().as_i64()?;
                    stack.push(Val::I64(i64::from(a as i32))); // i64.extend32_s
                }
                // select: pop cond, then b, then a; push `cond ? a : b`.
                0x1b => {
                    let cond = pop!().as_i32()?;
                    let b = pop!();
                    let a = pop!();
                    stack.push(if cond != 0 { a } else { b });
                }
                // --- floating point ---
                0x43 => {
                    let v = f32::from_le_bytes(r.bytes(4)?.try_into().unwrap());
                    stack.push(Val::F32(v)); // f32.const
                }
                0x44 => {
                    let v = f64::from_le_bytes(r.bytes(8)?.try_into().unwrap());
                    stack.push(Val::F64(v)); // f64.const
                }
                0x2a => {
                    let a = mem_addr!(r, 4);
                    let v = f32::from_le_bytes(store.mem[a..a + 4].try_into().unwrap());
                    stack.push(Val::F32(v)); // f32.load
                }
                0x2b => {
                    let a = mem_addr!(r, 8);
                    let v = f64::from_le_bytes(store.mem[a..a + 8].try_into().unwrap());
                    stack.push(Val::F64(v)); // f64.load
                }
                0x38 => {
                    let (a, v) = {
                        let v = pop!().as_f32()?;
                        let a = mem_addr!(r, 4);
                        (a, v)
                    };
                    store.mem[a..a + 4].copy_from_slice(&v.to_le_bytes()); // f32.store
                }
                0x39 => {
                    let (a, v) = {
                        let v = pop!().as_f64()?;
                        let a = mem_addr!(r, 8);
                        (a, v)
                    };
                    store.mem[a..a + 8].copy_from_slice(&v.to_le_bytes()); // f64.store
                }
                // f32 comparisons
                0x5b => cmp_f32!(|a, b| a == b),
                0x5c => cmp_f32!(|a, b| a != b),
                0x5d => cmp_f32!(|a, b| a < b),
                0x5e => cmp_f32!(|a, b| a > b),
                0x5f => cmp_f32!(|a, b| a <= b),
                0x60 => cmp_f32!(|a, b| a >= b),
                // f64 comparisons
                0x61 => cmp_f64!(|a, b| a == b),
                0x62 => cmp_f64!(|a, b| a != b),
                0x63 => cmp_f64!(|a, b| a < b),
                0x64 => cmp_f64!(|a, b| a > b),
                0x65 => cmp_f64!(|a, b| a <= b),
                0x66 => cmp_f64!(|a, b| a >= b),
                // f32 unary / arithmetic
                0x8b => {
                    let a = pop!().as_f32()?;
                    stack.push(Val::F32(f32_abs(a)));
                }
                0x8c => {
                    let a = pop!().as_f32()?;
                    stack.push(Val::F32(-a));
                }
                0x8d => {
                    let a = pop!().as_f32()?;
                    stack.push(Val::F32(f32_ceil(a)));
                }
                0x8e => {
                    let a = pop!().as_f32()?;
                    stack.push(Val::F32(f32_floor(a)));
                }
                0x8f => {
                    let a = pop!().as_f32()?;
                    stack.push(Val::F32(f32_trunc(a)));
                }
                0x90 => {
                    let a = pop!().as_f32()?;
                    stack.push(Val::F32(f32_nearest(a)));
                }
                0x91 => {
                    let a = pop!().as_f32()?;
                    stack.push(Val::F32(f32_sqrt(a)));
                }
                0x92 => bin_f32!(|a, b| a + b),
                0x93 => bin_f32!(|a, b| a - b),
                0x94 => bin_f32!(|a, b| a * b),
                0x95 => bin_f32!(|a, b| a / b),
                0x96 => bin_f32!(f32_min),
                0x97 => bin_f32!(f32_max),
                0x98 => bin_f32!(f32_copysign),
                // f64 unary / arithmetic
                0x99 => {
                    let a = pop!().as_f64()?;
                    stack.push(Val::F64(f64_abs(a)));
                }
                0x9a => {
                    let a = pop!().as_f64()?;
                    stack.push(Val::F64(-a));
                }
                0x9b => {
                    let a = pop!().as_f64()?;
                    stack.push(Val::F64(f64_ceil(a)));
                }
                0x9c => {
                    let a = pop!().as_f64()?;
                    stack.push(Val::F64(f64_floor(a)));
                }
                0x9d => {
                    let a = pop!().as_f64()?;
                    stack.push(Val::F64(f64_trunc(a)));
                }
                0x9e => {
                    let a = pop!().as_f64()?;
                    stack.push(Val::F64(f64_nearest(a)));
                }
                0x9f => {
                    let a = pop!().as_f64()?;
                    stack.push(Val::F64(f64_sqrt(a)));
                }
                0xa0 => bin_f64!(|a, b| a + b),
                0xa1 => bin_f64!(|a, b| a - b),
                0xa2 => bin_f64!(|a, b| a * b),
                0xa3 => bin_f64!(|a, b| a / b),
                0xa4 => bin_f64!(f64_min),
                0xa5 => bin_f64!(f64_max),
                0xa6 => bin_f64!(f64_copysign),
                // structured control: block / loop
                0x02 | 0x03 => {
                    let _blocktype = r.byte()?; // 0x40 (empty) or a value type
                    let is_loop = op == 0x03;
                    let (consumed, flow) =
                        self.exec_block(&code[r.pos..], locals, stack, store, host, is_loop)?;
                    r.pos += consumed;
                    match flow {
                        Flow::Return => return Ok(Flow::Return),
                        Flow::Branch(0) if is_loop => {
                            // a `br 0` to a loop re-runs it; handled inside
                            // exec_block, so this arm is unreachable in practice.
                        }
                        Flow::Branch(n) => return Ok(Flow::Branch(n.saturating_sub(1))),
                        Flow::Normal => {}
                    }
                }
                // if / else / end: pop the condition, run the chosen arm.
                0x04 => {
                    let _blocktype = r.byte()?;
                    let cond = pop!().as_i32()? != 0;
                    let body = &code[r.pos..];
                    let inner_len = block_len(body)?; // includes the matching `end`
                    let inner = &body[..inner_len - 1]; // exclude `end`
                    let else_at = else_split(inner)?;
                    let arm = if cond {
                        &inner[..else_at.unwrap_or(inner.len())]
                    } else {
                        match else_at {
                            Some(e) => &inner[e + 1..], // after the `else` byte
                            None => &inner[inner.len()..],
                        }
                    };
                    r.pos += inner_len;
                    match self.exec(arm, locals, stack, store, host)? {
                        Flow::Return => return Ok(Flow::Return),
                        Flow::Branch(n) => return Ok(Flow::Branch(n.saturating_sub(1))),
                        Flow::Normal => {}
                    }
                }
                0x0c => return Ok(Flow::Branch(r.u32()?)), // br
                0x0d => {
                    let depth = r.u32()?;
                    if pop!().as_i32()? != 0 {
                        return Ok(Flow::Branch(depth));
                    }
                }
                // br_table: a computed branch — pick labels[index], else default.
                0x0e => {
                    let count = r.u32()? as usize;
                    let mut labels = Vec::with_capacity(count);
                    for _ in 0..count {
                        labels.push(r.u32()?);
                    }
                    let default = r.u32()?;
                    let idx = pop!().as_i32()? as u32 as usize;
                    return Ok(Flow::Branch(labels.get(idx).copied().unwrap_or(default)));
                }
                // 0xfc-prefixed: the saturating truncations (non-trapping — Rust's
                // float→int `as` cast already saturates: NaN→0, out-of-range→clamp).
                0xfc => {
                    let sub = r.u32()?;
                    match sub {
                        0x00 => {
                            let a = pop!().as_f32()?;
                            stack.push(Val::I32(a as i32)); // i32.trunc_sat_f32_s
                        }
                        0x01 => {
                            let a = pop!().as_f32()?;
                            stack.push(Val::I32(a as u32 as i32)); // i32.trunc_sat_f32_u
                        }
                        0x02 => {
                            let a = pop!().as_f64()?;
                            stack.push(Val::I32(a as i32)); // i32.trunc_sat_f64_s
                        }
                        0x03 => {
                            let a = pop!().as_f64()?;
                            stack.push(Val::I32(a as u32 as i32)); // i32.trunc_sat_f64_u
                        }
                        0x04 => {
                            let a = pop!().as_f32()?;
                            stack.push(Val::I64(a as i64)); // i64.trunc_sat_f32_s
                        }
                        0x05 => {
                            let a = pop!().as_f32()?;
                            stack.push(Val::I64(a as u64 as i64)); // i64.trunc_sat_f32_u
                        }
                        0x06 => {
                            let a = pop!().as_f64()?;
                            stack.push(Val::I64(a as i64)); // i64.trunc_sat_f64_s
                        }
                        0x07 => {
                            let a = pop!().as_f64()?;
                            stack.push(Val::I64(a as u64 as i64)); // i64.trunc_sat_f64_u
                        }
                        0x0a => {
                            // memory.copy: two reserved memory indices, then n/src/dst.
                            r.byte()?;
                            r.byte()?;
                            let n = pop!().as_i32()? as u32 as usize;
                            let src = pop!().as_i32()? as u32 as usize;
                            let dst = pop!().as_i32()? as u32 as usize;
                            match (src.checked_add(n), dst.checked_add(n)) {
                                (Some(es), Some(ed))
                                    if es <= store.mem.len() && ed <= store.mem.len() =>
                                {
                                    store.mem.copy_within(src..es, dst); // overlap-safe
                                }
                                _ => return Err(WasmRtError("out of bounds memory access")),
                            }
                        }
                        0x0b => {
                            // memory.fill: one reserved memory index, then n/val/dst.
                            r.byte()?;
                            let n = pop!().as_i32()? as u32 as usize;
                            let val = pop!().as_i32()? as u8;
                            let dst = pop!().as_i32()? as u32 as usize;
                            match dst.checked_add(n) {
                                Some(end) if end <= store.mem.len() => {
                                    store.mem[dst..end].fill(val);
                                }
                                _ => return Err(WasmRtError("out of bounds memory access")),
                            }
                        }
                        0x08 => {
                            // memory.init: data segment index, reserved memory index,
                            // then n/src/dst. Copies from the (passive) segment bytes.
                            let seg = r.u32()? as usize;
                            r.byte()?;
                            let n = pop!().as_i32()? as u32 as usize;
                            let src = pop!().as_i32()? as u32 as usize;
                            let dst = pop!().as_i32()? as u32 as usize;
                            // A dropped segment behaves as zero-length.
                            let empty: Vec<u8> = Vec::new();
                            let bytes = match self.data.get(seg) {
                                Some(_) if store.dropped.get(seg).copied().unwrap_or(true) => {
                                    &empty
                                }
                                Some((_, b)) => b,
                                None => return Err(WasmRtError("memory.init: bad segment")),
                            };
                            match (src.checked_add(n), dst.checked_add(n)) {
                                (Some(es), Some(ed))
                                    if es <= bytes.len() && ed <= store.mem.len() =>
                                {
                                    store.mem[dst..ed].copy_from_slice(&bytes[src..es]);
                                }
                                _ => return Err(WasmRtError("out of bounds memory access")),
                            }
                        }
                        0x09 => {
                            // data.drop: release the segment's bytes.
                            let seg = r.u32()? as usize;
                            if let Some(d) = store.dropped.get_mut(seg) {
                                *d = true;
                            }
                        }
                        _ => return Err(WasmRtError("unsupported 0xfc opcode")),
                    }
                }
                _ => return Err(WasmRtError("unsupported opcode")),
            }
        }
        Ok(Flow::Normal)
    }

    /// Executes a structured block, returning `(bytes consumed up to and
    /// including its matching `end`, resulting flow)`. A `loop` re-enters on
    /// `br 0`; a `block` exits on `br 0`.
    fn exec_block(
        &self,
        code: &[u8],
        locals: &mut [Val],
        stack: &mut Vec<Val>,
        store: &mut Store,
        host: ImportHost,
        is_loop: bool,
    ) -> Result<(usize, Flow), WasmRtError> {
        let inner_len = block_len(code)?;
        let inner = &code[..inner_len - 1]; // exclude the matching `end`
        loop {
            match self.exec(inner, locals, stack, store, host)? {
                Flow::Branch(0) => {
                    if is_loop {
                        continue; // re-run the loop body
                    }
                    return Ok((inner_len, Flow::Normal)); // br 0 to a block = exit
                }
                Flow::Branch(n) => return Ok((inner_len, Flow::Branch(n))),
                Flow::Return => return Ok((inner_len, Flow::Return)),
                Flow::Normal => return Ok((inner_len, Flow::Normal)),
            }
        }
    }
}

/// A live, **instantiated** module: linear memory and globals that persist across
/// calls (unlike [`Module::call`], which uses a throwaway store). This is the
/// stateful object the JS↔WASM boundary exchanges data through — host code reads
/// and writes the instance's linear memory to pass strings, arrays, and buffers
/// in and out, and a mutable global keeps state between invocations.
/// A snapshot of an instance's mutable state (linear memory, globals, dropped data
/// segments), carried across separate `call_export` invocations so a JS
/// `WebAssembly.Instance` keeps persistent state instead of re-initializing per
/// call. Owned (no module borrow), so a host can stash it between calls.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct InstanceState {
    /// linear memory bytes
    pub mem: Vec<u8>,
    /// global values, in index order
    pub globals: Vec<Val>,
    /// per-data-segment dropped flags
    pub dropped: Vec<bool>,
}

/// An instantiated module: the decoded `module` plus its mutable `store` (linear
/// memory, globals) and the host functions bound to its imports.
pub struct Instance<'m> {
    module: &'m Module,
    store: Store,
    /// Host functions backing the module's function imports (index-aligned with
    /// `Module::func_imports`); dispatched by `call`'s default import host.
    host_funcs: Vec<HostFunc>,
}

impl<'m> Instance<'m> {
    /// Instantiates `module` with no imports.
    ///
    /// # Errors
    /// Returns `WasmRtError` if a data segment is out of bounds, or the module
    /// declares function imports (use [`with_imports`](Self::with_imports)).
    pub fn new(module: &'m Module) -> Result<Self, WasmRtError> {
        Self::with_imports(module, Vec::new())
    }

    /// Instantiates `module`, binding `host_funcs` to its function imports (in
    /// declaration order — see [`Module::import_names`]). This is how a host
    /// (e.g. JS) function becomes callable from inside the module.
    ///
    /// # Errors
    /// `WasmRtError("import count mismatch")` if the number of host functions
    /// doesn't match the module's function imports, or a data segment is out of
    /// bounds.
    pub fn with_imports(
        module: &'m Module,
        host_funcs: Vec<HostFunc>,
    ) -> Result<Self, WasmRtError> {
        Self::instantiate(module, host_funcs, Vec::new())
    }

    /// Instantiates `module` binding both function imports (`host_funcs`) and
    /// **global imports** (`import_globals`, in declaration order — the values
    /// the host supplies for the module's imported globals).
    ///
    /// # Errors
    /// `WasmRtError("import count mismatch")` if either list's length doesn't
    /// match the module's imports, or a data segment is out of bounds.
    pub fn instantiate(
        module: &'m Module,
        host_funcs: Vec<HostFunc>,
        import_globals: Vec<Val>,
    ) -> Result<Self, WasmRtError> {
        Self::instantiate_full(module, host_funcs, import_globals, None)
    }

    /// Instantiates `module` binding function imports, global imports, and an
    /// optional **imported linear memory** (`import_memory` — the host-owned
    /// bytes the module reads/writes; required iff the module imports memory).
    ///
    /// # Errors
    /// `WasmRtError("import count mismatch")` if any import list doesn't match the
    /// module's declared imports, or a data segment is out of bounds.
    pub fn instantiate_full(
        module: &'m Module,
        host_funcs: Vec<HostFunc>,
        import_globals: Vec<Val>,
        import_memory: Option<Vec<u8>>,
    ) -> Result<Self, WasmRtError> {
        if host_funcs.len() != module.n_imported_funcs()
            || import_globals.len() != module.n_imported_globals()
            || module.mem_imported != import_memory.is_some()
        {
            return Err(WasmRtError("import count mismatch"));
        }
        let mut store = module.new_store()?;
        // Imported globals occupy the first slots of the global space.
        for (i, v) in import_globals.into_iter().enumerate() {
            store.globals[i] = v;
        }
        // An imported memory replaces the store's default-allocated bytes; data
        // segments (applied by `new_store`) carry over by re-applying them.
        if let Some(mut mem) = import_memory {
            for (off, bytes) in &module.data {
                let Some(off) = off else { continue };
                let start = *off as usize;
                let end = start
                    .checked_add(bytes.len())
                    .filter(|e| *e <= mem.len())
                    .ok_or(WasmRtError("data segment out of bounds"))?;
                mem[start..end].copy_from_slice(bytes);
            }
            store.mem = mem;
        }
        let mut inst = Self {
            module,
            store,
            host_funcs,
        };
        // The `start` function runs automatically at instantiation (after memory
        // and globals are set up), initializing the instance.
        if let Some(start) = module.start {
            inst.call(start, &[])?;
        }
        Ok(inst)
    }

    /// Calls function `index` with `args` over this instance's **persistent**
    /// state — memory writes and global mutations are visible to later calls.
    ///
    /// # Errors
    /// Returns `WasmRtError` on a type mismatch, a missing function, or an
    /// unsupported instruction / trap.
    pub fn call(&mut self, index: u32, args: &[Val]) -> Result<Vec<Val>, WasmRtError> {
        // Destructure so the import host (borrowing `host_funcs`) and the store
        // (borrowed mutably) are independent field borrows, not both via `self`.
        let Instance {
            module,
            store,
            host_funcs,
        } = self;
        let mut host = |i: usize, a: &[Val]| {
            host_funcs
                .get(i)
                .ok_or(WasmRtError("missing host function for import"))?(a)
        };
        module.call_with_store(index, args, store, &mut host)
    }

    /// Instantiates `module` whose function imports are dispatched through an
    /// external [`ImportHost`] (e.g. one that calls JS) supplied per call to
    /// [`call_export_with_host`](Self::call_export_with_host), rather than bound
    /// `HostFunc`s. Only function imports are supported on this path.
    ///
    /// # Errors
    /// `WasmRtError` if the module imports a global or memory (unsupported here),
    /// or a data segment is out of bounds.
    pub fn with_host_imports(module: &'m Module) -> Result<Self, WasmRtError> {
        if module.n_imported_globals() != 0 || module.mem_imported {
            return Err(WasmRtError(
                "unsupported import kind for host instantiation",
            ));
        }
        let store = module.new_store()?;
        let mut inst = Self {
            module,
            store,
            host_funcs: Vec::new(),
        };
        if let Some(start) = module.start {
            inst.call(start, &[])?; // start with no host imports
        }
        Ok(inst)
    }

    /// Like [`with_host_imports`](Self::with_host_imports), but also seeds the
    /// module's imported **globals** with host-supplied values (in
    /// [`global_import_names`](Module::global_import_names) order). Function imports
    /// still dispatch through the per-call host; memory imports remain unsupported.
    ///
    /// # Errors
    /// `WasmRtError` if the module imports memory, the global count doesn't match,
    /// or a data segment is out of bounds.
    pub fn with_host_imports_and_globals(
        module: &'m Module,
        import_globals: Vec<Val>,
    ) -> Result<Self, WasmRtError> {
        if module.mem_imported {
            return Err(WasmRtError(
                "unsupported import kind for host instantiation",
            ));
        }
        if import_globals.len() != module.n_imported_globals() {
            return Err(WasmRtError("import count mismatch"));
        }
        let mut store = module.new_store()?;
        // Imported globals occupy the first slots of the global space.
        for (i, v) in import_globals.into_iter().enumerate() {
            store.globals[i] = v;
        }
        let mut inst = Self {
            module,
            store,
            host_funcs: Vec::new(),
        };
        if let Some(start) = module.start {
            inst.call(start, &[])?;
        }
        Ok(inst)
    }

    /// Calls function `index` over this instance, dispatching imported functions
    /// through `host` (which may invoke host-engine functions — e.g. JS). The
    /// instance's own `host_funcs` are ignored on this path.
    ///
    /// # Errors
    /// As [`call`](Self::call).
    pub fn call_with_host(
        &mut self,
        index: u32,
        args: &[Val],
        host: ImportHost,
    ) -> Result<Vec<Val>, WasmRtError> {
        self.module
            .call_with_store(index, args, &mut self.store, host)
    }

    /// Resolves an exported function by name and calls it, dispatching imports
    /// through `host`.
    ///
    /// # Errors
    /// `WasmRtError("no such export")` if `name` is not exported, else as
    /// [`call_with_host`](Self::call_with_host).
    pub fn call_export_with_host(
        &mut self,
        name: &str,
        args: &[Val],
        host: ImportHost,
    ) -> Result<Vec<Val>, WasmRtError> {
        let idx = self
            .module
            .export(name)
            .ok_or(WasmRtError("no such export"))?;
        self.call_with_host(idx, args, host)
    }

    /// Resolves an exported function by name and calls it.
    ///
    /// # Errors
    /// `WasmRtError("no such export")` if `name` is not an exported function,
    /// else as [`call`](Self::call).
    pub fn call_export(&mut self, name: &str, args: &[Val]) -> Result<Vec<Val>, WasmRtError> {
        let idx = self
            .module
            .export(name)
            .ok_or(WasmRtError("no such export"))?;
        self.call(idx, args)
    }

    /// The current value of global `index` (imports occupy the low indices), or
    /// `None` if out of range.
    #[must_use]
    pub fn global_value(&self, index: u32) -> Option<Val> {
        self.store.globals.get(index as usize).copied()
    }

    /// Snapshots the instance's mutable state (memory, globals, dropped segments).
    #[must_use]
    pub fn export_state(&self) -> InstanceState {
        InstanceState {
            mem: self.store.mem.clone(),
            globals: self.store.globals.clone(),
            dropped: self.store.dropped.clone(),
        }
    }

    /// Overwrites the instance's mutable state with a previously exported snapshot
    /// (e.g. to resume a JS `WebAssembly.Instance` with the state from its prior
    /// call). The module identity must match the one the state came from.
    pub fn import_state(&mut self, state: &InstanceState) {
        self.store.mem = state.mem.clone();
        self.store.globals.clone_from(&state.globals);
        self.store.dropped.clone_from(&state.dropped);
    }

    /// Calls an exported function with **JS values** (`NanBox`), marshaling each
    /// argument to the parameter's WASM type and each result back to a JS number.
    /// This is the JS↔WASM call boundary the engine uses to invoke WASM from JS.
    ///
    /// # Errors
    /// `WasmRtError("no such export")`, `"argument count mismatch"`, or
    /// `"argument not coercible to wasm value"` (a non-numeric JS argument), else
    /// as [`call`](Self::call).
    pub fn call_export_js(
        &mut self,
        name: &str,
        js_args: &[crate::nanbox::NanBox],
    ) -> Result<Vec<crate::nanbox::NanBox>, WasmRtError> {
        let idx = self
            .module
            .export(name)
            .ok_or(WasmRtError("no such export"))?;
        // Snapshot the parameter types so the immutable borrow ends before `call`.
        let params: Vec<ValType> = {
            let ty = self
                .module
                .func_type(idx)
                .ok_or(WasmRtError("no such function"))?;
            if js_args.len() != ty.params.len() {
                return Err(WasmRtError("argument count mismatch"));
            }
            ty.params.clone()
        };
        let args: Vec<Val> = params
            .iter()
            .zip(js_args)
            .map(|(t, v)| {
                Val::from_nanbox(*v, *t).ok_or(WasmRtError("argument not coercible to wasm value"))
            })
            .collect::<Result<_, _>>()?;
        let results = self.call(idx, &args)?;
        Ok(results.into_iter().map(Val::to_nanbox).collect())
    }

    /// The instance's linear memory (the bytes the JS side reads).
    #[must_use]
    pub fn memory(&self) -> &[u8] {
        &self.store.mem
    }

    /// The instance's linear memory, mutable (the JS side writes here before a
    /// call that consumes the data).
    #[must_use]
    pub fn memory_mut(&mut self) -> &mut [u8] {
        &mut self.store.mem
    }

    /// Reads `len` bytes of linear memory at `offset` (e.g. to pull a result
    /// buffer back to the host), or `None` if out of bounds.
    #[must_use]
    pub fn read_memory(&self, offset: usize, len: usize) -> Option<&[u8]> {
        self.store.mem.get(offset..offset.checked_add(len)?)
    }

    /// Writes `bytes` into linear memory at `offset` (e.g. to hand input to a
    /// WASM function).
    ///
    /// # Errors
    /// `WasmRtError("out of bounds memory access")` if the range doesn't fit.
    pub fn write_memory(&mut self, offset: usize, bytes: &[u8]) -> Result<(), WasmRtError> {
        let end = offset
            .checked_add(bytes.len())
            .filter(|e| *e <= self.store.mem.len())
            .ok_or(WasmRtError("out of bounds memory access"))?;
        self.store.mem[offset..end].copy_from_slice(bytes);
        Ok(())
    }
}

/// The completion of an instruction stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Flow {
    /// Fell off the end / `end`.
    Normal,
    /// A `return`.
    Return,
    /// A `br`/`br_if` targeting the block `n` levels out.
    Branch(u32),
}

/// The byte offset of an `if` body's depth-0 `else` (0x05), or `None` if the `if`
/// has no else clause. `code` is the if-body (sans the trailing `end`).
fn else_split(code: &[u8]) -> Result<Option<usize>, WasmRtError> {
    let mut r = Reader::new(code);
    let mut depth = 0i32;
    while !r.done() {
        let at = r.pos;
        let op = r.byte()?;
        match op {
            0x02..=0x04 => {
                depth += 1;
                r.byte()?; // blocktype
            }
            0x0b => depth -= 1, // end of a nested block (the body has no top `end`)
            0x05 if depth == 0 => return Ok(Some(at)),
            // Skip immediates so a literal `0x05` inside an operand isn't mistaken
            // for an `else` (mirrors `block_len`).
            0x20..=0x24 | 0x0c | 0x0d | 0x10 => {
                r.u32()?;
            }
            0x0e => {
                // br_table: count + (count+1) label depths.
                let c = r.u32()?;
                for _ in 0..=c {
                    r.u32()?;
                }
            }
            0x11 => {
                r.u32()?;
                r.u32()?;
            }
            0x28..=0x3e => {
                r.u32()?;
                r.u32()?;
            }
            0x3f | 0x40 => {
                r.byte()?;
            }
            0x41 => {
                r.i32()?;
            }
            0x42 => {
                r.i64()?;
            }
            0x43 => {
                r.bytes(4)?;
            }
            0x44 => {
                r.bytes(8)?;
            }
            0xfc => {
                r.skip_fc()?;
            }
            _ => {}
        }
    }
    Ok(None)
}

/// The byte length of a structured block's body up to and including its matching
/// `end` (0x0b), accounting for nested `block`/`loop`/`if`.
fn block_len(code: &[u8]) -> Result<usize, WasmRtError> {
    let mut r = Reader::new(code);
    let mut depth = 0i32;
    while !r.done() {
        let op = r.byte()?;
        match op {
            0x02..=0x04 => {
                depth += 1;
                r.byte()?; // blocktype
            }
            0x0b => {
                if depth == 0 {
                    return Ok(r.pos);
                }
                depth -= 1;
            }
            // Skip immediates of the ops that carry them.
            0x20 | 0x21 | 0x22 | 0x23 | 0x24 | 0x0c | 0x0d | 0x10 => {
                r.u32()?;
            }
            0x0e => {
                // br_table: count + (count+1) label depths.
                let c = r.u32()?;
                for _ in 0..=c {
                    r.u32()?;
                }
            }
            // call_indirect carries two immediates (typeidx, tableidx).
            0x11 => {
                r.u32()?;
                r.u32()?;
            }
            // Memory load/store ops carry a memarg (align + offset, two LEBs).
            0x28..=0x3e => {
                r.u32()?;
                r.u32()?;
            }
            // memory.size / memory.grow carry a reserved byte.
            0x3f | 0x40 => {
                r.byte()?;
            }
            0x41 => {
                r.i32()?;
            }
            0x42 => {
                r.i64()?;
            }
            0x43 => {
                r.bytes(4)?; // f32.const
            }
            0x44 => {
                r.bytes(8)?; // f64.const
            }
            0xfc => {
                r.skip_fc()?;
            }
            _ => {}
        }
    }
    Err(WasmRtError("unterminated block"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// `(module (func (export "add") (param i32 i32) (result i32)
    ///    local.get 0 local.get 1 i32.add))`
    fn module_add() -> Vec<u8> {
        let mut m = vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
        // type section: 1 type, (i32 i32) -> (i32)
        m.extend([0x01, 0x07, 0x01, 0x60, 0x02, 0x7f, 0x7f, 0x01, 0x7f]);
        // function section: 1 function, type 0
        m.extend([0x03, 0x02, 0x01, 0x00]);
        // export section: 1 export "add" func 0
        m.extend([0x07, 0x07, 0x01, 0x03, b'a', b'd', b'd', 0x00, 0x00]);
        // code section: 1 body: 0 locals, local.get 0, local.get 1, i32.add, end
        m.extend([
            0x0a, 0x09, 0x01, 0x07, 0x00, 0x20, 0x00, 0x20, 0x01, 0x6a, 0x0b,
        ]);
        m
    }

    #[test]
    fn decode_and_run_add() {
        let m = Module::decode(&module_add()).expect("decode");
        let f = m.export("add").expect("export add");
        let r = m.call(f, &[Val::I32(20), Val::I32(22)]).expect("call");
        assert_eq!(r, vec![Val::I32(42)]);
        assert_eq!(
            m.call(f, &[Val::I32(-5), Val::I32(8)]).unwrap(),
            vec![Val::I32(3)]
        );
    }

    #[test]
    fn arithmetic_and_const() {
        // (func (export "f") (result i32) i32.const 6 i32.const 7 i32.mul)
        let mut m = vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
        m.extend([0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7f]); // () -> i32
        m.extend([0x03, 0x02, 0x01, 0x00]);
        m.extend([0x07, 0x05, 0x01, 0x01, b'f', 0x00, 0x00]);
        // body: i32.const 6, i32.const 7, i32.mul, end
        m.extend([
            0x0a, 0x09, 0x01, 0x07, 0x00, 0x41, 0x06, 0x41, 0x07, 0x6c, 0x0b,
        ]);
        let module = Module::decode(&m).unwrap();
        let f = module.export("f").unwrap();
        assert_eq!(module.call(f, &[]).unwrap(), vec![Val::I32(42)]);
    }

    #[test]
    fn loop_countdown_sum() {
        // A hand-assembled loop computing sum(1..=n) with a local accumulator.
        //   (func (export "sum") (param i32) (result i32) (local i32)  ;; l1 = acc
        //     block
        //       loop
        //         local.get 0      ;; n
        //         i32.eqz
        //         br_if 1          ;; if n==0, exit block
        //         local.get 1
        //         local.get 0
        //         i32.add
        //         local.set 1      ;; acc += n
        //         local.get 0
        //         i32.const 1
        //         i32.sub
        //         local.set 0      ;; n -= 1
        //         br 0             ;; repeat loop
        //       end
        //     end
        //     local.get 1)
        let body: Vec<u8> = vec![
            0x01, 0x01, 0x7f, // 1 local run: 1 x i32
            0x02, 0x40, // block (empty type)
            0x03, 0x40, // loop (empty type)
            0x20, 0x00, // local.get 0
            0x45, // i32.eqz
            0x0d, 0x01, // br_if 1 (exit block)
            0x20, 0x01, 0x20, 0x00, 0x6a, 0x21, 0x01, // acc += n
            0x20, 0x00, 0x41, 0x01, 0x6b, 0x21, 0x00, // n -= 1
            0x0c, 0x00, // br 0 (repeat loop)
            0x0b, // end loop
            0x0b, // end block
            0x20, 0x01, // local.get 1 (acc)
            0x0b, // end func
        ];
        let mut m = vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
        m.extend([0x01, 0x06, 0x01, 0x60, 0x01, 0x7f, 0x01, 0x7f]); // (i32)->i32
        m.extend([0x03, 0x02, 0x01, 0x00]);
        m.extend([0x07, 0x07, 0x01, 0x03, b's', b'u', b'm', 0x00, 0x00]);
        m.push(0x0a);
        m.push((body.len() + 2) as u8); // section size: count byte + size byte + body
        m.push(0x01); // 1 body
        m.push(body.len() as u8);
        m.extend(body);
        let module = Module::decode(&m).expect("decode loop");
        let f = module.export("sum").unwrap();
        for n in [0i32, 1, 5, 10, 100] {
            assert_eq!(
                module.call(f, &[Val::I32(n)]).unwrap(),
                vec![Val::I32(n * (n + 1) / 2)],
                "sum 1..={n}"
            );
        }
    }

    #[test]
    fn memory_store_then_load() {
        // (memory 1)
        // (func (export "f") (param i32) (result i32)
        //   i32.const 0  local.get 0  i32.store          ;; mem[0] = arg
        //   i32.const 0  i32.load                          ;; return mem[0]
        //   i32.const 4  i32.const 99  i32.store           ;; mem[4] = 99
        //   i32.const 4  i32.load  i32.add)                ;; + mem[4]
        // memarg = align(0x02 for i32) + offset(0x00)
        let body: Vec<u8> = vec![
            0x00, // 0 local runs
            0x41, 0x00, 0x20, 0x00, 0x36, 0x02, 0x00, // i32.store mem[0]=arg
            0x41, 0x00, 0x28, 0x02, 0x00, // i32.load mem[0]
            // i32.const 99 needs 2 LEB bytes (0x63 alone is -29: sign bit set).
            0x41, 0x04, 0x41, 0xe3, 0x00, 0x36, 0x02, 0x00, // i32.store mem[4]=99
            0x41, 0x04, 0x28, 0x02, 0x00, // i32.load mem[4]
            0x6a, // i32.add
            0x0b, // end
        ];
        let mut m = vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
        m.extend([0x01, 0x06, 0x01, 0x60, 0x01, 0x7f, 0x01, 0x7f]); // (i32)->i32
        m.extend([0x03, 0x02, 0x01, 0x00]);
        m.extend([0x05, 0x03, 0x01, 0x00, 0x01]); // memory: 1 mem, min 1 page
        m.extend([0x07, 0x05, 0x01, 0x01, b'f', 0x00, 0x00]);
        m.push(0x0a);
        m.push((body.len() + 2) as u8);
        m.push(0x01);
        m.push(body.len() as u8);
        m.extend(body);
        let module = Module::decode(&m).expect("decode mem module");
        let f = module.export("f").unwrap();
        // f(arg) = arg + 99
        assert_eq!(module.call(f, &[Val::I32(1)]).unwrap(), vec![Val::I32(100)]);
        assert_eq!(
            module.call(f, &[Val::I32(-50)]).unwrap(),
            vec![Val::I32(49)]
        );
    }

    #[test]
    fn data_segment_and_load8() {
        // (memory 1) (data (i32.const 2) "\07\09")
        // (func (export "g") (result i32)
        //   i32.const 2 i32.load8_u  i32.const 3 i32.load8_u  i32.add)  ;; 7 + 9
        let body: Vec<u8> = vec![
            0x00, 0x41, 0x02, 0x2d, 0x00, 0x00, // i32.load8_u mem[2]
            0x41, 0x03, 0x2d, 0x00, 0x00, // i32.load8_u mem[3]
            0x6a, 0x0b,
        ];
        let mut m = vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
        m.extend([0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7f]); // ()->i32
        m.extend([0x03, 0x02, 0x01, 0x00]);
        m.extend([0x05, 0x03, 0x01, 0x00, 0x01]); // memory min 1
        m.extend([0x07, 0x05, 0x01, 0x01, b'g', 0x00, 0x00]);
        m.push(0x0a);
        m.push((body.len() + 2) as u8);
        m.push(0x01);
        m.push(body.len() as u8);
        m.extend(body);
        // data section: 1 segment, mode 0, offset (i32.const 2; end), 2 bytes 07 09
        m.extend([0x0b, 0x08, 0x01, 0x00, 0x41, 0x02, 0x0b, 0x02, 0x07, 0x09]);
        let module = Module::decode(&m).expect("decode data module");
        let g = module.export("g").unwrap();
        assert_eq!(module.call(g, &[]).unwrap(), vec![Val::I32(16)]); // 7 + 9
    }

    #[test]
    fn mutable_global_accumulates() {
        // (global $g (mut i32) (i32.const 100))
        // (func (export "add") (param i32) (result i32)
        //   global.get 0  local.get 0  i32.add  global.set 0   ;; g += arg
        //   global.get 0)                                        ;; return g
        let body: Vec<u8> = vec![
            0x00, // 0 locals
            0x23, 0x00, 0x20, 0x00, 0x6a, 0x24, 0x00, // g = g + arg
            0x23, 0x00, // global.get 0
            0x0b,
        ];
        let mut m = vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
        m.extend([0x01, 0x06, 0x01, 0x60, 0x01, 0x7f, 0x01, 0x7f]); // (i32)->i32
        m.extend([0x03, 0x02, 0x01, 0x00]);
        // global section: 1 global, i32, mutable(1), init i32.const 100, end
        m.extend([0x06, 0x07, 0x01, 0x7f, 0x01, 0x41, 0xe4, 0x00, 0x0b]);
        m.extend([0x07, 0x07, 0x01, 0x03, b'a', b'd', b'd', 0x00, 0x00]);
        m.push(0x0a);
        m.push((body.len() + 2) as u8);
        m.push(0x01);
        m.push(body.len() as u8);
        m.extend(body);
        let module = Module::decode(&m).expect("decode global module");
        let f = module.export("add").unwrap();
        // Each call starts from a fresh instance: g initialized to 100.
        assert_eq!(module.call(f, &[Val::I32(5)]).unwrap(), vec![Val::I32(105)]);
        assert_eq!(
            module.call(f, &[Val::I32(-30)]).unwrap(),
            vec![Val::I32(70)]
        );
    }

    #[test]
    fn immutable_global_set_traps() {
        // (global i32 (i32.const 7)) immutable; func sets it -> trap.
        let body: Vec<u8> = vec![0x00, 0x41, 0x01, 0x24, 0x00, 0x41, 0x00, 0x0b];
        let mut m = vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
        m.extend([0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7f]);
        m.extend([0x03, 0x02, 0x01, 0x00]);
        m.extend([0x06, 0x06, 0x01, 0x7f, 0x00, 0x41, 0x07, 0x0b]); // immutable global
        m.extend([0x07, 0x05, 0x01, 0x01, b'f', 0x00, 0x00]);
        m.push(0x0a);
        m.push((body.len() + 2) as u8);
        m.push(0x01);
        m.push(body.len() as u8);
        m.extend(body);
        let module = Module::decode(&m).unwrap();
        let f = module.export("f").unwrap();
        assert_eq!(
            module.call(f, &[]),
            Err(WasmRtError("set of immutable/undefined global"))
        );
    }

    #[test]
    fn memory_out_of_bounds_traps() {
        // (memory 1) (func (export "h") (result i32) i32.const 100000 i32.load)
        // address 100000 is past one 64 KiB page.
        let body: Vec<u8> = vec![0x00, 0x41, 0xa0, 0x8d, 0x06, 0x28, 0x02, 0x00, 0x0b];
        let mut m = vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
        m.extend([0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7f]);
        m.extend([0x03, 0x02, 0x01, 0x00]);
        m.extend([0x05, 0x03, 0x01, 0x00, 0x01]);
        m.extend([0x07, 0x05, 0x01, 0x01, b'h', 0x00, 0x00]);
        m.push(0x0a);
        m.push((body.len() + 2) as u8);
        m.push(0x01);
        m.push(body.len() as u8);
        m.extend(body);
        let module = Module::decode(&m).unwrap();
        let h = module.export("h").unwrap();
        assert_eq!(
            module.call(h, &[]),
            Err(WasmRtError("out of bounds memory access"))
        );
    }

    #[test]
    fn f64_arithmetic_and_sqrt() {
        // (func (export "f") (param f64 f64) (result f64)
        //   local.get 0  local.get 1  f64.add  f64.sqrt)   ;; sqrt(a + b)
        let body: Vec<u8> = vec![0x00, 0x20, 0x00, 0x20, 0x01, 0xa0, 0x9f, 0x0b];
        let mut m = vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
        m.extend([0x01, 0x07, 0x01, 0x60, 0x02, 0x7c, 0x7c, 0x01, 0x7c]); // (f64 f64)->f64
        m.extend([0x03, 0x02, 0x01, 0x00]);
        m.extend([0x07, 0x05, 0x01, 0x01, b'f', 0x00, 0x00]);
        m.push(0x0a);
        m.push((body.len() + 2) as u8);
        m.push(0x01);
        m.push(body.len() as u8);
        m.extend(body);
        let module = Module::decode(&m).expect("decode f64 module");
        let f = module.export("f").unwrap();
        // sqrt(9 + 16) = 5
        let r = module.call(f, &[Val::F64(9.0), Val::F64(16.0)]).unwrap();
        match r[..] {
            [Val::F64(v)] => assert!((v - 5.0).abs() < 1e-9, "got {v}"),
            _ => panic!("expected one f64, got {r:?}"),
        }
        // sqrt(2) ≈ 1.4142135623730951
        let r = module.call(f, &[Val::F64(2.0), Val::F64(0.0)]).unwrap();
        match r[..] {
            [Val::F64(v)] => assert!((v - core::f64::consts::SQRT_2).abs() < 1e-12, "got {v}"),
            _ => panic!("expected one f64"),
        }
    }

    #[test]
    fn f64_const_and_compare() {
        // (func (export "lt") (param f64) (result i32)
        //   local.get 0  f64.const 3.5  f64.lt)   ;; arg < 3.5
        let mut body: Vec<u8> = vec![0x00, 0x20, 0x00, 0x44];
        body.extend(3.5f64.to_le_bytes());
        body.extend([0x63, 0x0b]); // f64.lt, end
        let mut m = vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
        m.extend([0x01, 0x06, 0x01, 0x60, 0x01, 0x7c, 0x01, 0x7f]); // (f64)->i32
        m.extend([0x03, 0x02, 0x01, 0x00]);
        m.extend([0x07, 0x06, 0x01, 0x02, b'l', b't', 0x00, 0x00]);
        m.push(0x0a);
        m.push((body.len() + 2) as u8);
        m.push(0x01);
        m.push(body.len() as u8);
        m.extend(body);
        let module = Module::decode(&m).expect("decode compare module");
        let lt = module.export("lt").unwrap();
        assert_eq!(
            module.call(lt, &[Val::F64(2.0)]).unwrap(),
            vec![Val::I32(1)]
        );
        assert_eq!(
            module.call(lt, &[Val::F64(9.0)]).unwrap(),
            vec![Val::I32(0)]
        );
    }

    #[test]
    fn float_helpers_match_semantics() {
        assert_eq!(f64_sqrt(144.0), 12.0);
        assert_eq!(f64_sqrt(0.0), 0.0);
        assert!(f64_sqrt(-1.0).is_nan());
        assert_eq!(f64_abs(-3.5), 3.5);
        assert_eq!(f64_min(-0.0, 0.0), -0.0);
        assert_eq!(f64_max(-0.0, 0.0), 0.0);
        assert!(f64_min(f64::NAN, 1.0).is_nan());
        assert_eq!(f64_min(2.0, 5.0), 2.0);
    }

    #[test]
    fn call_indirect_dispatches_through_table() {
        // Two functions of type (i32,i32)->i32: add (func 0) and sub (func 1).
        // A dispatcher (func 2, type (i32,i32,i32)->i32) calls table[arg2](a,b).
        //   (table 2 funcref) (elem (i32.const 0) 0 1)
        //   func 2: local.get 0  local.get 1  local.get 2  call_indirect (type 0)
        let mut m = vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
        // types: 0 = (i32 i32)->i32, 1 = (i32 i32 i32)->i32
        m.extend([
            0x01, 0x0e, 0x02, // section, size, 2 types
            0x60, 0x02, 0x7f, 0x7f, 0x01, 0x7f, // (i32 i32)->i32
            0x60, 0x03, 0x7f, 0x7f, 0x7f, 0x01, 0x7f, // (i32 i32 i32)->i32
        ]);
        // functions: func0:type0, func1:type0, func2:type1
        m.extend([0x03, 0x04, 0x03, 0x00, 0x00, 0x01]);
        // table: 1 table, funcref(0x70), limits min 2
        m.extend([0x04, 0x04, 0x01, 0x70, 0x00, 0x02]);
        // export "dispatch" = func 2
        m.extend([
            0x07, 0x0c, 0x01, 0x08, b'd', b'i', b's', b'p', b'a', b't', b'c', b'h', 0x00, 0x02,
        ]);
        // element segment: active table 0, offset i32.const 0, funcs [0,1]
        m.extend([0x09, 0x08, 0x01, 0x00, 0x41, 0x00, 0x0b, 0x02, 0x00, 0x01]);
        // code: 3 bodies
        let add_body = [0x00, 0x20, 0x00, 0x20, 0x01, 0x6a, 0x0b]; // a+b
        let sub_body = [0x00, 0x20, 0x00, 0x20, 0x01, 0x6b, 0x0b]; // a-b
        // dispatcher: local.get 0,1,2 ; call_indirect type 0 table 0
        let disp_body = [
            0x00, 0x20, 0x00, 0x20, 0x01, 0x20, 0x02, 0x11, 0x00, 0x00, 0x0b,
        ];
        let mut code = vec![0x0a, 0x00, 0x03]; // section id, size placeholder, 3 bodies
        for b in [&add_body[..], &sub_body[..], &disp_body[..]] {
            code.push(b.len() as u8);
            code.extend_from_slice(b);
        }
        code[1] = (code.len() - 2) as u8; // patch section size
        m.extend(code);

        let module = Module::decode(&m).expect("decode call_indirect module");
        let d = module.export("dispatch").unwrap();
        // dispatch(10, 3, 0) -> add -> 13 ; dispatch(10, 3, 1) -> sub -> 7
        assert_eq!(
            module
                .call(d, &[Val::I32(10), Val::I32(3), Val::I32(0)])
                .unwrap(),
            vec![Val::I32(13)]
        );
        assert_eq!(
            module
                .call(d, &[Val::I32(10), Val::I32(3), Val::I32(1)])
                .unwrap(),
            vec![Val::I32(7)]
        );
        // Out-of-range table index traps.
        assert_eq!(
            module.call(d, &[Val::I32(1), Val::I32(1), Val::I32(5)]),
            Err(WasmRtError("undefined element (call_indirect)"))
        );
    }

    #[test]
    fn instance_memory_persists_and_is_host_accessible() {
        // (memory 1)
        // (func (export "store") (param i32 i32) i32.store)   ;; mem[p0] = p1
        // (func (export "load") (param i32) (result i32) local.get 0 i32.load)
        let store_body: Vec<u8> = vec![0x00, 0x20, 0x00, 0x20, 0x01, 0x36, 0x02, 0x00, 0x0b];
        let load_body: Vec<u8> = vec![0x00, 0x20, 0x00, 0x28, 0x02, 0x00, 0x0b];
        let mut m = vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
        // types: 0 = (i32 i32)->(), 1 = (i32)->i32
        m.extend([
            0x01, 0x0b, 0x02, 0x60, 0x02, 0x7f, 0x7f, 0x00, 0x60, 0x01, 0x7f, 0x01, 0x7f,
        ]);
        m.extend([0x03, 0x03, 0x02, 0x00, 0x01]); // funcs: f0:type0, f1:type1
        m.extend([0x05, 0x03, 0x01, 0x00, 0x01]); // memory min 1
        m.extend([
            0x07, 0x10, 0x02, // exports: 2
            0x05, b's', b't', b'o', b'r', b'e', 0x00, 0x00, // "store" func0
            0x04, b'l', b'o', b'a', b'd', 0x00, 0x01, // "load" func1
        ]);
        let mut code = vec![0x0a, 0x00, 0x02];
        for b in [&store_body[..], &load_body[..]] {
            code.push(b.len() as u8);
            code.extend_from_slice(b);
        }
        code[1] = (code.len() - 2) as u8;
        m.extend(code);

        let module = Module::decode(&m).expect("decode");
        let mut inst = Instance::new(&module).expect("instantiate");

        // A WASM store is visible to a later WASM load (memory persists).
        inst.call_export("store", &[Val::I32(16), Val::I32(12345)])
            .unwrap();
        assert_eq!(
            inst.call_export("load", &[Val::I32(16)]).unwrap(),
            vec![Val::I32(12345)]
        );
        // ...and visible to the host, which reads the raw little-endian bytes.
        assert_eq!(inst.read_memory(16, 4).unwrap(), &12345i32.to_le_bytes());

        // The host writes input into memory; WASM reads it back.
        inst.write_memory(32, &999i32.to_le_bytes()).unwrap();
        assert_eq!(
            inst.call_export("load", &[Val::I32(32)]).unwrap(),
            vec![Val::I32(999)]
        );
        // Out-of-bounds host writes are rejected.
        assert!(inst.write_memory(PAGE_SIZE, &[1, 2, 3, 4]).is_err());
        assert!(inst.read_memory(PAGE_SIZE, 4).is_none());
    }

    /// End-to-end: compile a JS numeric function to a WASM binary with the
    /// `wasm` compiler, then decode and run it with this engine — the two WASM
    /// halves (JS→WASM lowering and the execution engine) meeting in the middle.
    #[test]
    fn runs_wasm_compiled_from_js() {
        use crate::nanbox::{NanBox, Unpacked};
        for (src, name, args, expect) in [
            (
                "function add(a, b){ return a + b; }",
                "add",
                alloc::vec![3.0, 4.0],
                7.0,
            ),
            (
                "function poly(x){ return x * x - 2 * x + 1; }",
                "poly",
                alloc::vec![5.0],
                16.0,
            ),
            (
                "function avg(a, b){ return (a + b) / 2; }",
                "avg",
                alloc::vec![3.0, 10.0],
                6.5,
            ),
        ] {
            let program = crate::parser::Parser::parse_program(src).expect("parse");
            let binary = crate::wasm::compile_module_binary(&program).expect("compile to wasm");
            let module = Module::decode(&binary).expect("decode compiled wasm");
            let mut inst = Instance::new(&module).expect("instantiate");
            // Invoke through the JS-value boundary with NanBox numbers.
            let js_args: Vec<NanBox> = args.iter().map(|a| NanBox::number(*a)).collect();
            let out = inst.call_export_js(name, &js_args).expect("call");
            assert_eq!(out.len(), 1);
            match out[0].unpack() {
                Unpacked::Number(v) => assert!((v - expect).abs() < 1e-12, "{name}: {v}"),
                _ => panic!("expected a number"),
            }
        }
    }

    #[test]
    fn js_value_marshaling_across_the_boundary() {
        use crate::nanbox::{NanBox, Unpacked};
        // Value conversions both ways.
        assert_eq!(Val::I32(42).to_nanbox().unpack(), Unpacked::Number(42.0));
        assert_eq!(Val::F64(3.5).to_nanbox().unpack(), Unpacked::Number(3.5));
        assert_eq!(
            Val::from_nanbox(NanBox::number(7.9), ValType::I32),
            Some(Val::I32(7))
        );
        assert_eq!(
            Val::from_nanbox(NanBox::number(2.5), ValType::F64),
            Some(Val::F64(2.5))
        );
        assert_eq!(
            Val::from_nanbox(NanBox::boolean(true), ValType::I32),
            Some(Val::I32(1))
        );
        // A non-numeric JS value (null) is not coercible.
        assert_eq!(Val::from_nanbox(NanBox::null(), ValType::I32), None);

        // call_export_js: a WASM (i32,i32)->i32 add, invoked with JS numbers.
        let body: Vec<u8> = vec![0x00, 0x20, 0x00, 0x20, 0x01, 0x6a, 0x0b];
        let mut m = vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
        m.extend([0x01, 0x07, 0x01, 0x60, 0x02, 0x7f, 0x7f, 0x01, 0x7f]);
        m.extend([0x03, 0x02, 0x01, 0x00]);
        m.extend([0x07, 0x07, 0x01, 0x03, b'a', b'd', b'd', 0x00, 0x00]);
        m.push(0x0a);
        m.push((body.len() + 2) as u8);
        m.push(0x01);
        m.push(body.len() as u8);
        m.extend(body);
        let module = Module::decode(&m).unwrap();
        let mut inst = Instance::new(&module).unwrap();
        let out = inst
            .call_export_js("add", &[NanBox::number(20.0), NanBox::number(22.0)])
            .unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].unpack(), Unpacked::Number(42.0));
        // A non-numeric JS argument is rejected at the boundary.
        assert!(
            inst.call_export_js("add", &[NanBox::null(), NanBox::number(1.0)])
                .is_err()
        );
        // Wrong arity is rejected.
        assert!(inst.call_export_js("add", &[NanBox::number(1.0)]).is_err());
    }

    #[test]
    fn host_function_import_is_callable_from_wasm() {
        // (import "env" "triple" (func (type 0)))           ;; func 0 (i32)->i32
        // (func (export "run") (param i32) (result i32)
        //   local.get 0 call 0  i32.const 1 i32.add)         ;; triple(p0) + 1
        let mut m = vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
        m.extend([0x01, 0x06, 0x01, 0x60, 0x01, 0x7f, 0x01, 0x7f]); // type0 (i32)->i32
        // import section: 1 import, "env"."triple", func type 0
        m.extend([
            0x02, 0x0e, 0x01, 0x03, b'e', b'n', b'v', 0x06, b't', b'r', b'i', b'p', b'l', b'e',
            0x00, 0x00,
        ]);
        // function section: 1 defined func, type 0 (this is func index 1)
        m.extend([0x03, 0x02, 0x01, 0x00]);
        // export "run" = func 1 (the defined function)
        m.extend([0x07, 0x07, 0x01, 0x03, b'r', b'u', b'n', 0x00, 0x01]);
        // code: body of func 1: local.get 0, call 0 (import), i32.const 1, i32.add, end
        let body: Vec<u8> = vec![0x00, 0x20, 0x00, 0x10, 0x00, 0x41, 0x01, 0x6a, 0x0b];
        m.push(0x0a);
        m.push((body.len() + 2) as u8);
        m.push(0x01);
        m.push(body.len() as u8);
        m.extend(body);

        let module = Module::decode(&m).expect("decode import module");
        assert_eq!(module.import_names(), vec![("env", "triple")]);

        // Supply the host function (here a Rust closure; in the engine it bridges
        // to a JS function): triple(x) = x * 3.
        let host: HostFunc = alloc::boxed::Box::new(|args: &[Val]| {
            let x = args[0].as_i32()?;
            Ok(vec![Val::I32(x.wrapping_mul(3))])
        });
        let mut inst = Instance::with_imports(&module, vec![host]).expect("instantiate");
        // run(10) = triple(10) + 1 = 31.
        assert_eq!(
            inst.call_export("run", &[Val::I32(10)]).unwrap(),
            vec![Val::I32(31)]
        );
        assert_eq!(
            inst.call_export("run", &[Val::I32(-2)]).unwrap(),
            vec![Val::I32(-5)]
        );

        // A missing import binding is rejected at instantiation.
        assert!(Instance::with_imports(&module, vec![]).is_err());
    }

    /// Helper: a single-result `(i32,i32)->i32` module whose body is `local.get 0
    /// local.get 1 <opcode bytes>`.
    fn binop_i32_module(opcode: &[u8]) -> Vec<u8> {
        let mut body: Vec<u8> = vec![0x00, 0x20, 0x00, 0x20, 0x01];
        body.extend_from_slice(opcode);
        body.push(0x0b);
        let mut m = vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
        m.extend([0x01, 0x07, 0x01, 0x60, 0x02, 0x7f, 0x7f, 0x01, 0x7f]);
        m.extend([0x03, 0x02, 0x01, 0x00]);
        m.extend([0x07, 0x05, 0x01, 0x01, b'f', 0x00, 0x00]);
        m.push(0x0a);
        m.push((body.len() + 2) as u8);
        m.push(0x01);
        m.push(body.len() as u8);
        m.extend(body);
        m
    }

    #[test]
    fn extended_integer_ops() {
        let run = |opcode: &[u8], a: i32, b: i32| -> Val {
            let m = binop_i32_module(opcode);
            Module::decode(&m)
                .unwrap()
                .call(0, &[Val::I32(a), Val::I32(b)])
                .unwrap()[0]
        };
        // Unsigned compare: -1 (0xFFFFFFFF) is the largest u32.
        assert_eq!(run(&[0x49], -1, 1), Val::I32(0)); // lt_u: -1 < 1 ? no (huge)
        assert_eq!(run(&[0x4b], -1, 1), Val::I32(1)); // gt_u: yes
        // div_u / rem_u.
        assert_eq!(run(&[0x6e], -2, 3), Val::I32((((-2i32) as u32) / 3) as i32)); // div_u
        assert_eq!(run(&[0x70], 17, 5), Val::I32(2)); // rem_u
        assert_eq!(run(&[0x6f], -17, 5), Val::I32(-2)); // rem_s
        // shr_u vs shr_s.
        assert_eq!(
            run(&[0x76], -8, 1),
            Val::I32((((-8i32) as u32) >> 1) as i32)
        ); // shr_u
        // rotl / rotr.
        assert_eq!(
            run(&[0x77], 0x1234_5678, 8),
            Val::I32(0x1234_5678i32.rotate_left(8))
        );
        assert_eq!(
            run(&[0x78], 0x1234_5678, 4),
            Val::I32(0x1234_5678i32.rotate_right(4))
        );

        // Unary clz/ctz/popcnt via a 1-arg module.
        let unary = |opcode: u8, x: i32| -> i32 {
            let mut body: Vec<u8> = vec![0x00, 0x20, 0x00, opcode, 0x0b];
            let mut m = vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
            m.extend([0x01, 0x06, 0x01, 0x60, 0x01, 0x7f, 0x01, 0x7f]);
            m.extend([0x03, 0x02, 0x01, 0x00]);
            m.extend([0x07, 0x05, 0x01, 0x01, b'f', 0x00, 0x00]);
            m.push(0x0a);
            m.push((body.len() + 2) as u8);
            m.push(0x01);
            m.push(body.len() as u8);
            m.append(&mut body);
            match Module::decode(&m).unwrap().call(0, &[Val::I32(x)]).unwrap()[0] {
                Val::I32(v) => v,
                _ => panic!(),
            }
        };
        assert_eq!(unary(0x67, 1), 31); // clz(1)
        assert_eq!(unary(0x68, 8), 3); // ctz(8)
        assert_eq!(unary(0x69, 0xff), 8); // popcnt(0xff)
    }

    #[test]
    fn i64_ops_and_conversions() {
        // (func (export "f") (param i32) (result i32)
        //   local.get 0 i64.extend_i32_s   ;; i64
        //   i64.const 1000000000000 i64.add ;; + 1e12 (needs i64)
        //   i32.wrap_i64)                    ;; back to i32 (wraps)
        let mut body: Vec<u8> = vec![0x00, 0x20, 0x00, 0xac]; // local.get0; i64.extend_i32_s
        body.push(0x42); // i64.const
        // 1000000000000 = 0xE8D4A51000; LEB128 signed:
        body.extend([0x80, 0xa0, 0x94, 0xa5, 0x8d, 0x1d]);
        body.extend([0x7c, 0xa7, 0x0b]); // i64.add; i32.wrap_i64; end
        let mut m = vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
        m.extend([0x01, 0x06, 0x01, 0x60, 0x01, 0x7f, 0x01, 0x7f]);
        m.extend([0x03, 0x02, 0x01, 0x00]);
        m.extend([0x07, 0x05, 0x01, 0x01, b'f', 0x00, 0x00]);
        m.push(0x0a);
        m.push((body.len() + 2) as u8);
        m.push(0x01);
        m.push(body.len() as u8);
        m.extend(body);
        let module = Module::decode(&m).expect("decode i64 module");
        // f(5) = wrap(5 + 1e12). 1e12 mod 2^32 = 1000000000000 & 0xFFFFFFFF.
        let expect = (5i64 + 1_000_000_000_000i64) as i32;
        assert_eq!(
            module.call(0, &[Val::I32(5)]).unwrap(),
            vec![Val::I32(expect)]
        );
    }

    #[test]
    fn br_table_computed_branch() {
        // (func (export "sw") (param i32) (result i32)
        //   (block (block (block
        //     local.get 0
        //     br_table 0 1 2 2      ;; 0->b0, 1->b1, else->b2(default)
        //   ) ;; b0 reached → return 10
        //     i32.const 10 return)
        //   ;; b1 → return 20
        //     i32.const 20 return)
        //   ;; b2/default → return 30
        //   i32.const 30 return)
        let body: Vec<u8> = vec![
            0x00, // 0 locals
            0x02, 0x40, // block (outer, depth target for default)
            0x02, 0x40, // block
            0x02, 0x40, // block (innermost)
            0x20, 0x00, // local.get 0
            0x0e, 0x02, 0x00, 0x01, 0x02, // br_table {0,1} default 2
            0x0b, // end innermost  -> falls here for index 0
            0x41, 0x0a, 0x0f, // i32.const 10; return
            0x0b, // end middle     -> index 1
            0x41, 0x14, 0x0f, // i32.const 20; return
            0x0b, // end outer      -> default
            0x41, 0x1e, 0x0f, // i32.const 30; return
            0x0b, // end func
        ];
        let mut m = vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
        m.extend([0x01, 0x06, 0x01, 0x60, 0x01, 0x7f, 0x01, 0x7f]);
        m.extend([0x03, 0x02, 0x01, 0x00]);
        m.extend([0x07, 0x06, 0x01, 0x02, b's', b'w', 0x00, 0x00]);
        m.push(0x0a);
        m.push((body.len() + 2) as u8);
        m.push(0x01);
        m.push(body.len() as u8);
        m.extend(body);
        let module = Module::decode(&m).expect("decode br_table module");
        assert_eq!(module.call(0, &[Val::I32(0)]).unwrap(), vec![Val::I32(10)]);
        assert_eq!(module.call(0, &[Val::I32(1)]).unwrap(), vec![Val::I32(20)]);
        assert_eq!(module.call(0, &[Val::I32(2)]).unwrap(), vec![Val::I32(30)]); // default
        assert_eq!(module.call(0, &[Val::I32(99)]).unwrap(), vec![Val::I32(30)]); // out-of-range → default
    }

    #[test]
    fn if_else_control_flow() {
        // (func (export "abs") (param i32) (result i32)
        //   local.get 0 i32.const 0 i32.lt_s   ;; n < 0 ?
        //   if (result i32)
        //     i32.const 0 local.get 0 i32.sub   ;; -n
        //   else
        //     local.get 0                       ;; n
        //   end)
        let body: Vec<u8> = vec![
            0x00, 0x20, 0x00, 0x41, 0x00, 0x48, // n < 0
            0x04, 0x7f, // if (result i32)
            0x41, 0x00, 0x20, 0x00, 0x6b, // then: 0 - n
            0x05, // else
            0x20, 0x00, // n
            0x0b, // end if
            0x0b, // end func
        ];
        let mut m = vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
        m.extend([0x01, 0x06, 0x01, 0x60, 0x01, 0x7f, 0x01, 0x7f]); // (i32)->i32
        m.extend([0x03, 0x02, 0x01, 0x00]);
        m.extend([0x07, 0x07, 0x01, 0x03, b'a', b'b', b's', 0x00, 0x00]);
        m.push(0x0a);
        m.push((body.len() + 2) as u8);
        m.push(0x01);
        m.push(body.len() as u8);
        m.extend(body);
        let module = Module::decode(&m).expect("decode if/else module");
        assert_eq!(module.call(0, &[Val::I32(-7)]).unwrap(), vec![Val::I32(7)]);
        assert_eq!(module.call(0, &[Val::I32(5)]).unwrap(), vec![Val::I32(5)]);
        assert_eq!(module.call(0, &[Val::I32(0)]).unwrap(), vec![Val::I32(0)]);

        // An `if` with no else (the then-branch only runs when true).
        // (func (export "clampneg") (param i32) (result i32)
        //   local.get 0 local.get 0 i32.const 0 i32.lt_s
        //   if  drop i32.const 0  end)   ;; if n<0 { return 0 } else keep n
        // Simpler: push n; if (n<0) replace... use a local. Keep it minimal:
        // (func (export "nz") (param i32) (result i32) (local i32)
        //   local.get 0 local.set 1
        //   local.get 0 if  i32.const 1 local.set 1  end
        //   local.get 1)
        let body2: Vec<u8> = vec![
            0x01, 0x01, 0x7f, // 1 local: i32
            0x20, 0x00, 0x21, 0x01, // l1 = arg
            0x20, 0x00, // arg (cond)
            0x04, 0x40, // if (no result)
            0x41, 0x01, 0x21, 0x01, // l1 = 1
            0x0b, // end if
            0x20, 0x01, // l1
            0x0b, // end func
        ];
        let mut m2 = vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
        m2.extend([0x01, 0x06, 0x01, 0x60, 0x01, 0x7f, 0x01, 0x7f]);
        m2.extend([0x03, 0x02, 0x01, 0x00]);
        m2.extend([0x07, 0x06, 0x01, 0x02, b'n', b'z', 0x00, 0x00]);
        m2.push(0x0a);
        m2.push((body2.len() + 2) as u8);
        m2.push(0x01);
        m2.push(body2.len() as u8);
        m2.extend(body2);
        let module2 = Module::decode(&m2).expect("decode if-no-else module");
        assert_eq!(module2.call(0, &[Val::I32(0)]).unwrap(), vec![Val::I32(0)]); // cond false
        assert_eq!(module2.call(0, &[Val::I32(9)]).unwrap(), vec![Val::I32(1)]); // cond true
    }

    #[test]
    fn type_validation_rejects_ill_typed_bodies() {
        // Helper: a (i32)->i32 function with the given body bytes (no trailing end).
        let module = |tail: &[u8]| -> Vec<u8> {
            let mut body = vec![0x00]; // 0 locals
            body.extend_from_slice(tail);
            body.push(0x0b);
            let mut m = vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
            m.extend([0x01, 0x06, 0x01, 0x60, 0x01, 0x7f, 0x01, 0x7f]); // (i32)->i32
            m.extend([0x03, 0x02, 0x01, 0x00]);
            m.extend([0x07, 0x05, 0x01, 0x01, b'f', 0x00, 0x00]);
            m.push(0x0a);
            m.push((body.len() + 2) as u8);
            m.push(0x01);
            m.push(body.len() as u8);
            m.extend(body);
            m
        };
        // Well-typed: local.get 0; i32.const 1; i32.add  → i32.
        assert!(Module::decode(&module(&[0x20, 0x00, 0x41, 0x01, 0x6a])).is_ok());
        // Type mismatch: i64.const 1; i32.add (i32.add wants i32 operands).
        assert!(
            Module::decode(&module(&[0x20, 0x00, 0x42, 0x01, 0x6a])).is_err(),
            "i32.add on an i64 operand must be rejected"
        );
        // Wrong result type: the body leaves an f64 but the function returns i32.
        // (local.get 0; f64.convert_i32_s)
        assert!(
            Module::decode(&module(&[0x20, 0x00, 0xb7])).is_err(),
            "returning f64 from an i32 function must be rejected"
        );
        // Stack underflow: i32.add with only one operand.
        assert!(
            Module::decode(&module(&[0x20, 0x00, 0x6a])).is_err(),
            "i32.add with one operand must be rejected"
        );
        // 0xfc trunc_sat is type-checked: i32.trunc_sat_f32_s (0xfc 0x00) wants an
        // f32 operand, so feeding it the i32 param must be rejected.
        assert!(
            Module::decode(&module(&[0x20, 0x00, 0xfc, 0x00])).is_err(),
            "i32.trunc_sat_f32_s on an i32 operand must be rejected"
        );
        // Well-typed trunc_sat: f32.convert then trunc_sat round-trips to i32.
        // (local.get 0; f32.convert_i32_s (0xb2); i32.trunc_sat_f32_s (0xfc 0x00))
        assert!(
            Module::decode(&module(&[0x20, 0x00, 0xb2, 0xfc, 0x00])).is_ok(),
            "f32→i32 saturating round-trip is well-typed"
        );

        // Control-flow aware: an `if (result i32)` whose then-branch yields i64.
        // local.get 0; if (result i32) { i64.const 1 } else { i32.const 0 } end
        // The then-arm leaves an i64 where the block declares i32 → rejected.
        assert!(
            Module::decode(&module(&[
                0x20, 0x00, // local.get 0 (condition)
                0x04, 0x7f, // if (result i32)
                0x42, 0x01, // i64.const 1   <-- wrong type
                0x05, // else
                0x41, 0x00, // i32.const 0
                0x0b, // end if
            ]))
            .is_err(),
            "if-result type mismatch must be rejected"
        );

        // A well-typed block: (block (result i32) i32.const 7) is accepted.
        assert!(
            Module::decode(&module(&[0x02, 0x7f, 0x41, 0x07, 0x0b])).is_ok(),
            "well-typed block must be accepted"
        );

        // After `unreachable`, the stack is polymorphic: `unreachable i32.add` is
        // valid (the add's operands come from the polymorphic stack).
        assert!(
            Module::decode(&module(&[0x00, 0x6a])).is_ok(),
            "unreachable makes the rest of the block polymorphic"
        );

        // f32 typing: `f32.add` (0x92) on i32 operands is rejected.
        // local.get 0; local.get 0; f32.add  → wrong (operands are i32).
        assert!(
            Module::decode(&module(&[0x20, 0x00, 0x20, 0x00, 0x92])).is_err(),
            "f32.add on i32 operands must be rejected"
        );
    }

    #[test]
    fn validate_rejects_bad_cross_references() {
        // A valid baseline: (func (export "f") (result i32) i32.const 1).
        let valid = || -> Vec<u8> {
            let mut m = vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
            m.extend([0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7f]); // type ()->i32
            m.extend([0x03, 0x02, 0x01, 0x00]); // func0: type0
            m.extend([0x07, 0x05, 0x01, 0x01, b'f', 0x00, 0x00]); // export f=func0
            m.extend([0x0a, 0x06, 0x01, 0x04, 0x00, 0x41, 0x01, 0x0b]); // code
            m
        };
        assert!(Module::decode(&valid()).is_ok(), "baseline must decode");

        // Export referencing a non-existent function index (5).
        let mut bad_export = valid();
        let n = bad_export.len();
        bad_export[n - 9] = 0x05; // the export's function index byte → 5
        assert!(
            Module::decode(&bad_export).is_err(),
            "bad export index rejected"
        );

        // Function section declaring 2 functions but only 1 code body.
        let mut m = vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
        m.extend([0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7f]);
        m.extend([0x03, 0x03, 0x02, 0x00, 0x00]); // 2 funcs, both type0
        m.extend([0x0a, 0x06, 0x01, 0x04, 0x00, 0x41, 0x01, 0x0b]); // only 1 body
        assert!(
            Module::decode(&m).is_err(),
            "function/code count mismatch rejected"
        );

        // A function declaring type index 9 (only 1 type exists).
        let mut m2 = vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
        m2.extend([0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7f]);
        m2.extend([0x03, 0x02, 0x01, 0x09]); // func0: type9 (invalid)
        m2.extend([0x0a, 0x06, 0x01, 0x04, 0x00, 0x41, 0x01, 0x0b]);
        assert!(Module::decode(&m2).is_err(), "invalid type index rejected");

        // A body reading local 3 in a (i32)->i32 function (only local 0 exists).
        // body = local.get 3; end.
        let mut m3 = vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
        m3.extend([0x01, 0x06, 0x01, 0x60, 0x01, 0x7f, 0x01, 0x7f]); // (i32)->i32
        m3.extend([0x03, 0x02, 0x01, 0x00]);
        m3.extend([0x07, 0x05, 0x01, 0x01, b'f', 0x00, 0x00]);
        m3.extend([0x0a, 0x06, 0x01, 0x04, 0x00, 0x20, 0x03, 0x0b]); // local.get 3
        assert!(Module::decode(&m3).is_err(), "out-of-range local rejected");

        // A body calling function 4 (only function 0 exists).
        let mut m4 = vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
        m4.extend([0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7f]);
        m4.extend([0x03, 0x02, 0x01, 0x00]);
        m4.extend([0x07, 0x05, 0x01, 0x01, b'f', 0x00, 0x00]);
        m4.extend([0x0a, 0x06, 0x01, 0x04, 0x00, 0x10, 0x04, 0x0b]); // call 4
        assert!(
            Module::decode(&m4).is_err(),
            "out-of-range call target rejected"
        );
    }

    #[test]
    fn nop_and_unreachable() {
        // (func (export "f") (param i32) (result i32)
        //   nop local.get 0 nop i32.const 1 i32.add nop)   ;; nops are no-ops
        let body: Vec<u8> = vec![
            0x00, // 0 locals
            0x01, 0x20, 0x00, 0x01, 0x41, 0x01, 0x6a,
            0x01, // nop; get0; nop; const1; add; nop
            0x0b,
        ];
        let mut m = vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
        m.extend([0x01, 0x06, 0x01, 0x60, 0x01, 0x7f, 0x01, 0x7f]);
        m.extend([0x03, 0x02, 0x01, 0x00]);
        m.extend([0x07, 0x05, 0x01, 0x01, b'f', 0x00, 0x00]);
        m.push(0x0a);
        m.push((body.len() + 2) as u8);
        m.push(0x01);
        m.push(body.len() as u8);
        m.extend(body);
        let module = Module::decode(&m).expect("decode nop module");
        assert_eq!(module.call(0, &[Val::I32(41)]).unwrap(), vec![Val::I32(42)]);

        // (func (export "boom") (result i32) unreachable) — traps.
        let body2: Vec<u8> = vec![0x00, 0x00, 0x0b]; // 0 locals; unreachable; end
        let mut m2 = vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
        m2.extend([0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7f]); // ()->i32
        m2.extend([0x03, 0x02, 0x01, 0x00]);
        m2.extend([0x07, 0x08, 0x01, 0x04, b'b', b'o', b'o', b'm', 0x00, 0x00]);
        m2.push(0x0a);
        m2.push((body2.len() + 2) as u8);
        m2.push(0x01);
        m2.push(body2.len() as u8);
        m2.extend(body2);
        let module2 = Module::decode(&m2).expect("decode unreachable module");
        assert!(module2.call(0, &[]).is_err(), "unreachable must trap");
    }

    #[test]
    fn i64_narrow_memory_ops() {
        // (memory 1)
        // (func (export "rt") (param i64) (result i64)
        //   i32.const 0 local.get 0 i64.store32   ;; store low 32 bits at addr 0
        //   i32.const 0 i64.load32_u)             ;; reload them, zero-extended
        let body: Vec<u8> = vec![
            0x00, // 0 locals
            0x41, 0x00, 0x20, 0x00, 0x3e, 0x02,
            0x00, // i32.const 0; local.get 0; i64.store32 a=2 o=0
            0x41, 0x00, 0x35, 0x02, 0x00, // i32.const 0; i64.load32_u a=2 o=0
            0x0b,
        ];
        let mut m = vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
        m.extend([0x01, 0x06, 0x01, 0x60, 0x01, 0x7e, 0x01, 0x7e]); // (i64)->i64
        m.extend([0x03, 0x02, 0x01, 0x00]);
        m.extend([0x05, 0x03, 0x01, 0x00, 0x01]); // memory: 1 page
        m.extend([0x07, 0x06, 0x01, 0x02, b'r', b't', 0x00, 0x00]);
        m.push(0x0a);
        m.push((body.len() + 2) as u8);
        m.push(0x01);
        m.push(body.len() as u8);
        m.extend(body);
        let module = Module::decode(&m).expect("decode i64 narrow mem module");
        // Storing 0x1_FFFF_FFFF and loading32_u keeps only the low 32 bits (zero-ext).
        let r = module.call(0, &[Val::I64(0x1_FFFF_FFFF)]).unwrap();
        assert_eq!(r, vec![Val::I64(0xFFFF_FFFF)]);
        // A small positive value round-trips exactly.
        assert_eq!(
            module.call(0, &[Val::I64(12345)]).unwrap(),
            vec![Val::I64(12345)]
        );
    }

    #[test]
    fn reinterpret_and_f32_conversions() {
        // (func (export "bits") (param f64) (result i64) local.get 0 i64.reinterpret_f64)
        let body: Vec<u8> = vec![0x00, 0x20, 0x00, 0xbd, 0x0b];
        let mut m = vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
        m.extend([0x01, 0x06, 0x01, 0x60, 0x01, 0x7c, 0x01, 0x7e]); // (f64)->i64
        m.extend([0x03, 0x02, 0x01, 0x00]);
        m.extend([0x07, 0x08, 0x01, 0x04, b'b', b'i', b't', b's', 0x00, 0x00]);
        m.push(0x0a);
        m.push((body.len() + 2) as u8);
        m.push(0x01);
        m.push(body.len() as u8);
        m.extend(body);
        let module = Module::decode(&m).expect("decode reinterpret module");
        // The bit pattern of 1.0_f64 is 0x3FF0000000000000.
        let r = module.call(0, &[Val::F64(1.0)]).unwrap();
        assert_eq!(r, vec![Val::I64(0x3FF0_0000_0000_0000)]);

        // (func (export "cf") (param i32) (result f32) local.get 0 f32.convert_i32_s)
        let body2: Vec<u8> = vec![0x00, 0x20, 0x00, 0xb2, 0x0b];
        let mut m2 = vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
        m2.extend([0x01, 0x06, 0x01, 0x60, 0x01, 0x7f, 0x01, 0x7d]); // (i32)->f32
        m2.extend([0x03, 0x02, 0x01, 0x00]);
        m2.extend([0x07, 0x06, 0x01, 0x02, b'c', b'f', 0x00, 0x00]);
        m2.push(0x0a);
        m2.push((body2.len() + 2) as u8);
        m2.push(0x01);
        m2.push(body2.len() as u8);
        m2.extend(body2);
        let module2 = Module::decode(&m2).expect("decode convert module");
        assert_eq!(
            module2.call(0, &[Val::I32(-42)]).unwrap(),
            vec![Val::F32(-42.0)]
        );
    }

    #[test]
    fn select_and_conversions() {
        // (func (export "max") (param i32 i32) (result i32)
        //   local.get 0 local.get 1            ;; a, b
        //   local.get 0 local.get 1 i32.gt_s   ;; a > b
        //   select)                            ;; (a>b) ? a : b
        let body: Vec<u8> = vec![
            0x00, 0x20, 0x00, 0x20, 0x01, 0x20, 0x00, 0x20, 0x01, 0x4a, 0x1b, 0x0b,
        ];
        let mut m = vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
        m.extend([0x01, 0x07, 0x01, 0x60, 0x02, 0x7f, 0x7f, 0x01, 0x7f]);
        m.extend([0x03, 0x02, 0x01, 0x00]);
        m.extend([0x07, 0x07, 0x01, 0x03, b'm', b'a', b'x', 0x00, 0x00]);
        m.push(0x0a);
        m.push((body.len() + 2) as u8);
        m.push(0x01);
        m.push(body.len() as u8);
        m.extend(body);
        let module = Module::decode(&m).expect("decode select module");
        assert_eq!(
            module.call(0, &[Val::I32(7), Val::I32(3)]).unwrap(),
            vec![Val::I32(7)]
        );
        assert_eq!(
            module.call(0, &[Val::I32(2), Val::I32(9)]).unwrap(),
            vec![Val::I32(9)]
        );

        // (func (export "half") (param i32) (result f64)
        //   local.get 0 f64.convert_i32_s f64.const 2.0 f64.div)
        let mut body2: Vec<u8> = vec![0x00, 0x20, 0x00, 0xb7, 0x44];
        body2.extend(2.0f64.to_le_bytes());
        body2.extend([0xa3, 0x0b]); // f64.div, end
        let mut m2 = vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
        m2.extend([0x01, 0x06, 0x01, 0x60, 0x01, 0x7f, 0x01, 0x7c]); // (i32)->f64
        m2.extend([0x03, 0x02, 0x01, 0x00]);
        m2.extend([0x07, 0x08, 0x01, 0x04, b'h', b'a', b'l', b'f', 0x00, 0x00]);
        m2.push(0x0a);
        m2.push((body2.len() + 2) as u8);
        m2.push(0x01);
        m2.push(body2.len() as u8);
        m2.extend(body2);
        let module2 = Module::decode(&m2).expect("decode convert module");
        assert_eq!(
            module2.call(0, &[Val::I32(7)]).unwrap(),
            vec![Val::F64(3.5)]
        );
    }

    #[test]
    fn start_function_runs_at_instantiation() {
        // (global (mut i32) (i32.const 0))
        // (func $init  i32.const 111 global.set 0)     ;; func 0, the start fn
        // (func (export "get") (result i32) global.get 0)  ;; func 1
        // (start 0)
        // i32.const 111 needs 2 LEB bytes (0x6f alone is -17: sign bit set).
        let init_body: Vec<u8> = vec![0x00, 0x41, 0xef, 0x00, 0x24, 0x00, 0x0b]; // global0 = 111
        let get_body: Vec<u8> = vec![0x00, 0x23, 0x00, 0x0b];
        let mut m = vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
        // types: 0 = ()->(), 1 = ()->i32
        m.extend([0x01, 0x08, 0x02, 0x60, 0x00, 0x00, 0x60, 0x00, 0x01, 0x7f]);
        m.extend([0x03, 0x03, 0x02, 0x00, 0x01]); // func0:type0, func1:type1
        m.extend([0x06, 0x06, 0x01, 0x7f, 0x01, 0x41, 0x00, 0x0b]); // global0 = 0, mut
        m.extend([0x07, 0x07, 0x01, 0x03, b'g', b'e', b't', 0x00, 0x01]); // export "get"=func1
        m.extend([0x08, 0x01, 0x00]); // start section: func 0
        let mut code = vec![0x0a, 0x00, 0x02];
        for b in [&init_body[..], &get_body[..]] {
            code.push(b.len() as u8);
            code.extend_from_slice(b);
        }
        code[1] = (code.len() - 2) as u8;
        m.extend(code);

        let module = Module::decode(&m).expect("decode module with start");
        // `get` returns 111 immediately — the start function already ran.
        let mut inst = Instance::new(&module).expect("instantiate");
        assert_eq!(inst.call_export("get", &[]).unwrap(), vec![Val::I32(111)]);
    }

    #[test]
    fn imported_memory_supplied_by_host() {
        // (import "env" "memory" (memory 1))
        // (func (export "get") (param i32) (result i32) local.get 0 i32.load)
        let body: Vec<u8> = vec![0x00, 0x20, 0x00, 0x28, 0x02, 0x00, 0x0b];
        let mut m = vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
        m.extend([0x01, 0x06, 0x01, 0x60, 0x01, 0x7f, 0x01, 0x7f]); // (i32)->i32
        // import: "env"."memory" memory limits {min 1}
        m.extend([
            0x02, 0x0f, 0x01, 0x03, b'e', b'n', b'v', 0x06, b'm', b'e', b'm', b'o', b'r', b'y',
            0x02, 0x00, 0x01,
        ]);
        m.extend([0x03, 0x02, 0x01, 0x00]); // func 0: type 0
        m.extend([0x07, 0x07, 0x01, 0x03, b'g', b'e', b't', 0x00, 0x00]); // export "get"
        m.push(0x0a);
        m.push((body.len() + 2) as u8);
        m.push(0x01);
        m.push(body.len() as u8);
        m.extend(body);

        let module = Module::decode(&m).expect("decode imported-memory module");
        // The host owns the memory and pre-fills it: mem[8] = 0xdead_beef_u32 LE.
        let mut host_mem = alloc::vec![0u8; PAGE_SIZE];
        host_mem[8..12].copy_from_slice(&0x1234_5678u32.to_le_bytes());
        let mut inst =
            Instance::instantiate_full(&module, alloc::vec![], alloc::vec![], Some(host_mem))
                .expect("instantiate with imported memory");
        // The module reads the host-supplied bytes.
        assert_eq!(
            inst.call_export("get", &[Val::I32(8)]).unwrap(),
            vec![Val::I32(0x1234_5678)]
        );
        // A module importing memory must be given one.
        assert!(Instance::new(&module).is_err());
    }

    #[test]
    fn imported_global_supplied_by_host() {
        // (import "env" "base" (global i32))   ;; global 0, imported
        // (global (mut i32) (i32.const 100))    ;; global 1, defined
        // (func (export "sum") (result i32) global.get 0 global.get 1 i32.add)
        let body: Vec<u8> = vec![0x00, 0x23, 0x00, 0x23, 0x01, 0x6a, 0x0b];
        let mut m = vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
        m.extend([0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7f]); // type ()->i32
        // import: "env"."base" global i32 (immutable)
        m.extend([
            0x02, 0x0d, 0x01, 0x03, b'e', b'n', b'v', 0x04, b'b', b'a', b's', b'e', 0x03, 0x7f,
            0x00,
        ]);
        m.extend([0x03, 0x02, 0x01, 0x00]); // func 0: type 0
        // defined global1 = 100 (mut). i32.const 100 needs 2 LEB bytes (0x64 alone
        // is -28: the 0x40 sign bit is set).
        m.extend([0x06, 0x07, 0x01, 0x7f, 0x01, 0x41, 0xe4, 0x00, 0x0b]);
        m.extend([0x07, 0x07, 0x01, 0x03, b's', b'u', b'm', 0x00, 0x00]); // export "sum"
        m.push(0x0a);
        m.push((body.len() + 2) as u8);
        m.push(0x01);
        m.push(body.len() as u8);
        m.extend(body);

        let module = Module::decode(&m).expect("decode imported-global module");
        // Supply the imported global value; sum = base + 100.
        let mut inst = Instance::instantiate(&module, alloc::vec![], alloc::vec![Val::I32(7)])
            .expect("instantiate with imported global");
        assert_eq!(inst.call_export("sum", &[]).unwrap(), vec![Val::I32(107)]);

        // A different supplied value flows through.
        let mut inst2 =
            Instance::instantiate(&module, alloc::vec![], alloc::vec![Val::I32(-50)]).unwrap();
        assert_eq!(inst2.call_export("sum", &[]).unwrap(), vec![Val::I32(50)]);

        // Omitting the imported global value is rejected.
        assert!(Instance::new(&module).is_err());
    }

    #[test]
    fn instance_global_accumulates_across_calls() {
        // A mutable global persists across Instance calls (unlike Module::call,
        // which resets per invocation).
        // (global (mut i32) (i32.const 0))
        // (func (export "inc") (param i32) (result i32)
        //   global.get 0 local.get 0 i32.add global.set 0 global.get 0)
        let body: Vec<u8> = vec![
            0x00, 0x23, 0x00, 0x20, 0x00, 0x6a, 0x24, 0x00, 0x23, 0x00, 0x0b,
        ];
        let mut m = vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
        m.extend([0x01, 0x06, 0x01, 0x60, 0x01, 0x7f, 0x01, 0x7f]);
        m.extend([0x03, 0x02, 0x01, 0x00]);
        m.extend([0x06, 0x06, 0x01, 0x7f, 0x01, 0x41, 0x00, 0x0b]); // mutable global = 0
        m.extend([0x07, 0x07, 0x01, 0x03, b'i', b'n', b'c', 0x00, 0x00]);
        m.push(0x0a);
        m.push((body.len() + 2) as u8);
        m.push(0x01);
        m.push(body.len() as u8);
        m.extend(body);
        let module = Module::decode(&m).expect("decode");
        let mut inst = Instance::new(&module).expect("instantiate");
        assert_eq!(
            inst.call_export("inc", &[Val::I32(5)]).unwrap(),
            vec![Val::I32(5)]
        );
        assert_eq!(
            inst.call_export("inc", &[Val::I32(10)]).unwrap(),
            vec![Val::I32(15)]
        );
        assert_eq!(
            inst.call_export("inc", &[Val::I32(100)]).unwrap(),
            vec![Val::I32(115)]
        );
        // A fresh instance starts over.
        let mut inst2 = Instance::new(&module).unwrap();
        assert_eq!(
            inst2.call_export("inc", &[Val::I32(7)]).unwrap(),
            vec![Val::I32(7)]
        );
    }

    #[test]
    fn leb128_signed() {
        let mut r = Reader::new(&[0x7f]); // -1
        assert_eq!(r.i32().unwrap(), -1);
        let mut r = Reader::new(&[0x80, 0x7f]); // -128
        assert_eq!(r.i32().unwrap(), -128);
        let mut r = Reader::new(&[0xe5, 0x8e, 0x26]); // 624485
        assert_eq!(r.u32().unwrap(), 624_485);
    }

    #[test]
    fn rejects_bad_magic() {
        assert!(Module::decode(&[0, 0, 0, 0, 1, 0, 0, 0]).is_err());
    }
}
