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
    /// Linear memory minimum size, in 64 KiB pages (`None` = no memory).
    mem_min_pages: Option<u32>,
    /// Active data segments: `(constant byte offset, bytes)`.
    data: Vec<(u32, Vec<u8>)>,
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
}

/// A host function backing an `import` — receives the WASM call's arguments and
/// returns its results. This is how a JS (or Rust) function is callable from
/// inside a WASM module.
pub type HostFunc = alloc::boxed::Box<dyn Fn(&[Val]) -> Result<Vec<Val>, WasmRtError>>;

/// One linear-memory page, in bytes (WebAssembly fixes this at 64 KiB).
const PAGE_SIZE: usize = 65536;

/// The mutable instance state threaded through execution: linear memory and the
/// current global values.
struct Store {
    mem: Vec<u8>,
    globals: Vec<Val>,
    /// Host functions backing the module's imports (index-aligned with
    /// `Module::func_imports`). Empty for an import-free `Module::call`.
    host_funcs: Vec<HostFunc>,
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
                9 => Self::decode_elements(&mut s, &mut m)?,
                10 => Self::decode_code(&mut s, &mut m)?,
                11 => Self::decode_data(&mut s, &mut m)?,
                // Other sections (import/start) are skipped for now.
                _ => {}
            }
        }
        Ok(m)
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
                    let flag = s.byte()?;
                    s.u32()?;
                    if flag == 1 {
                        s.u32()?;
                    }
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

    fn decode_memory(s: &mut Reader, m: &mut Module) -> Result<(), WasmRtError> {
        let count = s.u32()?;
        if count > 0 {
            // limits: flag (0 = min only, 1 = min+max), then min (and max).
            let flag = s.byte()?;
            let min = s.u32()?;
            if flag == 1 {
                let _max = s.u32()?;
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
            // Mode 0 = active, memory 0, with an `i32.const off; end` offset expr.
            if mode != 0 {
                return Err(WasmRtError("unsupported data segment mode"));
            }
            let off = read_const_i32_expr(s)?;
            let len = s.u32()? as usize;
            let bytes = s.bytes(len)?.to_vec();
            m.data.push((off as u32, bytes));
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
            if kind == 0x00 {
                // a function export
                m.exports.push((name, index));
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
        Ok(Store {
            mem,
            globals,
            host_funcs: Vec::new(),
        })
    }

    /// Calls function `index` with `args`, returning the results. Allocates a
    /// fresh instance (memory + globals) for the invocation.
    ///
    /// # Errors
    /// Returns `WasmRtError` on a type mismatch, a missing function, or an
    /// unsupported instruction.
    pub fn call(&self, index: u32, args: &[Val]) -> Result<Vec<Val>, WasmRtError> {
        let mut store = self.new_store()?;
        self.call_with_store(index, args, &mut store)
    }

    /// Like [`call`](Self::call) but over a caller-provided instance store
    /// (shared across nested `call`s).
    fn call_with_store(
        &self,
        index: u32,
        args: &[Val],
        store: &mut Store,
    ) -> Result<Vec<Val>, WasmRtError> {
        let ty = self
            .func_types
            .get(index as usize)
            .and_then(|t| self.types.get(*t as usize))
            .ok_or(WasmRtError("no such function"))?;
        if args.len() != ty.params.len() {
            return Err(WasmRtError("argument count mismatch"));
        }
        // A function import dispatches to its host function.
        let n_imp = self.n_imported_funcs();
        if (index as usize) < n_imp {
            let host = store
                .host_funcs
                .get(index as usize)
                .ok_or(WasmRtError("missing host function for import"))?;
            return host(args);
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
        self.exec(&body.code, &mut locals, &mut stack, store)?;
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
                0x0b => return Ok(Flow::Normal), // end
                0x0f => return Ok(Flow::Return), // return
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
                    let res = self.call_with_store(callee, &cargs, store)?;
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
                    let res = self.call_with_store(func, &cargs, store)?;
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
                0x3f => {
                    let _reserved = r.byte()?;
                    stack.push(Val::I32((store.mem.len() / PAGE_SIZE) as i32)); // memory.size
                }
                0x40 => {
                    let _reserved = r.byte()?;
                    let delta = pop!().as_i32()? as u32 as usize;
                    let old = store.mem.len() / PAGE_SIZE;
                    // Grow by `delta` pages (no max enforced here); -1 on failure.
                    if store
                        .mem
                        .len()
                        .checked_add(delta * PAGE_SIZE)
                        .filter(|n| *n <= 0x1_0000 * PAGE_SIZE)
                        .is_some()
                    {
                        store.mem.resize(store.mem.len() + delta * PAGE_SIZE, 0);
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
                0x4a => bin_i32!(|a, b| i32::from(a > b)), // gt_s
                0x4c => bin_i32!(|a, b| i32::from(a <= b)), // le_s
                0x4e => bin_i32!(|a, b| i32::from(a >= b)), // ge_s
                0x6a => bin_i32!(i32::wrapping_add),
                0x6b => bin_i32!(i32::wrapping_sub),
                0x6c => bin_i32!(i32::wrapping_mul),
                0x6d => {
                    let b = pop!().as_i32()?;
                    let a = pop!().as_i32()?;
                    if b == 0 {
                        return Err(WasmRtError("integer divide by zero"));
                    }
                    stack.push(Val::I32(a.wrapping_div(b))); // div_s
                }
                0x71 => bin_i32!(|a, b| a & b),
                0x72 => bin_i32!(|a, b| a | b),
                0x73 => bin_i32!(|a, b| a ^ b),
                0x74 => bin_i32!(|a: i32, b: i32| a.wrapping_shl(b as u32)),
                0x75 => bin_i32!(|a: i32, b: i32| a.wrapping_shr(b as u32)), // shr_s
                // i64 arithmetic
                0x7c => bin_i64!(i64::wrapping_add),
                0x7d => bin_i64!(i64::wrapping_sub),
                0x7e => bin_i64!(i64::wrapping_mul),
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
                // f64 unary / arithmetic
                0x99 => {
                    let a = pop!().as_f64()?;
                    stack.push(Val::F64(f64_abs(a)));
                }
                0x9a => {
                    let a = pop!().as_f64()?;
                    stack.push(Val::F64(-a));
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
                // structured control: block / loop / if
                0x02 | 0x03 => {
                    let _blocktype = r.byte()?; // 0x40 (empty) or a value type
                    let is_loop = op == 0x03;
                    let (consumed, flow) =
                        self.exec_block(&code[r.pos..], locals, stack, store, is_loop)?;
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
                0x0c => return Ok(Flow::Branch(r.u32()?)), // br
                0x0d => {
                    let depth = r.u32()?;
                    if pop!().as_i32()? != 0 {
                        return Ok(Flow::Branch(depth));
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
        is_loop: bool,
    ) -> Result<(usize, Flow), WasmRtError> {
        let inner_len = block_len(code)?;
        let inner = &code[..inner_len - 1]; // exclude the matching `end`
        loop {
            match self.exec(inner, locals, stack, store)? {
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
pub struct Instance<'m> {
    module: &'m Module,
    store: Store,
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
        if host_funcs.len() != module.n_imported_funcs()
            || import_globals.len() != module.n_imported_globals()
        {
            return Err(WasmRtError("import count mismatch"));
        }
        let mut store = module.new_store()?;
        store.host_funcs = host_funcs;
        // Imported globals occupy the first slots of the global space.
        for (i, v) in import_globals.into_iter().enumerate() {
            store.globals[i] = v;
        }
        Ok(Self { module, store })
    }

    /// Calls function `index` with `args` over this instance's **persistent**
    /// state — memory writes and global mutations are visible to later calls.
    ///
    /// # Errors
    /// Returns `WasmRtError` on a type mismatch, a missing function, or an
    /// unsupported instruction / trap.
    pub fn call(&mut self, index: u32, args: &[Val]) -> Result<Vec<Val>, WasmRtError> {
        self.module.call_with_store(index, args, &mut self.store)
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
