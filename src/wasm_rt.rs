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
}

impl Val {
    fn as_i32(self) -> Result<i32, WasmRtError> {
        match self {
            Val::I32(v) => Ok(v),
            Val::I64(_) => Err(WasmRtError("type mismatch: expected i32")),
        }
    }
    fn as_i64(self) -> Result<i64, WasmRtError> {
        match self {
            Val::I64(v) => Ok(v),
            Val::I32(_) => Err(WasmRtError("type mismatch: expected i64")),
        }
    }
}

/// A value type in a function signature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValType {
    /// `i32`
    I32,
    /// `i64`
    I64,
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
        _ => Err(WasmRtError("unsupported value type")),
    }
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
                3 => Self::decode_functions(&mut s, &mut m)?,
                7 => Self::decode_exports(&mut s, &mut m)?,
                10 => Self::decode_code(&mut s, &mut m)?,
                // Other sections (import/table/memory/global/data) are skipped
                // for now; they land with the non-numeric surface.
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

    fn decode_functions(s: &mut Reader, m: &mut Module) -> Result<(), WasmRtError> {
        let count = s.u32()?;
        for _ in 0..count {
            m.func_types.push(s.u32()?);
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

    /// The index of an exported function, or `None`.
    #[must_use]
    pub fn export(&self, name: &str) -> Option<u32> {
        self.exports
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, i)| *i)
    }

    /// Calls function `index` with `args`, returning the results.
    ///
    /// # Errors
    /// Returns `WasmRtError` on a type mismatch, a missing function, or an
    /// unsupported instruction.
    pub fn call(&self, index: u32, args: &[Val]) -> Result<Vec<Val>, WasmRtError> {
        let ty = self
            .func_types
            .get(index as usize)
            .and_then(|t| self.types.get(*t as usize))
            .ok_or(WasmRtError("no such function"))?;
        let body = self
            .bodies
            .get(index as usize)
            .ok_or(WasmRtError("no such function body"))?;
        if args.len() != ty.params.len() {
            return Err(WasmRtError("argument count mismatch"));
        }
        // Locals = parameters, then zero-initialized declared locals.
        let mut locals: Vec<Val> = args.to_vec();
        for lt in &body.locals {
            locals.push(match lt {
                ValType::I32 => Val::I32(0),
                ValType::I64 => Val::I64(0),
            });
        }
        let mut stack: Vec<Val> = Vec::new();
        self.exec(&body.code, &mut locals, &mut stack)?;
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
    ) -> Result<Flow, WasmRtError> {
        let mut r = Reader::new(code);
        macro_rules! pop {
            () => {
                stack.pop().ok_or(WasmRtError("stack underflow"))?
            };
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
                    let res = self.call(callee, &cargs)?;
                    stack.extend(res);
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
                // structured control: block / loop / if
                0x02 | 0x03 => {
                    let _blocktype = r.byte()?; // 0x40 (empty) or a value type
                    let is_loop = op == 0x03;
                    let (consumed, flow) =
                        self.exec_block(&code[r.pos..], locals, stack, is_loop)?;
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
        is_loop: bool,
    ) -> Result<(usize, Flow), WasmRtError> {
        let inner_len = block_len(code)?;
        let inner = &code[..inner_len - 1]; // exclude the matching `end`
        loop {
            match self.exec(inner, locals, stack)? {
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
            0x20 | 0x21 | 0x22 | 0x0c | 0x0d | 0x10 => {
                r.u32()?;
            }
            0x41 => {
                r.i32()?;
            }
            0x42 => {
                r.i64()?;
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
