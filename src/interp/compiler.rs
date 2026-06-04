//! The AST→bytecode compiler for the supported language subset.
//!
//! Lowers a program to a [`Module`] of register-bytecode [`Chunk`]s that
//! [`super::vm`] executes, reusing the interpreter's value semantics so the
//! bytecode path stays consistent with the tree-walker. Constructs outside the
//! subset return [`CompileError`] so the caller can fall back to the
//! tree-walker.
//!
//! Supported: literals, template literals, globals + block-scoped locals, the
//! full operator set (arithmetic/comparison incl. `!=`/`!==`, bitwise/shift,
//! `&&`/`||`/`??`, unary `-`/`+`/`!`, `typeof`/`void`/`delete`, `++`/`--`, the
//! ternary, `in`, `instanceof`), object and array literals (with array/object
//! spread), member/index access and writes, calls and method calls (with
//! `this` + built-in prototype dispatch, `call`/`apply`, and call-argument
//! spread), `new`, assignment (incl. compound, on identifiers and members),
//! `if`/`else`, `while`/`do-while`/`for`/`for-of`/`for-in` with
//! `break`/`continue`, `switch`, `try`/`catch`, `throw`, blocks, `return`,
//! destructuring declarations + parameters (array/object patterns, defaults,
//! array & object rest), **functions** (declarations hoisted, function/arrow
//! expressions, rest parameters), **closures** that capture enclosing variables
//! (boxed in shared cells, with transitive capture), and **classes** —
//! constructor, instance/static methods + fields, getters/setters, and
//! `extends`/`super` inheritance, and `try`/`catch`/`finally`. Object-literal
//! getters/setters compile too. Not yet: generators/async, computed/private
//! class keys, a `finally` whose guarded region escapes via
//! `return`/`break`/`continue`, and captured (hoisted) function *declarations* —
//! these return a `CompileError`
//! so the caller falls back to the tree-walker.

use crate::ast::{
    Arrow, ArrowBody, AssignOp, BinaryOp, BindingTarget, Expr, Function, LogicalOp, Param,
    PropertyKey, Stmt, UnaryOp,
};
use crate::bytecode::{Chunk, Const, ConstIdx, Module, Op, Reg};
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

/// A reason a program could not be compiled to bytecode (yet).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompileError {
    /// What was unsupported.
    pub message: String,
}

impl CompileError {
    fn unsupported(what: &str) -> Self {
        Self {
            message: format!("bytecode compiler: unsupported {what}"),
        }
    }
}

/// Compiles a program to a module whose entry chunk (#0) returns the value of
/// the final expression statement (REPL-style).
pub fn compile_program(body: &[Stmt]) -> Result<Module, CompileError> {
    // Top-level locals can be captured by functions too.
    let captured = captured_names(&[], &FnBody::Block(body));
    let mut c = Compiler {
        module: alloc::vec![Chunk::new("<main>")],
        funcs: alloc::vec![FnState::new(0, captured)],
    };
    c.compile_body(body, true)?;
    let main = c.funcs.pop().expect("entry function state");
    c.module[0].register_count = main.next_reg;
    Ok(Module { chunks: c.module })
}

/// A local binding: its name, backing register, and whether that register holds
/// a "cell" (a one-element array shared with closures that capture it) rather
/// than the value directly.
#[derive(Clone)]
struct Local {
    name: String,
    reg: Reg,
    is_cell: bool,
}

/// How a function reaches one of its upvalues at closure-creation time.
#[derive(Clone, Copy)]
enum UpvalSource {
    /// A cell in an enclosing function's local register.
    ParentLocal(Reg),
    /// An upvalue of the enclosing function (transitive capture).
    ParentUpval(u16),
}

/// An upvalue captured by a function: the captured name and where its cell
/// comes from in the immediately enclosing function.
#[derive(Clone)]
struct Upvalue {
    name: String,
    source: UpvalSource,
}

/// How a referenced name resolves in the current function.
enum Binding {
    /// A local register; `cell` means the value lives in `reg[0]` of a cell.
    Local { reg: Reg, cell: bool },
    /// An upvalue (captured variable) at the given index.
    Upvalue(u16),
    /// A global variable.
    Global,
}

/// Per-function compilation state.
struct FnState {
    chunk_idx: usize,
    next_reg: Reg,
    /// In-scope locals (innermost last; shadowing wins).
    locals: Vec<Local>,
    /// Names of this function's locals/params that are captured by nested
    /// functions and so are stored in cells (computed before compiling).
    captured: alloc::collections::BTreeSet<String>,
    /// Upvalues captured from enclosing functions, in capture order.
    upvalues: Vec<Upvalue>,
    /// The enclosing-loop stack, for resolving `break`/`continue` jumps.
    loops: Vec<LoopCtx>,
}

impl FnState {
    fn new(chunk_idx: usize, captured: alloc::collections::BTreeSet<String>) -> Self {
        Self {
            chunk_idx,
            // Register 0 is reserved for `this` (bound by the VM on each call);
            // parameters and locals start at register 1.
            next_reg: 1,
            locals: Vec::new(),
            captured,
            upvalues: Vec::new(),
            loops: Vec::new(),
        }
    }
}

/// The register holding the current function's `this` value.
const THIS_REG: Reg = 0;

/// The synthetic binding name a subclass's methods capture to reach their
/// superclass for `super(...)` / `super.m(...)`.
const SUPER_NAME: &str = "%super%";

/// Unresolved `break`/`continue` jump sites for one loop or switch.
struct LoopCtx {
    break_jumps: Vec<usize>,
    continue_jumps: Vec<usize>,
    /// True for real loops (a `continue` target); false for `switch` (only
    /// `break` applies).
    is_loop: bool,
}

struct Compiler {
    module: Vec<Chunk>,
    /// Function-compilation stack; the last entry is the current function.
    funcs: Vec<FnState>,
}

impl Compiler {
    // --- current-function accessors ------------------------------------

    fn ci(&self) -> usize {
        self.funcs.last().expect("a current function").chunk_idx
    }
    fn emit(&mut self, op: Op) -> usize {
        let i = self.ci();
        self.module[i].emit(op)
    }
    fn add_const(&mut self, c: Const) -> ConstIdx {
        let i = self.ci();
        self.module[i].add_constant(c)
    }
    fn code_len(&self) -> usize {
        self.module[self.ci()].code.len()
    }
    fn reg(&mut self) -> Reg {
        let f = self.funcs.last_mut().expect("a current function");
        let r = f.next_reg;
        f.next_reg += 1;
        r
    }
    /// Looks up a local in the current function (innermost wins).
    fn resolve(&self, name: &str) -> Option<Local> {
        self.funcs
            .last()
            .expect("a current function")
            .locals
            .iter()
            .rev()
            .find(|l| l.name == name)
            .cloned()
    }
    /// Declares a local backed by `reg`; `is_cell` marks a captured (boxed) one.
    fn declare_local_cell(&mut self, name: String, reg: Reg, is_cell: bool) {
        self.funcs
            .last_mut()
            .expect("a current function")
            .locals
            .push(Local { name, reg, is_cell });
    }
    /// Declares a plain (non-captured) local.
    fn declare_local(&mut self, name: String, reg: Reg) {
        self.declare_local_cell(name, reg, false);
    }
    /// Pre-declares (as empty cells) the captured simple `let`/`const`/`var`
    /// bindings declared directly in `stmts`, so initializers can forward- and
    /// self-reference them (recursive / mutually-recursive closures). Only
    /// simple identifiers are hoisted; patterns and params box per-site.
    fn predeclare_captured_cells(&mut self, stmts: &[Stmt]) {
        for s in stmts {
            let Stmt::Var(decl) = s else { continue };
            for d in &decl.declarations {
                if let BindingTarget::Ident(id) = &d.target
                    && self.is_captured_here(&id.name)
                    && self.resolve(&id.name).is_none_or(|l| !l.is_cell)
                {
                    let undef = self.reg();
                    self.emit(Op::LoadUndefined { dst: undef });
                    let cell = self.new_cell(undef);
                    self.declare_local_cell(id.name.clone().into_string(), cell, true);
                }
            }
        }
    }
    /// Whether `name` is captured by nested functions in the current function
    /// (and so must be stored in a cell).
    fn is_captured_here(&self, name: &str) -> bool {
        self.funcs
            .last()
            .expect("a current function")
            .captured
            .contains(name)
    }

    // --- name resolution (locals / upvalues / globals) ------------------

    /// Resolves a name to a binding, threading upvalue capture through the
    /// enclosing functions as needed.
    fn lookup_binding(&mut self, name: &str) -> Result<Binding, CompileError> {
        if let Some(local) = self.resolve(name) {
            return Ok(Binding::Local {
                reg: local.reg,
                cell: local.is_cell,
            });
        }
        let top = self.funcs.len() - 1;
        if let Some(idx) = self.resolve_upvalue(top, name)? {
            return Ok(Binding::Upvalue(idx));
        }
        Ok(Binding::Global)
    }

    /// Resolves `name` as an upvalue of function `fi`, recording the capture
    /// chain. Returns the upvalue index, or `None` if it is not an enclosing
    /// local. A reference to an enclosing local that was *not* boxed (the
    /// capture analysis missed it) is reported as unsupported so the caller
    /// falls back to the tree-walker.
    fn resolve_upvalue(&mut self, fi: usize, name: &str) -> Result<Option<u16>, CompileError> {
        if fi == 0 {
            return Ok(None);
        }
        let parent = fi - 1;
        // A local of the enclosing function?
        if let Some(local) = self.funcs[parent]
            .locals
            .iter()
            .rev()
            .find(|l| l.name == name)
            .cloned()
        {
            if !local.is_cell {
                return Err(CompileError::unsupported("captured (closure) variable"));
            }
            return Ok(Some(self.add_upvalue(
                fi,
                name,
                UpvalSource::ParentLocal(local.reg),
            )));
        }
        // An upvalue of the enclosing function (transitive capture)?
        if let Some(pidx) = self.resolve_upvalue(parent, name)? {
            return Ok(Some(self.add_upvalue(
                fi,
                name,
                UpvalSource::ParentUpval(pidx),
            )));
        }
        Ok(None)
    }

    /// Adds (or reuses) an upvalue on function `fi`, returning its index.
    fn add_upvalue(&mut self, fi: usize, name: &str, source: UpvalSource) -> u16 {
        let ups = &mut self.funcs[fi].upvalues;
        for (i, up) in ups.iter().enumerate() {
            if up.name == name {
                return i as u16;
            }
        }
        ups.push(Upvalue {
            name: name.into(),
            source,
        });
        (ups.len() - 1) as u16
    }

    /// Reads the binding `b` into a fresh register and returns it.
    fn read_binding(&mut self, b: &Binding, name: &str) -> Reg {
        match *b {
            Binding::Local { reg, cell: false } => {
                let dst = self.reg();
                self.emit(Op::Move { dst, src: reg });
                dst
            }
            Binding::Local { reg, cell: true } => self.cell_get(reg),
            Binding::Upvalue(idx) => {
                let cell = self.reg();
                self.emit(Op::GetUpvalue { dst: cell, idx });
                self.cell_get(cell)
            }
            Binding::Global => {
                let dst = self.reg();
                let k = self.add_const(Const::Str(name.into()));
                self.emit(Op::GetGlobal { dst, name: k });
                dst
            }
        }
    }

    /// Writes `value` into the binding `b`.
    fn write_binding(&mut self, b: &Binding, name: &str, value: Reg) {
        match *b {
            Binding::Local { reg, cell: false } => {
                self.emit(Op::Move {
                    dst: reg,
                    src: value,
                });
            }
            Binding::Local { reg, cell: true } => self.cell_set(reg, value),
            Binding::Upvalue(idx) => {
                let cell = self.reg();
                self.emit(Op::GetUpvalue { dst: cell, idx });
                self.cell_set(cell, value);
            }
            Binding::Global => {
                let k = self.add_const(Const::Str(name.into()));
                self.emit(Op::SetGlobal {
                    name: k,
                    src: value,
                });
            }
        }
    }

    /// `dst = cell[0]` — reads the value held in a cell.
    fn cell_get(&mut self, cell: Reg) -> Reg {
        let idx = self.reg();
        self.emit(Op::LoadInt { dst: idx, value: 0 });
        let dst = self.reg();
        self.emit(Op::GetElem {
            dst,
            obj: cell,
            index: idx,
        });
        dst
    }

    /// `cell[0] = value` — stores a value into a cell.
    fn cell_set(&mut self, cell: Reg, value: Reg) {
        let idx = self.reg();
        self.emit(Op::LoadInt { dst: idx, value: 0 });
        self.emit(Op::SetElem {
            obj: cell,
            index: idx,
            src: value,
        });
    }

    /// Creates a fresh cell holding `value`, returning the cell register.
    fn new_cell(&mut self, value: Reg) -> Reg {
        let cell = self.reg();
        self.emit(Op::NewArray { dst: cell, len: 1 });
        self.cell_set(cell, value);
        cell
    }

    // --- statements -----------------------------------------------------

    /// Compiles a statement list. With `return_last`, the value of the final
    /// expression statement is returned; otherwise the chunk falls off the end.
    fn compile_body(&mut self, body: &[Stmt], return_last: bool) -> Result<(), CompileError> {
        self.predeclare_captured_cells(body);
        for stmt in body {
            if let Stmt::Function(func) = stmt {
                self.compile_function_decl(func)?;
            }
        }
        let mut last: Option<Reg> = None;
        for stmt in body {
            if matches!(stmt, Stmt::Function(_)) {
                continue;
            }
            last = self.stmt(stmt)?;
        }
        if return_last {
            match last {
                Some(reg) => self.emit(Op::Return { src: reg }),
                None => self.emit(Op::ReturnUndefined),
            };
        }
        Ok(())
    }

    fn stmt(&mut self, stmt: &Stmt) -> Result<Option<Reg>, CompileError> {
        match stmt {
            Stmt::Expr { expression, .. } => Ok(Some(self.expr(expression)?)),
            Stmt::Empty { .. } => Ok(None),
            Stmt::Function(_) => Ok(None),
            Stmt::Class(class) => {
                let Some(id) = &class.id else {
                    return Err(CompileError::unsupported("anonymous class declaration"));
                };
                let ctor = self.compile_class(class)?;
                let name = self.add_const(Const::Str(id.name.clone().into_string()));
                self.emit(Op::SetGlobal { name, src: ctor });
                Ok(None)
            }
            Stmt::Return { argument, .. } => {
                let src = match argument {
                    Some(e) => self.expr(e)?,
                    None => {
                        let r = self.reg();
                        self.emit(Op::LoadUndefined { dst: r });
                        r
                    }
                };
                self.emit(Op::Return { src });
                Ok(None)
            }
            Stmt::Var(decl) => {
                for d in &decl.declarations {
                    // A captured simple binding has had its cell pre-declared at
                    // scope entry (so forward/self references resolve — e.g.
                    // recursive or mutually-recursive `const f = … f … `); just
                    // store the initializer into it.
                    if let BindingTarget::Ident(id) = &d.target
                        && self.is_captured_here(&id.name)
                    {
                        let cell = self.resolve(&id.name).expect("predeclared cell").reg;
                        if let Some(init) = &d.init {
                            let value = self.expr(init)?;
                            self.cell_set(cell, value);
                        }
                        continue;
                    }
                    let value = match &d.init {
                        Some(init) => self.expr(init)?,
                        None => {
                            let r = self.reg();
                            self.emit(Op::LoadUndefined { dst: r });
                            r
                        }
                    };
                    self.bind_pattern(&d.target, value)?;
                }
                Ok(None)
            }
            Stmt::Block { body, .. } => {
                let mark = self.funcs.last().expect("fn").locals.len();
                self.predeclare_captured_cells(body);
                for s in body {
                    if let Stmt::Function(f) = s {
                        self.compile_function_decl(f)?;
                    }
                }
                for s in body {
                    if !matches!(s, Stmt::Function(_)) {
                        self.stmt(s)?;
                    }
                }
                self.funcs.last_mut().expect("fn").locals.truncate(mark);
                Ok(None)
            }
            Stmt::If {
                test,
                consequent,
                alternate,
                ..
            } => {
                let cond = self.expr(test)?;
                let jf = self.emit(Op::JumpIfFalse { cond, offset: 0 });
                self.stmt(consequent)?;
                if let Some(alt) = alternate {
                    let jmp = self.emit(Op::Jump { offset: 0 });
                    self.patch_to_here(jf);
                    self.stmt(alt)?;
                    self.patch_to_here(jmp);
                } else {
                    self.patch_to_here(jf);
                }
                Ok(None)
            }
            Stmt::While { test, body, .. } => {
                let top = self.code_len();
                let cond = self.expr(test)?;
                let jf = self.emit(Op::JumpIfFalse { cond, offset: 0 });
                self.push_loop();
                self.stmt(body)?;
                let ctx = self.pop_loop();
                // `continue` re-evaluates the test.
                for j in &ctx.continue_jumps {
                    self.patch_jump(*j, top);
                }
                let back = self.emit(Op::Jump { offset: 0 });
                self.patch_jump(back, top);
                self.patch_to_here(jf);
                for j in &ctx.break_jumps {
                    self.patch_to_here(*j);
                }
                Ok(None)
            }
            Stmt::DoWhile { body, test, .. } => {
                let top = self.code_len();
                self.push_loop();
                self.stmt(body)?;
                let ctx = self.pop_loop();
                // `continue` jumps to the post-body test.
                let test_pos = self.code_len();
                for j in &ctx.continue_jumps {
                    self.patch_jump(*j, test_pos);
                }
                let cond = self.expr(test)?;
                let back = self.emit(Op::JumpIfTrue { cond, offset: 0 });
                self.patch_jump(back, top);
                for j in &ctx.break_jumps {
                    self.patch_to_here(*j);
                }
                Ok(None)
            }
            Stmt::For {
                init,
                test,
                update,
                body,
                ..
            } => self.compile_for(init.as_ref(), test.as_deref(), update.as_deref(), body),
            Stmt::Throw { argument, .. } => {
                let src = self.expr(argument)?;
                self.emit(Op::Throw { src });
                Ok(None)
            }
            Stmt::Try {
                block,
                handler,
                finalizer,
                ..
            } => self.compile_try(block, handler.as_ref(), finalizer.as_deref()),
            Stmt::Switch {
                discriminant,
                cases,
                ..
            } => self.compile_switch(discriminant, cases),
            Stmt::ForOf {
                left, right, body, ..
            } => self.compile_for_each(left, right, body, false),
            Stmt::ForIn {
                left, right, body, ..
            } => self.compile_for_each(left, right, body, true),
            Stmt::Break { label: None, .. } => {
                let j = self.emit(Op::Jump { offset: 0 });
                self.add_break(j)?;
                Ok(None)
            }
            Stmt::Continue { label: None, .. } => {
                let j = self.emit(Op::Jump { offset: 0 });
                self.add_continue(j)?;
                Ok(None)
            }
            _ => Err(CompileError::unsupported("statement in bytecode mode")),
        }
    }

    /// Compiles a C-style `for (init; test; update) body`.
    fn compile_for(
        &mut self,
        init: Option<&crate::ast::ForInit>,
        test: Option<&Expr>,
        update: Option<&Expr>,
        body: &Stmt,
    ) -> Result<Option<Reg>, CompileError> {
        use crate::ast::{ForInit, VarDeclKind};
        let mark = self.funcs.last().expect("fn").locals.len();
        match init {
            Some(ForInit::Var(decl)) => {
                // A captured `let`/`const` loop variable needs a *fresh* binding
                // per iteration (so body closures capture distinct values). The
                // VM uses one cell for the whole loop, so hand these to the
                // tree-walker, which implements per-iteration environments.
                if matches!(decl.kind, VarDeclKind::Let | VarDeclKind::Const) {
                    for d in &decl.declarations {
                        if let BindingTarget::Ident(id) = &d.target
                            && self.is_captured_here(&id.name)
                        {
                            return Err(CompileError::unsupported(
                                "captured for-loop binding (per-iteration scope)",
                            ));
                        }
                    }
                }
                self.stmt(&Stmt::Var(decl.clone()))?;
            }
            Some(ForInit::Expr(e)) => {
                self.expr(e)?;
            }
            None => {}
        }
        let top = self.code_len();
        let exit = match test {
            Some(t) => {
                let cond = self.expr(t)?;
                Some(self.emit(Op::JumpIfFalse { cond, offset: 0 }))
            }
            None => None,
        };
        self.push_loop();
        self.stmt(body)?;
        let ctx = self.pop_loop();
        // `continue` runs the update step.
        let update_pos = self.code_len();
        for j in &ctx.continue_jumps {
            self.patch_jump(*j, update_pos);
        }
        if let Some(u) = update {
            self.expr(u)?;
        }
        let back = self.emit(Op::Jump { offset: 0 });
        self.patch_jump(back, top);
        if let Some(ej) = exit {
            self.patch_to_here(ej);
        }
        for j in &ctx.break_jumps {
            self.patch_to_here(*j);
        }
        self.funcs.last_mut().expect("fn").locals.truncate(mark);
        Ok(None)
    }

    /// Compiles `try { … } catch (e) { … } finally { … }`. With a `finally`
    /// block, the guarded code must not `return`/`break`/`continue` *out* of the
    /// try/catch (the VM has no completion-record unwinding yet) — those cases
    /// fall back to the tree-walker; the `finally` body is duplicated on the
    /// normal and exceptional exit paths.
    fn compile_try(
        &mut self,
        block: &[Stmt],
        handler: Option<&crate::ast::CatchClause>,
        finalizer: Option<&[Stmt]>,
    ) -> Result<Option<Reg>, CompileError> {
        if handler.is_none() && finalizer.is_none() {
            return Err(CompileError::unsupported("`try` without `catch`/`finally`"));
        }
        // The `finally` lowering can't yet thread abrupt completions through.
        if finalizer.is_some()
            && (escapes_via_abrupt(block) || handler.is_some_and(|h| escapes_via_abrupt(&h.body)))
        {
            return Err(CompileError::unsupported(
                "`finally` with return/break/continue",
            ));
        }

        // --- guarded region (try block) ---
        let err_reg = self.reg();
        let ph = self.emit(Op::PushHandler {
            catch: 0,
            err: err_reg,
        });
        let mark = self.funcs.last().expect("fn").locals.len();
        for s in block {
            self.stmt(s)?;
        }
        self.funcs.last_mut().expect("fn").locals.truncate(mark);
        self.emit(Op::PopHandler);
        // Normal completion of the try block jumps over the handler.
        let skip_handler = self.emit(Op::Jump { offset: 0 });

        // --- handler entry (try block threw; error in `err_reg`) ---
        let catch_pc = self.code_len();
        let ci = self.ci();
        self.module[ci].code[ph] = Op::PushHandler {
            catch: (catch_pc as i64 - (ph as i64 + 1)) as i32,
            err: err_reg,
        };

        if let Some(handler) = handler {
            // catch (e) { … } — optionally with a finally on both exits.
            if let Some(fin) = finalizer {
                // Guard the catch body so the finally also runs if it throws.
                let err2 = self.reg();
                let ph2 = self.emit(Op::PushHandler {
                    catch: 0,
                    err: err2,
                });
                self.compile_catch_body(handler, err_reg)?;
                self.emit(Op::PopHandler);
                let skip_rethrow = self.emit(Op::Jump { offset: 0 });
                // Catch body threw → run finally, then rethrow.
                let h2_pc = self.code_len();
                let ci = self.ci();
                self.module[ci].code[ph2] = Op::PushHandler {
                    catch: (h2_pc as i64 - (ph2 as i64 + 1)) as i32,
                    err: err2,
                };
                self.compile_finally(fin)?;
                self.emit(Op::Throw { src: err2 });
                self.patch_to_here(skip_rethrow);
            } else {
                self.compile_catch_body(handler, err_reg)?;
            }
        } else if let Some(fin) = finalizer {
            // try { … } finally { … } — run finally on the error path, rethrow.
            self.compile_finally(fin)?;
            self.emit(Op::Throw { src: err_reg });
        }

        // Normal path lands here.
        self.patch_to_here(skip_handler);
        if let Some(fin) = finalizer {
            self.compile_finally(fin)?;
        }
        Ok(None)
    }

    /// Compiles a catch clause body, binding its parameter to `err_reg`.
    fn compile_catch_body(
        &mut self,
        handler: &crate::ast::CatchClause,
        err_reg: Reg,
    ) -> Result<(), CompileError> {
        let mark = self.funcs.last().expect("fn").locals.len();
        if let Some(BindingTarget::Ident(id)) = &handler.param {
            self.declare_local(id.name.clone().into_string(), err_reg);
        } else if handler.param.is_some() {
            return Err(CompileError::unsupported("catch binding pattern"));
        }
        for s in &handler.body {
            self.stmt(s)?;
        }
        self.funcs.last_mut().expect("fn").locals.truncate(mark);
        Ok(())
    }

    /// Compiles a `finally` block (a scoped statement list).
    fn compile_finally(&mut self, fin: &[Stmt]) -> Result<(), CompileError> {
        let mark = self.funcs.last().expect("fn").locals.len();
        for s in fin {
            self.stmt(s)?;
        }
        self.funcs.last_mut().expect("fn").locals.truncate(mark);
        Ok(())
    }

    /// Binds `value_reg` to a binding target — a simple identifier or an
    /// array/object destructuring pattern (recursively).
    fn bind_pattern(&mut self, target: &BindingTarget, value_reg: Reg) -> Result<(), CompileError> {
        match target {
            BindingTarget::Ident(id) => {
                let name = id.name.clone().into_string();
                if self.is_captured_here(&name) {
                    // Captured by a nested function: box it in a cell.
                    let cell = self.new_cell(value_reg);
                    self.declare_local_cell(name, cell, true);
                } else {
                    let slot = self.reg();
                    self.emit(Op::Move {
                        dst: slot,
                        src: value_reg,
                    });
                    self.declare_local(name, slot);
                }
                Ok(())
            }
            BindingTarget::Array(pat) => {
                use crate::ast::ArrayPatternElement;
                for (i, el) in pat.elements.iter().enumerate() {
                    match el {
                        ArrayPatternElement::Hole => {}
                        ArrayPatternElement::Item {
                            target, default, ..
                        } => {
                            // element = value[i]
                            let idx = self.reg();
                            self.emit(Op::LoadInt {
                                dst: idx,
                                value: i as i32,
                            });
                            let element = self.reg();
                            self.emit(Op::GetElem {
                                dst: element,
                                obj: value_reg,
                                index: idx,
                            });
                            let element = self.apply_default(element, default.as_ref())?;
                            self.bind_pattern(target, element)?;
                        }
                        ArrayPatternElement::Rest { target, .. } => {
                            // rest = value.slice(i)
                            let key = self.reg();
                            let k = self.add_const(Const::Str(String::from("slice")));
                            self.emit(Op::LoadConst { dst: key, k });
                            let from = self.reg();
                            self.emit(Op::LoadInt {
                                dst: from,
                                value: i as i32,
                            });
                            // The single argument occupies a fresh window slot.
                            let args_base = self.funcs.last().expect("fn").next_reg;
                            let slot = self.reg();
                            self.emit(Op::Move {
                                dst: slot,
                                src: from,
                            });
                            let rest = self.reg();
                            self.emit(Op::CallMethod {
                                dst: rest,
                                recv: value_reg,
                                key,
                                args_base,
                                argc: 1,
                            });
                            self.bind_pattern(target, rest)?;
                        }
                    }
                }
                Ok(())
            }
            BindingTarget::Object(pat) => {
                for prop in &pat.properties {
                    let value = match &prop.key {
                        PropertyKey::Ident(name) => {
                            let dst = self.reg();
                            let key = self.add_const(Const::Str(name.clone().into_string()));
                            self.emit(Op::GetProp {
                                dst,
                                obj: value_reg,
                                key,
                            });
                            dst
                        }
                        PropertyKey::Str(s) => {
                            let dst = self.reg();
                            let key = self.add_const(Const::Str(s.clone().into_string()));
                            self.emit(Op::GetProp {
                                dst,
                                obj: value_reg,
                                key,
                            });
                            dst
                        }
                        PropertyKey::Computed(e) => {
                            let index = self.expr(e)?;
                            let dst = self.reg();
                            self.emit(Op::GetElem {
                                dst,
                                obj: value_reg,
                                index,
                            });
                            dst
                        }
                        _ => return Err(CompileError::unsupported("object pattern key")),
                    };
                    let value = self.apply_default(value, prop.default.as_ref())?;
                    self.bind_pattern(&prop.value, value)?;
                }
                // `{ a, ...rest }`: rest gets a copy of the source minus the
                // already-bound keys.
                if let Some(rest) = &pat.rest {
                    let rest_obj = self.reg();
                    self.emit(Op::NewObject { dst: rest_obj });
                    self.emit_object_assign(rest_obj, value_reg);
                    for prop in &pat.properties {
                        let key_reg = match &prop.key {
                            PropertyKey::Ident(name) => {
                                let r = self.reg();
                                let k = self.add_const(Const::Str(name.clone().into_string()));
                                self.emit(Op::LoadConst { dst: r, k });
                                r
                            }
                            PropertyKey::Str(s) => {
                                let r = self.reg();
                                let k = self.add_const(Const::Str(s.clone().into_string()));
                                self.emit(Op::LoadConst { dst: r, k });
                                r
                            }
                            // A computed excluded key can't be resolved
                            // statically for the rest set.
                            _ => return Err(CompileError::unsupported("computed key with rest")),
                        };
                        let removed = self.reg();
                        self.emit(Op::DeleteMember {
                            dst: removed,
                            obj: rest_obj,
                            key: key_reg,
                        });
                    }
                    self.bind_pattern(rest, rest_obj)?;
                }
                Ok(())
            }
        }
    }

    /// Emits `Object.assign(dst, src)` (copies `src`'s own enumerable props into
    /// `dst`, mutating it in place).
    fn emit_object_assign(&mut self, dst: Reg, src: Reg) {
        let object_global = self.reg();
        let g = self.add_const(Const::Str(String::from("Object")));
        self.emit(Op::GetGlobal {
            dst: object_global,
            name: g,
        });
        let key = self.reg();
        let k = self.add_const(Const::Str(String::from("assign")));
        self.emit(Op::LoadConst { dst: key, k });
        let base = self.funcs.last().expect("fn").next_reg;
        let s1 = self.reg();
        self.emit(Op::Move { dst: s1, src: dst });
        let s2 = self.reg();
        self.emit(Op::Move { dst: s2, src });
        let ret = self.reg();
        self.emit(Op::CallMethod {
            dst: ret,
            recv: object_global,
            key,
            args_base: base,
            argc: 2,
        });
    }

    /// If `default` is present, replaces `value_reg` with the default when it is
    /// `undefined`. Returns the register holding the resolved value.
    fn apply_default(
        &mut self,
        value_reg: Reg,
        default: Option<&Expr>,
    ) -> Result<Reg, CompileError> {
        let Some(default) = default else {
            return Ok(value_reg);
        };
        let undef = self.reg();
        self.emit(Op::LoadUndefined { dst: undef });
        let is_undef = self.reg();
        self.emit(Op::StrictEq {
            dst: is_undef,
            a: value_reg,
            b: undef,
        });
        // If not undefined, skip the default assignment.
        let jf = self.emit(Op::JumpIfFalse {
            cond: is_undef,
            offset: 0,
        });
        let d = self.expr(default)?;
        self.emit(Op::Move {
            dst: value_reg,
            src: d,
        });
        self.patch_to_here(jf);
        Ok(value_reg)
    }

    /// Compiles `for (x of iterable)` (`keys = false`) or `for (x in obj)`
    /// (`keys = true`). The source is materialized into an array of values/keys
    /// and walked by index; only a simple identifier loop variable is supported.
    fn compile_for_each(
        &mut self,
        left: &crate::ast::ForLeft,
        right: &Expr,
        body: &Stmt,
        keys: bool,
    ) -> Result<Option<Reg>, CompileError> {
        use crate::ast::ForLeft;
        let var_name = match left {
            ForLeft::Decl {
                target: BindingTarget::Ident(id),
                ..
            } => id.name.clone().into_string(),
            ForLeft::Target(t) => {
                if let Expr::Ident(id) = &**t {
                    id.name.clone().into_string()
                } else {
                    return Err(CompileError::unsupported("for-of/in target"));
                }
            }
            ForLeft::Decl { .. } => {
                return Err(CompileError::unsupported("for-of/in destructuring"));
            }
        };

        let iterable = self.expr(right)?;
        let arr = self.reg();
        if keys {
            self.emit(Op::IterKeys {
                dst: arr,
                src: iterable,
            });
        } else {
            self.emit(Op::IterValues {
                dst: arr,
                src: iterable,
            });
        }
        let len = self.reg();
        let klen = self.add_const(Const::Str(String::from("length")));
        self.emit(Op::GetProp {
            dst: len,
            obj: arr,
            key: klen,
        });
        let idx = self.reg();
        self.emit(Op::LoadInt { dst: idx, value: 0 });

        let mark = self.funcs.last().expect("fn").locals.len();
        let var_slot = self.reg();
        self.declare_local(var_name, var_slot);

        let top = self.code_len();
        let cond = self.reg();
        self.emit(Op::Lt {
            dst: cond,
            a: idx,
            b: len,
        });
        let exit = self.emit(Op::JumpIfFalse { cond, offset: 0 });
        // x = arr[idx]
        self.emit(Op::GetElem {
            dst: var_slot,
            obj: arr,
            index: idx,
        });
        self.push_loop();
        self.stmt(body)?;
        let ctx = self.pop_loop();
        // `continue` runs the index increment.
        let inc_pos = self.code_len();
        for j in &ctx.continue_jumps {
            self.patch_jump(*j, inc_pos);
        }
        let one = self.reg();
        self.emit(Op::LoadInt { dst: one, value: 1 });
        let next = self.reg();
        self.emit(Op::Add {
            dst: next,
            a: idx,
            b: one,
        });
        self.emit(Op::Move {
            dst: idx,
            src: next,
        });
        let back = self.emit(Op::Jump { offset: 0 });
        self.patch_jump(back, top);
        self.patch_to_here(exit);
        for j in &ctx.break_jumps {
            self.patch_to_here(*j);
        }
        self.funcs.last_mut().expect("fn").locals.truncate(mark);
        Ok(None)
    }

    /// Compiles a `switch` statement: each case test is compared (`===`) to the
    /// discriminant, dispatching to the matching clause; clauses fall through,
    /// `break` exits, and `default` catches non-matches.
    fn compile_switch(
        &mut self,
        discriminant: &Expr,
        cases: &[crate::ast::SwitchCase],
    ) -> Result<Option<Reg>, CompileError> {
        let disc = self.expr(discriminant)?;
        // Emit the dispatch ladder: for each `case`, `JumpIfTrue` to its body.
        let mut case_jumps = Vec::new();
        let mut default_jump = None;
        for case in cases {
            match &case.test {
                Some(test) => {
                    let t = self.expr(test)?;
                    let eq = self.reg();
                    self.emit(Op::StrictEq {
                        dst: eq,
                        a: disc,
                        b: t,
                    });
                    case_jumps.push(self.emit(Op::JumpIfTrue {
                        cond: eq,
                        offset: 0,
                    }));
                }
                None => case_jumps.push(usize::MAX), // placeholder for `default`
            }
        }
        // No case matched: jump to `default` (if any) or past the switch.
        let fallthrough = self.emit(Op::Jump { offset: 0 });

        self.push_ctx(false); // a switch is a `break` target only
        let mark = self.funcs.last().expect("fn").locals.len();
        for (i, case) in cases.iter().enumerate() {
            let body_pc = self.code_len();
            if case.test.is_some() {
                self.patch_jump(case_jumps[i], body_pc);
            } else {
                default_jump = Some(body_pc);
            }
            for s in &case.body {
                self.stmt(s)?;
            }
        }
        self.funcs.last_mut().expect("fn").locals.truncate(mark);
        // Patch the no-match jump to `default` (or to the end).
        let end = self.code_len();
        self.patch_jump(fallthrough, default_jump.unwrap_or(end));
        let ctx = self.pop_loop();
        for j in &ctx.break_jumps {
            self.patch_jump(*j, end);
        }
        Ok(None)
    }

    // --- loop context (break/continue) ----------------------------------

    fn push_loop(&mut self) {
        self.push_ctx(true);
    }
    fn push_ctx(&mut self, is_loop: bool) {
        self.funcs.last_mut().expect("fn").loops.push(LoopCtx {
            break_jumps: Vec::new(),
            continue_jumps: Vec::new(),
            is_loop,
        });
    }
    fn pop_loop(&mut self) -> LoopCtx {
        self.funcs
            .last_mut()
            .expect("fn")
            .loops
            .pop()
            .expect("a current loop")
    }
    fn add_break(&mut self, idx: usize) -> Result<(), CompileError> {
        match self.funcs.last_mut().expect("fn").loops.last_mut() {
            Some(l) => {
                l.break_jumps.push(idx);
                Ok(())
            }
            None => Err(CompileError::unsupported("`break` outside a loop/switch")),
        }
    }
    fn add_continue(&mut self, idx: usize) -> Result<(), CompileError> {
        // `continue` targets the nearest enclosing *loop* (skipping switches).
        match self
            .funcs
            .last_mut()
            .expect("fn")
            .loops
            .iter_mut()
            .rev()
            .find(|l| l.is_loop)
        {
            Some(l) => {
                l.continue_jumps.push(idx);
                Ok(())
            }
            None => Err(CompileError::unsupported("`continue` outside a loop")),
        }
    }

    /// Compiles a function declaration and binds it as a global.
    fn compile_function_decl(&mut self, func: &Function) -> Result<(), CompileError> {
        let Some(id) = &func.id else {
            return Err(CompileError::unsupported("anonymous function declaration"));
        };
        let (chunk_idx, upvalues) =
            self.compile_function(&func.params, FnBody::Block(&func.body), &id.name)?;
        // Hoisted function declarations are bound before later locals exist, so
        // a declaration that captures an enclosing variable can't build its
        // closure here — fall back to the tree-walker.
        if !upvalues.is_empty() {
            return Err(CompileError::unsupported("captured function declaration"));
        }
        let dst = self.reg();
        let k = self.add_const(Const::Func(chunk_idx as u32));
        self.emit(Op::LoadConst { dst, k });
        let name = self.add_const(Const::Str(id.name.clone().into_string()));
        self.emit(Op::SetGlobal { name, src: dst });
        Ok(())
    }

    /// Compiles a class to a constructor-function value carrying its prototype
    /// (methods) and static members. Supports a constructor, instance/static
    /// methods, and instance/static fields with simple keys. `extends`/`super`,
    /// accessors, computed/private keys, generator/async methods, static
    /// blocks, and captured members fall back to the tree-walker.
    fn compile_class(&mut self, class: &crate::ast::Class) -> Result<Reg, CompileError> {
        use crate::ast::{ClassMember, MethodKind};
        // For `class B extends A`, evaluate the superclass and stash it in a
        // synthetic captured cell (`%super%`) so the constructor/methods can
        // reach it as an upvalue for `super(...)` / `super.m(...)`.
        let super_mark = self.funcs.last().expect("fn").locals.len();
        let has_super = class.super_class.is_some();
        if let Some(sc) = &class.super_class {
            let super_val = self.expr(sc)?;
            let cell = self.new_cell(super_val);
            self.declare_local_cell(String::from(SUPER_NAME), cell, true);
        }
        let mut ctor: Option<&Function> = None;
        let mut instance_methods: Vec<(String, &Function)> = Vec::new();
        let mut static_methods: Vec<(String, &Function)> = Vec::new();
        let mut instance_fields: Vec<(String, Option<&Expr>)> = Vec::new();
        let mut static_fields: Vec<(String, Option<&Expr>)> = Vec::new();
        // Accessors: (name, is_getter, function).
        let mut instance_accessors: Vec<(String, bool, &Function)> = Vec::new();
        let mut static_accessors: Vec<(String, bool, &Function)> = Vec::new();
        for member in &class.body {
            match member {
                ClassMember::Method(m) => {
                    if m.value.is_generator || m.value.is_async {
                        return Err(CompileError::unsupported("generator/async method"));
                    }
                    match m.kind {
                        MethodKind::Constructor => ctor = Some(&m.value),
                        MethodKind::Method => {
                            let name = simple_key(&m.key)?;
                            if m.is_static {
                                static_methods.push((name, &m.value));
                            } else {
                                instance_methods.push((name, &m.value));
                            }
                        }
                        MethodKind::Get | MethodKind::Set => {
                            let name = simple_key(&m.key)?;
                            let is_getter = matches!(m.kind, MethodKind::Get);
                            if m.is_static {
                                static_accessors.push((name, is_getter, &m.value));
                            } else {
                                instance_accessors.push((name, is_getter, &m.value));
                            }
                        }
                    }
                }
                ClassMember::Field(f) => {
                    let name = simple_key(&f.key)?;
                    if f.is_static {
                        static_fields.push((name, f.value.as_ref()));
                    } else {
                        instance_fields.push((name, f.value.as_ref()));
                    }
                }
                ClassMember::StaticBlock { .. } => {
                    return Err(CompileError::unsupported("class static block"));
                }
            }
        }

        // The constructor chunk runs the instance-field initializers, then the
        // declared constructor body. A subclass with no explicit constructor
        // gets the default `constructor(...args) { super(...args); }`.
        let (ctor_chunk, ctor_ups) = if has_super && ctor.is_none() {
            self.compile_default_subclass_ctor(&instance_fields)?
        } else {
            self.compile_constructor(ctor, &instance_fields)?
        };
        let ctor_reg = self.emit_closure(ctor_chunk, &ctor_ups);

        // Build the prototype object with the instance methods.
        let proto = self.reg();
        self.emit(Op::NewObject { dst: proto });
        for (name, func) in instance_methods {
            let m = self.class_method_value(func)?;
            let key = self.add_const(Const::Str(name));
            self.emit(Op::SetProp {
                obj: proto,
                key,
                src: m,
            });
        }
        let proto_key = self.add_const(Const::Str(String::from("prototype")));
        self.emit(Op::SetProp {
            obj: ctor_reg,
            key: proto_key,
            src: proto,
        });

        // Inheritance: chain the prototypes (`B.prototype.__proto__ =
        // A.prototype`) and the constructors (`B.__proto__ = A`, for static
        // inheritance), using `Object.setPrototypeOf`.
        if has_super {
            let super_val = self.read_ident(SUPER_NAME)?;
            let super_proto = self.reg();
            let pk = self.add_const(Const::Str(String::from("prototype")));
            self.emit(Op::GetProp {
                dst: super_proto,
                obj: super_val,
                key: pk,
            });
            self.emit_set_prototype_of(proto, super_proto);
            let super_val2 = self.read_ident(SUPER_NAME)?;
            self.emit_set_prototype_of(ctor_reg, super_val2);
        }

        // Instance getters/setters go on the prototype.
        for (name, is_getter, func) in instance_accessors {
            let f = self.class_method_value(func)?;
            self.emit_define_accessor(proto, &name, is_getter, f);
        }

        // Static methods and fields live on the constructor object itself.
        for (name, func) in static_methods {
            let m = self.class_method_value(func)?;
            let key = self.add_const(Const::Str(name));
            self.emit(Op::SetProp {
                obj: ctor_reg,
                key,
                src: m,
            });
        }
        for (name, is_getter, func) in static_accessors {
            let f = self.class_method_value(func)?;
            self.emit_define_accessor(ctor_reg, &name, is_getter, f);
        }
        for (name, init) in static_fields {
            let val = match init {
                Some(e) => self.expr(e)?,
                None => {
                    let r = self.reg();
                    self.emit(Op::LoadUndefined { dst: r });
                    r
                }
            };
            let key = self.add_const(Const::Str(name));
            self.emit(Op::SetProp {
                obj: ctor_reg,
                key,
                src: val,
            });
        }
        // Drop the synthetic `%super%` binding.
        self.funcs
            .last_mut()
            .expect("fn")
            .locals
            .truncate(super_mark);
        Ok(ctor_reg)
    }

    /// Installs an accessor on `obj`: `Object.defineProperty(obj, name,
    /// { get|set: func, enumerable: true, configurable: true })`.
    fn emit_define_accessor(&mut self, obj: Reg, name: &str, is_getter: bool, func: Reg) {
        // Build the descriptor object.
        let desc = self.reg();
        self.emit(Op::NewObject { dst: desc });
        let accessor_key = self.add_const(Const::Str(String::from(if is_getter {
            "get"
        } else {
            "set"
        })));
        self.emit(Op::SetProp {
            obj: desc,
            key: accessor_key,
            src: func,
        });
        for flag in ["enumerable", "configurable"] {
            let t = self.reg();
            self.emit(Op::LoadBool {
                dst: t,
                value: true,
            });
            let fk = self.add_const(Const::Str(String::from(flag)));
            self.emit(Op::SetProp {
                obj: desc,
                key: fk,
                src: t,
            });
        }
        // The property name as a register.
        let name_reg = self.reg();
        let nk = self.add_const(Const::Str(String::from(name)));
        self.emit(Op::LoadConst {
            dst: name_reg,
            k: nk,
        });
        // Object.defineProperty(obj, name, desc).
        let object_global = self.reg();
        let g = self.add_const(Const::Str(String::from("Object")));
        self.emit(Op::GetGlobal {
            dst: object_global,
            name: g,
        });
        let key = self.reg();
        let k = self.add_const(Const::Str(String::from("defineProperty")));
        self.emit(Op::LoadConst { dst: key, k });
        let base = self.funcs.last().expect("fn").next_reg;
        for src in [obj, name_reg, desc] {
            let slot = self.reg();
            self.emit(Op::Move { dst: slot, src });
        }
        let ret = self.reg();
        self.emit(Op::CallMethod {
            dst: ret,
            recv: object_global,
            key,
            args_base: base,
            argc: 3,
        });
    }

    /// `Object.setPrototypeOf(target, proto)`.
    fn emit_set_prototype_of(&mut self, target: Reg, proto: Reg) {
        let object_global = self.reg();
        let g = self.add_const(Const::Str(String::from("Object")));
        self.emit(Op::GetGlobal {
            dst: object_global,
            name: g,
        });
        let key = self.reg();
        let k = self.add_const(Const::Str(String::from("setPrototypeOf")));
        self.emit(Op::LoadConst { dst: key, k });
        let base = self.funcs.last().expect("fn").next_reg;
        let s1 = self.reg();
        self.emit(Op::Move {
            dst: s1,
            src: target,
        });
        let s2 = self.reg();
        self.emit(Op::Move {
            dst: s2,
            src: proto,
        });
        let ret = self.reg();
        self.emit(Op::CallMethod {
            dst: ret,
            recv: object_global,
            key,
            args_base: base,
            argc: 2,
        });
    }

    /// Compiles one class method to a function value (may capture `%super%` or
    /// other enclosing variables as upvalues).
    fn class_method_value(&mut self, func: &Function) -> Result<Reg, CompileError> {
        let (idx, upvalues) =
            self.compile_function(&func.params, FnBody::Block(&func.body), "<method>")?;
        Ok(self.emit_closure(idx, &upvalues))
    }

    /// Compiles the implicit subclass constructor
    /// `constructor(...args) { super(...args); <field inits> }`.
    fn compile_default_subclass_ctor(
        &mut self,
        fields: &[(String, Option<&Expr>)],
    ) -> Result<(usize, Vec<Upvalue>), CompileError> {
        let chunk_idx = self.module.len();
        self.module.push(Chunk::new("<constructor>"));
        self.module[chunk_idx].param_count = 1;
        self.module[chunk_idx].has_rest = true;

        let mut state = FnState::new(chunk_idx, alloc::collections::BTreeSet::new());
        // Register 1 receives the collected rest array (the forwarded args).
        let args_reg = state.next_reg;
        state.next_reg += 1;
        self.funcs.push(state);

        let result: Result<(), CompileError> = (|| {
            // super.apply(this, args)
            let super_val = self.read_ident(SUPER_NAME)?;
            self.emit_apply(super_val, THIS_REG, args_reg);
            // Field initializers run after the super call.
            for (name, init) in fields {
                let val = match init {
                    Some(e) => self.expr(e)?,
                    None => {
                        let r = self.reg();
                        self.emit(Op::LoadUndefined { dst: r });
                        r
                    }
                };
                let key = self.add_const(Const::Str(name.clone()));
                self.emit(Op::SetProp {
                    obj: THIS_REG,
                    key,
                    src: val,
                });
            }
            Ok(())
        })();
        if let Err(e) = result {
            self.funcs.pop();
            return Err(e);
        }
        self.emit(Op::ReturnUndefined);
        let finished = self.funcs.pop().expect("fn");
        self.module[chunk_idx].register_count = finished.next_reg;
        Ok((chunk_idx, finished.upvalues))
    }

    /// Compiles a class constructor chunk: instance-field initializers
    /// (`this.f = …`) followed by the declared constructor body. Returns the
    /// chunk index and any captured upvalues (e.g. `%super%`).
    fn compile_constructor(
        &mut self,
        ctor: Option<&Function>,
        fields: &[(String, Option<&Expr>)],
    ) -> Result<(usize, Vec<Upvalue>), CompileError> {
        let params: &[Param] = ctor.map_or(&[], |f| &f.params);
        if params.iter().any(|p| p.rest || p.default.is_some()) {
            return Err(CompileError::unsupported("constructor rest/default param"));
        }
        let chunk_idx = self.module.len();
        self.module.push(Chunk::new("<constructor>"));
        self.module[chunk_idx].param_count = params.len() as u16;

        let mut state = FnState::new(chunk_idx, alloc::collections::BTreeSet::new());
        let mut param_slots = Vec::new();
        for _ in params {
            let r = state.next_reg;
            state.next_reg += 1;
            param_slots.push(r);
        }
        self.funcs.push(state);
        for (p, slot) in params.iter().zip(param_slots) {
            if let BindingTarget::Ident(id) = &p.target {
                self.declare_local(id.name.clone().into_string(), slot);
            } else {
                self.funcs.pop();
                return Err(CompileError::unsupported("constructor pattern param"));
            }
        }

        // Instance fields: `this.name = <init>` (or `undefined`).
        let result: Result<(), CompileError> = (|| {
            for (name, init) in fields {
                let val = match init {
                    Some(e) => self.expr(e)?,
                    None => {
                        let r = self.reg();
                        self.emit(Op::LoadUndefined { dst: r });
                        r
                    }
                };
                let key = self.add_const(Const::Str(name.clone()));
                self.emit(Op::SetProp {
                    obj: THIS_REG,
                    key,
                    src: val,
                });
            }
            if let Some(f) = ctor {
                self.compile_body(&f.body, false)?;
            }
            Ok(())
        })();
        if let Err(e) = result {
            self.funcs.pop();
            return Err(e);
        }
        self.emit(Op::ReturnUndefined);
        let finished = self.funcs.pop().expect("fn");
        self.module[chunk_idx].register_count = finished.next_reg;
        Ok((chunk_idx, finished.upvalues))
    }

    /// Compiles a function body into a new chunk; returns the chunk index and
    /// the upvalues it captured (so the caller can build a closure over it).
    fn compile_function(
        &mut self,
        params: &[Param],
        body: FnBody,
        name: &str,
    ) -> Result<(usize, Vec<Upvalue>), CompileError> {
        // Only a trailing rest parameter is supported (`f(a, ...rest)`), and it
        // must be a simple identifier.
        let has_rest = params.last().is_some_and(|p| p.rest);
        if params.iter().enumerate().any(|(i, p)| {
            p.rest && (i + 1 != params.len() || !matches!(p.target, BindingTarget::Ident(_)))
        }) {
            return Err(CompileError::unsupported("rest parameter form"));
        }

        let chunk_idx = self.module.len();
        self.module.push(Chunk::new(name));
        self.module[chunk_idx].param_count = params.len() as u16;
        self.module[chunk_idx].has_rest = has_rest;

        // Pre-pass: which of this function's params/locals are captured by
        // nested functions (and so must be boxed in cells)?
        let captured = captured_names(params, &body);
        let mut state = FnState::new(chunk_idx, captured);
        // Reserve a positional register per parameter (the VM binds args here).
        let mut param_slots = Vec::new();
        for _ in params {
            let r = state.next_reg;
            state.next_reg += 1;
            param_slots.push(r);
        }
        self.funcs.push(state);

        // Bind each parameter: a plain (non-captured) identifier maps to its
        // slot directly; defaults, captures, and patterns go through the
        // general binding path.
        for (p, slot) in params.iter().zip(param_slots) {
            // The rest parameter's slot already holds the collected array (the
            // VM fills it); bind its name, boxing it if captured.
            if p.rest {
                if let BindingTarget::Ident(id) = &p.target {
                    let rest_name = id.name.clone().into_string();
                    if self.is_captured_here(&rest_name) {
                        let cell = self.new_cell(slot);
                        self.declare_local_cell(rest_name, cell, true);
                    } else {
                        self.declare_local(rest_name, slot);
                    }
                }
                continue;
            }
            let bound = self.apply_default(slot, p.default.as_ref())?;
            match &p.target {
                BindingTarget::Ident(id)
                    if p.default.is_none() && !self.is_captured_here(&id.name) =>
                {
                    self.declare_local(id.name.clone().into_string(), slot);
                }
                _ => self.bind_pattern(&p.target, bound)?,
            }
        }

        match body {
            FnBody::Block(stmts) => self.compile_body(stmts, false)?,
            FnBody::Expr(expr) => {
                let r = self.expr(expr)?;
                self.emit(Op::Return { src: r });
            }
        }
        self.emit(Op::ReturnUndefined);

        let finished = self.funcs.pop().expect("fn");
        self.module[chunk_idx].register_count = finished.next_reg;
        Ok((chunk_idx, finished.upvalues))
    }

    /// Emits the value for a compiled function: a plain function constant when
    /// it has no upvalues, or a `MakeClosure` capturing the upvalue cells.
    fn emit_closure(&mut self, chunk_idx: usize, upvalues: &[Upvalue]) -> Reg {
        if upvalues.is_empty() {
            let dst = self.reg();
            let k = self.add_const(Const::Func(chunk_idx as u32));
            self.emit(Op::LoadConst { dst, k });
            return dst;
        }
        // Gather the captured cells into a contiguous register window.
        let base = self.funcs.last().expect("fn").next_reg;
        for up in upvalues {
            let slot = self.reg();
            match up.source {
                UpvalSource::ParentLocal(reg) => {
                    self.emit(Op::Move {
                        dst: slot,
                        src: reg,
                    });
                }
                UpvalSource::ParentUpval(idx) => {
                    self.emit(Op::GetUpvalue { dst: slot, idx });
                }
            }
        }
        let dst = self.reg();
        self.emit(Op::MakeClosure {
            dst,
            chunk: chunk_idx as u32,
            upvals_base: base,
            count: upvalues.len() as u16,
        });
        dst
    }

    fn patch_to_here(&mut self, idx: usize) {
        let target = self.code_len();
        self.patch_jump(idx, target);
    }

    fn patch_jump(&mut self, idx: usize, target: usize) {
        let offset = (target as i64 - (idx as i64 + 1)) as i32;
        let ci = self.ci();
        self.module[ci].code[idx] = match &self.module[ci].code[idx] {
            Op::Jump { .. } => Op::Jump { offset },
            Op::JumpIfFalse { cond, .. } => Op::JumpIfFalse {
                cond: *cond,
                offset,
            },
            Op::JumpIfTrue { cond, .. } => Op::JumpIfTrue {
                cond: *cond,
                offset,
            },
            other => other.clone(),
        };
    }

    // --- expressions ----------------------------------------------------

    fn expr(&mut self, expr: &Expr) -> Result<Reg, CompileError> {
        match expr {
            Expr::Number { value, .. } => {
                let dst = self.reg();
                if value.fract() == 0.0
                    && *value >= f64::from(i32::MIN)
                    && *value <= f64::from(i32::MAX)
                {
                    self.emit(Op::LoadInt {
                        dst,
                        value: *value as i32,
                    });
                } else {
                    let k = self.add_const(Const::Number(*value));
                    self.emit(Op::LoadConst { dst, k });
                }
                Ok(dst)
            }
            Expr::Str { value, .. } => {
                let dst = self.reg();
                let k = self.add_const(Const::Str(value.clone().into_string()));
                self.emit(Op::LoadConst { dst, k });
                Ok(dst)
            }
            Expr::Bool { value, .. } => {
                let dst = self.reg();
                self.emit(Op::LoadBool { dst, value: *value });
                Ok(dst)
            }
            Expr::Null(_) => {
                let dst = self.reg();
                self.emit(Op::LoadNull { dst });
                Ok(dst)
            }
            Expr::This(_) => {
                let dst = self.reg();
                self.emit(Op::Move { dst, src: THIS_REG });
                Ok(dst)
            }
            // A bare `super` (only valid inside a subclass method) resolves to
            // the captured superclass constructor.
            Expr::Super(_) => self.read_ident(SUPER_NAME),
            Expr::Ident(id) => self.read_ident(&id.name),
            Expr::Assign {
                op, target, value, ..
            } => self.assign(*op, target, value),
            Expr::Unary { op, argument, .. } => self.unary(*op, argument),
            Expr::Update {
                op,
                prefix,
                argument,
                ..
            } => self.update(*op, *prefix, argument),
            Expr::Binary {
                op, left, right, ..
            } => self.binary(*op, left, right),
            Expr::Logical {
                op, left, right, ..
            } => self.logical(*op, left, right),
            Expr::Member {
                object,
                property,
                optional,
                ..
            } => {
                if *optional {
                    return Err(CompileError::unsupported("optional chaining"));
                }
                self.member(object, property)
            }
            Expr::Call {
                callee,
                arguments,
                optional,
                ..
            } => {
                if *optional {
                    return Err(CompileError::unsupported("optional call"));
                }
                self.call(callee, arguments)
            }
            Expr::Class(class) => self.compile_class(class),
            Expr::Function(func) => {
                let (idx, upvalues) = self.compile_function(
                    &func.params,
                    FnBody::Block(&func.body),
                    func.id.as_ref().map_or("<anonymous>", |id| &id.name),
                )?;
                Ok(self.emit_closure(idx, &upvalues))
            }
            Expr::Arrow(arrow) => self.arrow(arrow),
            Expr::New {
                callee, arguments, ..
            } => {
                let callee_reg = self.expr(callee)?;
                let (args_base, argc) = self.lower_args(arguments)?;
                let dst = self.reg();
                self.emit(Op::New {
                    dst,
                    callee: callee_reg,
                    args_base,
                    argc,
                });
                Ok(dst)
            }
            Expr::Array { elements, .. } => self.array_literal(elements),
            Expr::Object { members, .. } => self.object_literal(members),
            Expr::Conditional {
                test,
                consequent,
                alternate,
                ..
            } => self.conditional(test, consequent, alternate),
            Expr::Template(t) => self.template(t),
            _ => Err(CompileError::unsupported("expression")),
        }
    }

    /// `test ? consequent : alternate`, producing the chosen value in `dst`.
    fn conditional(
        &mut self,
        test: &Expr,
        consequent: &Expr,
        alternate: &Expr,
    ) -> Result<Reg, CompileError> {
        let dst = self.reg();
        let cond = self.expr(test)?;
        let jf = self.emit(Op::JumpIfFalse { cond, offset: 0 });
        let c = self.expr(consequent)?;
        self.emit(Op::Move { dst, src: c });
        let jmp = self.emit(Op::Jump { offset: 0 });
        self.patch_to_here(jf);
        let a = self.expr(alternate)?;
        self.emit(Op::Move { dst, src: a });
        self.patch_to_here(jmp);
        Ok(dst)
    }

    /// A template literal: the quasis and interpolations are concatenated with
    /// `Add` (string coercion, since the quasis are strings).
    fn template(&mut self, t: &crate::ast::TemplateLiteral) -> Result<Reg, CompileError> {
        let cooked = |q: &crate::ast::TemplateElement| {
            q.cooked
                .as_ref()
                .map_or(String::new(), |c| c.clone().into_string())
        };
        // Start with the first quasi.
        let acc = self.reg();
        let k0 = self.add_const(Const::Str(cooked(&t.quasis[0])));
        self.emit(Op::LoadConst { dst: acc, k: k0 });
        for (i, expr) in t.expressions.iter().enumerate() {
            // acc = acc + <expr>
            let e = self.expr(expr)?;
            let next = self.reg();
            self.emit(Op::Add {
                dst: next,
                a: acc,
                b: e,
            });
            // acc = next + quasi[i+1]
            let qreg = self.reg();
            let kq = self.add_const(Const::Str(cooked(&t.quasis[i + 1])));
            self.emit(Op::LoadConst { dst: qreg, k: kq });
            let joined = self.reg();
            self.emit(Op::Add {
                dst: joined,
                a: next,
                b: qreg,
            });
            self.emit(Op::Move {
                dst: acc,
                src: joined,
            });
        }
        Ok(acc)
    }

    fn array_literal(
        &mut self,
        elements: &[crate::ast::ArrayElement],
    ) -> Result<Reg, CompileError> {
        use crate::ast::ArrayElement;
        // The spread path builds the array with `concat`; the common path
        // (no spread) uses direct indexed writes.
        if elements
            .iter()
            .any(|e| matches!(e, ArrayElement::Spread(_)))
        {
            return self.array_literal_spread(elements);
        }
        let dst = self.reg();
        self.emit(Op::NewArray {
            dst,
            len: elements.len() as u32,
        });
        for (i, el) in elements.iter().enumerate() {
            let value = match el {
                ArrayElement::Item(e) => self.expr(e)?,
                ArrayElement::Hole => continue,
                ArrayElement::Spread(_) => unreachable!("handled above"),
            };
            let index = self.reg();
            self.emit(Op::LoadInt {
                dst: index,
                value: i as i32,
            });
            self.emit(Op::SetElem {
                obj: dst,
                index,
                src: value,
            });
        }
        Ok(dst)
    }

    /// Builds an array literal containing spreads: start empty, then `concat`
    /// each element (a one-element array for an item, the iterated values for a
    /// spread).
    fn array_literal_spread(
        &mut self,
        elements: &[crate::ast::ArrayElement],
    ) -> Result<Reg, CompileError> {
        use crate::ast::ArrayElement;
        let result = self.reg();
        self.emit(Op::NewArray {
            dst: result,
            len: 0,
        });
        for el in elements {
            let chunk = match el {
                ArrayElement::Item(e) => {
                    let v = self.expr(e)?;
                    let one = self.reg();
                    self.emit(Op::NewArray { dst: one, len: 1 });
                    let idx = self.reg();
                    self.emit(Op::LoadInt { dst: idx, value: 0 });
                    self.emit(Op::SetElem {
                        obj: one,
                        index: idx,
                        src: v,
                    });
                    one
                }
                ArrayElement::Hole => {
                    let one = self.reg();
                    self.emit(Op::NewArray { dst: one, len: 1 });
                    one
                }
                ArrayElement::Spread(e) => {
                    let v = self.expr(e)?;
                    let items = self.reg();
                    self.emit(Op::IterValues { dst: items, src: v });
                    items
                }
            };
            self.concat_into(result, chunk);
        }
        Ok(result)
    }

    /// `result = result.concat(arg)` (used when building spread arrays).
    fn concat_into(&mut self, result: Reg, arg: Reg) {
        let key = self.reg();
        let k = self.add_const(Const::Str(String::from("concat")));
        self.emit(Op::LoadConst { dst: key, k });
        let args_base = self.funcs.last().expect("fn").next_reg;
        let slot = self.reg();
        self.emit(Op::Move {
            dst: slot,
            src: arg,
        });
        let joined = self.reg();
        self.emit(Op::CallMethod {
            dst: joined,
            recv: result,
            key,
            args_base,
            argc: 1,
        });
        self.emit(Op::Move {
            dst: result,
            src: joined,
        });
    }

    fn object_literal(
        &mut self,
        members: &[crate::ast::ObjectMember],
    ) -> Result<Reg, CompileError> {
        use crate::ast::ObjectMember;
        let dst = self.reg();
        self.emit(Op::NewObject { dst });
        for member in members {
            // `{ ...src }` copies `src`'s own enumerable properties via
            // `Object.assign(dst, src)` (which mutates and returns `dst`).
            if let ObjectMember::Spread { value, .. } = member {
                let src = self.expr(value)?;
                self.emit_object_assign(dst, src);
                continue;
            }
            // `{ get x() {…} }` / `{ set x(v) {…} }` install an accessor.
            if let ObjectMember::Accessor {
                is_getter,
                key,
                value,
                ..
            } = member
            {
                let name = simple_key(key)?;
                let func = self.class_method_value(value)?;
                self.emit_define_accessor(dst, &name, *is_getter, func);
                continue;
            }
            let ObjectMember::Property { key, value, .. } = member else {
                return Err(CompileError::unsupported("object member"));
            };
            let val = self.expr(value)?;
            match key {
                PropertyKey::Ident(name) => {
                    let k = self.add_const(Const::Str(name.clone().into_string()));
                    self.emit(Op::SetProp {
                        obj: dst,
                        key: k,
                        src: val,
                    });
                }
                PropertyKey::Str(s) => {
                    let k = self.add_const(Const::Str(s.clone().into_string()));
                    self.emit(Op::SetProp {
                        obj: dst,
                        key: k,
                        src: val,
                    });
                }
                PropertyKey::Number(n) => {
                    let index = self.reg();
                    let k = self.add_const(Const::Number(*n));
                    self.emit(Op::LoadConst { dst: index, k });
                    self.emit(Op::SetElem {
                        obj: dst,
                        index,
                        src: val,
                    });
                }
                PropertyKey::Computed(expr) => {
                    let index = self.expr(expr)?;
                    self.emit(Op::SetElem {
                        obj: dst,
                        index,
                        src: val,
                    });
                }
                PropertyKey::Private(_) => {
                    return Err(CompileError::unsupported("private object key"));
                }
            }
        }
        Ok(dst)
    }

    fn arrow(&mut self, arrow: &Arrow) -> Result<Reg, CompileError> {
        let body = match &arrow.body {
            ArrowBody::Expr(e) => FnBody::Expr(e),
            ArrowBody::Block(b) => FnBody::Block(b),
        };
        let (idx, upvalues) = self.compile_function(&arrow.params, body, "<arrow>")?;
        Ok(self.emit_closure(idx, &upvalues))
    }

    /// Reads `name` (local copy, or global) into a fresh register; a reference
    /// to an enclosing function's local is an unsupported capture.
    fn read_ident(&mut self, name: &str) -> Result<Reg, CompileError> {
        if name == "undefined" {
            let dst = self.reg();
            self.emit(Op::LoadUndefined { dst });
            return Ok(dst);
        }
        let binding = self.lookup_binding(name)?;
        Ok(self.read_binding(&binding, name))
    }

    fn assign(&mut self, op: AssignOp, target: &Expr, value: &Expr) -> Result<Reg, CompileError> {
        match target {
            Expr::Ident(id) => self.assign_ident(op, &id.name, value),
            Expr::Member {
                object,
                property,
                optional: false,
                ..
            } => self.assign_member(op, object, property, value),
            _ => Err(CompileError::unsupported("assignment target")),
        }
    }

    fn assign_ident(
        &mut self,
        op: AssignOp,
        name: &str,
        value: &Expr,
    ) -> Result<Reg, CompileError> {
        let rhs = self.expr(value)?;
        let result = match compound_binop(op) {
            None => rhs,
            Some(binop) => {
                let cur = self.read_ident(name)?;
                let dst = self.reg();
                self.emit_binop(binop, dst, cur, rhs)?;
                dst
            }
        };
        let binding = self.lookup_binding(name)?;
        self.write_binding(&binding, name, result);
        Ok(result)
    }

    /// Compiles `obj.prop OP= value` / `obj[key] OP= value`.
    fn assign_member(
        &mut self,
        op: AssignOp,
        object: &Expr,
        property: &PropertyKey,
        value: &Expr,
    ) -> Result<Reg, CompileError> {
        let obj = self.expr(object)?;
        // Resolve the property to either a string-constant key or an index reg.
        let key: PropertySlot = match property {
            PropertyKey::Ident(name) => {
                PropertySlot::Const(self.add_const(Const::Str(name.clone().into_string())))
            }
            PropertyKey::Str(s) => {
                PropertySlot::Const(self.add_const(Const::Str(s.clone().into_string())))
            }
            PropertyKey::Computed(expr) => PropertySlot::Index(self.expr(expr)?),
            _ => return Err(CompileError::unsupported("member assignment key")),
        };
        let rhs = self.expr(value)?;
        let result = match compound_binop(op) {
            None => rhs,
            Some(binop) => {
                // Read the current member, fold, and write the result back.
                let cur = self.reg();
                match key {
                    PropertySlot::Const(k) => self.emit(Op::GetProp {
                        dst: cur,
                        obj,
                        key: k,
                    }),
                    PropertySlot::Index(idx) => self.emit(Op::GetElem {
                        dst: cur,
                        obj,
                        index: idx,
                    }),
                };
                let dst = self.reg();
                self.emit_binop(binop, dst, cur, rhs)?;
                dst
            }
        };
        match key {
            PropertySlot::Const(k) => self.emit(Op::SetProp {
                obj,
                key: k,
                src: result,
            }),
            PropertySlot::Index(idx) => self.emit(Op::SetElem {
                obj,
                index: idx,
                src: result,
            }),
        };
        Ok(result)
    }

    /// Compiles `++x` / `x++` / `--x` / `x--` on an identifier or member.
    fn update(
        &mut self,
        op: crate::ast::UpdateOp,
        prefix: bool,
        argument: &Expr,
    ) -> Result<Reg, CompileError> {
        use crate::ast::UpdateOp;
        let binop = match op {
            UpdateOp::Inc => BinaryOp::Add,
            UpdateOp::Dec => BinaryOp::Sub,
        };
        // Read the current value (coerced to a number via `cur - 0` is implicit
        // in the arithmetic), compute the updated value, write it back, and
        // yield the new value (prefix) or the original numeric value (postfix).
        match argument {
            Expr::Ident(id) => {
                let binding = self.lookup_binding(&id.name)?;
                let cur = self.read_binding(&binding, &id.name);
                let old = self.coerce_number(cur);
                let one = self.reg();
                self.emit(Op::LoadInt { dst: one, value: 1 });
                let updated = self.reg();
                self.emit_binop(binop, updated, old, one)?;
                self.write_binding(&binding, &id.name, updated);
                Ok(if prefix { updated } else { old })
            }
            Expr::Member {
                object,
                property,
                optional: false,
                ..
            } => {
                let obj = self.expr(object)?;
                let key = match property {
                    PropertyKey::Ident(name) => {
                        PropertySlot::Const(self.add_const(Const::Str(name.clone().into_string())))
                    }
                    PropertyKey::Str(s) => {
                        PropertySlot::Const(self.add_const(Const::Str(s.clone().into_string())))
                    }
                    PropertyKey::Computed(e) => PropertySlot::Index(self.expr(e)?),
                    _ => return Err(CompileError::unsupported("update member key")),
                };
                let cur = self.reg();
                match key {
                    PropertySlot::Const(k) => self.emit(Op::GetProp {
                        dst: cur,
                        obj,
                        key: k,
                    }),
                    PropertySlot::Index(idx) => self.emit(Op::GetElem {
                        dst: cur,
                        obj,
                        index: idx,
                    }),
                };
                let old = self.coerce_number(cur);
                let one = self.reg();
                self.emit(Op::LoadInt { dst: one, value: 1 });
                let updated = self.reg();
                self.emit_binop(binop, updated, old, one)?;
                match key {
                    PropertySlot::Const(k) => self.emit(Op::SetProp {
                        obj,
                        key: k,
                        src: updated,
                    }),
                    PropertySlot::Index(idx) => self.emit(Op::SetElem {
                        obj,
                        index: idx,
                        src: updated,
                    }),
                };
                Ok(if prefix { updated } else { old })
            }
            _ => Err(CompileError::unsupported("update target")),
        }
    }

    /// Coerces a register to a number via `value - 0`, returning a fresh reg.
    fn coerce_number(&mut self, value: Reg) -> Reg {
        let zero = self.reg();
        self.emit(Op::LoadInt {
            dst: zero,
            value: 0,
        });
        let dst = self.reg();
        self.emit(Op::Sub {
            dst,
            a: value,
            b: zero,
        });
        dst
    }

    fn unary(&mut self, op: UnaryOp, argument: &Expr) -> Result<Reg, CompileError> {
        // `typeof`, `void`, and `delete` handle the operand specially.
        match op {
            UnaryOp::Typeof => return self.type_of(argument),
            UnaryOp::Delete => return self.delete(argument),
            UnaryOp::Void => {
                self.expr(argument)?; // evaluate for side effects, discard
                let dst = self.reg();
                self.emit(Op::LoadUndefined { dst });
                return Ok(dst);
            }
            UnaryOp::Plus => {
                // Unary `+` coerces to number: `0 + x` reuses Add's coercion.
                let zero = self.reg();
                self.emit(Op::LoadInt {
                    dst: zero,
                    value: 0,
                });
                let x = self.expr(argument)?;
                let dst = self.reg();
                self.emit(Op::Sub { dst, a: x, b: zero }); // x - 0 → ToNumber(x)
                return Ok(dst);
            }
            _ => {}
        }
        let src = self.expr(argument)?;
        let dst = self.reg();
        match op {
            UnaryOp::Minus => self.emit(Op::Neg { dst, src }),
            UnaryOp::Not => self.emit(Op::Not { dst, src }),
            _ => return Err(CompileError::unsupported("unary operator")),
        };
        Ok(dst)
    }

    /// Compiles `delete argument`. `delete obj.prop` / `delete obj[k]` removes
    /// the member; deleting anything else evaluates to `true`.
    fn delete(&mut self, argument: &Expr) -> Result<Reg, CompileError> {
        if let Expr::Member {
            object,
            property,
            optional: false,
            ..
        } = argument
        {
            let obj = self.expr(object)?;
            let key = match property {
                PropertyKey::Ident(name) => {
                    let r = self.reg();
                    let k = self.add_const(Const::Str(name.clone().into_string()));
                    self.emit(Op::LoadConst { dst: r, k });
                    r
                }
                PropertyKey::Str(s) => {
                    let r = self.reg();
                    let k = self.add_const(Const::Str(s.clone().into_string()));
                    self.emit(Op::LoadConst { dst: r, k });
                    r
                }
                PropertyKey::Computed(e) => self.expr(e)?,
                _ => return Err(CompileError::unsupported("delete key")),
            };
            let dst = self.reg();
            self.emit(Op::DeleteMember { dst, obj, key });
            return Ok(dst);
        }
        // `delete <non-member>` → true.
        let dst = self.reg();
        self.emit(Op::LoadBool { dst, value: true });
        Ok(dst)
    }

    /// Compiles `typeof argument`. A bare global identifier uses the
    /// non-throwing form so `typeof unbound === 'undefined'`.
    fn type_of(&mut self, argument: &Expr) -> Result<Reg, CompileError> {
        if let Expr::Ident(id) = argument
            && id.name.as_ref() != "undefined"
            && matches!(self.lookup_binding(&id.name)?, Binding::Global)
        {
            let dst = self.reg();
            let name = self.add_const(Const::Str(id.name.clone().into_string()));
            self.emit(Op::TypeOfGlobal { dst, name });
            return Ok(dst);
        }
        let src = self.expr(argument)?;
        let dst = self.reg();
        self.emit(Op::TypeOf { dst, src });
        Ok(dst)
    }

    fn binary(&mut self, op: BinaryOp, left: &Expr, right: &Expr) -> Result<Reg, CompileError> {
        let a = self.expr(left)?;
        let b = self.expr(right)?;
        let dst = self.reg();
        self.emit_binop(op, dst, a, b)?;
        Ok(dst)
    }

    fn emit_binop(&mut self, op: BinaryOp, dst: Reg, a: Reg, b: Reg) -> Result<(), CompileError> {
        // `!=` / `!==` compose as the negation of `==` / `===`.
        if matches!(op, BinaryOp::NotEq | BinaryOp::NotEqEq) {
            let eq = if op == BinaryOp::NotEq {
                Op::Eq { dst, a, b }
            } else {
                Op::StrictEq { dst, a, b }
            };
            self.emit(eq);
            self.emit(Op::Not { dst, src: dst });
            return Ok(());
        }
        let inst = match op {
            BinaryOp::Add => Op::Add { dst, a, b },
            BinaryOp::Sub => Op::Sub { dst, a, b },
            BinaryOp::Mul => Op::Mul { dst, a, b },
            BinaryOp::Div => Op::Div { dst, a, b },
            BinaryOp::Mod => Op::Mod { dst, a, b },
            BinaryOp::Exp => Op::Pow { dst, a, b },
            BinaryOp::EqEq => Op::Eq { dst, a, b },
            BinaryOp::EqEqEq => Op::StrictEq { dst, a, b },
            BinaryOp::Lt => Op::Lt { dst, a, b },
            BinaryOp::LtEq => Op::Le { dst, a, b },
            BinaryOp::Gt => Op::Gt { dst, a, b },
            BinaryOp::GtEq => Op::Ge { dst, a, b },
            // The rest (bitwise/shift, `in`, `instanceof`) go through the
            // generic `Binary` op, dispatched to the shared evaluator.
            other => match binop_code(other) {
                Some(op) => Op::Binary { dst, a, b, op },
                None => return Err(CompileError::unsupported("binary operator")),
            },
        };
        self.emit(inst);
        Ok(())
    }

    fn logical(&mut self, op: LogicalOp, left: &Expr, right: &Expr) -> Result<Reg, CompileError> {
        let dst = self.reg();
        let l = self.expr(left)?;
        self.emit(Op::Move { dst, src: l });
        let jump_idx = match op {
            LogicalOp::And => self.emit(Op::JumpIfFalse {
                cond: dst,
                offset: 0,
            }),
            LogicalOp::Or => self.emit(Op::JumpIfTrue {
                cond: dst,
                offset: 0,
            }),
            LogicalOp::Nullish => {
                // `a ?? b`: take `b` only when `a` is null/undefined. Loose
                // `a == null` is true for exactly those two values.
                let nullreg = self.reg();
                self.emit(Op::LoadNull { dst: nullreg });
                let is_nullish = self.reg();
                self.emit(Op::Eq {
                    dst: is_nullish,
                    a: dst,
                    b: nullreg,
                });
                // If not nullish, jump past the `b` branch (keep `a`).
                self.emit(Op::JumpIfFalse {
                    cond: is_nullish,
                    offset: 0,
                })
            }
        };
        let r = self.expr(right)?;
        self.emit(Op::Move { dst, src: r });
        self.patch_to_here(jump_idx);
        Ok(dst)
    }

    fn member(&mut self, object: &Expr, property: &PropertyKey) -> Result<Reg, CompileError> {
        let obj = self.expr(object)?;
        match property {
            PropertyKey::Ident(name) => {
                let dst = self.reg();
                let key = self.add_const(Const::Str(name.clone().into_string()));
                self.emit(Op::GetProp { dst, obj, key });
                Ok(dst)
            }
            PropertyKey::Str(s) => {
                let dst = self.reg();
                let key = self.add_const(Const::Str(s.clone().into_string()));
                self.emit(Op::GetProp { dst, obj, key });
                Ok(dst)
            }
            PropertyKey::Computed(expr) => {
                let index = self.expr(expr)?;
                let dst = self.reg();
                self.emit(Op::GetElem { dst, obj, index });
                Ok(dst)
            }
            _ => Err(CompileError::unsupported("member key")),
        }
    }

    fn call(
        &mut self,
        callee: &Expr,
        arguments: &[crate::ast::Argument],
    ) -> Result<Reg, CompileError> {
        // `super(args)` runs the superclass constructor on the current `this`,
        // lowered to `super.call(this, …args)`.
        if matches!(callee, Expr::Super(_)) {
            let super_val = self.read_ident(SUPER_NAME)?;
            return self.emit_call_with_this(super_val, THIS_REG, arguments);
        }
        // `super.m(args)` invokes a superclass prototype method on `this`,
        // lowered to `super.prototype.m.call(this, …args)`.
        if let Expr::Member {
            object,
            property,
            optional: false,
            ..
        } = callee
            && matches!(&**object, Expr::Super(_))
        {
            let name = simple_key(property)?;
            let method = self.super_proto_member(&name)?;
            return self.emit_call_with_this(method, THIS_REG, arguments);
        }
        // A method call `obj.m(args)` / `obj[k](args)` binds `obj` as `this` and
        // dispatches built-in prototype methods (CallMethod).
        if let Expr::Member {
            object,
            property,
            optional: false,
            ..
        } = callee
        {
            return self.method_call(object, property, arguments);
        }
        let callee_reg = self.expr(callee)?;
        // A spread argument (`f(...xs)`) is lowered to `f.apply(undefined, arr)`.
        if has_spread(arguments) {
            let args_array = self.build_args_array(arguments)?;
            let undef = self.reg();
            self.emit(Op::LoadUndefined { dst: undef });
            return Ok(self.emit_apply(callee_reg, undef, args_array));
        }
        let (args_base, argc) = self.lower_args(arguments)?;
        let dst = self.reg();
        self.emit(Op::Call {
            dst,
            callee: callee_reg,
            args_base,
            argc,
        });
        Ok(dst)
    }

    /// Reads `super.prototype.<name>` (the superclass's prototype method).
    fn super_proto_member(&mut self, name: &str) -> Result<Reg, CompileError> {
        let super_val = self.read_ident(SUPER_NAME)?;
        let proto = self.reg();
        let pk = self.add_const(Const::Str(String::from("prototype")));
        self.emit(Op::GetProp {
            dst: proto,
            obj: super_val,
            key: pk,
        });
        let method = self.reg();
        let mk = self.add_const(Const::Str(String::from(name)));
        self.emit(Op::GetProp {
            dst: method,
            obj: proto,
            key: mk,
        });
        Ok(method)
    }

    /// Emits `callee.call(this_reg, …arguments)` — invokes `callee` with an
    /// explicit `this`. Used to lower `super(...)` / `super.m(...)`.
    fn emit_call_with_this(
        &mut self,
        callee: Reg,
        this_reg: Reg,
        arguments: &[crate::ast::Argument],
    ) -> Result<Reg, CompileError> {
        if has_spread(arguments) {
            // `callee.apply(this, argsArray)`.
            let args_array = self.build_args_array(arguments)?;
            return Ok(self.emit_apply(callee, this_reg, args_array));
        }
        // Evaluate the key and all argument values first, then lay them out in a
        // contiguous window: [this, arg0, arg1, …].
        let key = self.reg();
        let k = self.add_const(Const::Str(String::from("call")));
        self.emit(Op::LoadConst { dst: key, k });
        let mut arg_vals = Vec::with_capacity(arguments.len());
        for arg in arguments {
            let crate::ast::Argument::Item(e) = arg else {
                unreachable!("spread handled above");
            };
            arg_vals.push(self.expr(e)?);
        }
        let base = self.funcs.last().expect("fn").next_reg;
        let s_this = self.reg();
        self.emit(Op::Move {
            dst: s_this,
            src: this_reg,
        });
        for v in arg_vals {
            let slot = self.reg();
            self.emit(Op::Move { dst: slot, src: v });
        }
        let dst = self.reg();
        self.emit(Op::CallMethod {
            dst,
            recv: callee,
            key,
            args_base: base,
            argc: (arguments.len() + 1) as u16,
        });
        Ok(dst)
    }

    fn method_call(
        &mut self,
        object: &Expr,
        property: &PropertyKey,
        arguments: &[crate::ast::Argument],
    ) -> Result<Reg, CompileError> {
        let recv = self.expr(object)?;
        // The key goes into a register (a constant for `.name`, the computed
        // expression otherwise).
        let key = match property {
            PropertyKey::Ident(name) => {
                let r = self.reg();
                let k = self.add_const(Const::Str(name.clone().into_string()));
                self.emit(Op::LoadConst { dst: r, k });
                r
            }
            PropertyKey::Str(s) => {
                let r = self.reg();
                let k = self.add_const(Const::Str(s.clone().into_string()));
                self.emit(Op::LoadConst { dst: r, k });
                r
            }
            PropertyKey::Computed(expr) => self.expr(expr)?,
            _ => return Err(CompileError::unsupported("method key")),
        };
        // A spread argument (`obj.m(...xs)`) is lowered to
        // `obj.m.apply(obj, arr)`: fetch the method value, then apply with
        // `obj` as `this`.
        if has_spread(arguments) {
            let method = self.reg();
            self.emit(Op::GetElem {
                dst: method,
                obj: recv,
                index: key,
            });
            let args_array = self.build_args_array(arguments)?;
            return Ok(self.emit_apply(method, recv, args_array));
        }
        let (args_base, argc) = self.lower_args(arguments)?;
        let dst = self.reg();
        self.emit(Op::CallMethod {
            dst,
            recv,
            key,
            args_base,
            argc,
        });
        Ok(dst)
    }

    /// Builds an array holding all call arguments (items + flattened spreads),
    /// for the `apply`-based spread-call lowering.
    fn build_args_array(
        &mut self,
        arguments: &[crate::ast::Argument],
    ) -> Result<Reg, CompileError> {
        use crate::ast::Argument;
        let result = self.reg();
        self.emit(Op::NewArray {
            dst: result,
            len: 0,
        });
        for arg in arguments {
            let chunk = match arg {
                Argument::Item(e) => {
                    let v = self.expr(e)?;
                    let one = self.reg();
                    self.emit(Op::NewArray { dst: one, len: 1 });
                    let idx = self.reg();
                    self.emit(Op::LoadInt { dst: idx, value: 0 });
                    self.emit(Op::SetElem {
                        obj: one,
                        index: idx,
                        src: v,
                    });
                    one
                }
                Argument::Spread(e) => {
                    let v = self.expr(e)?;
                    let items = self.reg();
                    self.emit(Op::IterValues { dst: items, src: v });
                    items
                }
            };
            self.concat_into(result, chunk);
        }
        Ok(result)
    }

    /// Emits `callee.apply(this_reg, args_array)`, returning the result reg.
    fn emit_apply(&mut self, callee: Reg, this_reg: Reg, args_array: Reg) -> Reg {
        let key = self.reg();
        let k = self.add_const(Const::Str(String::from("apply")));
        self.emit(Op::LoadConst { dst: key, k });
        let base = self.funcs.last().expect("fn").next_reg;
        let s1 = self.reg();
        self.emit(Op::Move {
            dst: s1,
            src: this_reg,
        });
        let s2 = self.reg();
        self.emit(Op::Move {
            dst: s2,
            src: args_array,
        });
        let dst = self.reg();
        self.emit(Op::CallMethod {
            dst,
            recv: callee,
            key,
            args_base: base,
            argc: 2,
        });
        dst
    }

    /// Lowers call arguments into a fresh contiguous register window, returning
    /// `(args_base, argc)`.
    fn lower_args(
        &mut self,
        arguments: &[crate::ast::Argument],
    ) -> Result<(Reg, u16), CompileError> {
        use crate::ast::Argument;
        let mut arg_regs = Vec::new();
        for arg in arguments {
            match arg {
                Argument::Item(e) => arg_regs.push(self.expr(e)?),
                Argument::Spread(_) => return Err(CompileError::unsupported("spread argument")),
            }
        }
        let args_base = self.funcs.last().expect("fn").next_reg;
        for &src in &arg_regs {
            let slot = self.reg();
            self.emit(Op::Move { dst: slot, src });
        }
        Ok((args_base, arg_regs.len() as u16))
    }
}

/// How a member-assignment property was lowered: a string-constant key (for
/// `SetProp`) or a computed index register (for `SetElem`).
enum PropertySlot {
    Const(ConstIdx),
    Index(Reg),
}

/// A function body source: a block of statements or a single expression (arrow).
enum FnBody<'b> {
    Block(&'b [Stmt]),
    Expr(&'b Expr),
}

/// Maps the operators without a dedicated instruction to their generic
/// [`Op::Binary`] code; the others return `None`.
fn binop_code(op: BinaryOp) -> Option<u8> {
    use crate::bytecode::binop;
    Some(match op {
        BinaryOp::BitAnd => binop::BIT_AND,
        BinaryOp::BitOr => binop::BIT_OR,
        BinaryOp::BitXor => binop::BIT_XOR,
        BinaryOp::Shl => binop::SHL,
        BinaryOp::Shr => binop::SHR,
        BinaryOp::Ushr => binop::USHR,
        BinaryOp::In => binop::IN,
        BinaryOp::Instanceof => binop::INSTANCEOF,
        _ => return None,
    })
}

/// Whether a statement list can complete abruptly *out of itself* via
/// `return`, or a `break`/`continue` that targets an enclosing loop/switch
/// (i.e. one not contained in a loop/switch within these statements). Such
/// completions would need to run an enclosing `finally` first, which the VM
/// can't yet thread — so they trigger a tree-walker fallback.
fn escapes_via_abrupt(stmts: &[Stmt]) -> bool {
    stmts.iter().any(|s| stmt_escapes(s, 0))
}

fn stmt_escapes(s: &Stmt, loop_depth: usize) -> bool {
    match s {
        Stmt::Return { .. } => true,
        Stmt::Break { .. } | Stmt::Continue { .. } => loop_depth == 0,
        Stmt::Block { body, .. } => body.iter().any(|s| stmt_escapes(s, loop_depth)),
        Stmt::If {
            consequent,
            alternate,
            ..
        } => {
            stmt_escapes(consequent, loop_depth)
                || alternate
                    .as_ref()
                    .is_some_and(|a| stmt_escapes(a, loop_depth))
        }
        Stmt::While { body, .. }
        | Stmt::DoWhile { body, .. }
        | Stmt::For { body, .. }
        | Stmt::ForIn { body, .. }
        | Stmt::ForOf { body, .. } => stmt_escapes(body, loop_depth + 1),
        Stmt::Labeled { body, .. } => stmt_escapes(body, loop_depth),
        Stmt::Switch { cases, .. } => cases
            .iter()
            .any(|c| c.body.iter().any(|s| stmt_escapes(s, loop_depth + 1))),
        Stmt::Try {
            block,
            handler,
            finalizer,
            ..
        } => {
            block.iter().any(|s| stmt_escapes(s, loop_depth))
                || handler
                    .as_ref()
                    .is_some_and(|h| h.body.iter().any(|s| stmt_escapes(s, loop_depth)))
                || finalizer
                    .as_ref()
                    .is_some_and(|f| f.iter().any(|s| stmt_escapes(s, loop_depth)))
        }
        // Function/class bodies are separate scopes; their returns don't escape.
        _ => false,
    }
}

/// Extracts a simple (identifier/string) property key as a string; computed,
/// numeric, and private keys are reported as unsupported.
fn simple_key(key: &PropertyKey) -> Result<String, CompileError> {
    match key {
        PropertyKey::Ident(name) => Ok(name.clone().into_string()),
        PropertyKey::Str(s) => Ok(s.clone().into_string()),
        _ => Err(CompileError::unsupported("computed/private class key")),
    }
}

/// Whether a call's argument list contains a spread (`f(...xs)`).
fn has_spread(arguments: &[crate::ast::Argument]) -> bool {
    arguments
        .iter()
        .any(|a| matches!(a, crate::ast::Argument::Spread(_)))
}

/// Maps a compound assignment operator to its binary op; `=` returns `None`.
fn compound_binop(op: AssignOp) -> Option<BinaryOp> {
    match op {
        AssignOp::Assign => None,
        AssignOp::AddAssign => Some(BinaryOp::Add),
        AssignOp::SubAssign => Some(BinaryOp::Sub),
        AssignOp::MulAssign => Some(BinaryOp::Mul),
        AssignOp::DivAssign => Some(BinaryOp::Div),
        AssignOp::ModAssign => Some(BinaryOp::Mod),
        AssignOp::ExpAssign => Some(BinaryOp::Exp),
        _ => None,
    }
}

// --- capture analysis (which locals must be boxed in cells) ---------------
//
// Incompleteness here is safe: a missed capture leaves an enclosing local
// un-boxed, so the closure's reference resolves to a non-cell local and the
// whole program falls back to the tree-walker; an over-reported capture just
// boxes a local that didn't need it (still correct).

use alloc::collections::BTreeSet;

/// The names of `params`/body locals that are referenced inside nested
/// functions (and so must be stored in cells so closures can share them).
fn captured_names(params: &[Param], body: &FnBody) -> BTreeSet<String> {
    let mut declared = BTreeSet::new();
    for p in params {
        binding_names(&p.target, &mut declared);
    }
    let mut nested = BTreeSet::new();
    match body {
        FnBody::Block(stmts) => {
            for s in *stmts {
                declared_in_stmt(s, &mut declared);
                idents_in_stmt(s, &mut nested, true);
            }
        }
        FnBody::Expr(e) => idents_in_expr(e, &mut nested, true),
    }
    declared.intersection(&nested).cloned().collect()
}

/// Collects the names bound by a binding target (pattern).
fn binding_names(t: &BindingTarget, out: &mut BTreeSet<String>) {
    use crate::ast::ArrayPatternElement;
    match t {
        BindingTarget::Ident(id) => {
            out.insert(id.name.clone().into_string());
        }
        BindingTarget::Array(p) => {
            for el in &p.elements {
                match el {
                    ArrayPatternElement::Item { target, .. }
                    | ArrayPatternElement::Rest { target, .. } => binding_names(target, out),
                    ArrayPatternElement::Hole => {}
                }
            }
        }
        BindingTarget::Object(p) => {
            for prop in &p.properties {
                binding_names(&prop.value, out);
            }
            if let Some(rest) = &p.rest {
                binding_names(rest, out);
            }
        }
    }
}

/// Collects names declared directly in a statement (recursing through control
/// flow but not into nested functions).
fn declared_in_stmt(s: &Stmt, out: &mut BTreeSet<String>) {
    use crate::ast::{ForInit, ForLeft};
    match s {
        Stmt::Var(d) => {
            for decl in &d.declarations {
                binding_names(&decl.target, out);
            }
        }
        Stmt::Function(f) => {
            if let Some(id) = &f.id {
                out.insert(id.name.clone().into_string());
            }
        }
        Stmt::Block { body, .. } => {
            for st in body {
                declared_in_stmt(st, out);
            }
        }
        Stmt::If {
            consequent,
            alternate,
            ..
        } => {
            declared_in_stmt(consequent, out);
            if let Some(a) = alternate {
                declared_in_stmt(a, out);
            }
        }
        Stmt::For { init, body, .. } => {
            if let Some(ForInit::Var(d)) = init {
                for decl in &d.declarations {
                    binding_names(&decl.target, out);
                }
            }
            declared_in_stmt(body, out);
        }
        Stmt::ForIn { left, body, .. } | Stmt::ForOf { left, body, .. } => {
            if let ForLeft::Decl { target, .. } = left {
                binding_names(target, out);
            }
            declared_in_stmt(body, out);
        }
        Stmt::While { body, .. } | Stmt::DoWhile { body, .. } | Stmt::Labeled { body, .. } => {
            declared_in_stmt(body, out);
        }
        Stmt::Switch { cases, .. } => {
            for c in cases {
                for st in &c.body {
                    declared_in_stmt(st, out);
                }
            }
        }
        Stmt::Try {
            block,
            handler,
            finalizer,
            ..
        } => {
            for st in block {
                declared_in_stmt(st, out);
            }
            if let Some(h) = handler {
                for st in &h.body {
                    declared_in_stmt(st, out);
                }
            }
            if let Some(f) = finalizer {
                for st in f {
                    declared_in_stmt(st, out);
                }
            }
        }
        _ => {}
    }
}

/// Collects identifier references in a statement. With `nested_only`, only the
/// identifiers that appear *inside nested functions* are collected (entering a
/// function flips `nested_only` off so its whole body is collected).
fn idents_in_stmt(s: &Stmt, out: &mut BTreeSet<String>, nested_only: bool) {
    use crate::ast::{ForInit, ForLeft};
    match s {
        Stmt::Expr { expression, .. } => idents_in_expr(expression, out, nested_only),
        Stmt::Block { body, .. } => {
            for st in body {
                idents_in_stmt(st, out, nested_only);
            }
        }
        Stmt::Var(d) => {
            for decl in &d.declarations {
                if let Some(init) = &decl.init {
                    idents_in_expr(init, out, nested_only);
                }
            }
        }
        Stmt::Function(f) => {
            // A nested function declaration: collect everything inside it.
            for p in &f.params {
                if let Some(def) = &p.default {
                    idents_in_expr(def, out, false);
                }
            }
            for st in &f.body {
                idents_in_stmt(st, out, false);
            }
        }
        Stmt::If {
            test,
            consequent,
            alternate,
            ..
        } => {
            idents_in_expr(test, out, nested_only);
            idents_in_stmt(consequent, out, nested_only);
            if let Some(a) = alternate {
                idents_in_stmt(a, out, nested_only);
            }
        }
        Stmt::For {
            init,
            test,
            update,
            body,
            ..
        } => {
            match init {
                Some(ForInit::Var(d)) => {
                    for decl in &d.declarations {
                        if let Some(i) = &decl.init {
                            idents_in_expr(i, out, nested_only);
                        }
                    }
                }
                Some(ForInit::Expr(e)) => idents_in_expr(e, out, nested_only),
                None => {}
            }
            if let Some(t) = test {
                idents_in_expr(t, out, nested_only);
            }
            if let Some(u) = update {
                idents_in_expr(u, out, nested_only);
            }
            idents_in_stmt(body, out, nested_only);
        }
        Stmt::ForIn {
            left, right, body, ..
        }
        | Stmt::ForOf {
            left, right, body, ..
        } => {
            if let ForLeft::Target(e) = left {
                idents_in_expr(e, out, nested_only);
            }
            idents_in_expr(right, out, nested_only);
            idents_in_stmt(body, out, nested_only);
        }
        Stmt::While { test, body, .. } | Stmt::DoWhile { body, test, .. } => {
            idents_in_expr(test, out, nested_only);
            idents_in_stmt(body, out, nested_only);
        }
        Stmt::Switch {
            discriminant,
            cases,
            ..
        } => {
            idents_in_expr(discriminant, out, nested_only);
            for c in cases {
                if let Some(t) = &c.test {
                    idents_in_expr(t, out, nested_only);
                }
                for st in &c.body {
                    idents_in_stmt(st, out, nested_only);
                }
            }
        }
        Stmt::Try {
            block,
            handler,
            finalizer,
            ..
        } => {
            for st in block {
                idents_in_stmt(st, out, nested_only);
            }
            if let Some(h) = handler {
                for st in &h.body {
                    idents_in_stmt(st, out, nested_only);
                }
            }
            if let Some(f) = finalizer {
                for st in f {
                    idents_in_stmt(st, out, nested_only);
                }
            }
        }
        Stmt::Return {
            argument: Some(e), ..
        }
        | Stmt::Throw { argument: e, .. } => idents_in_expr(e, out, nested_only),
        Stmt::Labeled { body, .. } => idents_in_stmt(body, out, nested_only),
        _ => {}
    }
}

/// Collects identifier references in an expression (see [`idents_in_stmt`]).
fn idents_in_expr(e: &Expr, out: &mut BTreeSet<String>, nested_only: bool) {
    use crate::ast::{Argument, ArrayElement, ArrowBody, ObjectMember};
    match e {
        Expr::Ident(id) => {
            if !nested_only {
                out.insert(id.name.clone().into_string());
            }
        }
        Expr::Member {
            object, property, ..
        } => {
            idents_in_expr(object, out, nested_only);
            if let PropertyKey::Computed(k) = property {
                idents_in_expr(k, out, nested_only);
            }
        }
        Expr::Call {
            callee, arguments, ..
        }
        | Expr::New {
            callee, arguments, ..
        } => {
            idents_in_expr(callee, out, nested_only);
            for a in arguments {
                match a {
                    Argument::Item(x) | Argument::Spread(x) => idents_in_expr(x, out, nested_only),
                }
            }
        }
        Expr::Unary { argument, .. }
        | Expr::Update { argument, .. }
        | Expr::Await { argument, .. } => idents_in_expr(argument, out, nested_only),
        Expr::Yield {
            argument: Some(a), ..
        } => idents_in_expr(a, out, nested_only),
        Expr::Binary { left, right, .. } | Expr::Logical { left, right, .. } => {
            idents_in_expr(left, out, nested_only);
            idents_in_expr(right, out, nested_only);
        }
        Expr::Conditional {
            test,
            consequent,
            alternate,
            ..
        } => {
            idents_in_expr(test, out, nested_only);
            idents_in_expr(consequent, out, nested_only);
            idents_in_expr(alternate, out, nested_only);
        }
        Expr::Assign { target, value, .. } => {
            idents_in_expr(target, out, nested_only);
            idents_in_expr(value, out, nested_only);
        }
        Expr::Sequence { expressions, .. } => {
            for x in expressions {
                idents_in_expr(x, out, nested_only);
            }
        }
        Expr::Array { elements, .. } => {
            for el in elements {
                match el {
                    ArrayElement::Item(x) | ArrayElement::Spread(x) => {
                        idents_in_expr(x, out, nested_only);
                    }
                    ArrayElement::Hole => {}
                }
            }
        }
        Expr::Object { members, .. } => {
            for m in members {
                match m {
                    ObjectMember::Property { key, value, .. } => {
                        if let PropertyKey::Computed(k) = key {
                            idents_in_expr(k, out, nested_only);
                        }
                        idents_in_expr(value, out, nested_only);
                    }
                    ObjectMember::Spread { value, .. } => idents_in_expr(value, out, nested_only),
                    ObjectMember::Accessor { value, .. } => {
                        for st in &value.body {
                            idents_in_stmt(st, out, false);
                        }
                    }
                }
            }
        }
        Expr::Template(t) => {
            for x in &t.expressions {
                idents_in_expr(x, out, nested_only);
            }
        }
        Expr::TaggedTemplate { tag, quasi, .. } => {
            idents_in_expr(tag, out, nested_only);
            for x in &quasi.expressions {
                idents_in_expr(x, out, nested_only);
            }
        }
        // Entering a function: collect *all* identifiers inside it.
        Expr::Function(f) => {
            for p in &f.params {
                if let Some(def) = &p.default {
                    idents_in_expr(def, out, false);
                }
            }
            for st in &f.body {
                idents_in_stmt(st, out, false);
            }
        }
        Expr::Arrow(a) => {
            for p in &a.params {
                if let Some(def) = &p.default {
                    idents_in_expr(def, out, false);
                }
            }
            match &a.body {
                ArrowBody::Block(b) => {
                    for st in b {
                        idents_in_stmt(st, out, false);
                    }
                }
                ArrowBody::Expr(x) => idents_in_expr(x, out, false),
            }
        }
        _ => {}
    }
}
