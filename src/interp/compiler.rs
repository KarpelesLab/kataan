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
//! `&&`/`||`/`??`, unary `-`/`!`, the ternary, `in`, `instanceof`), object and
//! array literals, member/index access and writes, calls and method calls
//! (with `this` + built-in prototype dispatch), `new`, assignment (incl.
//! compound, on identifiers and members), `if`/`else`, `while`/`do-while`/`for`
//! with `break`/`continue`, `switch`, `try`/`catch`, `throw`, blocks, `return`,
//! and **functions** (declarations hoisted, function/arrow expressions,
//! positional parameters). Not yet: closures that capture an enclosing
//! function's variable (an upvalue), `finally`, for-in/for-of, destructuring,
//! classes, generators, and spread — these return a `CompileError` so the
//! caller falls back to the tree-walker.

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
    let mut c = Compiler {
        module: alloc::vec![Chunk::new("<main>")],
        funcs: alloc::vec![FnState::new(0, Vec::new())],
    };
    c.compile_body(body, true)?;
    let main = c.funcs.pop().expect("entry function state");
    c.module[0].register_count = main.next_reg;
    Ok(Module { chunks: c.module })
}

/// Per-function compilation state.
struct FnState {
    chunk_idx: usize,
    next_reg: Reg,
    /// In-scope locals → backing register (innermost last; shadowing wins).
    locals: Vec<(String, Reg)>,
    /// Names visible from enclosing functions, to detect (unsupported) captures.
    enclosing: Vec<String>,
    /// The enclosing-loop stack, for resolving `break`/`continue` jumps.
    loops: Vec<LoopCtx>,
}

impl FnState {
    fn new(chunk_idx: usize, enclosing: Vec<String>) -> Self {
        Self {
            chunk_idx,
            // Register 0 is reserved for `this` (bound by the VM on each call);
            // parameters and locals start at register 1.
            next_reg: 1,
            locals: Vec::new(),
            enclosing,
            loops: Vec::new(),
        }
    }
}

/// The register holding the current function's `this` value.
const THIS_REG: Reg = 0;

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
    fn resolve(&self, name: &str) -> Option<Reg> {
        self.funcs
            .last()
            .expect("a current function")
            .locals
            .iter()
            .rev()
            .find(|(n, _)| n == name)
            .map(|(_, r)| *r)
    }
    fn declare_local(&mut self, name: String, reg: Reg) {
        self.funcs
            .last_mut()
            .expect("a current function")
            .locals
            .push((name, reg));
    }
    /// Whether `name` is a local of an enclosing function (an upvalue capture).
    fn is_capture(&self, name: &str) -> bool {
        self.funcs
            .last()
            .expect("a current function")
            .enclosing
            .iter()
            .any(|n| n == name)
    }

    // --- statements -----------------------------------------------------

    /// Compiles a statement list. With `return_last`, the value of the final
    /// expression statement is returned; otherwise the chunk falls off the end.
    fn compile_body(&mut self, body: &[Stmt], return_last: bool) -> Result<(), CompileError> {
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
                    let BindingTarget::Ident(id) = &d.target else {
                        return Err(CompileError::unsupported("destructuring declaration"));
                    };
                    let value = match &d.init {
                        Some(init) => self.expr(init)?,
                        None => {
                            let r = self.reg();
                            self.emit(Op::LoadUndefined { dst: r });
                            r
                        }
                    };
                    let slot = self.reg();
                    self.emit(Op::Move {
                        dst: slot,
                        src: value,
                    });
                    self.declare_local(id.name.clone().into_string(), slot);
                }
                Ok(None)
            }
            Stmt::Block { body, .. } => {
                let mark = self.funcs.last().expect("fn").locals.len();
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
        use crate::ast::ForInit;
        let mark = self.funcs.last().expect("fn").locals.len();
        match init {
            Some(ForInit::Var(decl)) => {
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

    /// Compiles `try { … } catch (e) { … }`. A `finally` block falls back to
    /// the tree-walker for now (its run-on-every-path semantics need more than
    /// a single guarded region).
    fn compile_try(
        &mut self,
        block: &[Stmt],
        handler: Option<&crate::ast::CatchClause>,
        finalizer: Option<&[Stmt]>,
    ) -> Result<Option<Reg>, CompileError> {
        if finalizer.is_some() {
            return Err(CompileError::unsupported("`finally`"));
        }
        let Some(handler) = handler else {
            return Err(CompileError::unsupported("`try` without `catch`"));
        };
        // The register the VM drops the thrown value into on a catch.
        let err_reg = self.reg();
        let ph = self.emit(Op::PushHandler {
            catch: 0,
            err: err_reg,
        });
        // Guarded region.
        let mark = self.funcs.last().expect("fn").locals.len();
        for s in block {
            self.stmt(s)?;
        }
        self.funcs.last_mut().expect("fn").locals.truncate(mark);
        self.emit(Op::PopHandler);
        let skip_catch = self.emit(Op::Jump { offset: 0 });

        // Catch entry: patch the handler's catch offset to here.
        let catch_pc = self.code_len();
        let ci = self.ci();
        self.module[ci].code[ph] = Op::PushHandler {
            catch: (catch_pc as i64 - (ph as i64 + 1)) as i32,
            err: err_reg,
        };
        // Bind the catch parameter (if any) to the error register.
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
        self.patch_to_here(skip_catch);
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
        let chunk_idx = self.compile_function(&func.params, FnBody::Block(&func.body), &id.name)?;
        let dst = self.reg();
        let k = self.add_const(Const::Func(chunk_idx as u32));
        self.emit(Op::LoadConst { dst, k });
        let name = self.add_const(Const::Str(id.name.clone().into_string()));
        self.emit(Op::SetGlobal { name, src: dst });
        Ok(())
    }

    /// Compiles a function body into a new chunk; returns the chunk index.
    fn compile_function(
        &mut self,
        params: &[Param],
        body: FnBody,
        name: &str,
    ) -> Result<usize, CompileError> {
        let mut param_names = Vec::new();
        for p in params {
            match &p.target {
                BindingTarget::Ident(id) if p.default.is_none() && !p.rest => {
                    param_names.push(id.name.clone().into_string());
                }
                _ => return Err(CompileError::unsupported("parameter pattern/default/rest")),
            }
        }

        let outer = self.funcs.last().expect("fn");
        let mut enclosing = outer.enclosing.clone();
        enclosing.extend(outer.locals.iter().map(|(n, _)| n.clone()));

        let chunk_idx = self.module.len();
        self.module.push(Chunk::new(name));
        self.module[chunk_idx].param_count = param_names.len() as u16;

        let mut state = FnState::new(chunk_idx, enclosing);
        for pname in &param_names {
            let r = state.next_reg;
            state.next_reg += 1;
            state.locals.push((pname.clone(), r));
        }
        self.funcs.push(state);

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
        Ok(chunk_idx)
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
            Expr::Ident(id) => self.read_ident(&id.name),
            Expr::Assign {
                op, target, value, ..
            } => self.assign(*op, target, value),
            Expr::Unary { op, argument, .. } => self.unary(*op, argument),
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
            Expr::Function(func) => {
                let idx = self.compile_function(
                    &func.params,
                    FnBody::Block(&func.body),
                    func.id.as_ref().map_or("<anonymous>", |id| &id.name),
                )?;
                let dst = self.reg();
                let k = self.add_const(Const::Func(idx as u32));
                self.emit(Op::LoadConst { dst, k });
                Ok(dst)
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
        let dst = self.reg();
        self.emit(Op::NewArray {
            dst,
            len: elements.len() as u32,
        });
        for (i, el) in elements.iter().enumerate() {
            let value = match el {
                ArrayElement::Item(e) => self.expr(e)?,
                ArrayElement::Hole => continue,
                ArrayElement::Spread(_) => return Err(CompileError::unsupported("array spread")),
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

    fn object_literal(
        &mut self,
        members: &[crate::ast::ObjectMember],
    ) -> Result<Reg, CompileError> {
        use crate::ast::ObjectMember;
        let dst = self.reg();
        self.emit(Op::NewObject { dst });
        for member in members {
            let ObjectMember::Property { key, value, .. } = member else {
                return Err(CompileError::unsupported("object spread/accessor"));
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
        let idx = self.compile_function(&arrow.params, body, "<arrow>")?;
        let dst = self.reg();
        let k = self.add_const(Const::Func(idx as u32));
        self.emit(Op::LoadConst { dst, k });
        Ok(dst)
    }

    /// Reads `name` (local copy, or global) into a fresh register; a reference
    /// to an enclosing function's local is an unsupported capture.
    fn read_ident(&mut self, name: &str) -> Result<Reg, CompileError> {
        if name == "undefined" {
            let dst = self.reg();
            self.emit(Op::LoadUndefined { dst });
            return Ok(dst);
        }
        if let Some(slot) = self.resolve(name) {
            let dst = self.reg();
            self.emit(Op::Move { dst, src: slot });
            return Ok(dst);
        }
        if self.is_capture(name) {
            return Err(CompileError::unsupported("captured (closure) variable"));
        }
        let dst = self.reg();
        let k = self.add_const(Const::Str(name.into()));
        self.emit(Op::GetGlobal { dst, name: k });
        Ok(dst)
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
        if let Some(slot) = self.resolve(name) {
            self.emit(Op::Move {
                dst: slot,
                src: result,
            });
        } else if self.is_capture(name) {
            return Err(CompileError::unsupported("captured (closure) variable"));
        } else {
            let nk = self.add_const(Const::Str(name.into()));
            self.emit(Op::SetGlobal {
                name: nk,
                src: result,
            });
        }
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

    fn unary(&mut self, op: UnaryOp, argument: &Expr) -> Result<Reg, CompileError> {
        let src = self.expr(argument)?;
        let dst = self.reg();
        match op {
            UnaryOp::Minus => self.emit(Op::Neg { dst, src }),
            UnaryOp::Not => self.emit(Op::Not { dst, src }),
            _ => return Err(CompileError::unsupported("unary operator")),
        };
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
