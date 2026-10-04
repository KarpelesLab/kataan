//! Eval code on the bytecode VM (`ROADMAP.md` §2.0).
//!
//! Code the host parses at run time — a direct or indirect `eval`, a
//! `$262.evalScript` script, a `Function(…)` body — compiles like the body of
//! a dynamic-scope function (`Compiler::dyn_fn`): every name resolves through
//! the host environments it runs in. The host performs
//! EvalDeclarationInstantiation's checks and its `var` / lexical bindings,
//! then runs the compiled body with four captured cells:
//!
//! 0. the eval's lexical environment (`ENV_NAME`),
//! 1. its variable environment (`VARENV_NAME`), where function declarations
//!    bind (through the host, `EK_DECL_FN`),
//! 2. `this`, and
//! 3. `new.target`, when it is in scope.
//!
//! The body returns the code's completion value (`Compiler::completion`).
//! Nested functions append to the run's function table; a body without any
//! runs from no table slot at all.

use super::{
    BTreeSet, Binding, CompileError, Compiler, Ctx, EK_DECL_FN, ENV_NAME, EnvSite, FnProto,
    FrameExit, GlobalWrite, NT_NAME, NanBox, Op, Program, Stmt, THIS_NAME, VmError, VmHost,
    hosted_ctx, pattern_names, run_frame,
};
use alloc::string::String;
use alloc::vec::Vec;

/// The hidden binding holding eval code's variable environment.
pub(super) const VARENV_NAME: &str = "\0varenv";

/// [`compile_eval_code`] flag: function declarations bind as a Script's
/// (non-deletable global bindings) — `$262.evalScript`.
pub(crate) const EVAL_SCRIPT: u8 = 1;
/// [`compile_eval_code`] flag: `new.target` is in scope (the fourth capture).
pub(crate) const EVAL_NEW_TARGET: u8 = 2;
/// [`compile_eval_code`] flag: the program is CreateDynamicFunction's
/// `(function anonymous(…) {…})` wrapper; the body returns the function
/// (named `anonymous`, with no binding of that name in its body).
pub(crate) const EVAL_DYN_FN: u8 = 4;

/// Compiles eval code `program` (strict or sloppy as the host decided).
/// Functions it defines are appended to `table` (their ids follow the
/// existing entries); the returned body itself takes no table slot.
///
/// # Errors
/// [`CompileError`] for a construct the VM does not compile; `table` is left
/// unchanged.
pub(crate) fn compile_eval_code(
    program: &Program,
    table: &mut Vec<FnProto>,
    strict: bool,
    flags: u8,
) -> Result<FnProto, CompileError> {
    let base = table.len();
    let protos = alloc::rc::Rc::new(core::cell::RefCell::new(core::mem::take(table)));
    let r = compile_eval_body(program, &protos, strict, flags);
    *table = alloc::rc::Rc::try_unwrap(protos)
        .expect("unique proto table")
        .into_inner();
    if r.is_err() {
        table.truncate(base);
    }
    r
}

fn compile_eval_body(
    program: &Program,
    protos: &alloc::rc::Rc<core::cell::RefCell<Vec<FnProto>>>,
    strict: bool,
    flags: u8,
) -> Result<FnProto, CompileError> {
    let body = &program.body;
    let strict = strict || super::body_starts_strict(body);
    let mut c = Compiler {
        fn_ids: alloc::rc::Rc::new(alloc::collections::BTreeMap::new()),
        classes: alloc::rc::Rc::new(alloc::collections::BTreeMap::new()),
        protos: alloc::rc::Rc::clone(protos),
        strict,
        hosted: true,
        dyn_fn: true,
        ..Compiler::default()
    };
    c.scopes.push(alloc::collections::BTreeMap::new());
    let mut names: Vec<&str> = alloc::vec![ENV_NAME, VARENV_NAME, THIS_NAME];
    if flags & EVAL_NEW_TARGET != 0 {
        names.push(NT_NAME);
    }
    let cap_regs: Vec<_> = names.iter().map(|_| c.alloc()).collect();
    c.this_reg = c.alloc();
    for (j, n) in names.iter().enumerate() {
        c.scopes[0].insert(
            String::from(*n),
            Binding {
                reg: cap_regs[j],
                cell: true,
                konst: false,
                global: None,
                tdz: false,
                mapped: false,
                fn_name: false,
            },
        );
    }
    let vb = c.scopes[0][VARENV_NAME];
    let venv = c.read_var(vb);
    c.var_env = Some(venv);
    if flags & EVAL_DYN_FN != 0 {
        let Some(Stmt::Expr { expression, .. }) = body.first() else {
            return Err(CompileError::Unsupported("dynamic function source"));
        };
        let crate::ast::Expr::Function(f) = &**expression else {
            return Err(CompileError::Unsupported("dynamic function source"));
        };
        c.next_closure_is_generator = f.is_generator;
        let closure = c.make_closure(&f.params, &f.body, f.is_async, "anonymous", false)?;
        c.ops.push(Op::Return { src: closure });
        return Ok(finish_eval_proto(c, names.len()));
    }
    // The host bound the top-level lexical declarations (in their TDZ) and the
    // `var`s; a lexical declaration initializes its binding.
    let mut lexical = BTreeSet::new();
    for stmt in body {
        match stmt {
            Stmt::Var(d) if d.kind != crate::ast::VarDeclKind::Var => {
                for dr in &d.declarations {
                    pattern_names(&dr.target, &mut lexical);
                }
            }
            Stmt::Class(class) => {
                if let Some(id) = &class.id {
                    lexical.insert(String::from(&*id.name));
                }
            }
            _ => {}
        }
    }
    for name in &lexical {
        let b = c.env_site_binding(name, EnvSite::Dyn, GlobalWrite::Put);
        c.scopes[0].insert(name.clone(), b);
    }
    // Annex B.3.3.3: every block function of sloppy eval code updates its
    // `var` binding when evaluated.
    if !strict {
        let mut cands = Vec::new();
        crate::nbexec::collect_block_function_names(body, &mut cands);
        c.annexb_spans
            .extend(cands.into_iter().map(|(_, span)| span));
    }
    // Function declarations, instantiated (in the variable environment) before
    // any statement runs.
    let script = flags & EVAL_SCRIPT;
    for stmt in body {
        if let Stmt::Function(func) = stmt {
            c.hoisted_fns
                .insert(func as *const crate::ast::Function as usize);
            let Some(id) = &func.id else { continue };
            c.next_closure_is_generator = func.is_generator;
            let closure = c.make_closure(
                &func.params,
                &func.body,
                func.is_async,
                id.name.as_ref(),
                false,
            )?;
            c.env_op(EK_DECL_FN, 0, alloc::vec![venv, closure], &id.name, script);
        }
    }
    let completion = c.constant(NanBox::undefined())?;
    c.completion = Some(completion);
    for stmt in body {
        c.stmt(stmt)?;
    }
    c.ops.push(Op::Return { src: completion });
    if c.reg_overflow {
        return Err(CompileError::Unsupported("too many registers"));
    }
    Ok(finish_eval_proto(c, names.len()))
}

fn finish_eval_proto(mut c: Compiler, n_captures: usize) -> FnProto {
    FnProto {
        n_regs: c.next_reg as usize,
        n_params: 0,
        n_captures,
        rest_from: None,
        is_async: false,
        length: 0,
        ops: core::mem::take(&mut c.ops),
        name: String::new(),
        legacy: false,
        class_ctor: false,
        derived: false,
        is_generator: false,
    }
}

/// Runs compiled eval code `proto` (see [`compile_eval_code`]) for the host,
/// with `captures` (the values of its captured cells, in order) and `table`
/// the run's function table.
///
/// # Errors
/// The eval code's throw, or a VM fault.
pub(crate) fn run_eval_code(
    host: &mut dyn VmHost,
    table: &alloc::rc::Rc<[FnProto]>,
    proto: &FnProto,
    captures: &[NanBox],
) -> Result<NanBox, VmError> {
    let mut realm = core::mem::take(host.realm_slot());
    let result = {
        let mut ctx = hosted_ctx(&mut realm, host);
        let r = run_eval_frame(&mut ctx, table, proto, captures);
        if r.is_ok() && !ctx.microtasks.is_empty() {
            Err(VmError::Unsupported)
        } else {
            r
        }
    };
    *host.realm_slot() = realm;
    result
}

fn run_eval_frame(
    ctx: &mut Ctx,
    table: &[FnProto],
    proto: &FnProto,
    captures: &[NanBox],
) -> Result<NanBox, VmError> {
    if ctx.realm.vm_total_depth >= ctx.realm.limits.max_call_depth {
        let e = super::vm_error(ctx, "RangeError", "Maximum call stack size exceeded");
        return Err(VmError::Thrown(e));
    }
    let mut regs: Vec<NanBox> = alloc::vec![NanBox::undefined(); proto.n_regs];
    for (j, v) in captures.iter().enumerate().take(proto.n_captures) {
        let cell = ctx.realm.new_array(alloc::vec![*v]);
        regs[proto.n_params + j] = NanBox::handle(cell.to_raw());
    }
    ctx.realm.vm_total_depth += 1;
    let r = run_frame(ctx, table, &proto.ops, &mut regs);
    ctx.realm.vm_total_depth -= 1;
    match r? {
        FrameExit::Return(v) => Ok(v.unwrap_or(NanBox::undefined())),
        _ => Err(VmError::Unsupported),
    }
}
