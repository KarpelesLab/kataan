//! ES modules on the bytecode VM (`ROADMAP.md` §2.0).
//!
//! The module *machinery* — loading, linking, `ResolveExport`, namespace
//! objects, the async evaluation order of top-level `await` — stays in the
//! interpreter's module loader (`nbexec::module`), which keeps each module's
//! environment as an interpreter [`Scope`](crate::env::Scope): namespace
//! objects and importers read export slots there, live. What moves to the VM is
//! the module *code*:
//!
//! - A module body compiles like a hosted script body whose top-level
//!   declarations live in the module environment rather than the global one:
//!   every environment access of module code (`LoadGlobal`, `StoreGlobal`,
//!   `InitGlobal`, …) carries a [`tagged`] name naming its module, which the
//!   host resolves in that module's environment — its own bindings, its
//!   imports (indirect bindings into the exporters' environments), then the
//!   global environment. Nested functions are module code too, so a function
//!   called from another module (or a promise job) still resolves its names in
//!   the module that defined it.
//! - Top-level function declarations are instantiated at link time by a small
//!   generated *init* function, so they are callable across an import cycle
//!   before their module's body has run.
//! - A body with top-level `await` compiles as an async function; the loader
//!   chains its async-evaluation bookkeeping on the promise its call returns.
//! - `import.meta` and dynamic `import()` are their own ops, stamped with the
//!   module index (the referrer).

use super::{
    BindingTarget, CompileError, Compiler, FnProto, MODULE_DEFAULT_SLOT, Op, Program, Stmt, VmError,
};
use crate::ast::{ExportDecl, Function, Ident, VarDecl, VarDeclKind, VarDeclarator};
use crate::nanbox::NanBox;
use alloc::collections::BTreeMap;
use alloc::rc::Rc;
use alloc::string::String;
use alloc::vec::Vec;

/// The marker that opens a module-environment name (see [`tagged`]); no
/// identifier can contain it.
const TAG: char = '\u{1}';

/// The environment name `name` of VM module `module`: `"\u{1}<module>:<name>"`.
fn tagged(module: u32, name: &str) -> String {
    alloc::format!("{TAG}{module}:{name}")
}

/// Splits a module-environment name (`"\u{1}<module>:<name>"`) into its module index and the
/// plain name; `None` for an ordinary (global-environment) name.
#[must_use]
pub fn split_module_name(name: &str) -> Option<(u32, &str)> {
    let rest = name.strip_prefix(TAG)?;
    let (module, name) = rest.split_once(':')?;
    Some((module.parse().ok()?, name))
}

/// The function-table entries of a compiled module (see
/// [`compile_module_into`]).
#[derive(Clone, Copy, Debug)]
pub struct ModuleProtos {
    /// The module body (an async function for a body with top-level `await`).
    pub main: u32,
    /// The link-time function instantiation: creates the module's top-level
    /// function declarations and binds them in its environment.
    pub init: u32,
}

/// A top-level function declaration of a module and the environment bindings
/// its value initializes (`(name, const)`).
struct TopFn {
    func: Function,
    binds: Vec<(String, bool)>,
    name: String,
}

/// `const *default* = <expr>;` — an anonymous `export default`'s binding.
fn default_binding(init: crate::ast::Expr) -> Stmt {
    let span = crate::common::Span::point(0);
    Stmt::Var(VarDecl {
        kind: VarDeclKind::Const,
        declarations: alloc::vec![VarDeclarator {
            target: BindingTarget::Ident(Ident {
                name: MODULE_DEFAULT_SLOT.into(),
                span,
            }),
            init: Some(init),
            span,
        }],
        span,
    })
}

/// Compiles the module `program` into the shared function `table` (after the
/// entries already there) as VM module number `module`.
///
/// The body is lowered to script shape first: `import` declarations and
/// re-exports disappear (the loader wired them at link time), `export <decl>`
/// becomes `<decl>`, and `export default <expr>` / an anonymous default class
/// becomes `const *default* = <expr>` (named `"default"`). Top-level function
/// declarations — exported or not, an anonymous default included — are
/// instantiated by the separate [`ModuleProtos::init`] function.
///
/// # Errors
/// [`CompileError`] for a construct the VM cannot run in module code; the
/// table is left unchanged.
pub fn compile_module_into(
    program: &Program,
    module: u32,
    is_async: bool,
    table: &mut Vec<FnProto>,
) -> Result<ModuleProtos, CompileError> {
    let mut body: Vec<Stmt> = Vec::new();
    let mut fns: Vec<TopFn> = Vec::new();
    let push = |s: Stmt, body: &mut Vec<Stmt>, fns: &mut Vec<TopFn>| match s {
        Stmt::Function(f) if f.id.is_some() => {
            let name = String::from(&*f.id.as_ref().expect("named").name);
            fns.push(TopFn {
                func: f,
                binds: alloc::vec![(name.clone(), false)],
                name,
            });
        }
        other => body.push(other),
    };
    for s in &program.body {
        match s {
            Stmt::Import(_) | Stmt::Export(ExportDecl::Named { .. } | ExportDecl::All { .. }) => {}
            Stmt::Export(ExportDecl::Decl { declaration, .. }) => {
                push((**declaration).clone(), &mut body, &mut fns);
            }
            // A *named* default function/class binds its own name, which the
            // loader resolves the `default` export to.
            Stmt::Export(ExportDecl::Default { declaration, .. }) => match &**declaration {
                Stmt::Function(f) if f.id.is_none() => fns.push(TopFn {
                    func: f.clone(),
                    binds: alloc::vec![(String::from(MODULE_DEFAULT_SLOT), true)],
                    name: String::from("default"),
                }),
                Stmt::Class(c) if c.id.is_none() => {
                    body.push(default_binding(crate::ast::Expr::Class(c.clone())));
                }
                Stmt::Expr { expression, .. } => body.push(default_binding((**expression).clone())),
                other => push(other.clone(), &mut body, &mut fns),
            },
            other => push(other.clone(), &mut body, &mut fns),
        }
    }

    let base = table.len() as u32;
    let n = fns.len() as u32;
    let init_id = base + n + 1;
    // No static dispatch: every top-level function is a module binding.
    let fn_ids = Rc::new(BTreeMap::new());
    let classes = Rc::new(BTreeMap::new());
    let protos = Rc::new(core::cell::RefCell::new(core::mem::take(table)));
    let placeholder = || FnProto {
        ops: Vec::new(),
        n_regs: 0,
        n_params: 0,
        n_captures: 0,
        rest_from: None,
        is_async: false,
        length: 0,
        name: String::new(),
        legacy: false,
        class_ctor: false,
        derived: false,
        is_generator: false,
        source_span: None,
        source: None,
    };
    protos
        .borrow_mut()
        .extend((base..=init_id).map(|_| placeholder()));
    let no_flags = BTreeMap::new();
    let compiled = (|| {
        let main = Compiler::compile_fn_inner(
            &fn_ids,
            &classes,
            &protos,
            &[],
            &[],
            &body,
            true,
            None,
            &[],
            None,
            is_async,
            true,
            true,
            false,
            &no_flags,
            None,
            &[],
            false,
            false,
        )?;
        protos.borrow_mut()[base as usize] = main;
        for (i, t) in fns.iter().enumerate() {
            let mut proto = Compiler::compile_fn_inner(
                &fn_ids,
                &classes,
                &protos,
                &t.func.params,
                &[],
                &t.func.body,
                false,
                None,
                &[],
                None,
                t.func.is_async,
                true,
                true,
                false,
                &no_flags,
                None,
                &[],
                t.func.is_generator,
                false,
            )?;
            proto.name.clone_from(&t.name);
            protos.borrow_mut()[base as usize + 1 + i] = proto;
        }
        // The init function: one canonical closure per declaration, made and
        // bound exactly as a hosted script body makes its own.
        let mut ops = Vec::new();
        for (i, t) in fns.iter().enumerate() {
            ops.push(Op::LoadFunc {
                dst: 1,
                func: base + 1 + i as u32,
            });
            if t.func.is_generator {
                ops.push(Op::InitGenerator { f: 1 });
            } else if !t.func.is_async {
                ops.push(Op::InitFnPrototype { f: 1 });
            }
            for (name, konst) in &t.binds {
                ops.push(Op::InitGlobal {
                    name: name.clone(),
                    src: 1,
                    konst: *konst,
                });
            }
        }
        ops.push(Op::LoadConst {
            dst: 1,
            value: NanBox::undefined(),
        });
        ops.push(Op::Return { src: 1 });
        protos.borrow_mut()[init_id as usize] = FnProto {
            ops,
            n_regs: 2,
            ..placeholder()
        };
        // Stamp the module into every environment access of its code.
        for proto in protos.borrow_mut()[base as usize..].iter_mut() {
            for op in &mut proto.ops {
                match op {
                    Op::LoadGlobal { name, .. }
                    | Op::TypeofGlobal { name, .. }
                    | Op::StoreGlobal { name, .. }
                    | Op::GlobalExists { name, .. }
                    | Op::DeleteGlobal { name, .. }
                    | Op::InitGlobal { name, .. } => *name = tagged(module, name),
                    // A dynamic-scope function's outermost environment is the
                    // module's.
                    Op::Env { kind, name, .. } if *kind == super::EK_ROOT => {
                        *name = tagged(module, "");
                    }
                    Op::ImportMeta { module: m, .. } | Op::DynImport { module: m, .. } => {
                        *m = module;
                    }
                    // A direct `eval` would run in the global scope, not the
                    // module's.
                    Op::DirectEval { .. } | Op::AnnexBGlobal { .. } => {
                        return Err(CompileError::Unsupported("direct eval in module code"));
                    }
                    _ => {}
                }
            }
        }
        Ok(())
    })();
    *table = Rc::try_unwrap(protos)
        .expect("unique proto table")
        .into_inner();
    match compiled {
        Ok(()) => Ok({
            crate::nbvm::resolve_fn_sources(&mut table[base as usize..], &program.source);
            ModuleProtos {
                main: base,
                init: init_id,
            }
        }),
        Err(e) => {
            table.truncate(base as usize);
            Err(e)
        }
    }
}

/// Whether `e` is a VM fault surfaced through the interpreter (not a JS throw).
fn is_fault(e: &crate::nbexec::ExecError) -> bool {
    !matches!(e, crate::nbexec::ExecError::Throw(_))
}

/// Parses `source` as a Script for a module entry's prelude or a script entry.
fn parse_script(source: &str) -> Result<Program, crate::nbexec::Thrown> {
    crate::parser::Parser::parse_program(source).map_err(|e| crate::nbexec::Thrown {
        phase: crate::nbexec::ErrorPhase::Parse,
        name: String::from("SyntaxError"),
        message: alloc::format!("{e}"),
    })
}

/// How a VM attempt at an entry ended when it did not produce the answer.
enum Attempt {
    /// A JS-level result to report as-is.
    Done(Result<(String, String), crate::nbexec::Thrown>),
    /// The VM could not run it: re-run on the tree-walker (or, under
    /// `KATAAN_VM_STRICT`, report this reason).
    Fallback(String),
}

/// Runs `program` as a hosted script inside `interp` over the module-growable
/// function table.
fn run_hosted_script<'p>(
    interp: &mut crate::nbexec::Interp<'p>,
    program: &'p Program,
) -> Result<NanBox, Attempt> {
    let mut table: Vec<FnProto> = interp
        .module_vm_table()
        .map(|t| t.to_vec())
        .unwrap_or_default();
    let main = match super::compile_program_into(program, true, &mut table) {
        Ok(main) => main,
        Err(e) => return Err(Attempt::Fallback(alloc::format!("compile: {e:?}"))),
    };
    let table: Rc<[FnProto]> = table.into();
    interp.install_module_vm_table(Rc::clone(&table));
    if let Err(e) = interp.prepare_script_for_vm(program) {
        return Err(Attempt::Done(Err(crate::nbexec::thrown_from_exec_error(
            interp,
            e,
            crate::nbexec::ErrorPhase::Runtime,
        ))));
    }
    match super::run_program_hosted(interp, &table, main as usize, &[]) {
        Ok(v) => Ok(v),
        Err(VmError::Thrown(v)) if super::vm_strict() => {
            Err(Attempt::Done(Err(crate::nbexec::thrown_from_exec_error(
                interp,
                crate::nbexec::ExecError::Throw(v),
                crate::nbexec::ErrorPhase::Runtime,
            ))))
        }
        Err(e) => Err(Attempt::Fallback(alloc::format!("runtime: {e:?}"))),
    }
}

/// The VM attempt at a module entry (see [`execute_module_entry`]).
fn attempt_module(
    entry_key: &str,
    host: &dyn crate::nbexec::module::ModuleHost,
    prelude: Option<&Program>,
    limits: crate::limits::Limits,
) -> Attempt {
    use crate::nbexec::ErrorPhase;
    let mut interp = crate::nbexec::Interp::new_with_limits(limits);
    if let Some(program) = prelude
        && let Err(a) = run_hosted_script(&mut interp, program)
    {
        return a;
    }
    interp.enable_vm_modules();
    let linked = interp
        .load_module_pub(entry_key, host)
        .and_then(|()| interp.link_module_pub(entry_key));
    let result = match linked {
        Err(e) if is_fault(&e) => return Attempt::Fallback(alloc::format!("link: {e:?}")),
        Err(e) => Err(interp.exec_error_to_thrown(e, ErrorPhase::Parse)),
        Ok(()) => match interp.evaluate_entry(entry_key) {
            Ok(ns) => Ok((String::from(interp.output()), interp.display(ns))),
            Err(e) if is_fault(&e) => return Attempt::Fallback(alloc::format!("runtime: {e:?}")),
            Err(e) => Err(interp.exec_error_to_thrown(e, ErrorPhase::Runtime)),
        },
    };
    finish(&interp, result)
}

/// The end of a successful VM attempt: a module the VM refused to compile ran
/// on the tree-walker, which `KATAAN_VM_STRICT` reports; under
/// `KATAAN_VM_PURE` so does any tree-walked statement. Outside strict mode a
/// throw re-runs the entry on the tree-walker, as the script entries do.
fn finish(
    interp: &crate::nbexec::Interp,
    result: Result<(String, String), crate::nbexec::Thrown>,
) -> Attempt {
    if interp.vm_module_faulted() {
        return Attempt::Fallback(String::from(
            interp.vm_module_note().unwrap_or("async module fault"),
        ));
    }
    if super::vm_strict()
        && let Some(note) = interp.vm_module_note()
    {
        return Attempt::Done(Err(super::vm_fallback(note)));
    }
    if let Err(t) = super::tree_walk_check(interp) {
        return Attempt::Done(Err(t));
    }
    match result {
        Err(t) if !super::vm_strict() => Attempt::Fallback(alloc::format!("throw: {}", t.name)),
        r => Attempt::Done(r),
    }
}

/// Settles an [`Attempt`]: its result, or the tree-walker's (`fallback`) — a
/// `VmFallback` error under `KATAAN_VM_STRICT`.
fn settle(
    attempt: Attempt,
    fallback: impl FnOnce() -> Result<(String, String), crate::nbexec::Thrown>,
) -> Result<(String, String), crate::nbexec::Thrown> {
    match attempt {
        Attempt::Done(r) => r,
        Attempt::Fallback(reason) if super::vm_strict() => {
            Err(super::vm_fallback(&alloc::format!("module: {reason}")))
        }
        Attempt::Fallback(_) => fallback(),
    }
}

/// [`super::execute_module_typed`] / [`super::execute_module_typed_with_prelude`]:
/// the prelude and every module of the graph run on the VM inside one
/// interpreter (which keeps the module records, environments and namespaces).
/// A failed attempt re-runs the whole entry on the tree-walker from a fresh
/// realm — only console output, which is discarded, was observable.
pub(super) fn execute_module_entry(
    entry_key: &str,
    host: &dyn crate::nbexec::module::ModuleHost,
    prelude: Option<&str>,
    limits: crate::limits::Limits,
) -> Result<(String, String), crate::nbexec::Thrown> {
    let fallback = || match prelude {
        Some(p) => {
            crate::nbexec::module::eval_module_typed_with_prelude(entry_key, host, p, limits)
        }
        None => crate::nbexec::module::eval_module_typed(entry_key, host, limits),
    };
    let program = match prelude.filter(|p| !p.is_empty()).map(parse_script) {
        Some(Ok(p)) if p.source_type == crate::ast::SourceType::Module => return fallback(),
        Some(Ok(p)) => Some(p),
        Some(Err(t)) => return Err(t),
        None => None,
    };
    settle(
        attempt_module(entry_key, host, program.as_ref(), limits),
        fallback,
    )
}

/// [`super::execute_script_typed_with_import_base`] on the VM: a script whose
/// dynamic `import()`s resolve relative to `base_path` and whose imported
/// modules run on the VM too.
pub(super) fn execute_script_with_import_base(
    source: &str,
    base_path: &str,
    limits: crate::limits::Limits,
) -> Result<(String, String), crate::nbexec::Thrown> {
    let fallback =
        || crate::nbexec::module::eval_script_typed_with_import_base(source, base_path, limits);
    let program = parse_script(source)?;
    if program.source_type == crate::ast::SourceType::Module {
        return fallback();
    }
    let attempt = (|| {
        let mut interp = crate::nbexec::Interp::new_with_limits(limits);
        interp.set_script_import_base(Some(String::from(base_path)));
        interp.enable_vm_modules();
        let result = match run_hosted_script(&mut interp, &program) {
            Ok(v) => Ok((String::from(interp.output()), interp.display(v))),
            Err(a) => return a,
        };
        finish(&interp, result)
    })();
    settle(attempt, fallback)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn module_names_round_trip() {
        let t = tagged(7, "x");
        assert_eq!(split_module_name(&t), Some((7, "x")));
        assert_eq!(split_module_name("x"), None);
        assert_eq!(split_module_name(&tagged(0, "a:b")), Some((0, "a:b")));
    }
}
