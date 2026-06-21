//! The ECMAScript **module** subsystem: a module record / resolve+load /
//! link / evaluate pipeline layered on the [`Interp`](super::Interp)
//! tree-walker, plus dynamic `import()` and `import.meta`.
//!
//! This is the abstract-operations machinery of ECMA-262 §16.2 reduced to the
//! shape that fits a tree-walking interpreter whose lexical environments are
//! shared `Rc<RefCell<…>>` [`Scope`]s:
//!
//! - **Parse** a source to a [`ModuleRecord`]: its import requests, its local /
//!   indirect / star exports, and the (leaked) AST body.
//! - **Resolve + Load** dependencies through a host [`ModuleHost`] hook,
//!   transitively, deduping by resolved key and tolerating cycles.
//! - **Link**: give every module its own [`Scope`]; wire each `import {x} from
//!   "m"` to the *export slot* of `m` (`ResolveExport`, including re-exports and
//!   `export *`), so a read sees a **live binding**. A reference before the
//!   source module has run is a **TDZ** `ReferenceError`. Missing / ambiguous
//!   exports are `SyntaxError`s surfaced at link time.
//! - **Evaluate** in DFS post-order (dependencies first), each module exactly
//!   once, draining microtasks so top-level `await` settles.
//! - **Namespace objects** (`import * as ns`, dynamic `import()`): a frozen,
//!   null-prototype exotic object with sorted string keys, `@@toStringTag`
//!   `"Module"`, and live bindings.
//!
//! Gated on `module` + `std` (the loader needs file I/O for the default host);
//! the no_std language core does not pull this in.

use super::{ExecError, Interp, N_SYNTAX_ERROR, NanBox, Thrown};
use crate::ast::{ExportDecl, ImportSpecifier, ModuleExportName, Program, Stmt};
use crate::env::Scope;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::rc::Rc;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// Loads, links, and evaluates the ES-module graph rooted at the resolved
/// `entry_key` through `host`, returning `(console_output, completion_string)`
/// on success or a structured [`Thrown`] (carrying the JS error *type*) on a
/// parse-, link-, or evaluation-phase failure — the module analogue of
/// [`eval_source_typed`](super::eval_source_typed), for the Test262 runner.
///
/// # Errors
/// Returns [`Thrown`] for any parse, link, or runtime failure in the graph.
pub fn eval_module_typed(
    entry_key: &str,
    host: &dyn ModuleHost,
    limits: crate::limits::Limits,
) -> Result<(String, String), Thrown> {
    use super::ErrorPhase;
    let mut interp = Interp::new_with_limits(limits);
    // Load + link are the *parse/resolution* phase (a missing/ambiguous export or
    // a malformed dependency is a parse-phase SyntaxError per the runner's
    // `negative: { phase: parse|resolution }` expectation). Evaluation is the
    // runtime phase.
    let linked = interp
        .load_module_pub(entry_key, host)
        .and_then(|()| interp.link_module_pub(entry_key));
    if let Err(e) = linked {
        return Err(interp.exec_error_to_thrown(e, ErrorPhase::Parse));
    }
    match interp.evaluate_entry(entry_key) {
        Ok(ns) => {
            let completion = interp.display(ns);
            Ok((String::from(interp.output()), completion))
        }
        Err(e) => Err(interp.exec_error_to_thrown(e, ErrorPhase::Runtime)),
    }
}

/// Like [`eval_module_typed`], but first evaluates `prelude` as ordinary script
/// code in the module realm's global (installing global functions/values such as
/// the Test262 harness `assert`/`Test262Error`), then loads, links, and
/// evaluates the module graph rooted at `entry_key`. The module body sees the
/// prelude's globals (a module can read any global binding).
///
/// # Errors
/// Returns [`Thrown`] for a prelude parse/throw or any graph failure.
pub fn eval_module_typed_with_prelude(
    entry_key: &str,
    host: &dyn ModuleHost,
    prelude: &str,
    limits: crate::limits::Limits,
) -> Result<(String, String), Thrown> {
    use super::ErrorPhase;
    let mut interp = Interp::new_with_limits(limits);
    // Evaluate the prelude as a script in the global environment.
    if !prelude.is_empty() {
        let program = match crate::parser::Parser::parse_program(prelude) {
            Ok(p) => alloc::boxed::Box::leak(alloc::boxed::Box::new(p)),
            Err(e) => {
                return Err(Thrown {
                    phase: ErrorPhase::Parse,
                    name: String::from("SyntaxError"),
                    message: alloc::format!("{e}"),
                });
            }
        };
        if let Err(e) = interp.run(program) {
            return Err(interp.exec_error_to_thrown(e, ErrorPhase::Runtime));
        }
    }
    let linked = interp
        .load_module_pub(entry_key, host)
        .and_then(|()| interp.link_module_pub(entry_key));
    if let Err(e) = linked {
        return Err(interp.exec_error_to_thrown(e, ErrorPhase::Parse));
    }
    match interp.evaluate_entry(entry_key) {
        Ok(ns) => {
            let completion = interp.display(ns);
            Ok((String::from(interp.output()), completion))
        }
        Err(e) => Err(interp.exec_error_to_thrown(e, ErrorPhase::Runtime)),
    }
}

/// Runs `source` as a script with a dynamic-`import()` base of `base_path` (so
/// `import("./x.js")` resolves relative to the script file). Mirrors
/// `eval_source_typed` otherwise.
///
/// # Errors
/// Returns [`Thrown`] for a parse failure or uncaught throw.
pub fn eval_script_typed_with_import_base(
    source: &str,
    base_path: &str,
    limits: crate::limits::Limits,
) -> Result<(String, String), Thrown> {
    use super::ErrorPhase;
    let program = match crate::parser::Parser::parse_program(source) {
        Ok(p) => alloc::boxed::Box::leak(alloc::boxed::Box::new(p)),
        Err(e) => {
            return Err(Thrown {
                phase: ErrorPhase::Parse,
                name: String::from("SyntaxError"),
                message: alloc::format!("{e}"),
            });
        }
    };
    let mut interp = Interp::new_with_limits(limits);
    interp.set_script_import_base(Some(base_path.to_string()));
    match interp.run(program) {
        Ok(value) => {
            let completion = interp.display(value);
            Ok((String::from(interp.output()), completion))
        }
        Err(e) => Err(interp.exec_error_to_thrown(e, ErrorPhase::Runtime)),
    }
}

/// Like [`eval_module_typed`] but returns a flattened message on failure (for
/// the CLI / embedders that do not need the structured error type).
///
/// # Errors
/// Returns a human-readable message on any failure.
pub fn eval_module(
    entry_key: &str,
    host: &dyn ModuleHost,
    limits: crate::limits::Limits,
) -> Result<(String, String), String> {
    match eval_module_typed(entry_key, host, limits) {
        Ok(ok) => Ok(ok),
        Err(t) => Err(if t.message.is_empty() {
            t.name
        } else {
            alloc::format!("{}: {}", t.name, t.message)
        }),
    }
}

/// A host hook that resolves a module specifier (relative to its referrer) to a
/// canonical key and loads the corresponding source text. The runner and CLI
/// supply file-relative resolution; an embedder may supply any scheme.
pub trait ModuleHost {
    /// Resolves `specifier` (as written in `import "<specifier>"`) against the
    /// `referrer` key (the importing module's key, or `None` for the entry
    /// module) to a *canonical, deduping* key. Two specifiers that denote the
    /// same module must return the same key.
    ///
    /// # Errors
    /// Returns a human-readable message if the specifier cannot be resolved.
    fn resolve(&self, specifier: &str, referrer: Option<&str>) -> Result<String, String>;

    /// Loads the source text for a resolved key.
    ///
    /// # Errors
    /// Returns a human-readable message if the source cannot be read.
    fn load(&self, key: &str) -> Result<String, String>;
}

/// A [`ModuleHost`] that resolves `import` specifiers as paths relative to the
/// referrer file and reads them from the filesystem. The entry module's key is
/// its absolute path; a relative specifier is joined onto the referrer's parent
/// directory and canonicalised so the same file is deduped under one key.
pub struct FileModuleHost;

impl ModuleHost for FileModuleHost {
    fn resolve(&self, specifier: &str, referrer: Option<&str>) -> Result<String, String> {
        use std::path::{Path, PathBuf};
        let base: PathBuf = match referrer {
            Some(r) => Path::new(r)
                .parent()
                .map_or_else(|| PathBuf::from("."), Path::to_path_buf),
            None => PathBuf::from("."),
        };
        let joined = base.join(specifier);
        // Canonicalise so `./a.js` and `a.js` and `../dir/a.js` dedupe to one
        // key. Fall back to the lexical join if the file does not exist yet (the
        // load step then reports a readable error).
        match std::fs::canonicalize(&joined) {
            Ok(p) => Ok(p.to_string_lossy().into_owned()),
            Err(_) => Ok(joined.to_string_lossy().into_owned()),
        }
    }

    fn load(&self, key: &str) -> Result<String, String> {
        std::fs::read_to_string(key).map_err(|e| alloc::format!("cannot load module {key}: {e}"))
    }
}

/// Where a module is in the link/evaluate lifecycle (a coarse subset of the
/// spec's `[[Status]]`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Status {
    /// Parsed and registered, dependencies not yet loaded.
    New,
    /// Dependencies loaded and parsed (the graph is complete below this node).
    Loaded,
    /// Environment allocated and imports wired.
    Linked,
    /// Body has run (or is running — set before evaluation to break cycles).
    Evaluated,
}

/// One `import` request of a module: the resolved key of the dependency plus the
/// bindings it introduces.
struct ImportEntry {
    /// The resolved key of the imported module.
    key: String,
    /// The specifier as written (for diagnostics).
    specifiers: Vec<ImportBind>,
}

/// A single binding introduced by an import declaration.
enum ImportBind {
    /// `import x from "m"` — bind the local name to `m`'s default export.
    Default(String),
    /// `import * as ns from "m"` — bind the local name to `m`'s namespace object.
    Namespace(String),
    /// `import { imported as local } from "m"`.
    Named { imported: String, local: String },
}

/// A re-export request (`export { x } from "m"` / `export * [as ns] from "m"`).
enum ReExport {
    /// `export { local as exported } from "m"` — re-expose `m`'s `local` export
    /// under `exported`.
    Named {
        key: String,
        local: String,
        exported: String,
    },
    /// `export * from "m"` — re-expose every named export of `m`.
    Star { key: String },
    /// `export * as ns from "m"` — expose `m`'s namespace under `ns`.
    StarAs { key: String, exported: String },
}

/// A parsed, registered module and its link/evaluate state.
struct ModuleRecord {
    /// This module's canonical key.
    key: String,
    /// The leaked module AST (so the interpreter's `&'a` borrows outlive the run,
    /// exactly like the `eval`/`Function` program cache).
    program: &'static Program,
    /// Resolved import requests.
    imports: Vec<ImportEntry>,
    /// Re-export requests (resolved keys).
    reexports: Vec<ReExport>,
    /// Local export names → the module-local binding name they read.
    /// (`export { a as b }` ⇒ `b -> a`; `export const c` ⇒ `c -> c`;
    /// `export default …` ⇒ `default -> *default*`.)
    local_exports: BTreeMap<String, String>,
    /// This module's lexical environment (allocated at link time).
    scope: Scope,
    /// The import alias table (`local -> (source scope, source local name)`),
    /// installed as `Interp::module_imports` while this module evaluates.
    import_aliases: Rc<BTreeMap<String, (Scope, String)>>,
    /// The lazily-built namespace exotic object.
    namespace: Option<NanBox>,
    /// `import.meta` for this module.
    meta: Option<NanBox>,
    status: Status,
    /// A captured evaluation error (so a re-entered, already-failed module
    /// rethrows the same value rather than re-running).
    eval_error: Option<NanBox>,
}

/// The set of loaded modules, keyed by resolved key, plus the active host.
pub struct ModuleRegistry {
    records: BTreeMap<String, ModuleRecord>,
}

impl ModuleRegistry {
    pub(crate) fn new() -> Self {
        Self {
            records: BTreeMap::new(),
        }
    }
}

/// The synthetic local name an `export default` value is bound under.
const DEFAULT_LOCAL: &str = "*default*";

impl<'a> Interp<'a> {
    /// Loads, links, and evaluates the module graph rooted at `entry_key`
    /// (already resolved by the host), then runs the event loop to quiescence.
    /// The host owns specifier resolution and source loading.
    ///
    /// Returns the entry module's namespace object on success. A parse/link
    /// failure surfaces as a `SyntaxError`; an evaluation throw propagates as
    /// the thrown value.
    ///
    /// # Errors
    /// Propagates any parse-, link-, or evaluation-phase failure.
    pub fn run_module(
        &mut self,
        entry_key: &str,
        host: &dyn ModuleHost,
    ) -> Result<NanBox, ExecError> {
        self.load_module(entry_key, host)?;
        self.link_module(entry_key)?;
        self.evaluate_entry(entry_key)
    }

    /// Public wrapper over [`Self::load_module`] (the loader is private; the
    /// phased entry points need to call it from the free functions).
    pub fn load_module_pub(
        &mut self,
        entry_key: &str,
        host: &dyn ModuleHost,
    ) -> Result<(), ExecError> {
        self.load_module(entry_key, host)
    }

    /// Public wrapper over [`Self::link_module`].
    pub fn link_module_pub(&mut self, entry_key: &str) -> Result<(), ExecError> {
        self.link_module(entry_key)
    }

    /// Evaluates an already-linked entry module's graph, drains the event loop
    /// (so top-level `await` and microtasks settle even on failure), and returns
    /// the entry's namespace object.
    pub fn evaluate_entry(&mut self, entry_key: &str) -> Result<NanBox, ExecError> {
        // Make the entry module the last-resort referrer for a dynamic `import()`
        // that runs in a *deferred* microtask (e.g. a `.then`/`await`
        // continuation), after the synchronous module body — and thus after
        // `active_module_key` has been restored — has returned. Without this a
        // self-import like `import("./self.js")` from such a continuation would
        // resolve against the process cwd.
        if self.script_import_base.is_none() {
            self.script_import_base = Some(entry_key.to_string());
        }
        let r = self.evaluate_module(entry_key);
        // Drain microtasks/timers even on failure so a rejected top-level-await
        // promise's reactions run (matching `run`'s post-body event loop).
        let _ = self.run_event_loop();
        r?;
        self.run_event_loop()?;
        self.namespace_object(entry_key)
    }

    /// Converts an [`ExecError`] surfaced from the module pipeline into a typed
    /// [`Thrown`], tagging it with `phase` (Parse for load/link, Runtime for
    /// evaluation). Mirrors `eval_source_typed`'s error rendering.
    pub fn exec_error_to_thrown(&self, e: ExecError, phase: super::ErrorPhase) -> Thrown {
        match e {
            ExecError::Throw(thrown) => {
                let (name, message) = super::error_name_message(self, thrown)
                    .unwrap_or_else(|| (self.display(thrown), String::new()));
                Thrown {
                    phase,
                    name,
                    message,
                }
            }
            other => Thrown {
                phase,
                name: String::from("Error"),
                message: alloc::format!("{other:?}"),
            },
        }
    }

    // --- Load -----------------------------------------------------------

    /// Transitively loads and parses `key` and its dependencies, deduping by key
    /// and tolerating import cycles (a key already present is not reloaded).
    fn load_module(&mut self, key: &str, host: &dyn ModuleHost) -> Result<(), ExecError> {
        if self.modules.records.contains_key(key) {
            return Ok(());
        }
        let source = host.load(key).map_err(|e| self.syntax_error(&e))?;
        let record = self.parse_module(key, &source, host)?;
        // Collect dependency keys before recursing (the borrow of `record` ends).
        let deps: Vec<String> = record
            .imports
            .iter()
            .map(|i| i.key.clone())
            .chain(record.reexports.iter().map(reexport_key))
            .collect();
        self.modules.records.insert(key.to_string(), record);
        for dep in deps {
            self.load_module(&dep, host)?;
        }
        if let Some(r) = self.modules.records.get_mut(key) {
            r.status = Status::Loaded;
        }
        Ok(())
    }

    /// Parses one module source into a [`ModuleRecord`], resolving each import /
    /// re-export specifier to its dependency key via the host.
    fn parse_module(
        &mut self,
        key: &str,
        source: &str,
        host: &dyn ModuleHost,
    ) -> Result<ModuleRecord, ExecError> {
        let program = crate::parser::Parser::parse_module(source)
            .map_err(|e| self.syntax_error(&alloc::format!("{e}")))?;
        // A source with no import/export is still a valid module (a script-shaped
        // module). Leak it like the eval cache so its AST is `'static`/`'a`.
        let program: &'static Program = alloc::boxed::Box::leak(alloc::boxed::Box::new(program));

        let mut imports: Vec<ImportEntry> = Vec::new();
        let mut reexports: Vec<ReExport> = Vec::new();
        let mut local_exports: BTreeMap<String, String> = BTreeMap::new();

        let resolve = |spec: &str, this: &mut Self| -> Result<String, ExecError> {
            host.resolve(spec, Some(key))
                .map_err(|e| this.syntax_error(&e))
        };

        for stmt in &program.body {
            match stmt {
                Stmt::Import(decl) => {
                    let dep = resolve(&decl.source, self)?;
                    let mut binds = Vec::new();
                    for s in &decl.specifiers {
                        match s {
                            ImportSpecifier::Default(id) => {
                                binds.push(ImportBind::Default(id.name.to_string()));
                            }
                            ImportSpecifier::Namespace(id) => {
                                binds.push(ImportBind::Namespace(id.name.to_string()));
                            }
                            ImportSpecifier::Named { imported, local } => {
                                binds.push(ImportBind::Named {
                                    imported: export_name(imported),
                                    local: local.name.to_string(),
                                });
                            }
                        }
                    }
                    imports.push(ImportEntry {
                        key: dep,
                        specifiers: binds,
                    });
                }
                Stmt::Export(ExportDecl::Named {
                    specifiers,
                    source: Some(src),
                    ..
                }) => {
                    let dep = resolve(src, self)?;
                    for sp in specifiers {
                        reexports.push(ReExport::Named {
                            key: dep.clone(),
                            local: export_name(&sp.local),
                            exported: export_name(&sp.exported),
                        });
                    }
                }
                Stmt::Export(ExportDecl::Named {
                    specifiers,
                    source: None,
                    ..
                }) => {
                    for sp in specifiers {
                        local_exports.insert(export_name(&sp.exported), export_name(&sp.local));
                    }
                }
                Stmt::Export(ExportDecl::All {
                    exported,
                    source: src,
                    ..
                }) => {
                    let dep = resolve(src, self)?;
                    match exported {
                        Some(name) => reexports.push(ReExport::StarAs {
                            key: dep,
                            exported: export_name(name),
                        }),
                        None => reexports.push(ReExport::Star { key: dep }),
                    }
                }
                Stmt::Export(ExportDecl::Default { declaration, .. }) => {
                    // A *named* `export default function f`/`class C` exports the
                    // `f`/`C` binding itself (so a later reassignment of `f` inside
                    // the function is observed through the `default` export — a live
                    // binding). An anonymous default binds the synthetic
                    // `*default*` slot.
                    let local = decl_name(declaration).map_or_else(
                        || DEFAULT_LOCAL.to_string(),
                        alloc::string::ToString::to_string,
                    );
                    local_exports.insert("default".to_string(), local);
                }
                Stmt::Export(ExportDecl::Decl { declaration, .. }) => {
                    for name in declared_names(declaration) {
                        local_exports.insert(name.clone(), name);
                    }
                }
                _ => {}
            }
        }

        Ok(ModuleRecord {
            key: key.to_string(),
            program,
            imports,
            reexports,
            local_exports,
            scope: Scope::root(),
            import_aliases: Rc::new(BTreeMap::new()),
            namespace: None,
            meta: None,
            status: Status::New,
            eval_error: None,
        })
    }

    // --- Link -----------------------------------------------------------

    /// Allocates each module's environment (a child of the global scope) and
    /// wires its imports to the exporting modules' binding slots, depth-first.
    /// Idempotent per module (a cycle re-entry is a no-op once linked).
    fn link_module(&mut self, key: &str) -> Result<(), ExecError> {
        match self.modules.records.get(key).map(|r| r.status) {
            Some(Status::New | Status::Loaded) => {}
            // Already linked/evaluated (or a cycle's back-edge): nothing to do.
            _ => return Ok(()),
        }
        // Allocate this module's scope and mark Linked *before* recursing so an
        // import cycle terminates.
        let scope = self.global_scope.child();
        if let Some(r) = self.modules.records.get_mut(key) {
            r.scope = scope;
            r.status = Status::Linked;
        }
        let dep_keys: Vec<String> = {
            let r = &self.modules.records[key];
            r.imports
                .iter()
                .map(|i| i.key.clone())
                .chain(r.reexports.iter().map(reexport_key))
                .collect()
        };
        for dep in &dep_keys {
            self.link_module(dep)?;
        }

        // Build the import alias table for this module: each imported binding
        // points at the exporting module's scope + local name (a live slot), or
        // is materialised eagerly (namespace object / default).
        let imports: Vec<DepBinds> = {
            let r = &self.modules.records[key];
            r.imports
                .iter()
                .map(|i| {
                    let binds = i
                        .specifiers
                        .iter()
                        .map(|b| match b {
                            ImportBind::Default(local) => (local.clone(), ImportKind::Default),
                            ImportBind::Namespace(local) => (local.clone(), ImportKind::Namespace),
                            ImportBind::Named { imported, local } => {
                                (local.clone(), ImportKind::Named(imported.clone()))
                            }
                        })
                        .collect();
                    DepBinds {
                        dep: i.key.clone(),
                        binds,
                    }
                })
                .collect()
        };

        let mut aliases: BTreeMap<String, (Scope, String)> = BTreeMap::new();
        for DepBinds { dep: dep_key, binds } in &imports {
            for (local, kind) in binds {
                match kind {
                    ImportKind::Default => {
                        let (src_scope, src_name) =
                            self.resolve_export(dep_key, "default", &mut BTreeSet::new())?;
                        aliases.insert(local.clone(), (src_scope, src_name));
                    }
                    ImportKind::Named(imported) => {
                        let (src_scope, src_name) =
                            self.resolve_export(dep_key, imported, &mut BTreeSet::new())?;
                        aliases.insert(local.clone(), (src_scope, src_name));
                    }
                    ImportKind::Namespace => {
                        // `import * as ns`: bind `ns` directly in this module's
                        // own scope to the dependency's namespace object (a
                        // constant binding, not a live slot).
                        let ns = self.namespace_object(dep_key)?;
                        let r = &self.modules.records[key];
                        r.scope.declare_const(local, ns);
                    }
                }
            }
        }
        // Validate this module's own re-exports resolve (link-time SyntaxError on
        // a missing/ambiguous re-exported name).
        let reexports: Vec<ReExport> = {
            let r = &self.modules.records[key];
            r.reexports
                .iter()
                .map(|re| match re {
                    ReExport::Named {
                        key,
                        local,
                        exported,
                    } => ReExport::Named {
                        key: key.clone(),
                        local: local.clone(),
                        exported: exported.clone(),
                    },
                    ReExport::Star { key } => ReExport::Star { key: key.clone() },
                    ReExport::StarAs { key, exported } => ReExport::StarAs {
                        key: key.clone(),
                        exported: exported.clone(),
                    },
                })
                .collect()
        };
        for re in &reexports {
            if let ReExport::Named { key, local, .. } = re {
                self.resolve_export(key, local, &mut BTreeSet::new())?;
            }
        }

        let aliases = Rc::new(aliases);
        if let Some(r) = self.modules.records.get_mut(key) {
            r.import_aliases = aliases;
        }
        // Instantiate this module's top-level function declarations into its
        // scope *now*, at link time (the spec's InitializeEnvironment step). A
        // function is thus callable across an import cycle before the defining
        // module's body has run — e.g. `b` may call `a`'s exported function even
        // when `a` is mid-evaluation.
        self.instantiate_module_functions(key)?;
        Ok(())
    }

    /// Pre-declares a module's top-level function declarations (including
    /// `export function`/`export default function`) in its scope, capturing that
    /// scope as their closure environment — the link-time function instantiation
    /// that makes functions usable across import cycles.
    fn instantiate_module_functions(&mut self, key: &str) -> Result<(), ExecError> {
        let (scope, program) = {
            let r = &self.modules.records[key];
            (r.scope.clone(), r.program)
        };
        let saved = core::mem::replace(&mut self.current, scope.clone());
        let saved_strict = core::mem::replace(&mut self.strict, true);
        for stmt in &program.body {
            let inner = match stmt {
                Stmt::Function(_) => stmt,
                Stmt::Export(ExportDecl::Decl { declaration, .. })
                | Stmt::Export(ExportDecl::Default { declaration, .. }) => declaration,
                _ => continue,
            };
            if let Stmt::Function(func) = inner {
                let is_default = matches!(stmt, Stmt::Export(ExportDecl::Default { .. }));
                match &func.id {
                    Some(id) => {
                        let value = self.make_function(
                            &func.params,
                            super::Body::Block(&func.body),
                            func.is_async,
                            func.is_generator,
                        );
                        self.set_fn_name(value, &id.name);
                        scope.declare(&id.name, value);
                        // `export default function f` also binds `*default*`.
                        if is_default {
                            scope.declare_const(DEFAULT_LOCAL, value);
                        }
                    }
                    // `export default function() {}` / `function*() {}` (anonymous)
                    // is a HoistableDeclaration: instantiate it now under
                    // `*default*` with the name "default", so an importer can call
                    // the default export before this module's body has run.
                    None if is_default => {
                        let value = self.make_function(
                            &func.params,
                            super::Body::Block(&func.body),
                            func.is_async,
                            func.is_generator,
                        );
                        self.set_fn_name(value, "default");
                        scope.declare_const(DEFAULT_LOCAL, value);
                    }
                    None => {}
                }
            }
        }
        self.current = saved;
        self.strict = saved_strict;
        Ok(())
    }

    /// `ResolveExport(module, name)` — finds the *binding slot* (scope + local
    /// name) that backs export `name` of `module`, following re-exports and
    /// `export *`. `seen` breaks cycles. A missing export, or two equally-good
    /// star re-exports of the same name, is a link-time `SyntaxError`.
    fn resolve_export(
        &mut self,
        key: &str,
        name: &str,
        seen: &mut BTreeSet<(String, String)>,
    ) -> Result<(Scope, String), ExecError> {
        if !seen.insert((key.to_string(), name.to_string())) {
            // A cycle in re-export resolution → ambiguous/unresolvable.
            return Err(self.syntax_error(&alloc::format!(
                "circular re-export resolving '{name}' from {key}"
            )));
        }
        let Some(record) = self.modules.records.get(key) else {
            return Err(self.syntax_error(&alloc::format!("module not loaded: {key}")));
        };
        // 1. A local export.
        if let Some(local) = record.local_exports.get(name) {
            return Ok((record.scope.clone(), local.clone()));
        }
        // 2. A direct re-export `export { local as name } from "m"`.
        let named: Vec<(String, String)> = record
            .reexports
            .iter()
            .filter_map(|re| match re {
                ReExport::Named {
                    key,
                    local,
                    exported,
                } if exported == name => Some((key.clone(), local.clone())),
                ReExport::StarAs { key, exported } if exported == name => {
                    Some((key.clone(), String::new()))
                }
                _ => None,
            })
            .collect();
        if let Some((dep, local)) = named.first() {
            if local.is_empty() {
                // `export * as name` — the slot is a namespace object; create a
                // synthetic const binding in this module's scope holding it.
                let ns = self.namespace_object(dep)?;
                let scope = self.modules.records[key].scope.clone();
                let synth = alloc::format!("*ns:{dep}*");
                scope.declare_const(&synth, ns);
                return Ok((scope, synth));
            }
            return self.resolve_export(dep, local, seen);
        }
        // 3. `export * from "m"` — search each star dependency; ambiguous if more
        //    than one resolves the name.
        let stars: Vec<String> = record
            .reexports
            .iter()
            .filter_map(|re| match re {
                ReExport::Star { key } => Some(key.clone()),
                _ => None,
            })
            .collect();
        let mut found: Option<(Scope, String)> = None;
        for dep in &stars {
            // `default` is never provided by `export *`.
            if name == "default" {
                continue;
            }
            if let Ok(slot) = self.resolve_export(dep, name, &mut seen.clone()) {
                if let Some(prev) = &found {
                    // Multiple `export *` paths are only *ambiguous* when they
                    // resolve to **distinct** bindings. Two star re-exports that
                    // ultimately denote the *same* slot (same scope + local) — or
                    // the same materialised value, e.g. two `export * as ns from
                    // "m"` of one module — are unambiguous (ResolveExport returns
                    // that single binding).
                    let same_slot = prev.0.ptr_eq(&slot.0) && prev.1 == slot.1;
                    let same_value = {
                        let a = prev.0.get(&prev.1);
                        let b = slot.0.get(&slot.1);
                        matches!((a, b), (Some(x), Some(y)) if x.as_handle() == y.as_handle() && x.as_handle().is_some())
                    };
                    if !same_slot && !same_value {
                        return Err(self.syntax_error(&alloc::format!(
                            "ambiguous export '{name}' (multiple `export *`)"
                        )));
                    }
                } else {
                    found = Some(slot);
                }
            }
        }
        if let Some(slot) = found {
            return Ok(slot);
        }
        Err(self.syntax_error(&alloc::format!(
            "module {key} has no export named '{name}'"
        )))
    }

    // --- Evaluate -------------------------------------------------------

    /// Evaluates `key` and (post-order) its dependencies, each exactly once.
    /// A module is marked `Evaluated` *before* its body runs so an import cycle
    /// does not re-enter it.
    fn evaluate_module(&mut self, key: &str) -> Result<(), ExecError> {
        match self.modules.records.get(key).map(|r| r.status) {
            Some(Status::Evaluated) => {
                // Re-entrant (cycle) or already done; rethrow a prior failure.
                if let Some(err) = self.modules.records.get(key).and_then(|r| r.eval_error) {
                    return Err(ExecError::Throw(err));
                }
                return Ok(());
            }
            Some(Status::Linked) => {}
            _ => return Err(self.syntax_error(&alloc::format!("module {key} not linked"))),
        }
        // Evaluate dependencies first (post-order).
        let deps: Vec<String> = {
            let r = &self.modules.records[key];
            r.imports
                .iter()
                .map(|i| i.key.clone())
                .chain(r.reexports.iter().map(reexport_key))
                .collect()
        };
        // Mark Evaluated up front to break cycles.
        if let Some(r) = self.modules.records.get_mut(key) {
            r.status = Status::Evaluated;
        }
        for dep in &deps {
            self.evaluate_module(dep)?;
        }
        let result = self.run_module_body(key);
        if let Err(ExecError::Throw(v)) = &result
            && let Some(r) = self.modules.records.get_mut(key)
        {
            r.eval_error = Some(*v);
        }
        result
    }

    /// Runs one module's top-level statements in its own environment, with its
    /// import aliases active and `import.meta` set up. Modules are always strict.
    fn run_module_body(&mut self, key: &str) -> Result<(), ExecError> {
        let (scope, program, aliases) = {
            let r = &self.modules.records[key];
            (r.scope.clone(), r.program, r.import_aliases.clone())
        };
        // `import.meta` (built lazily, once).
        let meta = self.module_meta(key);

        let saved_scope = core::mem::replace(&mut self.current, scope.clone());
        let saved_var = core::mem::replace(&mut self.var_scope, scope.clone());
        let saved_strict = core::mem::replace(&mut self.strict, true);
        let saved_imports = core::mem::replace(&mut self.module_imports, aliases);
        let saved_meta = self.import_meta.replace(meta);
        let saved_this = core::mem::replace(&mut self.this_val, NanBox::undefined());
        let saved_annexb = core::mem::take(&mut self.annexb_block_fns);
        let saved_active = self.active_module_key.replace(key.to_string());

        let result = self.exec_module_stmts(&program.body);

        self.current = saved_scope;
        self.var_scope = saved_var;
        self.strict = saved_strict;
        self.module_imports = saved_imports;
        self.import_meta = saved_meta;
        self.this_val = saved_this;
        self.annexb_block_fns = saved_annexb;
        self.active_module_key = saved_active;
        result
    }

    /// Hoists then executes a module body's statements, treating `import` as a
    /// no-op (bindings were wired at link time) and `export` by evaluating its
    /// inner declaration / default expression.
    fn exec_module_stmts(&mut self, stmts: &'a [Stmt]) -> Result<(), ExecError> {
        // `var` + function-declaration hoisting at the module variable
        // environment boundary (lexical `let`/`const`/`class` bind on execution).
        self.hoist_with(stmts, true)?;
        for stmt in stmts {
            match stmt {
                Stmt::Import(_) => {}
                Stmt::Export(decl) => self.exec_export(decl)?,
                other => {
                    self.exec(other)?;
                }
            }
        }
        Ok(())
    }

    /// Evaluates an `export` declaration's payload (the binding side; the export
    /// *slot* wiring already happened at link time).
    fn exec_export(&mut self, decl: &'a ExportDecl) -> Result<(), ExecError> {
        match decl {
            // Re-exports and bare `export { … }` bind nothing locally.
            ExportDecl::All { .. } => Ok(()),
            ExportDecl::Named { source: Some(_), .. } => Ok(()),
            ExportDecl::Named { source: None, .. } => Ok(()),
            ExportDecl::Decl { declaration, .. } => {
                self.exec(declaration)?;
                Ok(())
            }
            ExportDecl::Default { declaration, .. } => {
                // `export default function/class …` declares a *named* binding too
                // (the `default` export resolves to it, set up in `parse_module`);
                // `export default <expr>` binds the value under `*default*`.
                match &**declaration {
                    // A *named* function/class declaration: execute it (it hoists /
                    // binds its own name), then alias `*default*` to that value so
                    // an anonymous-export observer still sees it.
                    Stmt::Function(crate::ast::Function { id: Some(_), .. })
                    | Stmt::Class(crate::ast::Class { id: Some(_), .. }) => {
                        self.exec(declaration)?;
                        let value = decl_name(declaration)
                            .and_then(|n| self.current.get(n))
                            .unwrap_or_else(NanBox::undefined);
                        self.current.declare_const(DEFAULT_LOCAL, value);
                        Ok(())
                    }
                    // An *anonymous* `export default function(){}` / `class {}`:
                    // build the value as an expression and give it the name
                    // `"default"` (NamedEvaluation), bound under `*default*`.
                    Stmt::Function(func) => {
                        let value = self.make_function(
                            &func.params,
                            super::Body::Block(&func.body),
                            func.is_async,
                            func.is_generator,
                        );
                        self.set_fn_name(value, "default");
                        self.current.declare_const(DEFAULT_LOCAL, value);
                        Ok(())
                    }
                    Stmt::Class(class) => {
                        let value = self.make_class(class)?;
                        self.set_fn_name(value, "default");
                        self.current.declare_const(DEFAULT_LOCAL, value);
                        Ok(())
                    }
                    Stmt::Expr { expression, .. } => {
                        // `export default <expr>`: an anonymous function/class/arrow
                        // expression is named "default" by NamedEvaluation
                        // (`set_fn_name` is a no-op on a value that already has a
                        // name or is not a function/class).
                        let value = self.eval(expression)?;
                        if matches!(
                            &**expression,
                            crate::ast::Expr::Function(crate::ast::Function { id: None, .. })
                                | crate::ast::Expr::Class(crate::ast::Class { id: None, .. })
                                | crate::ast::Expr::Arrow(_)
                        ) {
                            self.set_fn_name(value, "default");
                        }
                        self.current.declare_const(DEFAULT_LOCAL, value);
                        Ok(())
                    }
                    other => {
                        self.exec(other)?;
                        Ok(())
                    }
                }
            }
        }
    }

    // --- Namespace objects ---------------------------------------------

    /// Returns (creating once) the **module namespace exotic object** for `key`:
    /// a frozen, null-prototype object whose own enumerable keys are the module's
    /// resolved export names in sorted order, each a live read-through of its
    /// binding slot, plus a non-enumerable `@@toStringTag` of `"Module"`.
    fn namespace_object(&mut self, key: &str) -> Result<NanBox, ExecError> {
        if let Some(ns) = self.modules.records.get(key).and_then(|r| r.namespace) {
            return Ok(ns);
        }
        // Allocate and cache the (initially empty) object *before* resolving its
        // exports, so a self-referential `export * as ns from "./self"` —
        // which resolves back into this same namespace — returns the in-progress
        // handle instead of recursing forever.
        let obj = self.realm.new_object_with_proto(None);
        let ns = NanBox::handle(obj.to_raw());
        if let Some(r) = self.modules.records.get_mut(key) {
            r.namespace = Some(ns);
        }
        let names = self.export_names(key, &mut BTreeSet::new())?;
        // Resolve every name to its slot. A name that resolves *ambiguously* (or
        // is otherwise unresolvable — only reachable via `export *`) is **omitted**
        // from the namespace per GetModuleNamespace, *not* an error. A direct
        // `import { x }` of such a name is still rejected at link time (that path
        // calls `resolve_export` separately and propagates the error).
        let mut slots: Vec<(String, Scope, String)> = Vec::new();
        for n in &names {
            if let Ok((s, l)) = self.resolve_export(key, n, &mut BTreeSet::new()) {
                slots.push((n.clone(), s, l));
            }
        }
        // Snapshot the current values; namespace properties read the *current*
        // binding value. (A live read-through would need an accessor per name;
        // we snapshot at first materialisation, which is correct for the common
        // case where the namespace is observed after the module has evaluated.)
        for (name, scope, local) in &slots {
            let value = scope.get(local).unwrap_or_else(NanBox::undefined);
            self.realm.set_property(obj, name, value);
        }
        // Record each export's backing slot so a later read of `ns.<name>`
        // refreshes from the live binding (§28.3 — namespace properties are live).
        let binding_map: BTreeMap<String, (Scope, String)> = slots
            .iter()
            .map(|(n, s, l)| (n.clone(), (s.clone(), l.clone())))
            .collect();
        self.module_namespaces.insert(obj.to_raw(), binding_map);
        // `@@toStringTag` = "Module", non-enumerable, non-writable.
        let tag_sym = self.well_known_symbol("toStringTag");
        let tag_key = self.member_key(tag_sym);
        let module_str = self.new_str("Module");
        self.realm.set_property(obj, &tag_key, module_str);
        self.realm.mark_hidden(obj, &tag_key);
        self.realm.set_readonly_property(obj, &tag_key);
        self.realm.set_non_configurable_property(obj, &tag_key);
        // Per §28.3 a module namespace exotic object's export bindings are
        // *writable* data properties (the binding value is live), but
        // **non-configurable**, and the object itself is non-extensible. (They are
        // not frozen — freezing would report `writable: false`, which the spec and
        // the namespace conformance tests reject.)
        for (name, _, _) in &slots {
            self.realm.set_non_configurable_property(obj, name);
        }
        self.realm.prevent_extensions(obj);
        Ok(ns)
    }

    /// `GetExportedNames(module)` — the sorted set of export names a namespace
    /// object exposes (locals + named re-exports + star-imported names, minus
    /// `default` for `export *`). `seen` breaks `export *` cycles.
    fn export_names(
        &mut self,
        key: &str,
        seen: &mut BTreeSet<String>,
    ) -> Result<Vec<String>, ExecError> {
        if !seen.insert(key.to_string()) {
            return Ok(Vec::new());
        }
        let Some(record) = self.modules.records.get(key) else {
            return Err(self.syntax_error(&alloc::format!("module not loaded: {key}")));
        };
        let mut names: BTreeSet<String> = record.local_exports.keys().cloned().collect();
        let mut star_deps: Vec<String> = Vec::new();
        for re in &record.reexports {
            match re {
                ReExport::Named { exported, .. } | ReExport::StarAs { exported, .. } => {
                    names.insert(exported.clone());
                }
                ReExport::Star { key } => star_deps.push(key.clone()),
            }
        }
        for dep in star_deps {
            for n in self.export_names(&dep, seen)? {
                if n != "default" {
                    names.insert(n);
                }
            }
        }
        Ok(names.into_iter().collect())
    }

    // --- import.meta ----------------------------------------------------

    /// Builds (once) the module's `import.meta` object: a plain object with a
    /// `url` property (the module key as a `file://`-ish URL).
    fn module_meta(&mut self, key: &str) -> NanBox {
        if let Some(m) = self.modules.records.get(key).and_then(|r| r.meta) {
            return m;
        }
        // `import.meta` is an ordinary object with a **null** `[[Prototype]]`
        // (OrdinaryObjectCreate(null) per §16.2.1.6.3), so it inherits no
        // `toString`/`valueOf`; `ToString(import.meta)` therefore throws.
        let obj = self.realm.new_object_with_proto(None);
        let url = if key.starts_with("file://") || key.contains("://") {
            key.to_string()
        } else {
            alloc::format!("file://{key}")
        };
        let url_val = self.new_str(&url);
        self.realm.set_property(obj, "url", url_val);
        let meta = NanBox::handle(obj.to_raw());
        if let Some(r) = self.modules.records.get_mut(key) {
            r.meta = Some(meta);
        }
        meta
    }

    // --- Dynamic import() -----------------------------------------------

    /// Evaluates `import(specifier)` — a synchronous-best-effort dynamic import
    /// that loads, links, and evaluates the referenced module and returns a
    /// promise fulfilled with its namespace object (or rejected on any
    /// resolve/load/link/evaluate failure). The specifier resolves relative to
    /// the *currently evaluating* module (or the entry, for a script).
    pub(crate) fn dynamic_import(
        &mut self,
        arguments: &'a [crate::ast::Argument],
    ) -> Result<NanBox, ExecError> {
        let promise = self.fresh_promise();
        // Evaluate the specifier argument (the first item; options are ignored).
        let spec = match arguments.first() {
            Some(crate::ast::Argument::Item(e)) => self.eval(e)?,
            _ => NanBox::undefined(),
        };
        let referrer = self.current_module_key();
        let host = FileModuleHost;
        let outcome: Result<NanBox, ExecError> = (|this: &mut Self| {
            // `ToString(specifier)` is part of the ImportCall steps, so a throwing
            // `toString`/`Symbol.toPrimitive`/`valueOf` *rejects the promise*
            // rather than propagating synchronously. Use the real ToString (not
            // the lossy display form) so a user `toString` override is honoured.
            let spec_str = this.coerce_to_string(spec)?;
            let dep = host
                .resolve(&spec_str, referrer.as_deref())
                .map_err(|e| this.type_error(&e))?;
            this.load_module(&dep, &host)?;
            this.link_module(&dep)?;
            this.evaluate_module(&dep)?;
            this.namespace_object(&dep)
        })(self);
        match outcome {
            Ok(ns) => self.settle(promise, ns, true),
            Err(ExecError::Throw(v)) => self.settle(promise, v, false),
            Err(other) => {
                let m = self.new_str(&alloc::format!("dynamic import failed: {other:?}"));
                let err = self.make_error(N_SYNTAX_ERROR, Some(m));
                self.settle(promise, err, false);
            }
        }
        Ok(NanBox::handle(promise.to_raw()))
    }

    /// The key of the module whose body is currently running, found by matching
    /// the active scope against each record's scope. Falls back to the script
    /// import base (so a script's `import()` resolves relative to its file).
    fn current_module_key(&self) -> Option<String> {
        self.modules
            .records
            .values()
            .find(|r| r.scope.ptr_eq(&self.current))
            .map(|r| r.key.clone())
            // The active scope may be a nested function's, not the module's own;
            // fall back to the module whose body is on the call stack.
            .or_else(|| self.active_module_key.clone())
            .or_else(|| self.script_import_base.clone())
    }

    /// Sets the base referrer for dynamic `import()` from script code (the
    /// script's own path), so `import("./sib.js")` resolves relative to it.
    pub fn set_script_import_base(&mut self, base: Option<String>) {
        self.script_import_base = base;
    }

}

/// What kind of slot an import binding maps to.
enum ImportKind {
    Default,
    Namespace,
    Named(String),
}

/// A dependency key paired with the `(local name, kind)` bindings an import from
/// it introduces — a flattened, borrow-free view built during linking.
struct DepBinds {
    dep: String,
    binds: Vec<(String, ImportKind)>,
}

/// The resolved dependency key of a re-export.
fn reexport_key(re: &ReExport) -> String {
    match re {
        ReExport::Named { key, .. } | ReExport::Star { key } | ReExport::StarAs { key, .. } => {
            key.clone()
        }
    }
}

/// The string form of a `ModuleExportName`.
fn export_name(n: &ModuleExportName) -> String {
    match n {
        ModuleExportName::Ident(s) | ModuleExportName::Str(s) => s.to_string(),
    }
}

/// The id of a function/class declaration statement, if any.
fn decl_name(stmt: &Stmt) -> Option<&str> {
    match stmt {
        Stmt::Function(f) => f.id.as_ref().map(|i| i.name.as_ref()),
        Stmt::Class(c) => c.id.as_ref().map(|i| i.name.as_ref()),
        _ => None,
    }
}

/// The names a `var`/`let`/`const`/function/class declaration binds (so
/// `export const a = 1, b = 2;` exports both `a` and `b`).
fn declared_names(stmt: &Stmt) -> Vec<String> {
    let mut out = Vec::new();
    match stmt {
        Stmt::Function(f) => {
            if let Some(id) = &f.id {
                out.push(id.name.to_string());
            }
        }
        Stmt::Class(c) => {
            if let Some(id) = &c.id {
                out.push(id.name.to_string());
            }
        }
        Stmt::Var(decl) => {
            for d in &decl.declarations {
                collect_pattern_names(&d.target, &mut out);
            }
        }
        _ => {}
    }
    out
}

/// Collects the binding names of a (possibly destructuring) declaration target.
fn collect_pattern_names(target: &crate::ast::BindingTarget, out: &mut Vec<String>) {
    use crate::ast::{ArrayPatternElement, BindingTarget};
    match target {
        BindingTarget::Ident(id) => out.push(id.name.to_string()),
        BindingTarget::Array(pat) => {
            for el in &pat.elements {
                match el {
                    ArrayPatternElement::Hole => {}
                    ArrayPatternElement::Item { target, .. }
                    | ArrayPatternElement::Rest { target, .. } => {
                        collect_pattern_names(target, out);
                    }
                }
            }
        }
        BindingTarget::Object(pat) => {
            for p in &pat.properties {
                collect_pattern_names(&p.value, out);
            }
            if let Some(r) = &pat.rest {
                collect_pattern_names(r, out);
            }
        }
    }
}

