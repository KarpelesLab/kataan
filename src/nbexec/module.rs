//! The ECMAScript **module** subsystem: a module record / resolve+load /
//! link / evaluate pipeline layered on the [`Interp`]
//! tree-walker, plus dynamic `import()` and `import.meta`.
//!
//! This is the abstract-operations machinery of ECMA-262 §16.2 reduced to the
//! shape that fits a tree-walking interpreter whose lexical environments are
//! shared `Rc<RefCell<…>>` [`Scope`]s:
//!
//! - **Parse** a source to a `ModuleRecord`: its import requests, its local /
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

use super::{ExecError, Interp, N_REFERENCE_ERROR, N_SYNTAX_ERROR, NanBox, Thrown};
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
    /// Body is currently running (set before evaluation to break cycles). A
    /// deferred-namespace access of a module in this state is a TypeError
    /// (import-defer: the module's bindings are not yet initialized).
    Evaluating,
    /// Body has finished running (successfully or with a captured `eval_error`).
    Evaluated,
}

/// One `import` request of a module: the resolved key of the dependency plus the
/// bindings it introduces.
struct ImportEntry {
    /// The resolved key of the imported module.
    key: String,
    /// The specifier as written (for diagnostics).
    specifiers: Vec<ImportBind>,
    /// `import defer * as ns from …` — this dependency is loaded and linked but
    /// not eagerly evaluated; its namespace triggers evaluation on first access.
    deferred: bool,
    /// The `type` import attribute (`with { type: "json" }`), selecting a
    /// non-JavaScript module kind for the dependency.
    type_attr: Option<String>,
}

/// The kind of a loaded module, selected by the `type` import attribute
/// (import-attributes proposal). A JavaScript module has a parsed AST body; a
/// JSON / text module is *synthetic* — a single frozen-shaped `default` export
/// built from the referenced file's contents, with no named exports.
enum ModuleKind {
    /// An ordinary ECMAScript module (the AST body drives evaluation).
    JavaScript,
    /// `with { type: "json" }` — the file is parsed with `JSON.parse` and its
    /// value becomes the module's `default` export.
    Json,
    /// `with { type: "text" }` — the file's raw text becomes the `default`
    /// export (the import-text proposal).
    Text,
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
        type_attr: Option<String>,
    },
    /// `export * from "m"` — re-expose every named export of `m`.
    Star {
        key: String,
        type_attr: Option<String>,
    },
    /// `export * as ns from "m"` — expose `m`'s namespace under `ns`.
    StarAs {
        key: String,
        exported: String,
        type_attr: Option<String>,
    },
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
    /// This module's `[[RequestedModules]]` in **source order** — every `import`
    /// / `export … from` dependency key, paired with whether the request is a
    /// deferred import. Dependencies are evaluated in this order (interleaving
    /// imports and re-exports as they appear in the source), which the separate
    /// `imports`/`reexports` lists would not preserve.
    requested: Vec<(String, bool)>,
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
    /// The lazily-built *deferred* namespace exotic object (import-defer): a
    /// distinct object from `namespace`, with `@@toStringTag` "Deferred Module".
    deferred_namespace: Option<NanBox>,
    /// `import.meta` for this module.
    meta: Option<NanBox>,
    status: Status,
    /// A captured evaluation error (so a re-entered, already-failed module
    /// rethrows the same value rather than re-running).
    eval_error: Option<NanBox>,
    /// JavaScript / JSON / text (import-attributes). A synthetic (JSON/text)
    /// module has an empty `program` and a single `default` export.
    kind: ModuleKind,
    /// For a synthetic (JSON/text) module: the already-built `default` export
    /// value (JSON parsed / text string), materialised at load time so a JSON
    /// parse failure surfaces in the load/resolution phase.
    default_value: Option<NanBox>,
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
        self.load_module(entry_key, host, None)?;
        self.link_module(entry_key)?;
        self.evaluate_entry(entry_key)
    }

    /// Public wrapper over `Self::load_module` (the loader is private; the
    /// phased entry points need to call it from the free functions).
    pub fn load_module_pub(
        &mut self,
        entry_key: &str,
        host: &dyn ModuleHost,
    ) -> Result<(), ExecError> {
        self.load_module(entry_key, host, None)
    }

    /// Public wrapper over `Self::link_module`.
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
                let (name, message) =
                    super::error_name_message(self, thrown).unwrap_or_else(|| {
                        // A throw lacking a `name` property (e.g. Test262Error, which
                        // carries only `message`): surface its `message` so the failure
                        // is diagnosable rather than the opaque `[object Object]`.
                        if let Some(raw) = thrown.as_handle()
                            && let Some(m) = self
                                .realm()
                                .get_property(crate::heap::Handle::from_raw(raw), "message")
                        {
                            let s = self.realm().to_display_string(m);
                            if !s.is_empty() {
                                return (String::from("Test262Error"), s);
                            }
                        }
                        (self.display(thrown), String::new())
                    });
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
    /// `type_attr` is the `type` import attribute of the request that reached
    /// `key` (`None` for the entry module), selecting a JSON / text synthetic
    /// module instead of a JavaScript one.
    fn load_module(
        &mut self,
        key: &str,
        host: &dyn ModuleHost,
        type_attr: Option<&str>,
    ) -> Result<(), ExecError> {
        if self.modules.records.contains_key(key) {
            return Ok(());
        }
        // The map key of a JSON / text module is suffixed with its type (so the
        // same file imported both as JavaScript and as text/JSON is two distinct
        // modules, per the spec's `(specifier, attributes)` module map key). The
        // host loads the underlying file, so strip the suffix first.
        let source = host
            .load(module_load_path(key))
            .map_err(|e| self.syntax_error(&e))?;
        // A `type` attribute selects a synthetic (JSON / text) module; otherwise
        // the file is an ordinary JavaScript module. A JSON parse error here is a
        // load/resolution-phase failure (a SyntaxError), matching the tests'
        // `negative: { phase: resolution }` expectation.
        let record = match type_attr {
            Some("json") => self.build_json_module(key, &source)?,
            Some("text") => self.build_text_module(key, &source),
            _ => self.parse_module(key, &source, host)?,
        };
        // Collect dependency keys (with their own `type` attributes) before
        // recursing — the borrow of `record` ends here.
        let deps: Vec<(String, Option<String>)> = record
            .imports
            .iter()
            .map(|i| (i.key.clone(), i.type_attr.clone()))
            .chain(record.reexports.iter().map(reexport_key_type))
            .collect();
        self.modules.records.insert(key.to_string(), record);
        for (dep, dep_type) in deps {
            self.load_module(&dep, host, dep_type.as_deref())?;
        }
        if let Some(r) = self.modules.records.get_mut(key) {
            r.status = Status::Loaded;
        }
        Ok(())
    }

    /// Builds a synthetic **JSON module** record for `key`: `JSON.parse(source)`
    /// becomes the sole `default` export. An empty AST body drives (no) link /
    /// evaluation. A malformed source propagates as a SyntaxError.
    fn build_json_module(&mut self, key: &str, source: &str) -> Result<ModuleRecord, ExecError> {
        let value = self.parse_json_source(source)?;
        Ok(self.synthetic_module(key, ModuleKind::Json, value))
    }

    /// Builds a synthetic **text module** record for `key`: the raw source text
    /// becomes the `default` export (import-text proposal).
    fn build_text_module(&mut self, key: &str, source: &str) -> ModuleRecord {
        let value = self.new_str(source);
        self.synthetic_module(key, ModuleKind::Text, value)
    }

    /// Assembles a synthetic module record (JSON / text): an empty program, a
    /// single local `default` export, and the pre-built export `value`.
    fn synthetic_module(&mut self, key: &str, kind: ModuleKind, value: NanBox) -> ModuleRecord {
        // A leaked empty program so the record's `&'static Program` invariant
        // holds (link/instantiate iterate an empty body — no-ops).
        let empty = Program {
            body: Vec::new(),
            source_type: crate::ast::SourceType::Module,
            span: crate::common::Span::new(0, 0),
            source: alloc::boxed::Box::from(""),
        };
        let program: &'static Program = alloc::boxed::Box::leak(alloc::boxed::Box::new(empty));
        let mut local_exports = BTreeMap::new();
        local_exports.insert("default".to_string(), DEFAULT_LOCAL.to_string());
        ModuleRecord {
            key: key.to_string(),
            program,
            imports: Vec::new(),
            reexports: Vec::new(),
            requested: Vec::new(),
            local_exports,
            scope: Scope::root(),
            import_aliases: Rc::new(BTreeMap::new()),
            namespace: None,
            deferred_namespace: None,
            meta: None,
            status: Status::New,
            eval_error: None,
            kind,
            default_value: Some(value),
        }
    }

    /// Full `JSON.parse` of a whole source string (value plus a trailing-content
    /// check), used to materialise a JSON module's default export.
    fn parse_json_source(&mut self, source: &str) -> Result<NanBox, ExecError> {
        let chars: Vec<char> = source.chars().collect();
        let mut pos = 0;
        let value = self.json_parse(&chars, &mut pos, 0)?;
        super::skip_ws(&chars, &mut pos);
        if pos != chars.len() {
            return Err(self.json_error("Unexpected token in JSON"));
        }
        Ok(value)
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
        let mut requested: Vec<(String, bool)> = Vec::new();
        let mut local_exports: BTreeMap<String, String> = BTreeMap::new();

        let resolve = |spec: &str, this: &mut Self| -> Result<String, ExecError> {
            host.resolve(spec, Some(key))
                .map_err(|e| this.syntax_error(&e))
        };

        for stmt in &program.body {
            match stmt {
                Stmt::Import(decl) => {
                    let dep = resolve(&decl.source, self)?;
                    let type_attr = attr_type(&decl.attributes);
                    let dep = module_map_key(&dep, type_attr.as_deref());
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
                    requested.push((dep.clone(), decl.deferred));
                    imports.push(ImportEntry {
                        key: dep,
                        specifiers: binds,
                        deferred: decl.deferred,
                        type_attr,
                    });
                }
                Stmt::Export(ExportDecl::Named {
                    specifiers,
                    source: Some(src),
                    attributes,
                    ..
                }) => {
                    let dep = resolve(src, self)?;
                    let type_attr = attr_type(attributes);
                    let dep = module_map_key(&dep, type_attr.as_deref());
                    requested.push((dep.clone(), false));
                    // An *empty* named re-export (`export {} from "mod"`) still
                    // contributes "mod" to [[RequestedModules]] — the module is
                    // loaded, parsed, and evaluated (so an early error in it
                    // surfaces) but re-exports nothing. Model it as a bare
                    // load-only import (like `import "mod"`).
                    if specifiers.is_empty() {
                        imports.push(ImportEntry {
                            key: dep.clone(),
                            specifiers: Vec::new(),
                            deferred: false,
                            type_attr: type_attr.clone(),
                        });
                    }
                    for sp in specifiers {
                        reexports.push(ReExport::Named {
                            key: dep.clone(),
                            local: export_name(&sp.local),
                            exported: export_name(&sp.exported),
                            type_attr: type_attr.clone(),
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
                    attributes,
                    ..
                }) => {
                    let dep = resolve(src, self)?;
                    let type_attr = attr_type(attributes);
                    let dep = module_map_key(&dep, type_attr.as_deref());
                    requested.push((dep.clone(), false));
                    match exported {
                        Some(name) => reexports.push(ReExport::StarAs {
                            key: dep,
                            exported: export_name(name),
                            type_attr,
                        }),
                        None => reexports.push(ReExport::Star {
                            key: dep,
                            type_attr,
                        }),
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
            requested,
            local_exports,
            scope: Scope::root(),
            import_aliases: Rc::new(BTreeMap::new()),
            namespace: None,
            deferred_namespace: None,
            meta: None,
            status: Status::New,
            eval_error: None,
            kind: ModuleKind::JavaScript,
            default_value: None,
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
            // Link every requested module (deferred included) in source order.
            r.requested.iter().map(|(k, _)| k.clone()).collect()
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
                        deferred: i.deferred,
                    }
                })
                .collect()
        };

        let mut aliases: BTreeMap<String, (Scope, String)> = BTreeMap::new();
        for DepBinds {
            dep: dep_key,
            binds,
            deferred,
        } in &imports
        {
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
                        // constant binding, not a live slot). `import defer * as ns`
                        // binds a *deferred* namespace that evaluates `dep_key` on
                        // first access.
                        let ns = if *deferred {
                            self.deferred_namespace_object(dep_key)?
                        } else {
                            self.namespace_object(dep_key)?
                        };
                        let r = &self.modules.records[key];
                        r.scope.declare_const(local, ns);
                    }
                }
            }
        }
        // Validate this module's own re-exports resolve (link-time SyntaxError on
        // a missing/ambiguous re-exported name).
        let named_reexports: Vec<(String, String)> = {
            let r = &self.modules.records[key];
            r.reexports
                .iter()
                .filter_map(|re| match re {
                    ReExport::Named { key, local, .. } => Some((key.clone(), local.clone())),
                    _ => None,
                })
                .collect()
        };
        for (dep, local) in &named_reexports {
            self.resolve_export(dep, local, &mut BTreeSet::new())?;
        }

        let aliases = Rc::new(aliases);
        if let Some(r) = self.modules.records.get_mut(key) {
            r.import_aliases = aliases.clone();
            // Tag the module's top-level scope with its imports so a function
            // defined here restores the right aliases when it runs (even when
            // called from another module) — see `Scope::module_imports`.
            r.scope.set_module_imports(aliases);
        }
        // Instantiate this module's top-level function declarations into its
        // scope *now*, at link time (the spec's InitializeEnvironment step). A
        // function is thus callable across an import cycle before the defining
        // module's body has run — e.g. `b` may call `a`'s exported function even
        // when `a` is mid-evaluation.
        self.instantiate_module_functions(key)?;
        // A synthetic (JSON / text) module has no body: bind its pre-built
        // `default` export value into its scope now, so a `default` import or a
        // namespace snapshot reads it (evaluation is a no-op for these).
        if let Some(r) = self.modules.records.get(key)
            && !matches!(r.kind, ModuleKind::JavaScript)
            && let Some(value) = r.default_value
        {
            let scope = r.scope.clone();
            scope.declare_const(DEFAULT_LOCAL, value);
        }
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
                        self.set_fn_source(value, func.span);
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
                        self.set_fn_source(value, func.span);
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
        // 1. A local export. But if the exported local name is itself an *imported*
        //    binding (`import { x } from m; export { x }` — or a default/renamed
        //    import re-exported), it has no real slot in this module's scope; follow
        //    the import to its source instead. (A namespace import re-exported IS a
        //    real const slot, so it falls through to the scope.)
        if let Some(local) = record.local_exports.get(name).cloned() {
            let via_import = record.imports.iter().find_map(|imp| {
                imp.specifiers.iter().find_map(|spec| match spec {
                    ImportBind::Named { imported, local: l } if *l == local => {
                        Some((imp.key.clone(), imported.clone()))
                    }
                    ImportBind::Default(l) if *l == local => {
                        Some((imp.key.clone(), "default".to_string()))
                    }
                    _ => None,
                })
            });
            let own_scope = record.scope.clone();
            if let Some((dep, imported)) = via_import {
                return self.resolve_export(&dep, &imported, seen);
            }
            return Ok((own_scope, local));
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
                    ..
                } if exported == name => Some((key.clone(), local.clone())),
                ReExport::StarAs { key, exported, .. } if exported == name => {
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
                ReExport::Star { key, .. } => Some(key.clone()),
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
        Err(self.syntax_error(&alloc::format!("module {key} has no export named '{name}'")))
    }

    // --- Evaluate -------------------------------------------------------

    /// Evaluates `key` and (post-order) its dependencies, each exactly once.
    /// A module is marked `Evaluated` *before* its body runs so an import cycle
    /// does not re-enter it.
    fn evaluate_module(&mut self, key: &str) -> Result<(), ExecError> {
        match self.modules.records.get(key).map(|r| r.status) {
            Some(Status::Evaluating | Status::Evaluated) => {
                // Re-entrant (cycle) or already done; rethrow a prior failure.
                // (A still-`Evaluating` module is on the stack — a normal import
                // cycle, which proceeds; the import-defer "accessed while
                // evaluating" TypeError is enforced in `force_deferred_namespace`,
                // not here.)
                if let Some(err) = self.modules.records.get(key).and_then(|r| r.eval_error) {
                    return Err(ExecError::Throw(err));
                }
                return Ok(());
            }
            Some(Status::Linked) => {}
            _ => return Err(self.syntax_error(&alloc::format!("module {key} not linked"))),
        }
        // Evaluate dependencies first (post-order), in source order
        // (`[[RequestedModules]]`). A *deferred* import (`import defer`) is not
        // itself evaluated here — that happens lazily on first namespace access —
        // but its **asynchronous** transitive dependencies still are, because a
        // module with top-level await cannot be evaluated synchronously later when
        // the namespace is touched. That is
        // `GatherAsynchronousTransitiveDependencies`.
        let requested: Vec<(String, bool)> = self.modules.records[key].requested.clone();
        let mut deps: Vec<String> = Vec::new();
        for (dep_key, deferred) in &requested {
            if *deferred {
                let mut seen = alloc::collections::BTreeSet::new();
                let mut gathered = Vec::new();
                self.gather_async_transitive_deps(dep_key, &mut seen, &mut gathered);
                for m in gathered {
                    if !deps.contains(&m) {
                        deps.push(m);
                    }
                }
            } else {
                deps.push(dep_key.clone());
            }
        }
        // Mark Evaluating up front to break cycles (and so a deferred-namespace
        // access of this in-flight module throws a TypeError).
        if let Some(r) = self.modules.records.get_mut(key) {
            r.status = Status::Evaluating;
        }
        for dep in &deps {
            self.evaluate_module(dep)?;
        }
        let result = self.run_module_body(key);
        if let Some(r) = self.modules.records.get_mut(key) {
            r.status = Status::Evaluated;
            if let Err(ExecError::Throw(v)) = &result {
                r.eval_error = Some(*v);
            }
        }
        result
    }

    /// Runs one module's top-level statements in its own environment, with its
    /// import aliases active and `import.meta` set up. Modules are always strict.
    ///
    /// A module whose body contains a **top-level `await`** (or `for await`) is a
    /// spec async module: its body is driven as a suspendable coroutine (the same
    /// engine that runs async functions), so an `await` suspends the module and
    /// yields to the microtask queue rather than draining the event loop inline —
    /// giving the specified tick interleaving. Such a body is run to settlement
    /// here (blocking on the event loop until the module's evaluation promise
    /// resolves/rejects), preserving the synchronous-completion contract the
    /// depth-first `evaluate_module` relies on for a dependency; the entry's
    /// trailing reactions (`$DONE`, further `.then`s) drain in `evaluate_entry`.
    /// `GatherAsynchronousTransitiveDependencies(module, seen)` — the modules
    /// reachable from `key` that must be evaluated *eagerly* even though `key`
    /// itself sits behind a deferred import, because they have top-level await.
    ///
    /// A deferred module is normally evaluated on first namespace access, which is
    /// a synchronous operation; a module with TLA cannot be evaluated
    /// synchronously, so the spec hoists those out of the deferral and evaluates
    /// them with the importing module. Walking stops at the first module with TLA
    /// on each path — evaluating it will evaluate its own dependencies anyway —
    /// and at any module already evaluating or evaluated.
    fn gather_async_transitive_deps(
        &self,
        key: &str,
        seen: &mut alloc::collections::BTreeSet<String>,
        out: &mut Vec<String>,
    ) {
        if !seen.insert(String::from(key)) {
            return;
        }
        let Some(r) = self.modules.records.get(key) else {
            return;
        };
        if matches!(r.status, Status::Evaluating | Status::Evaluated) {
            return;
        }
        if matches!(r.kind, ModuleKind::JavaScript) && module_body_has_await(&r.program.body) {
            if !out.iter().any(|m| m == key) {
                out.push(String::from(key));
            }
            return;
        }
        let requested: Vec<String> = r.requested.iter().map(|(k, _)| k.clone()).collect();
        for dep in &requested {
            self.gather_async_transitive_deps(dep, seen, out);
        }
    }

    fn run_module_body(&mut self, key: &str) -> Result<(), ExecError> {
        let is_tla = {
            let r = &self.modules.records[key];
            matches!(r.kind, ModuleKind::JavaScript) && module_body_has_await(&r.program.body)
        };
        if is_tla {
            return self.run_module_body_async(key);
        }
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

    /// Runs a **top-level-await** module body as a suspendable async coroutine
    /// (the async-function engine), draining the event loop until the module's
    /// evaluation promise settles. An `await` therefore suspends the whole module
    /// body and yields to pending microtasks — matching the spec's async-module
    /// tick interleaving — instead of draining the event loop inline at the
    /// `await`. A rejection of the evaluation promise becomes this module's
    /// evaluation error (propagated / cached by `evaluate_module`).
    fn run_module_body_async(&mut self, key: &str) -> Result<(), ExecError> {
        use crate::cell::PromiseStatus::{Pending, Rejected};
        let (scope, program) = {
            let r = &self.modules.records[key];
            (r.scope.clone(), r.program)
        };
        // `import.meta` (built lazily, once) — materialised before the coroutine
        // runs so an early `import.meta` read on the first burst sees it.
        let _ = self.module_meta(key);
        // Pre-declare the `*default*` lexical binding in its Temporal Dead Zone for
        // an anonymous `export default <expr>` (as `exec_module_stmts` does), so a
        // namespace access of `default` before the statement runs is a
        // ReferenceError. A *named* default (`export default function f`) hoists
        // ordinarily and is excluded.
        for stmt in &program.body {
            if let Stmt::Export(ExportDecl::Default { declaration, .. }) = stmt
                && !matches!(
                    &**declaration,
                    Stmt::Function(crate::ast::Function { id: Some(_), .. })
                )
                && !scope.has_local(DEFAULT_LOCAL)
            {
                scope.declare(DEFAULT_LOCAL, NanBox::tdz());
            }
        }
        // Capture a module-appropriate execution context into the coroutine frame:
        // `this` is undefined, no home object / new.target, always strict. The
        // per-resume module ambient state (import aliases, `import.meta`, active
        // key, var environment) is installed by `async_step` from the controller's
        // module key, so it is *not* part of the frame.
        let saved_this = core::mem::replace(&mut self.this_val, NanBox::undefined());
        let saved_target = core::mem::replace(&mut self.new_target, NanBox::undefined());
        let saved_home = self.current_home.take();
        let saved_home_static = core::mem::replace(&mut self.current_home_static, false);
        let saved_home_obj = self.current_home_object.take();
        let saved_strict = core::mem::replace(&mut self.strict, true);
        let (id, promise, controller) =
            self.make_async_frame(super::Body::Block(&program.body), scope);
        self.this_val = saved_this;
        self.new_target = saved_target;
        self.current_home = saved_home;
        self.current_home_static = saved_home_static;
        self.current_home_object = saved_home_obj;
        self.strict = saved_strict;
        // Tag the controller as a module coroutine so each resume re-establishes
        // the module's ambient state.
        let key_val = self.new_str(key);
        self.realm
            .set_hidden_property(controller, super::MODULE_KEY, key_val);
        // Drive the first synchronous burst (body up to the first top-level
        // `await`, or to completion for a body whose only await is unreachable).
        self.async_step(
            id,
            controller,
            super::generator::Resumption::Next(NanBox::undefined()),
        );
        // Block on the event loop until the module's evaluation promise settles, so
        // a *dependency* module completes before its importer's body runs (the DFS
        // contract). The entry module's trailing reactions drain in
        // `evaluate_entry`'s own event-loop pass.
        while self
            .realm
            .promise_state(promise)
            .is_some_and(|s| s.borrow().status == Pending)
            && (!self.microtasks.is_empty() || !self.macrotasks.is_empty())
        {
            if self.microtasks.is_empty() {
                self.run_one_macrotask()?;
            } else {
                self.run_one_microtask()?;
            }
        }
        // A rejected evaluation promise is this module's evaluation error.
        if let Some(state) = self.realm.promise_state(promise) {
            let st = state.borrow();
            if st.status == Rejected {
                return Err(ExecError::Throw(st.value));
            }
        }
        Ok(())
    }

    /// If `controller` is an async coroutine driving a module body (it carries a
    /// [`MODULE_KEY`] slot), install that module's ambient evaluation state (import
    /// aliases, `import.meta`, active-module key, and the module top-level variable
    /// environment), returning the prior state to restore afterwards. Returns
    /// `None` for an ordinary async-function controller (no module context).
    pub(crate) fn enter_module_context_for_controller(
        &mut self,
        controller: crate::heap::Handle,
    ) -> Option<ModuleContextSave> {
        let key = self
            .realm
            .get_property(controller, super::MODULE_KEY)
            .and_then(|v| v.as_handle())
            .map(crate::heap::Handle::from_raw)
            .and_then(|h| self.realm.string_value(h))?;
        let (var_scope, aliases) = {
            let r = self.modules.records.get(&key)?;
            (r.scope.clone(), r.import_aliases.clone())
        };
        let meta = self.module_meta(&key);
        Some(ModuleContextSave {
            var_scope: core::mem::replace(&mut self.var_scope, var_scope),
            module_imports: core::mem::replace(&mut self.module_imports, aliases),
            import_meta: self.import_meta.replace(meta),
            active_module_key: self.active_module_key.replace(key),
            annexb_block_fns: core::mem::take(&mut self.annexb_block_fns),
        })
    }

    /// Restores the ambient module-evaluation state saved by
    /// [`Self::enter_module_context_for_controller`].
    pub(crate) fn exit_module_context(&mut self, save: ModuleContextSave) {
        self.var_scope = save.var_scope;
        self.module_imports = save.module_imports;
        self.import_meta = save.import_meta;
        self.active_module_key = save.active_module_key;
        self.annexb_block_fns = save.annexb_block_fns;
    }

    /// Hoists then executes a module body's statements, treating `import` as a
    /// no-op (bindings were wired at link time) and `export` by evaluating its
    /// inner declaration / default expression.
    fn exec_module_stmts(&mut self, stmts: &'a [Stmt]) -> Result<(), ExecError> {
        // `var` + function-declaration hoisting at the module variable
        // environment boundary (lexical `let`/`const`/`class` bind on execution).
        self.hoist_with(stmts, true)?;
        // `export default <AssignmentExpression>` (and an anonymous default
        // function/class) binds the synthetic `*default*` lexical name, which is
        // in its Temporal Dead Zone until the `export default` statement runs — so
        // a namespace `[[Get]]`/`[[GetOwnProperty]]` of `default` before then
        // throws a ReferenceError. A *named* `export default function f` hoists
        // like an ordinary function declaration (never TDZ), so it is excluded.
        for stmt in stmts {
            if let Stmt::Export(ExportDecl::Default { declaration, .. }) = stmt
                && !matches!(
                    &**declaration,
                    Stmt::Function(crate::ast::Function { id: Some(_), .. })
                )
                && !self.current.has_local(DEFAULT_LOCAL)
            {
                self.current.declare(DEFAULT_LOCAL, NanBox::tdz());
            }
        }
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
    pub(crate) fn exec_export(&mut self, decl: &'a ExportDecl) -> Result<(), ExecError> {
        match decl {
            // Re-exports and bare `export { … }` bind nothing locally.
            ExportDecl::All { .. } => Ok(()),
            ExportDecl::Named {
                source: Some(_), ..
            } => Ok(()),
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
                        self.set_fn_source(value, func.span);
                        self.current.declare_const(DEFAULT_LOCAL, value);
                        Ok(())
                    }
                    Stmt::Class(class) => {
                        // NamedEvaluation gives the anonymous class the name "default"
                        // *before* its static initializers run, so a
                        // `static f = this.name` observes "default" (not "").
                        self.pending_class_name = Some("default");
                        let value = self.make_class(class);
                        self.pending_class_name = None;
                        let value = value?;
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
        self.populate_namespace(obj, key, false)?;
        Ok(ns)
    }

    /// Builds (once, cached) the **Deferred Module Namespace** exotic object for
    /// `key` (import-defer proposal). Structurally identical to the ordinary
    /// namespace (live export bindings) but a distinct object with `@@toStringTag`
    /// "Deferred Module"; until `key` is evaluated the handle is registered in
    /// `deferred_namespaces` so the first export access triggers evaluation.
    fn deferred_namespace_object(&mut self, key: &str) -> Result<NanBox, ExecError> {
        if let Some(ns) = self
            .modules
            .records
            .get(key)
            .and_then(|r| r.deferred_namespace)
        {
            return Ok(ns);
        }
        let obj = self.realm.new_object_with_proto(None);
        let ns = NanBox::handle(obj.to_raw());
        if let Some(r) = self.modules.records.get_mut(key) {
            r.deferred_namespace = Some(ns);
        }
        // Only arm the lazy-evaluation trigger when the module has not already
        // run (a defer of an already-evaluated module is just a namespace view).
        let already = matches!(
            self.modules.records.get(key).map(|r| r.status),
            Some(Status::Evaluated)
        );
        #[cfg(all(feature = "module", feature = "std"))]
        if !already {
            self.deferred_namespaces
                .insert(obj.to_raw(), key.to_string());
        }
        self.populate_namespace(obj, key, true)?;
        Ok(ns)
    }

    /// Shared body of [`Self::namespace_object`] /
    /// [`Self::deferred_namespace_object`]: resolves `key`'s exports into live
    /// data properties on the already-allocated, already-cached `obj`, sets
    /// `@@toStringTag` ("Module" or "Deferred Module"), and freezes the shape.
    fn populate_namespace(
        &mut self,
        obj: crate::heap::Handle,
        key: &str,
        deferred: bool,
    ) -> Result<(), ExecError> {
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
        // `@@toStringTag` = "Module" (or "Deferred Module" for a deferred
        // namespace), non-enumerable, non-writable, non-configurable.
        let tag_sym = self.well_known_symbol("toStringTag");
        let tag_key = self.member_key(tag_sym);
        let module_str = self.new_str(if deferred {
            "Deferred Module"
        } else {
            "Module"
        });
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
        // A module namespace is **sealed** (non-extensible + every property
        // non-configurable) but not frozen (its bindings stay writable). Mark the
        // seal flag so `Object.isSealed(ns)` computes `true` (not just
        // `preventExtensions`, which would leave the flag unset).
        self.realm.seal_object(obj);
        Ok(())
    }

    /// If `handle` is a module namespace exotic object and `key` names a string
    /// export, synchronise its stored data property with the *live* binding value
    /// (§28.3 — namespace properties are live) and, when that binding is still
    /// uninitialized (Temporal Dead Zone), return the `ReferenceError` its
    /// `[[GetOwnProperty]]` / `[[Get]]` must throw (per §10.4.6, which routes both
    /// through `GetBindingValue` with Strict = true). Symbol keys, non-export
    /// keys, and non-namespace objects are no-ops. This is the shared guard for
    /// the `[[GetOwnProperty]]`-based operations (`getOwnPropertyDescriptor`,
    /// `hasOwnProperty`, `Object.hasOwn`, `propertyIsEnumerable`).
    #[cfg(all(feature = "module", feature = "std"))]
    pub(crate) fn namespace_binding_tdz(
        &mut self,
        handle: crate::heap::Handle,
        key: &str,
    ) -> Result<(), ExecError> {
        if let Some((scope, local)) = self
            .module_namespaces
            .get(&handle.to_raw())
            .and_then(|m| m.get(key))
            .map(|(s, l)| (s.clone(), l.clone()))
        {
            let value = scope.get(&local).unwrap_or_else(NanBox::undefined);
            if value.is_tdz() {
                let msg = self.new_str(&alloc::format!(
                    "Cannot access '{key}' before initialization"
                ));
                return Err(ExecError::Throw(
                    self.make_error(N_REFERENCE_ERROR, Some(msg)),
                ));
            }
            // Refresh the snapshot so a `getOwnPropertyDescriptor` reports the
            // live value (the property is non-configurable but writable).
            self.realm.set_property(handle, key, value);
        }
        Ok(())
    }

    /// Whole-object enumeration guard: if `handle` is a module namespace with
    /// *any* export binding in its Temporal Dead Zone, return the `ReferenceError`
    /// that iterating its own keys (`Object.keys`, `for..in`, `Object.values` …)
    /// must throw — each such operation calls `[[GetOwnProperty]]` per key, which
    /// throws on the first uninitialized binding. Bindings are visited in the
    /// namespace's sorted-key order (the `BTreeMap` iteration order).
    #[cfg(all(feature = "module", feature = "std"))]
    pub(crate) fn namespace_enumeration_tdz(
        &mut self,
        handle: crate::heap::Handle,
    ) -> Result<(), ExecError> {
        let first_tdz = self.module_namespaces.get(&handle.to_raw()).and_then(|m| {
            m.iter()
                .find(|(_, (scope, local))| scope.get(local).is_some_and(|v| v.is_tdz()))
                .map(|(name, _)| name.clone())
        });
        if let Some(name) = first_tdz {
            let msg = self.new_str(&alloc::format!(
                "Cannot access '{name}' before initialization"
            ));
            return Err(ExecError::Throw(
                self.make_error(N_REFERENCE_ERROR, Some(msg)),
            ));
        }
        Ok(())
    }

    /// The module namespace exotic `[[DefineOwnProperty]]` (§10.4.6.11). Returns
    /// `Ok(None)` when `handle` is not a namespace or `key` is a Symbol (both
    /// fall through to `OrdinaryDefineOwnProperty` — so `@@toStringTag` and new
    /// symbols behave ordinarily), otherwise `Ok(Some(result))`:
    /// - a non-export String key → `false`;
    /// - a request that would change the binding (configurable, non-enumerable,
    ///   accessor, non-writable, or a differing value) → `false`;
    /// - an inert / compatible redefinition → `true`.
    ///
    /// A TDZ export binding makes the internal `[[GetOwnProperty]]` throw.
    #[cfg(all(feature = "module", feature = "std"))]
    pub(crate) fn namespace_define_own_property(
        &mut self,
        handle: crate::heap::Handle,
        key: &str,
        desc: crate::heap::Handle,
    ) -> Result<Option<bool>, ExecError> {
        if key.starts_with("\u{0}sym:") || !self.module_namespaces.contains_key(&handle.to_raw()) {
            return Ok(None);
        }
        // Step 2: `current = ? [[GetOwnProperty]](P)` — refreshes the live value
        // and throws a ReferenceError for a TDZ binding.
        self.namespace_binding_tdz(handle, key)?;
        // Step 3: a String key that is not an export → false.
        let is_export = self
            .module_namespaces
            .get(&handle.to_raw())
            .is_some_and(|m| m.contains_key(key));
        if !is_export {
            return Ok(Some(false));
        }
        // Steps 4-7: any attribute that would alter the fixed shape → false.
        let present_true = |i: &Self, k: &str| {
            i.realm.has_own(desc, k)
                && i.realm
                    .get_property(desc, k)
                    .is_some_and(|v| i.realm.truthy(v))
        };
        let present_false = |i: &Self, k: &str| {
            i.realm.has_own(desc, k)
                && !i
                    .realm
                    .get_property(desc, k)
                    .is_some_and(|v| i.realm.truthy(v))
        };
        if present_true(self, "configurable")
            || present_false(self, "enumerable")
            || self.realm.has_own(desc, "get")
            || self.realm.has_own(desc, "set")
            || present_false(self, "writable")
        {
            return Ok(Some(false));
        }
        // Step 8: a supplied [[Value]] must SameValue the current binding value.
        if self.realm.has_own(desc, "value") {
            let requested = self
                .realm
                .get_property(desc, "value")
                .unwrap_or_else(NanBox::undefined);
            let current = self
                .realm
                .get_property(handle, key)
                .unwrap_or_else(NanBox::undefined);
            return Ok(Some(self.realm.same_value(requested, current)));
        }
        // Step 9: an inert redefinition succeeds.
        Ok(Some(true))
    }

    /// import-defer lazy trigger for a keyed operation ([[Get]], [[GetOwnProperty]],
    /// [[HasProperty]], [[Delete]], [[DefineOwnProperty]]). If `handle` is an
    /// armed Deferred Module Namespace, evaluate its target module *now* — unless
    /// `name` is a Symbol key (the `\0sym:` sentinel) or the String `"then"` (the
    /// thenable guard, so `await import.defer(...)` does not force evaluation).
    /// An evaluation throw propagates (and is cached, so a re-access rethrows it).
    pub(crate) fn trigger_deferred_namespace(
        &mut self,
        handle: crate::heap::Handle,
        name: &str,
    ) -> Result<(), ExecError> {
        if name == "then" || name.starts_with("\u{0}sym:") {
            return Ok(());
        }
        self.force_deferred_namespace(handle)
    }

    /// import-defer trigger for a *chained* operation ([[Get]] / [[HasProperty]],
    /// including a `super` home object): walk `handle`'s prototype chain and
    /// evaluate the first Deferred Module Namespace reached. A closer object that
    /// owns `name` shadows it (the chain stops before the namespace, so no
    /// trigger). Symbol keys and `"then"` never trigger. Cheap no-op when no
    /// deferred namespace is armed (the common case).
    pub(crate) fn trigger_deferred_in_chain(
        &mut self,
        handle: crate::heap::Handle,
        name: &str,
    ) -> Result<(), ExecError> {
        if self.deferred_namespaces.is_empty() || name == "then" || name.starts_with("\u{0}sym:") {
            return Ok(());
        }
        let mut cur = Some(handle);
        let mut guard = 0usize;
        while let Some(h) = cur {
            if self.deferred_namespaces.contains_key(&h.to_raw()) {
                return self.force_deferred_namespace(h);
            }
            if self.realm.has_own(h, name) {
                return Ok(());
            }
            guard += 1;
            if guard > 100_000 {
                break;
            }
            cur = self.realm.object_proto(h);
        }
        Ok(())
    }

    /// import-defer trigger for a whole-object operation ([[OwnPropertyKeys]]),
    /// which always evaluates regardless of any key. A no-op unless `handle` is an
    /// armed Deferred Module Namespace.
    pub(crate) fn force_deferred_namespace(
        &mut self,
        handle: crate::heap::Handle,
    ) -> Result<(), ExecError> {
        let Some(dep) = self.deferred_namespaces.get(&handle.to_raw()).cloned() else {
            return Ok(());
        };
        // Accessing a deferred namespace whose synchronous evaluation would
        // require running a module that is *currently* on the evaluation stack
        // (a cycle reached through the deferred edge) is a TypeError — those
        // bindings are not yet initialized (import-defer). We must detect this
        // over the whole transitive (non-deferred) closure *before* evaluating
        // anything, so no side effects of the subgraph run.
        if self.deferred_closure_has_evaluating(&dep) {
            return Err(self.type_error(
                "Cannot access a deferred module namespace while the module is being evaluated",
            ));
        }
        self.evaluate_module(&dep)?;
        self.deferred_namespaces.remove(&handle.to_raw());
        // The deferred namespace's data properties were snapshotted at creation
        // time — before the module ran, so each held `undefined`. Now that the
        // module has evaluated, refresh them from their live bindings so a
        // *non-read* access (`getOwnPropertyDescriptor`, `ownKeys`) reports the
        // real values. (`read_member` refreshes on its own per-read; this covers
        // the trap paths that read the stored property directly.)
        if let Some(map) = self.module_namespaces.get(&handle.to_raw()) {
            let refreshed: Vec<(String, NanBox)> = map
                .iter()
                .map(|(name, (scope, local))| {
                    (
                        name.clone(),
                        scope.get(local).unwrap_or_else(NanBox::undefined),
                    )
                })
                .collect();
            for (name, value) in refreshed {
                self.realm.set_property(handle, &name, value);
            }
        }
        Ok(())
    }

    /// True if any module in `key`'s transitive *non-deferred* dependency
    /// closure (the set `evaluate_module` would synchronously run) is currently
    /// `Evaluating` — i.e. on the active evaluation stack. Used to reject a
    /// deferred-namespace force that would re-enter an in-flight module.
    fn deferred_closure_has_evaluating(&self, key: &str) -> bool {
        let mut stack = alloc::vec![key.to_string()];
        let mut seen = BTreeSet::new();
        while let Some(k) = stack.pop() {
            if !seen.insert(k.clone()) {
                continue;
            }
            let Some(r) = self.modules.records.get(&k) else {
                continue;
            };
            if r.status == Status::Evaluating {
                return true;
            }
            // Don't descend into an already-evaluated module: its subgraph ran
            // to completion and is no longer on the stack.
            if r.status == Status::Evaluated {
                continue;
            }
            for dep in r
                .imports
                .iter()
                .filter(|i| !i.deferred)
                .map(|i| i.key.clone())
                .chain(r.reexports.iter().map(reexport_key))
            {
                stack.push(dep);
            }
        }
        false
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
                ReExport::Star { key, .. } => star_deps.push(key.clone()),
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
        // Evaluate the specifier and the optional options argument *before* the
        // promise capability exists: a throw from either expression (or its
        // `GetValue`) propagates synchronously per `ImportCall` steps 2–4, and
        // both are evaluated in source order (`2nd-param-evaluation-sequence`).
        let spec = match arguments.first() {
            Some(crate::ast::Argument::Item(e)) => self.eval(e)?,
            _ => NanBox::undefined(),
        };
        let options = match arguments.get(1) {
            Some(crate::ast::Argument::Item(e)) => Some(self.eval(e)?),
            _ => None,
        };
        let promise = self.fresh_promise();
        let referrer = self.current_module_key();
        let host = FileModuleHost;
        let outcome: Result<NanBox, ExecError> = (|this: &mut Self| {
            // `ToString(specifier)` is part of the ImportCall steps, so a throwing
            // `toString`/`Symbol.toPrimitive`/`valueOf` *rejects the promise*
            // rather than propagating synchronously. Use the real ToString (not
            // the lossy display form) so a user `toString` override is honoured.
            let spec_str = this.coerce_to_string(spec)?;
            // Import attributes from the options `with` object (a non-object
            // options / `with`, or a non-string attribute value, rejects the
            // promise with a TypeError).
            let type_attr = this.import_call_attributes_type(options)?;
            let dep = host
                .resolve(&spec_str, referrer.as_deref())
                .map_err(|e| this.type_error(&e))?;
            let dep = module_map_key(&dep, type_attr.as_deref());
            this.load_module(&dep, &host, type_attr.as_deref())?;
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

    /// Evaluates `import.defer(specifier)` (import-defer proposal): loads and
    /// links the module but does *not* evaluate it, and returns a promise
    /// fulfilled with its **Deferred Module Namespace** (which evaluates lazily on
    /// first access — identical object to the static `import defer * as ns`).
    pub(crate) fn dynamic_import_deferred(
        &mut self,
        arguments: &'a [crate::ast::Argument],
    ) -> Result<NanBox, ExecError> {
        let promise = self.fresh_promise();
        let spec = match arguments.first() {
            Some(crate::ast::Argument::Item(e)) => self.eval(e)?,
            _ => NanBox::undefined(),
        };
        let referrer = self.current_module_key();
        let host = FileModuleHost;
        let outcome: Result<NanBox, ExecError> = (|this: &mut Self| {
            let spec_str = this.coerce_to_string(spec)?;
            let dep = host
                .resolve(&spec_str, referrer.as_deref())
                .map_err(|e| this.type_error(&e))?;
            this.load_module(&dep, &host, None)?;
            this.link_module(&dep)?;
            // Deliberately NOT evaluated — deferred until first namespace access —
            // except for the asynchronous transitive dependencies, which cannot be
            // evaluated synchronously when that access happens and so are hoisted
            // out of the deferral exactly as for a static `import defer`.
            let mut seen = alloc::collections::BTreeSet::new();
            let mut gathered = Vec::new();
            this.gather_async_transitive_deps(&dep, &mut seen, &mut gathered);
            for m in &gathered {
                this.evaluate_module(m)?;
            }
            this.deferred_namespace_object(&dep)
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

    /// `ShadowRealm.prototype.importValue` loading primitive: imports `specifier`
    /// **into** the ShadowRealm at `realm_idx` (its global scope + `globalThis` +
    /// intrinsics swapped in for the whole load / link / evaluate, so the module
    /// runs genuinely isolated in that realm), then returns the raw value of its
    /// `export_name` export. The specifier resolves relative to `referrer` (the
    /// importing module — a sibling fixture in the Test262 harness). A resolve /
    /// load (parse) / link / evaluate failure, or a missing export, is an
    /// `ExecError`; the caller (`shadow_realm_dispatch`) turns it into a
    /// caller-realm `TypeError` rejection per the ShadowRealm spec.
    pub(crate) fn shadow_realm_import(
        &mut self,
        realm_idx: usize,
        specifier: &str,
        referrer: Option<&str>,
        export_name: &str,
    ) -> Result<NanBox, ExecError> {
        let host = FileModuleHost;
        let dep = host
            .resolve(specifier, referrer)
            .map_err(|e| self.type_error(&e))?;
        let dep = module_map_key(&dep, None);

        // Swap the ShadowRealm's environment in for the duration (mirrors
        // `shadow_realm_run_program`), so the imported module's own scope roots at
        // *that* realm's global scope and its `[]`/`{}`/error intrinsics come from
        // it. Restored unconditionally afterward.
        let scope = self.created_realms[realm_idx].global_scope.clone();
        let global_this = self.created_realms[realm_idx].global_this;
        let intrinsics = self.created_realms[realm_idx].intrinsics;
        let saved_current = self.current.clone();
        let saved_global_scope = self.global_scope.clone();
        let saved_var_scope = self.var_scope.clone();
        let saved_global_this = self.global_this;
        let saved_this = self.this_val;
        let saved_new_target = self.new_target;
        let saved_strict = self.strict;
        let saved_realm = self.cur_realm;
        let saved_intrinsics = self.realm.intrinsics_snapshot();
        let child_intl = core::mem::take(&mut self.created_realms[realm_idx].intl_protos);
        self.current = scope.clone();
        self.global_scope = scope.clone();
        self.var_scope = scope;
        self.global_this = global_this;
        self.this_val = NanBox::undefined();
        self.new_target = NanBox::undefined();
        self.strict = true;
        self.cur_realm = Some(realm_idx);
        self.realm.restore_intrinsics(intrinsics);
        let saved_intl = self.realm.replace_intl_protos(child_intl);

        let outcome = (|this: &mut Self| -> Result<NanBox, ExecError> {
            // Load is the parse/resolution phase (a SyntaxError in the fixture
            // surfaces here); link wires imports; evaluate runs the body (a
            // top-level-await body blocks to settlement inside `run_module_body`).
            this.load_module(&dep, &host, None)?;
            this.link_module(&dep)?;
            this.evaluate_module(&dep)?;
            let ns = this.namespace_object(&dep)?;
            let ns_h = ns
                .as_handle()
                .map(crate::heap::Handle::from_raw)
                .ok_or_else(|| this.type_error("module namespace is not an object"))?;
            // The export must exist as an own property of the namespace, else a
            // TypeError (importValue rejects for a non-existent export name).
            let has = this
                .realm
                .own_property_names(ns_h)
                .unwrap_or_default()
                .iter()
                .any(|k| k == export_name);
            if !has {
                return Err(this.type_error(&alloc::format!(
                    "module has no export named '{export_name}'"
                )));
            }
            Ok(this
                .realm
                .get_property(ns_h, export_name)
                .unwrap_or_else(NanBox::undefined))
        })(self);

        self.created_realms[realm_idx].intl_protos = self.realm.replace_intl_protos(saved_intl);
        self.current = saved_current;
        self.global_scope = saved_global_scope;
        self.var_scope = saved_var_scope;
        self.global_this = saved_global_this;
        self.this_val = saved_this;
        self.new_target = saved_new_target;
        self.strict = saved_strict;
        self.cur_realm = saved_realm;
        self.realm.restore_intrinsics(saved_intrinsics);
        outcome
    }

    /// Processes a dynamic `import(specifier, options)` second argument per the
    /// `ImportCall` runtime semantics (import-attributes): validates `options`
    /// is an Object (or absent/undefined), reads its `with` attributes object,
    /// enumerates every own-enumerable string attribute (running getters /
    /// proxy traps, each value required to be a String), and returns the value
    /// of the `type` attribute if present. Any type violation (non-object
    /// options / `with`, non-string value) is a TypeError that rejects the
    /// promise; an abrupt getter / trap propagates.
    fn import_call_attributes_type(
        &mut self,
        options: Option<NanBox>,
    ) -> Result<Option<String>, ExecError> {
        let Some(options) = options else {
            return Ok(None);
        };
        if options.is_undefined() {
            return Ok(None);
        }
        if !self.is_object_value(options) {
            return Err(self.type_error("the import() options argument must be an object"));
        }
        let opt_h = crate::heap::Handle::from_raw(options.as_handle().unwrap());
        let with_val = self.read_member(opt_h, "with")?;
        if with_val.is_undefined() {
            return Ok(None);
        }
        if !self.is_object_value(with_val) {
            return Err(self.type_error("the `with` import attributes must be an object"));
        }
        let attrs_h = crate::heap::Handle::from_raw(with_val.as_handle().unwrap());
        let keys = self.enumerable_own_string_keys(attrs_h)?;
        let mut type_attr = None;
        for k in keys {
            let v = self.read_member(attrs_h, &k)?;
            let s = v
                .as_handle()
                .map(crate::heap::Handle::from_raw)
                .and_then(|h| self.realm.string_value(h));
            let Some(s) = s else {
                return Err(self.type_error("an import attribute value must be a string"));
            };
            if k == "type" {
                type_attr = Some(s);
            }
        }
        Ok(type_attr)
    }

    /// The own **enumerable** String-keyed property names of `handle`
    /// (`EnumerableOwnProperties`, key kind), routed through the proxy
    /// `ownKeys` / `getOwnPropertyDescriptor` protocol for a proxy.
    fn enumerable_own_string_keys(
        &mut self,
        handle: crate::heap::Handle,
    ) -> Result<Vec<String>, ExecError> {
        if self.realm.proxy_at(handle).is_some() {
            return Ok(self.proxy_own_enumerable_keys(handle)?.unwrap_or_default());
        }
        let mut out = Vec::new();
        for k in self.realm.own_property_names(handle).unwrap_or_default() {
            if self.realm.property_is_enumerable(handle, &k) {
                out.push(k);
            }
        }
        Ok(out)
    }

    /// The key of the module whose body is currently running, found by matching
    /// the active scope against each record's scope. Falls back to the script
    /// import base (so a script's `import()` resolves relative to its file).
    pub(crate) fn current_module_key(&self) -> Option<String> {
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

/// The ambient module-evaluation state saved while an async module coroutine
/// runs (restored after each `async_step`). See
/// [`Interp::enter_module_context_for_controller`].
pub(crate) struct ModuleContextSave {
    var_scope: Scope,
    module_imports: Rc<BTreeMap<String, (Scope, String)>>,
    import_meta: Option<NanBox>,
    active_module_key: Option<String>,
    annexb_block_fns: Vec<String>,
}

/// Whether a module body's top-level statements contain a reachable `await` (or
/// `for await`) — i.e. this is an async (top-level-await) module. Unlike the
/// async-function detector, this also descends into an `export`'s inner
/// declaration (`export const x = await f()`), which is otherwise opaque to the
/// statement walker.
fn module_body_has_await(stmts: &[Stmt]) -> bool {
    stmts.iter().any(|s| match s {
        Stmt::Export(ExportDecl::Decl { declaration, .. })
        | Stmt::Export(ExportDecl::Default { declaration, .. }) => {
            super::generator::stmt_has_await(declaration)
        }
        other => super::generator::stmt_has_await(other),
    })
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
    deferred: bool,
}

/// The resolved dependency key of a re-export.
fn reexport_key(re: &ReExport) -> String {
    match re {
        ReExport::Named { key, .. } | ReExport::Star { key, .. } | ReExport::StarAs { key, .. } => {
            key.clone()
        }
    }
}

/// The resolved dependency key of a re-export paired with its `type` import
/// attribute (so a JSON re-export target is loaded as a JSON module).
fn reexport_key_type(re: &ReExport) -> (String, Option<String>) {
    match re {
        ReExport::Named { key, type_attr, .. }
        | ReExport::Star { key, type_attr }
        | ReExport::StarAs { key, type_attr, .. } => (key.clone(), type_attr.clone()),
    }
}

/// The internal module-map key for a resolved specifier under a `type` import
/// attribute. A JSON / text module is keyed by `<path>\0type=<t>` so the same
/// file imported both as JavaScript and as JSON/text is two distinct module
/// records (the spec keys the module map by `(specifier, attributes)`). A
/// plain JavaScript import keeps the bare resolved path as its key.
fn module_map_key(resolved: &str, type_attr: Option<&str>) -> String {
    match type_attr {
        Some(t @ ("json" | "text")) => alloc::format!("{resolved}\u{0}type={t}"),
        _ => resolved.to_string(),
    }
}

/// The underlying file path of a (possibly type-suffixed) module map key — the
/// path the host actually loads.
fn module_load_path(key: &str) -> &str {
    match key.split_once('\u{0}') {
        Some((path, _)) => path,
        None => key,
    }
}

/// The value of the `type` import attribute (`with { type: "…" }`), if present.
fn attr_type(attrs: &[crate::ast::ImportAttribute]) -> Option<String> {
    attrs
        .iter()
        .find(|(k, _)| &**k == "type")
        .map(|(_, v)| v.to_string())
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
