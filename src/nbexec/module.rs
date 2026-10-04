//! The ECMAScript **module** subsystem: a module record / resolve+load /
//! link / evaluate pipeline in the [`Interp`], plus dynamic `import()` and
//! `import.meta`. Module *code* runs on the bytecode VM (`crate::nbvm`'s
//! `vm_module`); this is the abstract-operations machinery of ECMA-262 §16.2
//! around it, over shared `Rc<RefCell<…>>` [`Scope`] environments:
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

use super::{
    DYN_DEFER, DYN_KEY, DYN_PROMISE, DYN_TYPE, ExecError, Interp, N_ERROR_BASE, N_REFERENCE_ERROR,
    N_SYNTAX_ERROR, NanBox, SAFE_ALL_REMAINING, SAFE_ALL_TARGET, Thrown,
};
use crate::ast::{ExportDecl, ImportSpecifier, ModuleExportName, Program, Stmt};
use crate::env::Scope;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::rc::Rc;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

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

    /// Loads the raw bytes of a resolved key, for `import … with { type:
    /// "bytes" }` (the import-bytes proposal). The default reads the key as
    /// source text and returns its UTF-8 encoding, so a host that only serves
    /// text still works for text files; a host with real files should override
    /// it to serve binary content verbatim.
    ///
    /// # Errors
    /// Returns a human-readable message if the bytes cannot be read.
    fn load_bytes(&self, key: &str) -> Result<Vec<u8>, String> {
        self.load(key).map(String::into_bytes)
    }
}

/// A [`ModuleHost`] that resolves `import` specifiers as paths relative to the
/// referrer file and reads them from the filesystem. The entry module's key is
/// its absolute path; a relative specifier is joined onto the referrer's parent
/// directory and canonicalised so the same file is deduped under one key.
pub struct FileModuleHost;

/// The host-defined module specifier that denotes "a module which provides a
/// valid [Module Source]" (source-phase-imports). Test262's `INTERPRETING.md`
/// mandates that a host resolve the literal specifier `<module source>` to such
/// a module; real hosts use e.g. a WebAssembly Module Record. There is no
/// JavaScript module that qualifies (a Source Text Module Record's
/// `GetModuleSource` always throws), so this engine synthesizes one: a body-less,
/// export-less record whose `[[ModuleSource]]` is an `%AbstractModuleSource%`
/// instance.
///
/// [Module Source]: https://tc39.es/proposal-source-phase-imports/#sec-module-source-objects
pub const MODULE_SOURCE_KEY: &str = "<module source>";

impl ModuleHost for FileModuleHost {
    fn resolve(&self, specifier: &str, referrer: Option<&str>) -> Result<String, String> {
        use std::path::{Path, PathBuf};
        // `<module source>` is not a path: it names the host's module-source
        // module and is its own canonical key (see [`MODULE_SOURCE_KEY`]).
        if specifier == MODULE_SOURCE_KEY {
            return Ok(MODULE_SOURCE_KEY.to_string());
        }
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

    fn load_bytes(&self, key: &str) -> Result<Vec<u8>, String> {
        std::fs::read(key).map_err(|e| alloc::format!("cannot load module {key}: {e}"))
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
    /// `evaluating-async`: the module (or its strongly-connected component) has
    /// started evaluating but is still waiting on a top-level `await` — its own,
    /// or one in an asynchronous dependency. Its body may not have run at all yet
    /// (`[[PendingAsyncDependencies]]` > 0).
    EvaluatingAsync,
    /// Body has finished running (successfully or with a captured `eval_error`).
    Evaluated,
}

/// `[[AsyncEvaluationOrder]]` — unset for a module that never participates in an
/// asynchronous evaluation, an ascending integer while it does, and `done` once
/// it has settled. The integer orders `AsyncModuleExecutionFulfilled`'s
/// `execList`, which is what makes an async graph resume leaf-to-root in the
/// order evaluation *started*.
#[derive(Clone, Copy, PartialEq, Eq)]
enum AsyncOrder {
    Unset,
    Order(u32),
    Done,
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
    /// `with { type: "bytes" }` — the file's raw bytes become the `default`
    /// export, as a `Uint8Array` over an immutable `ArrayBuffer` (the
    /// import-bytes proposal's `CreateBytesModule`).
    Bytes,
    /// A host-provided **module source** module (source-phase-imports): it has no
    /// exports and no body, but it *does* have a `[[ModuleSource]]` — an
    /// `%AbstractModuleSource%` instance — so `import source x from …` binds it.
    /// See [`MODULE_SOURCE_KEY`].
    ModuleSource,
}

/// A single binding introduced by an import declaration.
enum ImportBind {
    /// `import x from "m"` — bind the local name to `m`'s default export.
    Default(String),
    /// `import * as ns from "m"` — bind the local name to `m`'s namespace object.
    Namespace(String),
    /// `import { imported as local } from "m"`.
    Named { imported: String, local: String },
    /// `import source x from "m"` — bind the local name to `m`'s
    /// `[[ModuleSource]]` object (source-phase-imports). Like a namespace import
    /// this is a real immutable slot in the *importing* module's own scope, not
    /// an alias into the dependency, which is what makes a re-export of it
    /// (`import source x from "m"; export { x }`) resolve to the module source
    /// itself — the spec's `[[BindingName]]: source` resolved binding.
    ///
    /// Deviation: the request still contributes to `[[RequestedModules]]`, so the
    /// dependency is linked and evaluated. The proposal loads only its source. The
    /// only specifier this engine resolves to a module *with* a source phase is
    /// [`MODULE_SOURCE_KEY`], whose record has no body, so the difference is not
    /// observable today.
    Source(String),
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
    /// `[[ModuleSource]]` — the module's source-phase representation, an
    /// `%AbstractModuleSource%` instance. Only a host-provided *module source*
    /// module has one (a Source Text Module Record's `GetModuleSource` always
    /// throws), so this is `None` for every ordinary JavaScript / JSON / text
    /// module and `import source` of one is a link-time SyntaxError.
    module_source: Option<NanBox>,
    /// For a synthetic (JSON/text) module: the already-built `default` export
    /// value (JSON parsed / text string), materialised at load time so a JSON
    /// parse failure surfaces in the load/resolution phase.
    default_value: Option<NanBox>,
    /// `[[HasTLA]]` — the body contains a top-level `await` (so it evaluates as a
    /// suspendable coroutine and settles asynchronously). Computed once at parse.
    has_tla: bool,
    /// `[[DFSIndex]]` / `[[DFSAncestorIndex]]` — Tarjan indices assigned by
    /// `InnerModuleEvaluation`; equal iff this module is its cycle's root.
    dfs_index: u32,
    dfs_ancestor_index: u32,
    /// `[[PendingAsyncDependencies]]` — how many dependencies are still
    /// `evaluating-async`. While non-zero this module's body has not run.
    pending_async_deps: u32,
    /// `[[AsyncEvaluationOrder]]`.
    async_order: AsyncOrder,
    /// `[[CycleRoot]]` — the module whose `[[TopLevelCapability]]` settles for
    /// this whole strongly-connected component.
    cycle_root: Option<String>,
    /// `[[AsyncParentModules]]` — modules that are waiting on this one.
    async_parents: Vec<String>,
    /// `[[TopLevelCapability]]` — the promise `Evaluate()` handed out (only ever
    /// set on a cycle root, or on a module whose evaluation failed outright).
    top_level_capability: Option<crate::heap::Handle>,
    /// The module's code on the bytecode VM: its VM module index and function
    /// table entries (see `crate::nbvm::compile_module_into`); `None` for a
    /// synthetic (JSON / text / bytes) module, or one the compiler refused
    /// (which fails to link).
    vm: Option<(u32, crate::nbvm::ModuleProtos)>,
    /// Whether compiling for the VM was already attempted.
    vm_tried: bool,
}

/// The set of loaded modules, keyed by resolved key, plus the active host.
pub struct ModuleRegistry {
    records: BTreeMap<String, ModuleRecord>,
    /// The agent-wide `[[AsyncEvaluationOrder]]` counter (the spec's
    /// *asyncEvaluationOrder* slot on the surrounding agent).
    async_order_counter: u32,
    /// Depth of the currently running `Evaluate()` — a module graph walk is in
    /// progress while non-zero. `Evaluate` asserts it never runs concurrently
    /// with another `Evaluate` in the same agent, so a dynamic `import()` reached
    /// from a module body must defer its own link/evaluate to a job.
    evaluating_depth: u32,
    /// VM module index → module key.
    vm_keys: Vec<String>,
    /// VM module index → the module's environment and import aliases, set at
    /// link time (what a VM environment access resolves against).
    vm_envs: Vec<Option<VmEnv>>,
    /// The function table holding every VM-compiled module (and the code
    /// compiled before them); it only grows.
    vm_table: Option<Rc<[crate::nbvm::FnProto]>>,
    /// Why the first module the VM refused could not run.
    vm_note: Option<String>,
    /// A VM fault happened where it could not propagate (see
    /// `Interp::vm_module_faulted`).
    vm_fault: bool,
}

/// A VM module's environment: its scope and its import alias table.
type VmEnv = (Scope, Rc<BTreeMap<String, (Scope, String)>>);

impl ModuleRegistry {
    /// Whether no module has been loaded into this registry — the GC safepoint's
    /// "no module graph is live" check (see `nbexec::gc`).
    pub(crate) fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub(crate) fn new() -> Self {
        Self {
            records: BTreeMap::new(),
            async_order_counter: 0,
            evaluating_depth: 0,
            vm_keys: Vec::new(),
            vm_envs: Vec::new(),
            vm_table: None,
            vm_note: None,
            vm_fault: false,
        }
    }

    /// The table a VM run should install: `table`, unless the module table
    /// has outgrown it (a dynamic `import()` compiled modules mid-run) — every
    /// table of one interpreter is a prefix of the next.
    pub(crate) fn prefer_module_table(
        &self,
        table: Option<Rc<[crate::nbvm::FnProto]>>,
    ) -> Option<Rc<[crate::nbvm::FnProto]>> {
        match (table, &self.vm_table) {
            (Some(t), Some(m)) if m.len() > t.len() => Some(Rc::clone(m)),
            (t, _) => t,
        }
    }
}

/// The synthetic local name an `export default` value is bound under.
pub(crate) const DEFAULT_LOCAL: &str = "*default*";

impl Interp {
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
        let promise = self.evaluate_module(entry_key);
        // Drain microtasks/timers so the graph's top-level `await`s (and any
        // trailing reactions) run — the entry's evaluation promise only settles
        // once the whole graph has completed.
        let _ = self.run_event_loop();
        let promise = promise?;
        // A rejected top-level capability is the graph's evaluation error.
        if let Some(state) = self.realm.promise_state(promise) {
            let st = state.borrow();
            if st.status == crate::cell::PromiseStatus::Rejected {
                let v = st.value;
                drop(st);
                return Err(ExecError::Throw(v));
            }
        }
        self.run_event_loop()?;
        self.namespace_object(entry_key)
    }

    /// Converts an [`ExecError`] surfaced from the module pipeline into a typed
    /// [`Thrown`], tagging it with `phase` (Parse for load/link, Runtime for
    /// evaluation). Mirrors `eval_source_typed`'s error rendering.
    pub fn exec_error_to_thrown(&self, e: ExecError, phase: super::ErrorPhase) -> Thrown {
        super::thrown_from_exec_error(self, e, phase)
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
        // The host-provided module-source module has no file behind it: it is
        // synthesized here rather than loaded (source-phase-imports).
        if key == MODULE_SOURCE_KEY {
            let record = self.build_module_source_module(key);
            self.modules.records.insert(key.to_string(), record);
            if let Some(r) = self.modules.records.get_mut(key) {
                r.status = Status::Loaded;
            }
            return Ok(());
        }
        // The map key of a JSON / text module is suffixed with its type (so the
        // same file imported both as JavaScript and as text/JSON is two distinct
        // modules, per the spec's `(specifier, attributes)` module map key). The
        // host loads the underlying file, so strip the suffix first. A load
        // failure is *host*-defined, not a syntax error: `HostLoadImportedModule`
        // may complete with any error value, and code that probes for a module
        // needs to tell "the host could not supply this module" apart from "the
        // module is not valid JavaScript", so it surfaces as a plain `Error`.
        // A `type` attribute selects a synthetic (JSON / text / bytes) module;
        // otherwise the file is an ordinary JavaScript module. A JSON parse error
        // here is a load/resolution-phase failure (a SyntaxError), matching the
        // tests' `negative: { phase: resolution }` expectation. A bytes module is
        // loaded as raw bytes — its file (an image, say) need not be text at all.
        let record = if type_attr == Some("bytes") {
            let bytes = host
                .load_bytes(module_load_path(key))
                .map_err(|e| self.module_load_error(&e))?;
            self.build_bytes_module(key, &bytes)
        } else {
            let source = host
                .load(module_load_path(key))
                .map_err(|e| self.module_load_error(&e))?;
            match type_attr {
                Some("json") => self.build_json_module(key, &source)?,
                Some("text") => self.build_text_module(key, &source),
                _ => self.parse_module(key, &source, host)?,
            }
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

    /// A host module-*loading* failure (the file could not be read / the host has
    /// no such module) as a plain `Error`. Distinct from the `SyntaxError`s the
    /// parse and link phases raise.
    fn module_load_error(&mut self, message: &str) -> ExecError {
        let m = self.new_str(message);
        ExecError::Throw(self.make_error(N_ERROR_BASE, Some(m)))
    }

    /// Builds a synthetic **JSON module** record for `key`: `JSON.parse(source)`
    /// becomes the sole `default` export. An empty AST body drives (no) link /
    /// evaluation. A malformed source propagates as a SyntaxError.
    fn build_json_module(&mut self, key: &str, source: &str) -> Result<ModuleRecord, ExecError> {
        let value = self.parse_json_source(source)?;
        Ok(self.synthetic_module(key, ModuleKind::Json, Some(value)))
    }

    /// Builds a synthetic **text module** record for `key`: the raw source text
    /// becomes the `default` export (import-text proposal).
    fn build_text_module(&mut self, key: &str, source: &str) -> ModuleRecord {
        let value = self.new_str(source);
        self.synthetic_module(key, ModuleKind::Text, Some(value))
    }

    /// Builds a synthetic **bytes module** record for `key` (import-bytes
    /// proposal, `CreateBytesModule`): the file's raw bytes become the `default`
    /// export as a `Uint8Array` spanning a fresh *immutable* `ArrayBuffer` — so
    /// the view is read-only and the buffer can be neither resized nor
    /// transferred.
    fn build_bytes_module(&mut self, key: &str, bytes: &[u8]) -> ModuleRecord {
        let buffer = self.make_array_buffer_from_bytes(bytes);
        self.realm.set_hidden_property(
            buffer,
            super::ARRAY_BUFFER_IMMUTABLE,
            NanBox::boolean(true),
        );
        // Kind 1 is `Uint8Array` (see `TYPED_ARRAY_KINDS`).
        let view = self
            .typed_array_over(buffer, 1, 0, bytes.len())
            .expect("a fresh ArrayBuffer backs the bytes module");
        self.synthetic_module(key, ModuleKind::Bytes, Some(NanBox::handle(view.to_raw())))
    }

    /// Builds the host-provided **module source** module (source-phase-imports):
    /// no body and no exports, but a `[[ModuleSource]]` — a fresh
    /// `%AbstractModuleSource%` instance — so `import source x from
    /// "<module source>"` has something to bind. See [`MODULE_SOURCE_KEY`].
    fn build_module_source_module(&mut self, key: &str) -> ModuleRecord {
        let src = self.module_source_object();
        let mut record = self.synthetic_module(key, ModuleKind::ModuleSource, None);
        record.local_exports.clear();
        record.module_source = Some(src);
        record
    }

    /// Assembles a synthetic module record (JSON / text / module source): an empty
    /// program plus, when `value` is present, a single local `default` export
    /// bound to it.
    fn synthetic_module(
        &mut self,
        key: &str,
        kind: ModuleKind,
        value: Option<NanBox>,
    ) -> ModuleRecord {
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
            module_source: None,
            default_value: value,
            has_tla: false,
            dfs_index: 0,
            dfs_ancestor_index: 0,
            pending_async_deps: 0,
            async_order: AsyncOrder::Unset,
            cycle_root: None,
            async_parents: Vec::new(),
            top_level_capability: None,
            vm: None,
            vm_tried: true,
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
                            ImportSpecifier::Source(id) => {
                                binds.push(ImportBind::Source(id.name.to_string()));
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
            module_source: None,
            default_value: None,
            has_tla: module_body_has_await(&program.body),
            dfs_index: 0,
            dfs_ancestor_index: 0,
            pending_async_deps: 0,
            async_order: AsyncOrder::Unset,
            cycle_root: None,
            async_parents: Vec::new(),
            top_level_capability: None,
            vm: None,
            vm_tried: false,
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
        // Compile what the loader just added (this module and its graph) for
        // the VM, before any of it is instantiated.
        self.vm_compile_pending();
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
                            ImportBind::Source(local) => (local.clone(), ImportKind::Source),
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
                    ImportKind::Source => {
                        // InitializeEnvironment step 7.c (source-phase-imports):
                        // bind the local name to the dependency's
                        // `[[ModuleSource]]`, or throw a SyntaxError when it has
                        // none (`GetModuleSource` of a Source Text Module Record
                        // always returns an abrupt completion).
                        let Some(src) = self
                            .modules
                            .records
                            .get(dep_key)
                            .and_then(|r| r.module_source)
                        else {
                            return Err(self.syntax_error(&alloc::format!(
                                "module {dep_key} has no source phase representation"
                            )));
                        };
                        let r = &self.modules.records[key];
                        r.scope.declare_const(local, src);
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
        // `import.meta` belongs to the *module*, not to the running context, so
        // tag the scope with it too (`Scope::module_meta`): a function defined
        // here and called from another module must still see this module's object.
        let meta = self.module_meta(key);
        if let Some(r) = self.modules.records.get_mut(key) {
            r.import_aliases = aliases.clone();
            // Tag the module's top-level scope with its imports so a function
            // defined here restores the right aliases when it runs (even when
            // called from another module) — see `Scope::module_imports`.
            r.scope.set_module_imports(aliases.clone());
            r.scope.set_module_meta(meta);
            if let Some((index, _)) = r.vm {
                let env = Some((r.scope.clone(), aliases));
                self.modules.vm_envs[index as usize] = env;
            }
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
        match self.modules.records[key].vm {
            Some((_, protos)) => self.vm_instantiate(key, protos),
            // A synthetic (JSON / text / bytes) module declares no functions.
            None if !matches!(self.modules.records[key].kind, ModuleKind::JavaScript) => Ok(()),
            // A module the bytecode compiler refused fails to link: module code
            // runs only on the VM.
            None => Err(ExecError::Unsupported(
                "the bytecode compiler refused this module",
            )),
        }
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

    /// `Evaluate()` (16.2.1.5.3) — evaluates the graph rooted at `key` and returns
    /// the module's **top-level capability** promise: fulfilled with `undefined`
    /// once the whole graph (including every top-level `await`) has completed,
    /// rejected with the evaluation error otherwise.
    ///
    /// The returned promise may still be pending: an asynchronous module does not
    /// block the walk, so the caller must chain on it rather than assume the body
    /// has run (see [`Self::evaluate_module_now`] for the callers that legitimately
    /// require synchronous completion).
    fn evaluate_module(&mut self, key: &str) -> Result<crate::heap::Handle, ExecError> {
        // Step 4: an already-(async-)evaluated module delegates to its cycle root,
        // which is where the capability for the whole component lives.
        let mut key = key.to_string();
        {
            let Some(r) = self.modules.records.get(&key) else {
                return Err(self.syntax_error(&alloc::format!("module {key} not linked")));
            };
            if matches!(r.status, Status::EvaluatingAsync | Status::Evaluated)
                && let Some(root) = &r.cycle_root
            {
                key = root.clone();
            }
        }
        if let Some(p) = self.modules.records[&key].top_level_capability {
            return Ok(p);
        }
        let capability = self.fresh_promise();
        self.modules
            .records
            .get_mut(&key)
            .expect("record")
            .top_level_capability = Some(capability);

        let mut stack: Vec<String> = Vec::new();
        self.modules.evaluating_depth += 1;
        let result = self.inner_module_evaluation(&key, &mut stack, 0);
        self.modules.evaluating_depth -= 1;

        match result {
            Err(ExecError::Throw(err)) => {
                // Step 9: every module still on the stack failed with this error.
                for m in &stack {
                    if let Some(r) = self.modules.records.get_mut(m) {
                        r.status = Status::Evaluated;
                        r.eval_error = Some(err);
                        r.async_order = AsyncOrder::Done;
                    }
                }
                self.settle(capability, err, false);
            }
            Err(other) => return Err(other),
            Ok(_) => {
                // Step 10: a module that is not asynchronously evaluating has
                // finished, so its capability resolves now.
                if self.modules.records[&key].status == Status::Evaluated {
                    self.settle(capability, NanBox::undefined(), true);
                }
            }
        }
        Ok(capability)
    }

    /// `Evaluate()` for the callers that require the graph to have *finished*
    /// (a `import defer` namespace force, whose asynchronous dependencies were
    /// hoisted out of the deferral precisely so this is synchronous). Surfaces a
    /// rejected top-level capability as the thrown evaluation error.
    fn evaluate_module_now(&mut self, key: &str) -> Result<(), ExecError> {
        let promise = self.evaluate_module(key)?;
        if let Some(state) = self.realm.promise_state(promise) {
            let st = state.borrow();
            if st.status == crate::cell::PromiseStatus::Rejected {
                return Err(ExecError::Throw(st.value));
            }
        }
        Ok(())
    }

    /// The dependency list `InnerModuleEvaluation` walks, in `[[RequestedModules]]`
    /// (source) order. A *deferred* import (`import defer`) contributes not itself
    /// but its **asynchronous** transitive dependencies
    /// (`GatherAsynchronousTransitiveDependencies`): the deferred module is
    /// evaluated lazily on first namespace access, which is a synchronous
    /// operation, so anything with top-level await below it must be hoisted out of
    /// the deferral and evaluated with the importer.
    fn evaluation_deps(&self, key: &str) -> Vec<String> {
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
        deps
    }

    /// `InnerModuleEvaluation(module, stack, index)` (16.2.1.5.3.1) — the
    /// depth-first walk that assigns the Tarjan indices, counts each importer's
    /// still-pending asynchronous dependencies (rather than blocking on them),
    /// executes what is ready, and settles each completed strongly-connected
    /// component's status and cycle root.
    fn inner_module_evaluation(
        &mut self,
        key: &str,
        stack: &mut Vec<String>,
        index: u32,
    ) -> Result<u32, ExecError> {
        match self.modules.records.get(key).map(|r| r.status) {
            Some(Status::EvaluatingAsync | Status::Evaluated) => {
                if let Some(err) = self.modules.records[key].eval_error {
                    return Err(ExecError::Throw(err));
                }
                return Ok(index);
            }
            // Already on the stack — a cycle back-edge.
            Some(Status::Evaluating) => return Ok(index),
            Some(Status::Linked) => {}
            _ => return Err(self.syntax_error(&alloc::format!("module {key} not linked"))),
        }
        {
            let r = self.modules.records.get_mut(key).expect("record");
            r.status = Status::Evaluating;
            r.dfs_index = index;
            r.dfs_ancestor_index = index;
            r.pending_async_deps = 0;
        }
        let mut index = index + 1;
        stack.push(key.to_string());

        for dep in self.evaluation_deps(key) {
            index = self.inner_module_evaluation(&dep, stack, index)?;
            // Step 11.c: a dependency still on the stack (status `evaluating`) is
            // part of this cycle and contributes its ancestor index; otherwise its
            // component is closed and what we wait on is its cycle root.
            let target = if stack.contains(&dep) {
                let ancestor = self.modules.records[&dep].dfs_ancestor_index;
                let r = self.modules.records.get_mut(key).expect("record");
                r.dfs_ancestor_index = r.dfs_ancestor_index.min(ancestor);
                dep.clone()
            } else {
                let root = self.modules.records[&dep]
                    .cycle_root
                    .clone()
                    .unwrap_or_else(|| dep.clone());
                if let Some(err) = self.modules.records[&root].eval_error {
                    return Err(ExecError::Throw(err));
                }
                root
            };
            // Step 11.c.v: an `[[AsyncEvaluationOrder]]` that is still an *integer*
            // means that module has started but not settled — including a module
            // of this very cycle that has already reached its own step 12.
            let Some(target_rec) = self.modules.records.get_mut(&target) else {
                continue;
            };
            if matches!(target_rec.async_order, AsyncOrder::Order(_)) {
                target_rec.async_parents.push(key.to_string());
                self.modules
                    .records
                    .get_mut(key)
                    .expect("record")
                    .pending_async_deps += 1;
            }
        }

        let (pending, has_tla) = {
            let r = &self.modules.records[key];
            (r.pending_async_deps, r.has_tla)
        };
        if pending > 0 || has_tla {
            let order = self.modules.async_order_counter;
            self.modules.async_order_counter += 1;
            self.modules
                .records
                .get_mut(key)
                .expect("record")
                .async_order = AsyncOrder::Order(order);
            if pending == 0 {
                self.execute_async_module(key);
            }
        } else {
            self.run_module_body(key)?;
        }

        let (dfs_index, dfs_ancestor) = {
            let r = &self.modules.records[key];
            (r.dfs_index, r.dfs_ancestor_index)
        };
        if dfs_ancestor == dfs_index {
            // This module roots a (possibly singleton) cycle: close the component.
            while let Some(m) = stack.pop() {
                let r = self.modules.records.get_mut(&m).expect("record");
                r.status = if matches!(r.async_order, AsyncOrder::Unset) {
                    Status::Evaluated
                } else {
                    Status::EvaluatingAsync
                };
                r.cycle_root = Some(key.to_string());
                if m == key {
                    break;
                }
            }
        }
        Ok(index)
    }

    /// `ExecuteAsyncModule(module)` (16.2.1.5.3.2) — starts a top-level-await
    /// module's body as a suspendable coroutine and hooks its evaluation promise
    /// with the `AsyncModuleExecutionFulfilled` / `AsyncModuleExecutionRejected`
    /// continuations. Returns immediately: the graph walk continues while the body
    /// is parked on its `await`.
    fn execute_async_module(&mut self, key: &str) {
        // Only a module the VM compiled links, so it has its VM protos.
        if let Some((_, protos)) = self.modules.records[key].vm {
            self.vm_execute_async_module(key, protos);
        }
    }

    /// `AsyncModuleExecutionFulfilled(module)` (16.2.1.5.3.3) — the module's body
    /// completed: mark it evaluated, settle its top-level capability, then run
    /// every ancestor that this unblocked, oldest `[[AsyncEvaluationOrder]]` first.
    pub(crate) fn async_module_execution_fulfilled(&mut self, key: &str) {
        {
            let Some(r) = self.modules.records.get_mut(key) else {
                return;
            };
            if r.status == Status::Evaluated {
                // Already settled by a rejection that reached it first.
                return;
            }
            r.async_order = AsyncOrder::Done;
            r.status = Status::Evaluated;
        }
        if let Some(cap) = self.modules.records[key].top_level_capability {
            self.settle(cap, NanBox::undefined(), true);
        }
        let mut exec_list: Vec<String> = Vec::new();
        self.gather_available_ancestors(key, &mut exec_list);
        exec_list.sort_by_key(|m| match self.modules.records[m].async_order {
            AsyncOrder::Order(n) => n,
            _ => u32::MAX,
        });
        for m in exec_list {
            let (status, has_tla) = {
                let r = &self.modules.records[&m];
                (r.status, r.has_tla)
            };
            if status == Status::Evaluated {
                // Its evaluation already failed; nothing left to run.
            } else if has_tla {
                self.execute_async_module(&m);
            } else {
                match self.run_module_body(&m) {
                    Err(ExecError::Throw(v)) => self.async_module_execution_rejected(&m, v),
                    Err(other) => {
                        let msg = self.new_str(&alloc::format!("{other:?}"));
                        let err = self.make_error(super::N_TYPE_ERROR, Some(msg));
                        self.async_module_execution_rejected(&m, err);
                    }
                    Ok(()) => {
                        let r = self.modules.records.get_mut(&m).expect("record");
                        r.async_order = AsyncOrder::Done;
                        r.status = Status::Evaluated;
                        if let Some(cap) = r.top_level_capability {
                            self.settle(cap, NanBox::undefined(), true);
                        }
                    }
                }
            }
        }
    }

    /// `AsyncModuleExecutionRejected(module, error)` (16.2.1.5.3.4) — records the
    /// evaluation error and propagates it to every module waiting on this one.
    pub(crate) fn async_module_execution_rejected(&mut self, key: &str, error: NanBox) {
        let parents = {
            let Some(r) = self.modules.records.get_mut(key) else {
                return;
            };
            if r.status == Status::Evaluated {
                return;
            }
            r.eval_error = Some(error);
            r.status = Status::Evaluated;
            r.async_order = AsyncOrder::Done;
            r.async_parents.clone()
        };
        // Own capability first, *then* the ancestors': a rejection settles the
        // graph's promises leaf-to-root (16.2.1.5.3.4 steps 9–10).
        if let Some(cap) = self.modules.records[key].top_level_capability {
            self.settle(cap, error, false);
        }
        for m in parents {
            self.async_module_execution_rejected(&m, error);
        }
    }

    /// `GatherAvailableAncestors(module, execList)` (16.2.1.5.3.5) — decrements
    /// each waiting importer's `[[PendingAsyncDependencies]]` and collects those
    /// that just reached zero (recursing through synchronous ones, which will
    /// unblock their own parents as soon as they run).
    fn gather_available_ancestors(&mut self, key: &str, exec_list: &mut Vec<String>) {
        let parents = self.modules.records[key].async_parents.clone();
        for m in parents {
            if exec_list.contains(&m) {
                continue;
            }
            let failed = {
                let root = self.modules.records[&m]
                    .cycle_root
                    .clone()
                    .unwrap_or_else(|| m.clone());
                self.modules
                    .records
                    .get(&root)
                    .is_some_and(|r| r.eval_error.is_some())
            };
            if failed {
                continue;
            }
            let (ready, has_tla) = {
                let r = self.modules.records.get_mut(&m).expect("record");
                r.pending_async_deps = r.pending_async_deps.saturating_sub(1);
                (r.pending_async_deps == 0, r.has_tla)
            };
            if ready {
                exec_list.push(m.clone());
                if !has_tla {
                    self.gather_available_ancestors(&m, exec_list);
                }
            }
        }
    }

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
        if matches!(
            r.status,
            Status::Evaluating | Status::EvaluatingAsync | Status::Evaluated
        ) {
            return;
        }
        if r.has_tla {
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

    /// `ExecuteModule()` for a **synchronous** module: runs its top-level
    /// statements in its own environment, with its import aliases active and
    /// `import.meta` set up. Modules are always strict. A top-level-await module
    /// goes through [`Self::execute_async_module`] instead.
    fn run_module_body(&mut self, key: &str) -> Result<(), ExecError> {
        // A synthetic module has no body; only a module the VM compiled links.
        match self.modules.records[key].vm {
            Some((_, protos)) => self.vm_call_proto(protos.main).map(|_| ()),
            None => Ok(()),
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
        // run *successfully* (a defer of an already-evaluated module is just a
        // namespace view). A module that already evaluated and **threw** stays
        // armed: `EnsureDeferredNamespaceEvaluation` always performs
        // `EvaluateSync`, which rethrows the cached `[[EvaluationError]]`.
        let already = matches!(
            self.modules.records.get(key),
            Some(r) if r.status == Status::Evaluated && r.eval_error.is_none()
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
        // The VM's own property access cannot see the live bindings.
        self.realm.mark_host_exotic(obj);
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
        // `EnsureDeferredNamespaceEvaluation` step 2: unless the module already
        // evaluated, its whole requested-module graph must be able to run
        // *synchronously* right now (`ReadyForSyncExecution`); otherwise the
        // access is a TypeError. This is checked over the whole closure *before*
        // evaluating anything, so no side effects of the subgraph run.
        let evaluated = matches!(
            self.modules.records.get(&dep).map(|r| r.status),
            Some(Status::Evaluated)
        );
        if !evaluated && !self.ready_for_sync_execution(&dep, &mut BTreeSet::new()) {
            return Err(self.type_error(
                "Cannot access a deferred module namespace while the module is being evaluated",
            ));
        }
        self.evaluate_module_now(&dep)?;
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

    /// `ReadyForSyncExecution(module, seen)` (import-defer) — whether forcing
    /// `key` can complete synchronously: every module in its `[[RequestedModules]]`
    /// closure (deferred requests included — the spec walks *all* of them) must be
    /// `evaluated`, or still `linked` with no top-level `await`. A module that is
    /// `evaluating` / `evaluating-async` is on the active evaluation stack with
    /// uninitialized bindings, and one with `[[HasTLA]]` cannot finish before the
    /// access returns; either makes the force a TypeError. `seen` breaks cycles
    /// (a module already being considered answers `true`).
    fn ready_for_sync_execution(&self, key: &str, seen: &mut BTreeSet<String>) -> bool {
        if !seen.insert(key.to_string()) {
            return true;
        }
        let Some(r) = self.modules.records.get(key) else {
            return true;
        };
        // An already-evaluated module's subgraph ran to completion; don't descend.
        if r.status == Status::Evaluated {
            return true;
        }
        if matches!(r.status, Status::Evaluating | Status::EvaluatingAsync) {
            return false;
        }
        if r.has_tla {
            return false;
        }
        let requested: Vec<String> = r.requested.iter().map(|(k, _)| k.clone()).collect();
        for dep in requested {
            if !self.ready_for_sync_execution(&dep, seen) {
                return false;
            }
        }
        true
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

    /// [`Self::dynamic_import_values`] with the referrer (the importing
    /// module's key, or a script's import base) given.
    fn dynamic_import_from(
        &mut self,
        referrer: Option<String>,
        spec: NanBox,
        options: Option<NanBox>,
    ) -> NanBox {
        let promise = self.fresh_promise();
        let host = FileModuleHost;
        let resolved: Result<(String, Option<String>), ExecError> = (|this: &mut Self| {
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
            Ok((module_map_key(&dep, type_attr.as_deref()), type_attr))
        })(self);
        match resolved {
            Ok((dep, type_attr)) => self.start_dynamic_import(&dep, type_attr.as_deref(), promise),
            Err(e) => self.reject_dynamic_import(promise, e),
        }
        NanBox::handle(promise.to_raw())
    }

    /// `FinishLoadingImportedModule` + `ContinueDynamicImport` for a resolved
    /// dynamic-import request: loads, links, and evaluates `dep`, settling
    /// `promise` with its namespace once the evaluation promise does.
    ///
    /// `Evaluate()` may not run while another `Evaluate()` is in progress in the
    /// same agent (16.2.1.5.3 step 1), so when this call is reached *from a module
    /// body* the whole continuation is deferred to a job — which is also what stops
    /// a dynamic import from preempting the enclosing graph's depth-first order.
    pub(crate) fn start_dynamic_import(
        &mut self,
        dep: &str,
        type_attr: Option<&str>,
        promise: crate::heap::Handle,
    ) {
        if self.modules.evaluating_depth > 0 {
            self.defer_dynamic_import(dep, type_attr, promise, false);
            return;
        }
        let host = FileModuleHost;
        let outcome: Result<(), ExecError> = (|this: &mut Self| {
            this.load_module(dep, &host, type_attr)?;
            this.link_module(dep)?;
            let evaluated = this.evaluate_module(dep)?;
            this.settle_namespace_when(evaluated, dep, promise, false);
            Ok(())
        })(self);
        if let Err(e) = outcome {
            self.reject_dynamic_import(promise, e);
        }
    }

    /// `ContinueDynamicImport` for the **defer** phase (`import.defer(specifier)`):
    /// loads and links `dep` but evaluates only its asynchronous transitive
    /// dependencies, then fulfils `promise` with the Deferred Module Namespace.
    pub(crate) fn start_dynamic_import_deferred(
        &mut self,
        dep: &str,
        promise: crate::heap::Handle,
    ) {
        if self.modules.evaluating_depth > 0 {
            self.defer_dynamic_import(dep, None, promise, true);
            return;
        }
        let host = FileModuleHost;
        let outcome: Result<(), ExecError> = (|this: &mut Self| {
            this.load_module(dep, &host, None)?;
            this.link_module(dep)?;
            // Deliberately NOT evaluated — deferred until first namespace access —
            // except for the asynchronous transitive dependencies, which cannot be
            // evaluated synchronously when that access happens and so are hoisted
            // out of the deferral exactly as for a static `import defer`.
            let mut seen = alloc::collections::BTreeSet::new();
            let mut gathered = Vec::new();
            this.gather_async_transitive_deps(dep, &mut seen, &mut gathered);
            let mut pending: Vec<crate::heap::Handle> = Vec::new();
            for m in &gathered {
                let p = this.evaluate_module(m)?;
                if this
                    .realm
                    .promise_state(p)
                    .is_some_and(|s| s.borrow().status == crate::cell::PromiseStatus::Pending)
                {
                    pending.push(p);
                }
            }
            if pending.is_empty() {
                // Every hoisted dependency already completed: the deferred
                // namespace is available now.
                for m in &gathered {
                    if let Some(err) = this.modules.records[m].eval_error {
                        return Err(ExecError::Throw(err));
                    }
                }
                let ns = this.deferred_namespace_object(dep)?;
                this.settle(promise, ns, true);
            } else {
                // Wait for the still-running ones before handing out the namespace.
                let all = this.safe_perform_promise_all(&pending);
                this.settle_namespace_when(all, dep, promise, true);
            }
            Ok(())
        })(self);
        if let Err(e) = outcome {
            self.reject_dynamic_import(promise, e);
        }
    }

    /// `SafePerformPromiseAll(promises)` — an aggregate promise that fulfils once
    /// every element has fulfilled and rejects with the first rejection. Unlike
    /// `Promise.all` it performs the reactions directly, so a patched
    /// `Promise.prototype.then` (or a `@@species` constructor) is never observed
    /// by import-defer's asynchronous-dependency hoisting.
    fn safe_perform_promise_all(
        &mut self,
        promises: &[crate::heap::Handle],
    ) -> crate::heap::Handle {
        let result = self.fresh_promise();
        if promises.is_empty() {
            self.settle(result, NanBox::undefined(), true);
            return result;
        }
        let state = self.realm.new_object();
        self.realm.set_hidden_property(
            state,
            SAFE_ALL_REMAINING,
            NanBox::number(promises.len() as f64),
        );
        self.realm
            .set_hidden_property(state, SAFE_ALL_TARGET, NanBox::handle(result.to_raw()));
        for p in promises {
            let on_f = self
                .realm
                .new_bound_native(super::N_SAFE_ALL_FULFILL, state);
            let on_r = self.realm.new_bound_native(super::N_SAFE_ALL_REJECT, state);
            self.register_then(
                *p,
                NanBox::handle(on_f.to_raw()),
                NanBox::handle(on_r.to_raw()),
                false,
            );
        }
        result
    }

    /// One element settlement of [`Self::safe_perform_promise_all`]: a rejection
    /// settles the aggregate immediately, a fulfilment once the count reaches zero.
    pub(crate) fn safe_promise_all_element(
        &mut self,
        state: crate::heap::Handle,
        value: NanBox,
        fulfilled: bool,
    ) {
        let Some(target) = self
            .realm
            .get_property(state, SAFE_ALL_TARGET)
            .and_then(|v| v.as_handle())
            .map(crate::heap::Handle::from_raw)
        else {
            return;
        };
        if !fulfilled {
            self.settle(target, value, false);
            return;
        }
        let remaining = self
            .realm
            .get_property(state, SAFE_ALL_REMAINING)
            .and_then(|v| v.as_number())
            .unwrap_or(0.0)
            - 1.0;
        self.realm
            .set_hidden_property(state, SAFE_ALL_REMAINING, NanBox::number(remaining));
        if remaining <= 0.0 {
            self.settle(target, NanBox::undefined(), true);
        }
    }

    /// Hooks `evaluated` (a module's top-level capability, or an aggregate of
    /// several) so that `promise` fulfils with `dep`'s (deferred) namespace object
    /// when it does, and rejects with the same reason when it rejects.
    fn settle_namespace_when(
        &mut self,
        evaluated: crate::heap::Handle,
        dep: &str,
        promise: crate::heap::Handle,
        deferred_phase: bool,
    ) {
        let state = self.realm.new_object();
        let key_val = self.new_str(dep);
        self.realm.set_hidden_property(state, DYN_KEY, key_val);
        self.realm
            .set_hidden_property(state, DYN_PROMISE, NanBox::handle(promise.to_raw()));
        self.realm
            .set_hidden_property(state, DYN_DEFER, NanBox::boolean(deferred_phase));
        let on_f = self
            .realm
            .new_bound_native(super::N_DYNAMIC_IMPORT_FULFILLED, state);
        let on_r = self
            .realm
            .new_bound_native(super::N_DYNAMIC_IMPORT_REJECTED, state);
        self.register_then(
            evaluated,
            NanBox::handle(on_f.to_raw()),
            NanBox::handle(on_r.to_raw()),
            false,
        );
    }

    /// Queues the load/link/evaluate continuation of a dynamic import as a job, so
    /// it runs once the enclosing `Evaluate()` has returned.
    fn defer_dynamic_import(
        &mut self,
        dep: &str,
        type_attr: Option<&str>,
        promise: crate::heap::Handle,
        deferred_phase: bool,
    ) {
        let state = self.realm.new_object();
        let key_val = self.new_str(dep);
        self.realm.set_hidden_property(state, DYN_KEY, key_val);
        if let Some(t) = type_attr {
            let t = self.new_str(t);
            self.realm.set_hidden_property(state, DYN_TYPE, t);
        }
        self.realm
            .set_hidden_property(state, DYN_PROMISE, NanBox::handle(promise.to_raw()));
        self.realm
            .set_hidden_property(state, DYN_DEFER, NanBox::boolean(deferred_phase));
        let job = self
            .realm
            .new_bound_native(super::N_DYNAMIC_IMPORT_JOB, state);
        let trigger = self.promise_resolve(NanBox::undefined());
        self.register_then(
            trigger,
            NanBox::handle(job.to_raw()),
            NanBox::undefined(),
            false,
        );
    }

    /// Runs a deferred dynamic-import job (the [`Self::defer_dynamic_import`]
    /// continuation), reading its request back out of the bound state object.
    pub(crate) fn run_dynamic_import_job(&mut self, state: crate::heap::Handle) {
        let key = self.hidden_string(state, DYN_KEY).unwrap_or_default();
        let type_attr = self.hidden_string(state, DYN_TYPE);
        let Some(promise) = self
            .realm
            .get_property(state, DYN_PROMISE)
            .and_then(|v| v.as_handle())
            .map(crate::heap::Handle::from_raw)
        else {
            return;
        };
        let deferred_phase = self
            .realm
            .get_property(state, DYN_DEFER)
            .is_some_and(|v| self.realm.truthy(v));
        // The enclosing `Evaluate()` has returned by now, so this runs the real
        // continuation rather than deferring again.
        if deferred_phase {
            self.start_dynamic_import_deferred(&key, promise);
        } else {
            self.start_dynamic_import(&key, type_attr.as_deref(), promise);
        }
    }

    /// Fulfils a dynamic import's promise with the imported module's namespace
    /// (the `ContinueDynamicImport` onFulfilled closure).
    pub(crate) fn dynamic_import_fulfilled(&mut self, state: crate::heap::Handle) {
        let key = self.hidden_string(state, DYN_KEY).unwrap_or_default();
        let Some(promise) = self
            .realm
            .get_property(state, DYN_PROMISE)
            .and_then(|v| v.as_handle())
            .map(crate::heap::Handle::from_raw)
        else {
            return;
        };
        let deferred_phase = self
            .realm
            .get_property(state, DYN_DEFER)
            .is_some_and(|v| self.realm.truthy(v));
        let ns = if deferred_phase {
            self.deferred_namespace_object(&key)
        } else {
            self.namespace_object(&key)
        };
        match ns {
            Ok(ns) => self.settle(promise, ns, true),
            Err(e) => self.reject_dynamic_import(promise, e),
        }
    }

    /// Rejects a dynamic import's promise with the module's evaluation error (the
    /// `ContinueDynamicImport` onRejected closure).
    pub(crate) fn dynamic_import_rejected(&mut self, state: crate::heap::Handle, error: NanBox) {
        if let Some(promise) = self
            .realm
            .get_property(state, DYN_PROMISE)
            .and_then(|v| v.as_handle())
            .map(crate::heap::Handle::from_raw)
        {
            self.settle(promise, error, false);
        }
    }

    /// Reads a hidden string slot off a bound-native state object.
    fn hidden_string(&self, state: crate::heap::Handle, slot: &str) -> Option<String> {
        self.realm
            .get_property(state, slot)
            .and_then(|v| v.as_handle())
            .map(crate::heap::Handle::from_raw)
            .and_then(|h| self.realm.string_value(h))
    }

    /// Settles a dynamic import's promise with a load/link/evaluate failure.
    fn reject_dynamic_import(&mut self, promise: crate::heap::Handle, e: ExecError) {
        match e {
            ExecError::Throw(v) => self.settle(promise, v, false),
            other => {
                let m = self.new_str(&alloc::format!("dynamic import failed: {other:?}"));
                let err = self.make_error(N_SYNTAX_ERROR, Some(m));
                self.settle(promise, err, false);
            }
        }
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
            // surfaces here); link wires imports; evaluate runs the body.
            // (`importValue` reads the export synchronously here, so a fixture
            // with top-level `await` would not have finished — the ShadowRealm
            // integration has no event loop of its own to drive it.)
            this.load_module(&dep, &host, None)?;
            this.link_module(&dep)?;
            this.evaluate_module_now(&dep)?;
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

/// Module code on the bytecode VM (`ROADMAP.md` §2.0): compilation into the
/// growing VM function table, link-time instantiation, running the bodies, and
/// the environment services VM module code calls back for (see
/// `crate::nbvm::compile_module_into` for the VM side).
impl Interp {
    /// The VM function table modules are compiled into, if any yet.
    pub(crate) fn module_vm_table(&self) -> Option<Rc<[crate::nbvm::FnProto]>> {
        self.modules.prefer_module_table(self.vm_table.clone())
    }

    /// Installs `table` — the previous table plus newly compiled code — as the
    /// VM function table, for the running VM code and for later runs.
    pub(crate) fn install_module_vm_table(&mut self, table: Rc<[crate::nbvm::FnProto]>) {
        self.realm.register_vm_fn_meta(&table);
        self.vm_table = Some(Rc::clone(&table));
        self.modules.vm_table = Some(table);
    }

    /// Why a module could not run on the VM (the first one).
    pub(crate) fn vm_module_note(&self) -> Option<&str> {
        self.modules.vm_note.as_deref()
    }

    /// Whether a VM fault happened where it could not propagate (an async
    /// module body): the entry must not trust the run.
    pub(crate) fn vm_module_faulted(&self) -> bool {
        self.modules.vm_fault
    }

    /// Compiles every loaded JavaScript module not yet tried for the VM. A
    /// module the VM refuses fails to link (see `instantiate_module_functions`).
    fn vm_compile_pending(&mut self) {
        let pending: Vec<String> = self
            .modules
            .records
            .values()
            .filter(|r| {
                !r.vm_tried
                    && matches!(r.kind, ModuleKind::JavaScript)
                    && matches!(r.status, Status::New | Status::Loaded)
            })
            .map(|r| r.key.clone())
            .collect();
        if pending.is_empty() {
            return;
        }
        let mut table: Vec<crate::nbvm::FnProto> = self
            .module_vm_table()
            .map(|t| t.to_vec())
            .unwrap_or_default();
        for key in pending {
            let index = self.modules.vm_keys.len() as u32;
            self.modules.vm_keys.push(key.clone());
            self.modules.vm_envs.push(None);
            let Some(r) = self.modules.records.get_mut(&key) else {
                continue;
            };
            r.vm_tried = true;
            match crate::nbvm::compile_module_into(r.program, index, r.has_tla, &mut table) {
                Ok(protos) => r.vm = Some((index, protos)),
                Err(e) => {
                    if self.modules.vm_note.is_none() {
                        self.modules.vm_note = Some(alloc::format!("module compile: {e:?}"));
                    }
                }
            }
        }
        self.install_module_vm_table(table.into());
    }

    /// `InitializeEnvironment` for a VM module: `var` bindings start
    /// `undefined`, lexical ones (and an anonymous default's `*default*`) in
    /// their temporal dead zone, and the init function binds the top-level
    /// function declarations.
    fn vm_instantiate(
        &mut self,
        key: &str,
        protos: crate::nbvm::ModuleProtos,
    ) -> Result<(), ExecError> {
        let (scope, program) = {
            let r = &self.modules.records[key];
            (r.scope.clone(), r.program)
        };
        let mut vars = Vec::new();
        super::collect_var_names(&program.body, &mut vars);
        for name in vars {
            if !scope.has_local(name) {
                scope.declare(name, NanBox::undefined());
            }
        }
        for stmt in &program.body {
            let inner = match stmt {
                Stmt::Export(
                    ExportDecl::Decl { declaration, .. } | ExportDecl::Default { declaration, .. },
                ) => &**declaration,
                other => other,
            };
            let lexical = match inner {
                Stmt::Var(d) => matches!(
                    d.kind,
                    crate::ast::VarDeclKind::Let | crate::ast::VarDeclKind::Const
                ),
                Stmt::Class(c) => c.id.is_some(),
                _ => false,
            };
            let mut names = if lexical {
                declared_names(inner)
            } else {
                Vec::new()
            };
            if matches!(stmt, Stmt::Export(ExportDecl::Default { .. }))
                && decl_name(inner).is_none()
            {
                names.push(DEFAULT_LOCAL.to_string());
            }
            for name in names {
                if !scope.has_local(&name) {
                    scope.declare(&name, NanBox::tdz());
                }
            }
        }
        self.vm_call_proto(protos.init).map(|_| ())
    }

    /// Calls the VM function-table entry `id` (a module body or init function)
    /// with `this` undefined.
    fn vm_call_proto(&mut self, id: u32) -> Result<NanBox, ExecError> {
        let f = self.realm.new_vm_function(id, Vec::new());
        self.call_with_this(NanBox::handle(f.to_raw()), NanBox::undefined(), &[])
    }

    /// `ExecuteAsyncModule` for a VM module: its body is an async function, so
    /// the call runs the first synchronous burst and returns the evaluation
    /// promise the loader's continuations chain on.
    fn vm_execute_async_module(&mut self, key: &str, protos: crate::nbvm::ModuleProtos) {
        let promise = match self.vm_call_proto(protos.main) {
            Ok(p) => p,
            Err(ExecError::Throw(v)) => {
                self.async_module_execution_rejected(key, v);
                return;
            }
            Err(other) => {
                // A VM fault has no JS value: record it so the entry falls
                // back, and fail the module so nothing waits on it.
                self.modules.vm_note = Some(alloc::format!("module runtime: {other:?}"));
                self.modules.vm_fault = true;
                let m = self.new_str("bytecode VM fault");
                let err = self.make_error(super::N_TYPE_ERROR, Some(m));
                self.async_module_execution_rejected(key, err);
                return;
            }
        };
        let Some(promise) = promise.as_handle().map(crate::heap::Handle::from_raw) else {
            return;
        };
        let state = self.new_str(key);
        let Some(state) = state.as_handle().map(crate::heap::Handle::from_raw) else {
            return;
        };
        let on_f = self
            .realm
            .new_bound_native(super::N_MODULE_FULFILLED, state);
        let on_r = self.realm.new_bound_native(super::N_MODULE_REJECTED, state);
        self.register_then(
            promise,
            NanBox::handle(on_f.to_raw()),
            NanBox::handle(on_r.to_raw()),
            false,
        );
    }

    /// Runs `f` with VM module `index`'s environment as the current one (its
    /// scope, its import aliases, strict mode) — how the host resolves a
    /// module-environment name of VM code. `None` for an unknown module.
    fn in_vm_module<R>(&mut self, index: u32, f: impl FnOnce(&mut Self) -> R) -> Option<R> {
        let (scope, aliases) = self.modules.vm_envs.get(index as usize)?.clone()?;
        let saved_current = core::mem::replace(&mut self.current, scope);
        let saved_imports = core::mem::replace(&mut self.module_imports, aliases);
        let saved_strict = core::mem::replace(&mut self.strict, true);
        let r = f(self);
        self.current = saved_current;
        self.module_imports = saved_imports;
        self.strict = saved_strict;
        Some(r)
    }

    /// VM module `index`'s environment (a dynamic-scope function's outermost
    /// one — see `crate::nbvm::EnvReq::Root`).
    pub(crate) fn vm_module_scope(&self, index: u32) -> Result<Scope, ExecError> {
        match self.modules.vm_envs.get(index as usize) {
            Some(Some((scope, _))) => Ok(scope.clone()),
            _ => Err(Self::no_vm_module()),
        }
    }

    /// The fault for an environment access of an unknown VM module.
    fn no_vm_module() -> ExecError {
        ExecError::Unsupported("unknown VM module")
    }

    /// `GetValue` of `name` in VM module `index`'s environment.
    pub(crate) fn vm_module_read(&mut self, index: u32, name: &str) -> Result<NanBox, ExecError> {
        self.in_vm_module(index, |this| this.read_ident_ref(name))
            .unwrap_or_else(|| Err(Self::no_vm_module()))
    }

    /// `typeof name` in VM module `index`'s environment: an import is always
    /// resolvable (and may be in its temporal dead zone).
    pub(crate) fn vm_module_typeof(
        &mut self,
        index: u32,
        name: &str,
    ) -> Result<&'static str, crate::nbvm::HostError> {
        self.in_vm_module(index, |this| {
            if this.module_imports.contains_key(name) {
                let v = this.read_ident_ref(name).map_err(super::exec_to_host)?;
                let t = this
                    .unary(crate::ast::UnaryOp::Typeof, v)
                    .map_err(super::exec_to_host)?;
                let t = this.realm.to_display_string(t);
                return Ok(super::TYPEOF_NAMES
                    .iter()
                    .copied()
                    .find(|n| *n == t)
                    .unwrap_or("object"));
            }
            crate::nbvm::VmHost::typeof_global(this, name)
        })
        .unwrap_or(Err(crate::nbvm::HostError::Fault))
    }

    /// `PutValue` of `name` in VM module `index`'s environment (module code is
    /// strict; an import is an immutable binding).
    pub(crate) fn vm_module_write(
        &mut self,
        index: u32,
        name: &str,
        value: NanBox,
    ) -> Result<(), ExecError> {
        self.in_vm_module(index, |this| {
            if this.module_imports.contains_key(name) {
                let m = this.new_str("Assignment to constant variable.");
                return Err(ExecError::Throw(
                    this.make_error(super::N_TYPE_ERROR, Some(m)),
                ));
            }
            this.assign_to_name(name, value)
        })
        .unwrap_or_else(|| Err(Self::no_vm_module()))
    }

    /// Initializes the module binding `name` (a top-level `let`/`const`/
    /// `class`, or a function the init function made).
    pub(crate) fn vm_module_init(
        &mut self,
        index: u32,
        name: &str,
        value: NanBox,
        konst: bool,
    ) -> Result<(), ExecError> {
        let Some(Some((scope, _))) = self.modules.vm_envs.get(index as usize) else {
            return Err(Self::no_vm_module());
        };
        if konst {
            scope.declare_const(name, value);
        } else {
            scope.declare(name, value);
        }
        Ok(())
    }

    /// Whether `name` resolves in VM module `index`'s environment.
    pub(crate) fn vm_module_exists(&mut self, index: u32, name: &str) -> bool {
        self.in_vm_module(index, |this| {
            this.module_imports.contains_key(name) || crate::nbvm::VmHost::global_exists(this, name)
        })
        .unwrap_or(false)
    }

    /// `for-in` over a module namespace from VM code: a binding in its
    /// temporal dead zone throws before any key is produced.
    pub(crate) fn vm_for_in_keys(&mut self, obj: NanBox) -> Result<NanBox, ExecError> {
        if let Some(raw) = obj.as_handle() {
            self.namespace_enumeration_tdz(crate::heap::Handle::from_raw(raw))?;
        }
        let keys = self.iterate_keys(obj);
        Ok(NanBox::handle(self.realm.new_array(keys).to_raw()))
    }

    /// `super[name] = v` from VM code whose receiver is a module namespace:
    /// unless an accessor on the home object's chain takes the write, the
    /// receiver's `[[DefineOwnProperty]]` runs — forcing a deferred namespace
    /// and checking the binding's TDZ.
    pub(crate) fn vm_super_set_namespace(
        &mut self,
        home: crate::heap::Handle,
        name: &str,
        receiver: NanBox,
    ) -> Result<(), ExecError> {
        let Some(th) = receiver.as_handle().map(crate::heap::Handle::from_raw) else {
            return Ok(());
        };
        if !self.realm.is_host_exotic(th) {
            return Ok(());
        }
        let mut cur = Some(home);
        while let Some(c) = cur {
            if self.realm.accessor(c, name).is_some() {
                return Ok(());
            }
            if self.realm.has_own(c, name) {
                break;
            }
            cur = self.realm.object_proto(c);
        }
        self.trigger_deferred_namespace(th, name)?;
        self.namespace_binding_tdz(th, name)
    }

    /// `import.meta` of VM module `index`.
    pub(crate) fn vm_import_meta(&mut self, index: u32) -> Option<NanBox> {
        let key = self.modules.vm_keys.get(index as usize)?.clone();
        Some(self.module_meta(&key))
    }

    /// A dynamic import from VM code (see `crate::nbvm::VmHost::dynamic_import`).
    pub(crate) fn vm_dynamic_import(
        &mut self,
        index: u32,
        spec: NanBox,
        options: Option<NanBox>,
        phase: u8,
    ) -> Result<NanBox, ExecError> {
        let referrer = match self.modules.vm_keys.get(index as usize) {
            Some(k) => Some(k.clone()),
            None => self.current_module_key(),
        };
        match phase {
            0 => Ok(self.dynamic_import_from(referrer, spec, options)),
            1 => {
                let promise = self.fresh_promise();
                let host = FileModuleHost;
                let resolved: Result<String, ExecError> = (|this: &mut Self| {
                    let spec_str = this.coerce_to_string(spec)?;
                    host.resolve(&spec_str, referrer.as_deref())
                        .map_err(|e| this.type_error(&e))
                })(self);
                match resolved {
                    Ok(dep) => self.start_dynamic_import_deferred(&dep, promise),
                    Err(e) => self.reject_dynamic_import(promise, e),
                }
                Ok(NanBox::handle(promise.to_raw()))
            }
            _ => {
                // `import.source(x)`: ToString the specifier (a throw rejects
                // with it), then reject with a SyntaxError.
                let p = self.fresh_promise();
                let rejection = match self.coerce_to_string(spec) {
                    Ok(_) => {
                        let m = self.new_str("source-phase / deferred import is not supported");
                        self.make_error(N_SYNTAX_ERROR, Some(m))
                    }
                    Err(ExecError::Throw(t)) => t,
                    Err(other) => return Err(other),
                };
                self.settle(p, rejection, false);
                Ok(NanBox::handle(p.to_raw()))
            }
        }
    }
}

/// Whether a module body's top-level statements contain a reachable `await` (or
/// `for await`) — i.e. this is an async (top-level-await) module. Unlike the
/// async-function detector, this also descends into an `export`'s inner
/// declaration (`export const x = await f()`), which is otherwise opaque to the
/// statement walker.
fn module_body_has_await(stmts: &[Stmt]) -> bool {
    stmts.iter().any(stmt_has_tla)
}

/// [`stmt_has_await`], extended to the places that walker
/// leaves to its eager path but that still make a module asynchronous
/// (`[[HasTLA]]`): binding-pattern defaults (`let { x = await y } = …`, a
/// `catch` parameter, a `for` head) and a class heritage (`class C extends
/// f(await x)`).
fn stmt_has_tla(s: &Stmt) -> bool {
    use crate::ast::{ForInit, ForLeft};
    let any = |b: &[Stmt]| b.iter().any(stmt_has_tla);
    stmt_has_await(s)
        || match s {
            Stmt::Export(ExportDecl::Decl { declaration, .. })
            | Stmt::Export(ExportDecl::Default { declaration, .. }) => stmt_has_tla(declaration),
            Stmt::Class(c) => c.super_class.as_deref().is_some_and(expr_awaits),
            Stmt::Expr { expression, .. } => expr_has_tla(expression),
            Stmt::Var(d) => d
                .declarations
                .iter()
                .any(|x| pattern_has_tla(&x.target) || x.init.as_ref().is_some_and(expr_has_tla)),
            Stmt::Block { body, .. } => any(body),
            Stmt::If {
                consequent,
                alternate,
                ..
            } => stmt_has_tla(consequent) || alternate.as_deref().is_some_and(stmt_has_tla),
            // An `await using` head awaits its disposal.
            Stmt::For { init, body, .. } => {
                matches!(init, Some(ForInit::Var(d)) if d.kind == crate::ast::VarDeclKind::AwaitUsing
                    || d.declarations.iter().any(|x| pattern_has_tla(&x.target)))
                    || stmt_has_tla(body)
            }
            Stmt::ForIn { left, body, .. } | Stmt::ForOf { left, body, .. } => {
                matches!(left, ForLeft::Decl { kind, target, .. }
                    if *kind == crate::ast::VarDeclKind::AwaitUsing || pattern_has_tla(target))
                    || stmt_has_tla(body)
            }
            Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => stmt_has_tla(body),
            Stmt::Labeled { body, .. } | Stmt::With { body, .. } => stmt_has_tla(body),
            Stmt::Switch { cases, .. } => cases.iter().any(|c| any(&c.body)),
            Stmt::Try {
                block,
                handler,
                finalizer,
                ..
            } => {
                any(block)
                    || handler.as_ref().is_some_and(|h| {
                        h.param.as_ref().is_some_and(pattern_has_tla) || any(&h.body)
                    })
                    || finalizer.as_deref().is_some_and(any)
            }
            _ => false,
        }
}

/// What [`stmt_has_tla`] adds for an expression: a class expression's
/// heritage.
fn expr_has_tla(e: &crate::ast::Expr) -> bool {
    match e {
        crate::ast::Expr::Class(c) => c.super_class.as_deref().is_some_and(expr_awaits),
        _ => false,
    }
}

/// Whether evaluating `e` may `await` (per [`stmt_has_tla`]).
fn expr_awaits(e: &crate::ast::Expr) -> bool {
    let s = Stmt::Expr {
        expression: alloc::boxed::Box::new(e.clone()),
        span: crate::common::Span::point(0),
    };
    stmt_has_await(&s) || expr_has_tla(e)
}

/// Whether a binding pattern's defaults or computed keys hold an `await`.
fn pattern_has_tla(t: &crate::ast::BindingTarget) -> bool {
    use crate::ast::{ArrayPatternElement, BindingTarget, PropertyKey};
    let has = expr_awaits;
    match t {
        BindingTarget::Ident(_) => false,
        BindingTarget::Array(p) => p.elements.iter().any(|el| match el {
            ArrayPatternElement::Hole => false,
            ArrayPatternElement::Item {
                target, default, ..
            } => pattern_has_tla(target) || default.as_ref().is_some_and(has),
            ArrayPatternElement::Rest { target, .. } => pattern_has_tla(target),
        }),
        BindingTarget::Object(p) => {
            p.properties.iter().any(|prop| {
                matches!(&prop.key, PropertyKey::Computed(k) if has(k))
                    || pattern_has_tla(&prop.value)
                    || prop.default.as_ref().is_some_and(has)
            }) || p.rest.as_deref().is_some_and(pattern_has_tla)
        }
    }
}

/// What kind of slot an import binding maps to.
enum ImportKind {
    Default,
    Namespace,
    Named(String),
    Source,
}

/// A dependency key paired with the `(local name, kind)` bindings an import from
/// it introduces — a flattened, borrow-free view built during linking.
struct DepBinds {
    dep: String,
    binds: Vec<(String, ImportKind)>,
    deferred: bool,
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
/// attribute. A JSON / text / bytes module is keyed by `<path>\0type=<t>` so the
/// same file imported both as JavaScript and as JSON/text is two distinct module
/// records (the spec keys the module map by `(specifier, attributes)`). A
/// plain JavaScript import keeps the bare resolved path as its key.
fn module_map_key(resolved: &str, type_attr: Option<&str>) -> String {
    match type_attr {
        Some(t @ ("json" | "text" | "bytes")) => alloc::format!("{resolved}\u{0}type={t}"),
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

// --- top-level `await` detection ---

/// Whether a statement contains a reachable `await` (or `for await`, or an
/// `await using` declaration) not nested inside a function/class boundary.
/// (`yield` counts too; it cannot occur at a module's top level.)
fn stmt_has_await(s: &Stmt) -> bool {
    match s {
        Stmt::Expr { expression, .. } => expr_has_await(expression),
        Stmt::Block { body, .. } => body.iter().any(stmt_has_await),
        Stmt::Empty { .. }
        | Stmt::Break { .. }
        | Stmt::Continue { .. }
        | Stmt::Debugger { .. }
        | Stmt::Function(_)
        | Stmt::Import(_)
        | Stmt::Export(_) => false,
        // A class declaration with a `yield`-bearing computed member key must be
        // driven through the machine so the key suspends (`class C { get [yield](){} }`).
        Stmt::Class(c) => class_computed_key_has_await(c),
        // An `await using` declaration is itself a suspension point of the async
        // coroutine: leaving the scope it belongs to performs `DisposeResources`,
        // whose step 4 `Await`s even when every resource was `null`/`undefined`
        // (so there was nothing to call). The declaration must therefore be lowered
        // into the machine — otherwise its whole enclosing block runs in one shot
        // through the eager walker, which cannot suspend, and the statements after
        // the block wrongly observe the same microtask.
        Stmt::Var(decl) => {
            matches!(decl.kind, crate::ast::VarDeclKind::AwaitUsing)
                || decl
                    .declarations
                    .iter()
                    .any(|d| d.init.as_ref().is_some_and(expr_has_await))
        }
        Stmt::If {
            test,
            consequent,
            alternate,
            ..
        } => {
            expr_has_await(test)
                || stmt_has_await(consequent)
                || alternate.as_deref().is_some_and(stmt_has_await)
        }
        Stmt::For {
            init,
            test,
            update,
            body,
            ..
        } => {
            init.as_ref().is_some_and(|i| match i {
                crate::ast::ForInit::Var(d) => d
                    .declarations
                    .iter()
                    .any(|x| x.init.as_ref().is_some_and(expr_has_await)),
                crate::ast::ForInit::Expr(e) => expr_has_await(e),
            }) || test.as_deref().is_some_and(expr_has_await)
                || update.as_deref().is_some_and(expr_has_await)
                || stmt_has_await(body)
        }
        // A `for await` loop is itself a suspension point of the async coroutine
        // (each iterated value is `await`ed), so it must be lowered into the
        // machine even when its operand and body are otherwise suspension-free.
        Stmt::ForOf {
            right,
            body,
            is_await: true,
            ..
        } => {
            let _ = (right, body);
            true
        }
        Stmt::ForIn {
            left, right, body, ..
        }
        | Stmt::ForOf {
            left, right, body, ..
        } => {
            // A yield can also hide in an assignment-target pattern's default/key
            // (`for ([ x = yield ] of …)`), which is bound per iteration.
            matches!(left, crate::ast::ForLeft::Target(e) if expr_has_await(e))
                || expr_has_await(right)
                || stmt_has_await(body)
        }
        Stmt::While { test, body, .. } => expr_has_await(test) || stmt_has_await(body),
        Stmt::DoWhile { body, test, .. } => stmt_has_await(body) || expr_has_await(test),
        Stmt::Switch {
            discriminant,
            cases,
            ..
        } => {
            expr_has_await(discriminant)
                || cases.iter().any(|c| {
                    c.test.as_ref().is_some_and(expr_has_await) || c.body.iter().any(stmt_has_await)
                })
        }
        Stmt::Try {
            block,
            handler,
            finalizer,
            ..
        } => {
            block.iter().any(stmt_has_await)
                || handler
                    .as_ref()
                    .is_some_and(|h| h.body.iter().any(stmt_has_await))
                || finalizer
                    .as_ref()
                    .is_some_and(|f| f.iter().any(stmt_has_await))
        }
        Stmt::Return { argument, .. } => argument.as_deref().is_some_and(expr_has_await),
        Stmt::Throw { argument, .. } => expr_has_await(argument),
        Stmt::Labeled { body, .. } => stmt_has_await(body),
        Stmt::With { object, body, .. } => expr_has_await(object) || stmt_has_await(body),
    }
}

/// Whether an expression may `await` (or `yield`) in the enclosing context.
/// Stops at nested function/arrow/class boundaries (their bodies have their own
/// context).
fn expr_has_await(e: &crate::ast::Expr) -> bool {
    match e {
        crate::ast::Expr::Yield { .. } => true,
        // Boundaries: a nested function/arrow introduces its own context.
        crate::ast::Expr::Function(_) | crate::ast::Expr::Arrow(_) => false,
        // A class body is a boundary (method bodies, field initializers have their
        // own context), EXCEPT a *computed member key* (`class { get [yield]() {} }`)
        // which is evaluated in the enclosing generator context at definition time.
        crate::ast::Expr::Class(c) => class_computed_key_has_await(c),
        crate::ast::Expr::Null(_)
        | crate::ast::Expr::Bool { .. }
        | crate::ast::Expr::Number { .. }
        | crate::ast::Expr::BigInt { .. }
        | crate::ast::Expr::Str { .. }
        | crate::ast::Expr::Regex { .. }
        | crate::ast::Expr::Ident(_)
        | crate::ast::Expr::PrivateName(..)
        | crate::ast::Expr::This(_)
        | crate::ast::Expr::Super(_)
        | crate::ast::Expr::NewTarget(_) => false,
        crate::ast::Expr::Template(t) => t.expressions.iter().any(expr_has_await),
        crate::ast::Expr::TaggedTemplate { tag, quasi, .. } => {
            expr_has_await(tag) || quasi.expressions.iter().any(expr_has_await)
        }
        crate::ast::Expr::Array { elements, .. } => elements.iter().any(|el| match el {
            crate::ast::ArrayElement::Hole => false,
            crate::ast::ArrayElement::Item(e) | crate::ast::ArrayElement::Spread(e) => {
                expr_has_await(e)
            }
        }),
        crate::ast::Expr::Object { members, .. } => members.iter().any(|m| match m {
            crate::ast::ObjectMember::Property { key, value, .. } => {
                key_has_await(key) || expr_has_await(value)
            }
            crate::ast::ObjectMember::Spread { value, .. } => expr_has_await(value),
            // An accessor's function body is a boundary, but its *computed* key
            // (`get [yield]()`) is evaluated in the enclosing generator context.
            crate::ast::ObjectMember::Accessor { key, .. } => key_has_await(key),
        }),
        crate::ast::Expr::Member {
            object, property, ..
        } => expr_has_await(object) || key_has_await(property),
        crate::ast::Expr::Call {
            callee, arguments, ..
        }
        | crate::ast::Expr::New {
            callee, arguments, ..
        } => {
            expr_has_await(callee)
                || arguments.iter().any(|a| match a {
                    crate::ast::Argument::Item(e) | crate::ast::Argument::Spread(e) => {
                        expr_has_await(e)
                    }
                })
        }
        // `await` is itself a suspension point of the *async* coroutine machine
        // (the same explicit-stack engine drives async functions). It can only
        // appear inside an async function, so treating it as a suspension point
        // unconditionally is correct: a plain `function*` body never contains a
        // top-level `await` (a nested async arrow's `await` is past a function
        // boundary, which this walker already stops at).
        crate::ast::Expr::Await { .. } => true,
        crate::ast::Expr::OptChain { expr, .. } => expr_has_await(expr),
        crate::ast::Expr::Unary { argument, .. } | crate::ast::Expr::Update { argument, .. } => {
            expr_has_await(argument)
        }
        crate::ast::Expr::Binary { left, right, .. }
        | crate::ast::Expr::Logical { left, right, .. } => {
            expr_has_await(left) || expr_has_await(right)
        }
        crate::ast::Expr::Conditional {
            test,
            consequent,
            alternate,
            ..
        } => expr_has_await(test) || expr_has_await(consequent) || expr_has_await(alternate),
        crate::ast::Expr::Assign { target, value, .. } => {
            expr_has_await(target) || expr_has_await(value)
        }
        crate::ast::Expr::Sequence { expressions, .. } => expressions.iter().any(expr_has_await),
    }
}

fn key_has_await(k: &crate::ast::PropertyKey) -> bool {
    matches!(k, crate::ast::PropertyKey::Computed(e) if expr_has_await(e))
}

/// Whether any of a class's *computed member keys* (`[expr]` on a method or
/// field) contains a `yield`/`await` reachable in the enclosing generator/async
/// context. Method bodies, field initializers, and the `extends` heritage are
/// their own contexts / left to the eager fallback, so only the member keys are
/// examined here.
fn class_computed_key_has_await(c: &crate::ast::Class) -> bool {
    c.body.iter().any(|m| match m {
        crate::ast::ClassMember::Method(m) => key_has_await(&m.key),
        crate::ast::ClassMember::Field(f) => key_has_await(&f.key),
        crate::ast::ClassMember::StaticBlock { .. } => false,
    })
}
