//! Central, tunable resource limits for the engine.
//!
//! Every cap that bounds untrusted work — recursion depths, allocation sizes,
//! the regex step budget, WebAssembly fuel — lives here in one [`Limits`] struct
//! with safe defaults equal to the engine's historical hard-coded constants. An
//! embedder overrides them by constructing a [`Realm`](crate::realm::Realm) with
//! [`Realm::with_limits`](crate::realm::Realm::with_limits) (or via the
//! `*_with_limits` entry points in [`nbvm`](crate::nbvm) /
//! [`nbexec`](crate::nbexec)).
//!
//! ## What is live vs. default-sourced
//! Limits enforced on a path that has a `Realm` in scope — the call/handler/JSON
//! depths, string/array/BigInt sizes, the object→dictionary threshold, and the
//! WebAssembly limits — are read **live** from `realm.limits`, so overriding them
//! takes effect at runtime. A few caps are enforced in standalone, pre-`Realm`
//! code (the JS parser depth, the regex engine, the rope's hard length ceiling);
//! those read the `DEFAULT_*` constants below so the canonical value still lives
//! in one place, but they are fixed at build time rather than per-`Realm`.
//!
//! This module is `no_std`-friendly (plain `Copy` data, no allocation).

/// Default maximum JS-level call/recursion depth before a `RangeError`.
pub const DEFAULT_MAX_CALL_DEPTH: usize = 3500;
/// Default maximum *tree-walk* eval/exec recursion depth before a `RangeError`.
///
/// The `nbexec` interpreter descends the native stack once per nested
/// expression/statement, and each level consumes far more native stack than a
/// bytecode call frame does. This cap is therefore set well below
/// [`DEFAULT_MAX_CALL_DEPTH`] so the guard fires (throwing a catchable
/// `RangeError`) before a small (~2 MB) host stack overflows and aborts.
///
/// The value is empirically chosen: on a 2 MB stack an optimized build of the
/// interpreter overflows on a left-deep `1+1+…+1` expression at roughly depth
/// 3700, so a cap of 1500 keeps a comfortable (~2.5×) safety margin while still
/// permitting genuinely deep nesting. It does not regress test262/conformance,
/// since deep *call* recursion resets the eval-depth counter at each function
/// frame (and is bounded separately by [`DEFAULT_MAX_CALL_DEPTH`]).
pub const DEFAULT_MAX_EVAL_DEPTH: usize = 1500;
/// Default maximum try/catch handler-stack depth.
pub const DEFAULT_MAX_HANDLER_DEPTH: usize = 100_000;
/// Default maximum `JSON.parse`/`JSON.stringify` nesting depth.
pub const DEFAULT_MAX_JSON_DEPTH: usize = 2000;
/// Default maximum recursion depth when stringifying a value for display.
pub const DEFAULT_MAX_DISPLAY_DEPTH: usize = 1000;
/// Default maximum string length (code units) before `RangeError`.
pub const DEFAULT_MAX_STRING_LEN: usize = 1 << 30;
/// Default maximum dense array / typed-array / `ArrayBuffer` length.
pub const DEFAULT_MAX_ARRAY_LEN: usize = 100_000_000;
/// Default maximum BigInt magnitude in bits before `RangeError`.
pub const DEFAULT_MAX_BIGINT_BITS: u64 = 1 << 30;
/// Default own-property count past which an object switches to dictionary mode.
pub const DEFAULT_OBJECT_DICTIONARY_THRESHOLD: usize = 128;
/// Default maximum parser recursion depth (SyntaxError past this).
pub const DEFAULT_MAX_PARSE_DEPTH: u32 = 300;
/// Default base of the regex backtracking step budget (per find operation;
/// scaled by the subject length at run time).
pub const DEFAULT_REGEX_STEP_BASE: u64 = 300_000;
/// Default maximum regex backtracking recursion depth.
pub const DEFAULT_REGEX_MAX_DEPTH: u32 = 2_000;
/// Default maximum regex *pattern* parser nesting depth (groups/lookaround).
pub const DEFAULT_REGEX_MAX_PARSE_DEPTH: u32 = 300;
/// Default maximum accepted `{n,m}` quantifier bound.
pub const DEFAULT_REGEX_MAX_QUANT: usize = 1_000_000;
/// Default maximum compiled regex program size (instructions).
pub const DEFAULT_REGEX_MAX_PROG_SIZE: usize = 100_000;

/// Default WebAssembly call-frame + block-nesting depth.
pub const DEFAULT_WASM_MAX_CALL_DEPTH: u32 = 1024;
/// Default WebAssembly fuel: instructions a single call may execute before it
/// traps with "out of fuel". `Some` by default so an infinite `loop`/`br 0`
/// terminates out of the box; set to `None` to disable (unbounded).
pub const DEFAULT_WASM_FUEL: Option<u64> = Some(1_000_000_000);
/// Default WebAssembly maximum linear-memory pages (4 GiB).
pub const DEFAULT_WASM_MAX_MEM_PAGES: u32 = 0x1_0000;
/// Default WebAssembly maximum declared table elements.
pub const DEFAULT_WASM_MAX_TABLE_ELEMS: u32 = 10_000_000;
/// Default WebAssembly maximum declared locals per function.
pub const DEFAULT_WASM_MAX_LOCALS: u32 = 50_000;

/// WebAssembly-specific resource limits (the subset threaded into the wasm
/// runtime, which has no `Realm`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WasmLimits {
    /// Maximum call-frame + structured-block nesting depth (native-stack guard).
    pub max_call_depth: u32,
    /// Instruction budget per call; `None` disables fuel metering (unbounded).
    pub fuel: Option<u64>,
    /// Maximum declared linear-memory pages.
    pub max_mem_pages: u32,
    /// Maximum declared table elements.
    pub max_table_elems: u32,
    /// Maximum declared locals per function.
    pub max_locals: u32,
}

impl Default for WasmLimits {
    fn default() -> Self {
        Self {
            max_call_depth: DEFAULT_WASM_MAX_CALL_DEPTH,
            fuel: DEFAULT_WASM_FUEL,
            max_mem_pages: DEFAULT_WASM_MAX_MEM_PAGES,
            max_table_elems: DEFAULT_WASM_MAX_TABLE_ELEMS,
            max_locals: DEFAULT_WASM_MAX_LOCALS,
        }
    }
}

/// Tunable resource limits for a [`Realm`](crate::realm::Realm). `Default`
/// reproduces the engine's historical behavior; see the module docs for which
/// fields are enforced live vs. at build time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    /// Maximum JS call/recursion depth before a `RangeError` (live).
    pub max_call_depth: usize,
    /// Maximum tree-walk `eval`/`exec` recursion depth before a `RangeError`
    /// (live). Bounds how deeply the `nbexec` interpreter may recurse on the
    /// native stack for nested expressions/statements. Kept separate from (and
    /// below) [`Self::max_call_depth`] because each tree-walk level consumes
    /// much more native stack than a bytecode call frame, so the JS call-depth
    /// cap is too high to protect a small host stack on this path.
    pub max_eval_depth: usize,
    /// Maximum try/catch handler-stack depth (live).
    pub max_handler_depth: usize,
    /// Maximum `JSON.parse`/`stringify` nesting depth (live).
    pub max_json_depth: usize,
    /// Maximum display/`toString` recursion depth (live).
    pub max_display_depth: usize,
    /// Maximum string length in code units (live).
    pub max_string_len: usize,
    /// Maximum dense array / typed-array / `ArrayBuffer` length (live).
    pub max_array_len: usize,
    /// Maximum BigInt magnitude in bits (live).
    pub max_bigint_bits: u64,
    /// Own-property count past which an object converts to dictionary mode,
    /// bounding shape-transition-tree growth (live).
    pub object_dictionary_threshold: usize,
    /// WebAssembly limits (live: threaded into the wasm runtime).
    pub wasm: WasmLimits,
}

// The JS-parser, regex-engine, and rope caps run in standalone code that has no
// `Realm` in scope, so they are not per-`Realm` fields; their canonical values
// are the `DEFAULT_MAX_PARSE_DEPTH` / `DEFAULT_REGEX_*` / `DEFAULT_MAX_STRING_LEN`
// constants above, which those modules reference directly. Keeping every cap's
// default in this one file preserves a single source of truth.

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_call_depth: DEFAULT_MAX_CALL_DEPTH,
            max_eval_depth: DEFAULT_MAX_EVAL_DEPTH,
            max_handler_depth: DEFAULT_MAX_HANDLER_DEPTH,
            max_json_depth: DEFAULT_MAX_JSON_DEPTH,
            max_display_depth: DEFAULT_MAX_DISPLAY_DEPTH,
            max_string_len: DEFAULT_MAX_STRING_LEN,
            max_array_len: DEFAULT_MAX_ARRAY_LEN,
            max_bigint_bits: DEFAULT_MAX_BIGINT_BITS,
            object_dictionary_threshold: DEFAULT_OBJECT_DICTIONARY_THRESHOLD,
            wasm: WasmLimits {
                max_call_depth: DEFAULT_WASM_MAX_CALL_DEPTH,
                fuel: DEFAULT_WASM_FUEL,
                max_mem_pages: DEFAULT_WASM_MAX_MEM_PAGES,
                max_table_elems: DEFAULT_WASM_MAX_TABLE_ELEMS,
                max_locals: DEFAULT_WASM_MAX_LOCALS,
            },
        }
    }
}
