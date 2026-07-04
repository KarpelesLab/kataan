# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- *(promise)* `class P extends Promise {}` subclassing — `super(executor)` now
  runs the executor and backs the (ordinary-object) subclass instance with a
  hidden `[[PromiseState]]` cell that `promise_state` follows, so construction,
  the static combinators (`P.resolve`/`all`/`race`/`allSettled`/…), `then`/`catch`,
  `await`, and microtask delivery all work on a subclass instance (a non-callable
  executor is a TypeError). Species-based `.then` chaining is still pending
  (ROADMAP §3.6 Promise).

### Fixed

- *(string)* `String.prototype.split` runs `ToString(separator)` before the
  `limit === 0` short-circuit (a throwing separator `toString` throws even at
  limit 0), and the bytecode VM faults an object/Symbol/wrapper separator (or an
  object `limit`) to the tree-walker so a custom `toString`/`@@split` and RegExp
  delegation are honored — previously `"axbxc".split({toString:()=>"x"})` did not
  split under `nbvm` (ROADMAP §3.6, §6 two-engines-one-truth).
- *(string)* `String.raw` uses throwing `ToObject` on the template and on
  `Get(template, "raw")` (a `null`/`undefined` or a non-object `raw` throws
  TypeError), and reads `raw` via `Get` so an inherited getter fires and its
  throw propagates — previously all silently succeeded (ROADMAP §3.6).
- *(string)* `String.prototype.replace`/`replaceAll` `ToString` the replacement
  value and the function-replacer's result (a custom `toString`/`@@toPrimitive`
  runs and a thrown value propagates) instead of rendering `"[object Object]"`
  (ROADMAP §3.6).
- *(functions)* the `Function` constructor `ToString`s each argument
  (CreateDynamicFunction): a custom `toString`/`valueOf` runs and a thrown value
  propagates — `new Function({toString(){throw 1}}, "")` throws `1` instead of
  stringifying to `"[object Object]"` and failing to parse (ROADMAP §3.6).
- *(proxy)* a **computed** write (`o[k]=v`, `arr[i]=v`) whose own slot is absent
  now runs `parent.[[Set]]` — an inherited setter, or a proxy on the prototype
  chain, handles it (with Receiver = the object) — matching the dot-key
  `assign_member` path. So `Object.setPrototypeOf(arr, proxyWithSetTrap); arr[0]=v`
  fires the trap instead of silently creating an own index. Default-`%Array.
  prototype%` arrays and present indices keep the dense fast path (ROADMAP §3.6).
- *(with)* the `with` object-environment `HasBinding` is a proxy-aware
  `HasProperty`, so `with (new Proxy(o, {has(){…}})) { name }` consults the `has`
  trap (a trapless proxy forwards to its target) instead of throwing a spurious
  ReferenceError; `@@unscopables` still applies (ROADMAP §3.5).
- *(array)* sparse-array holes are absent to `in` / `HasProperty`: the proxy-
  aware `[[HasProperty]]` used a bare `i < length` test that reported a hole as
  present, so `1 in [2,,3]`, `0 in new Array(3)`, `[2,,3].hasOwnProperty(1)`,
  and the generic array-like iteration (`forEach`/`map`/… presence probe) all
  counted holes. Now hole-aware (via `has_own`). `Array.prototype.flat`/
  `flatMap` also skip holes (FlattenIntoArray uses HasProperty) (ROADMAP §3.6).
- *(array)* `Array.prototype.flat`/`flatMap` on a generic array-like skip absent
  indices per FlattenIntoArray's HasProperty check (so a poisoned getter past
  `length` is never read), and `flatMap` passes the source object as the
  callback's 3rd argument (ROADMAP §3.6).
- *(array)* `Array.prototype.flat`/`flatMap` build the result with
  `ArraySpeciesCreate(O, 0)` + `CreateDataPropertyOrThrow` — a non-constructor
  `@@species` (or a non-extensible/non-configurable target) now throws a
  TypeError, and a subclass species is honored (ROADMAP §3.6).
- *(array)* `Array.of` honors a constructor `this` (`Array.of.call(C, …)` /
  subclass `C.of(…)`) — `Construct(C, «len»)` + `CreateDataPropertyOrThrow` —
  instead of always building a plain `%Array%` (ROADMAP §3.7).
- *(proxy)* route HasProperty through the proxy `has` trap in generic
  array-like iteration, so `[...new Proxy([1,2,3],{})]`, `Array.from(proxy)`,
  and `Array.prototype.*.call(proxy, …)` observe the proxied elements instead
  of reading holes (ROADMAP §3.7).
- *(proxy)* object spread (`{...proxy}`, CopyDataProperties) enumerates a proxy
  source through its `ownKeys`/`getOwnPropertyDescriptor`/`get` traps (strings
  and symbols, enumerable only) instead of copying nothing (ROADMAP §3.7).
- *(proxy)* object-rest patterns (`{...rest}` in a binding, assignment target,
  function parameter, or generator step) run the same proxy-aware
  CopyDataProperties, sharing one `copy_data_properties` helper; this also fixes
  symbol own keys being dropped from object rest on ordinary objects.
- *(proxy)* `JSON.stringify` enumerates a proxy through its `ownKeys`/
  `getOwnPropertyDescriptor`/`get` protocol (a proxy over an array serializes as
  an array via `IsArray`), and `Object.getOwnPropertyDescriptors` drives the
  proxy's `[[GetOwnProperty]]` — both previously returned `{}` (ROADMAP §3.7).
- *(proxy)* `Reflect.ownKeys` on a trap-less proxy forwards `[[OwnPropertyKeys]]`
  to the target instead of returning `[]` (ROADMAP §3.7).
- *(proxy)* a proxy used as a write *target* (`Object.assign(new Proxy({},{}),
  …)`) runs its own `[[Set]]` (set trap / trapless forward) instead of the
  cell-level extensibility gate wrongly throwing "object is not extensible"
  (ROADMAP §3.6).
- *(proxy)* `Reflect.set` with a **proxy receiver** — the ubiquitous passthrough
  `set(t,k,v,r){ return Reflect.set(t,k,v,r) }` — now writes to the receiver via
  `[[DefineOwnProperty]]` per OrdinarySetWithOwnDescriptor (not `[[Set]]`, which
  recursed / hit the cell gate), so the write persists; non-writable/accessor
  targets reject and an ordinary receiver is unchanged (ROADMAP §3.6).

## [0.0.6](https://github.com/KarpelesLab/kataan/compare/v0.0.5...v0.0.6) - 2026-07-04

### Added

- *(embed)* host-function registration API — `Interp::register_fn` /
  `register_global_fn` + `Ctx` (ROADMAP §4.0 milestone 1): register a Rust
  closure as a first-class JS function with spec-shaped `name`/`length`, build
  values, read/write properties, throw catchable errors, coerce arguments, and
  re-enter JS. Backed by a new `Cell::HostFn` registry cell; self-reentrancy is
  a clean `TypeError`.

### Other

- *(roadmap)* plan the host-function registration / embedding API (§4.0)
- add CI, crates.io, docs.rs, and MIT license badges to README

## [0.0.5](https://github.com/KarpelesLab/kataan/compare/v0.0.4...v0.0.5) - 2026-07-03

### Added

- *(generator)* suspend on yield in for-of/for-in assignment-target patterns
- *(generator)* suspend on yield in a destructuring member-target key
- *(generator)* suspend on yield in destructuring-assignment defaults
- *(generator)* suspend on yield inside object literals
- *(generator)* suspend on yield inside array literals
- *(builtins)* Array and Object are real callable function cells
- *(modules)* import-defer — exotic-object edges, super/this fixes (+ general)
- *(modules)* import-defer proposal (core: static `import defer * as ns`)
- *(arraybuffer)* immutable-arraybuffer proposal
- *(promise)* Promise.allKeyed / allSettledKeyed (await-dictionary proposal)
- *(object)* Symbol/BigInt wrapper objects (ToObject)
- *(eval)* new.target valid in direct eval inside function code
- *(iterators)* real %RegExpStringIteratorPrototype% for matchAll
- *(iterators)* real per-kind iterator prototypes (Array/String/Map/Set)
- *(class)* TDZ for `this` in derived constructors (must call super)
- *(functions)* formal-parameter TDZ (self/forward default reference)
- *(functions)* separate parameter and body environments
- *(eval)* EvalDeclarationInstantiation param/arguments conflict
- *(eval)* direct eval inherits the caller's super (home object)
- *(eval)* completion values carried through break/continue
- *(eval)* proper statement completion values (UpdateEmpty)
- *(intl)* Intl.PluralRules conformance
- *(intl)* Intl.DurationFormat conformance (+ NumberFormat -0/option fixes)
- *(intl)* Intl.RelativeTimeFormat conformance
- *(intl)* Intl.ListFormat conformance
- *(class)* lexically-scoped private names (#x)
- *(modules)* module completeness pass (parser, async, namespace, early errors)
- *(modules)* ES module system + dynamic import()
- *(language)* explicit resource management (using / await using)
- *(builtins)* complete getOrInsert/getOrInsertComputed (upsert)
- *(builtins)* Promise.try (ES2025)
- *(builtins)* Error.prototype.stack accessor
- *(builtins)* FinalizationRegistry, WeakRef.prototype.deref, Symbol.prototype
- *(builtins)* Math.sumPrecise + fix no_std build
- *(regex)* early errors for inline modifier groups (?ims-ims:)
- *(array)* property descriptors on indices + length (ArraySetLength)
- *(builtins)* Error.isError (ES2025) with [[ErrorData]] brand
- *(regex)* reject invalid \p{…} property escapes as SyntaxError
- *(regex)* parse-time literal validation + structural early errors
- *(builtins)* Uint8Array base64/hex (ES2025)
- RegExp.escape + async yield* requires Object iterator
- *(async)* async yield* delegation (async-iterator protocol)
- *(builtins)* Array.fromAsync
- *(builtins)* batch 13 part 2b — Array/RegExp/TypedArray conformance
- *(builtins)* batch 13 part 2a — Promise/JSON/Object conformance
- *(builtins)* batch 12 part 2 — String/TypedArray/DataView/Date conformance
- *(builtins)* sparse-array holes + ES2025 Map/Set conformance
- *(nbexec)* true lazy async/await with microtask ordering
- *(nbexec)* wire lazy generators into call/dispatch/iteration
- *(nbexec)* true lazy generators — explicit-stack suspension engine
- *(nbexec)* DisposableStack, AsyncDisposableStack, ShadowRealm + SuppressedError
- *(intl)* real branded .prototype for every Intl service + Locale/DurationFormat
- *(intl)* realm prototype cache + dispatch wiring for Intl branding
- *(nbexec)* native subclassing — class extends Map/Array/typed/Date/…
- *(nbexec)* GetPrototypeFromConstructor for built-in constructors
- *(intl)* spec-accurate option validation + resolvedOptions for NumberFormat/DateTimeFormat
- *(intl)* add Intl.getCanonicalLocales + Intl.supportedValuesOf
- *(regex)* inline modifiers, v-flag unicodeSets, CI word/property + XID
- *(builtins)* Array/Function/arguments conformance sweep (batch 9 part 2)
- *(annexB)* escape/unescape, String HTML methods, Date legacy, RegExp compile + legacy statics
- *(builtins)* Object/Iterator/Proxy/Reflect conformance sweep (batch 8 part 2)
- *(typedarray)* detachArrayBuffer native + ArrayBuffer/DataView/TypedArray conformance
- *(regexp)* ES2022 `d`-flag match indices (MakeIndicesArray)
- *(regexp)* first-class RegExp.prototype + spec-compliant exec/symbol methods (nbexec)
- *(builtins)* TypedArray of/from prototype link + spec-faithful from
- *(builtins)* spec-compliant TypedArray set + subarray (offset ToInteger, species)
- *(builtins)* TypedArraySpeciesCreate for map/filter/slice; fix construct return
- *(builtins)* DataView/TypedArray/ArrayBuffer prototype & ctor conformance
- *(class)* support extending an ordinary function (class X extends fn)
- *(super)* computed super member access, assignment, and delete
- *(assignment)* NamedEvaluation for plain identifier assignment
- *(global)* unify global-object properties and global var bindings
- *(class)* own name/length + static members as real own properties
- *(builtins)* Promise/Map/Set/Function/Reflect/Proxy sweep + regression cleanup (batch 6)
- *(with)* implement the with statement (object environment record)
- *(destructuring)* NamedEvaluation for binding/param defaults
- *(class)* inherit methods via prototype + VM prototype-chain reads
- *(class)* materialize class .prototype with methods/accessors
- *(builtins)* Array/Object/TypedArray conformance sweep (batch 5)
- *(date)* Date.prototype[Symbol.toPrimitive] + OrdinaryToPrimitive helper
- *(iterator)* add the Iterator global, %IteratorPrototype%, and helpers
- *(typed-arrays)* add BigInt64Array and BigUint64Array
- *(nbexec)* real eval() and the Function constructor (dynamic code)
- *(regex)* reject quantifiers on non-quantifiable assertions
- *(regex)* SyntaxError for invalid group names, duplicate names, bad refs, and u-mode strictness
- *(parser)* optional-chain tagged template + new.target placement
- *(parser)* destructuring-target early errors
- *(parser)* illegal break/continue/return early errors
- *(parser)* invalid assignment/update targets + strict binding names
- *(parser)* early-error validation pass for class & private-name rules
- *(regex)* full Unicode property escapes \p{…}/\P{…} (u-flag)
- *(typedarray)* add shared %TypedArray% intrinsic constructor hierarchy
- *(engine)* typed-error API for conformance checking
- *(limits)* add dedicated max_eval_depth knob (C2 follow-up)

### Fixed

- *(yield*)* propagate an abrupt IteratorClose when forwarding throw to a throw-less iterator
- *(yield*)* pass the inner result object through; don't read value when incomplete
- *(super)* RequireObjectCoercible on the super base for object methods
- *(proxy)* forward trapless has/delete to a proxy target recursively
- *(__proto__)* validate the __proto__ setter like SetPrototypeOf
- *(Object.prototype)* ToObject(this) in hasOwnProperty/propertyIsEnumerable/isPrototypeOf
- *(JSON.parse)* reject unescaped control characters in strings
- *(compound-assign)* capture the identifier reference before the RHS
- *(with)* lexically scope the with-statement object environment
- *(parser)* allow `let` as identifier in for-in left-hand side
- *(promise)* Promise.resolve/reject use NewPromiseCapability for custom C
- *(parser)* array/object literal contents are [+In] inside a for-header
- *(regex)* properties of strings are an early error in invalid positions
- *(parser/eval)* strict-reserved words as sloppy ident refs; direct eval inherits caller strictness
- *(Object.assign)* spec CopyDataProperties — key order, enumerability, errors
- *(Object.assign)* Set with Throw + read-only string indices
- *(class)* PrivateSet on a private method/accessor throws TypeError
- *(assignment)* ToPropertyKey before RHS in compound member assignment
- *(annexB)* spec-correct __defineGetter__/Setter & __lookupGetter__/Setter
- *(iterator)* tag boxed String wrapper's @@iterator prototype
- *(class)* evaluate computed member keys eagerly at class definition
- *(nbvm)* no-init `var` re-declaration must not clobber the binding
- *(with)* bare-name delete + NaN/undefined/Infinity resolve via with-object
- *(regex)* reject \p{…} property class as a range endpoint under u
- *(regex)* reject character-class range with low > high
- *(regex)* reject braced quantifier with nothing to repeat (Annex B)
- *(annexB/global)* preserve existing global binding + non-configurable decls
- *(annexB)* B.3.4 if/else function gets its own block binding
- *(annexB)* B.3.3 block-function hoisting through catch + switch
- *(array)* spec-conformant Array.prototype.concat
- *(async)* await adopts thenables (PromiseResolve semantics)
- *(async)* class async methods + async-generator promise results
- *(class)* super in static/field initializers + static field/method redecl
- *(annexB)* spec-accurate B.3.3/B.3.4 block-level function semantics
- *(nbvm)* route `for await` to the tree-walker (no bytecode await machinery)
- *(destructuring,with)* array GetIterator honors Symbol.iterator; with-statement var initializer
- *(operators,names)* ToNumber via ToPrimitive for update/unary-plus; accessor & class property names
- *(fn)* scope %ThrowTypeError% caller/arguments poison to restricted functions
- *(intl)* spec-accurate useGrouping + currency/unit/fraction-digit validation
- *(nbexec)* computed property keys use ToPropertyKey (throws on uncoercible)
- *(nbexec)* computed-key method call on a primitive boxes the receiver
- *(nbexec)* with-binding call this + method-call nullish-base order
- *(nbexec)* arrow functions capture lexical this/new.target/home at definition
- *(nbexec)* object-literal __proto__ primitive ignored + super getter receiver this
- *(nbexec)* BigInt operators unwrap object operands before type-mixing check
- *(nbexec)* spec evaluation order for compound assignment to member targets
- *(nbexec)* unresolvable compound-assignment / identifier read throws ReferenceError
- *(nbexec)* destructuring-assignment target reference evaluated before iterator step
- *(nbexec)* object-destructuring boxing + delete on globals/non-references
- *(nbexec)* spec-correct IteratorClose + array-destructuring iterator protocol
- *(class)* anonymous class NamedEvaluation sets name before static init
- *(class)* static field initializers run deferred, in order, with this=class
- *(class)* private methods/accessors are shared per class, not per instance
- *(class)* class bodies are strict code
- *(class)* validate non-identifier extends clauses too
- *(class)* TypeError when extends value is not a constructor or null
- *(scope)* sloppy implicit global binds on global scope, not current
- *(destructuring)* object rest invokes own getters once into data props
- *(class)* TypeError on private accessor missing the needed half
- *(object)* name Symbol-keyed concise methods ([desc] / empty)
- *(class)* private-name internal slots, static name/length override, constructor return semantics
- *(regexp)* build the spec string-method helpers without the `regex` feature
- *(intl)* make f16 binary16 helpers no_std-compatible (DataView Float16)
- *(builtins)* Array/TypedArray toLocaleString invokes element toLocaleString
- *(builtins)* preserve original-object identity for generic Array callbacks
- *(builtins)* Array generic ToObject(this) + sort comparefn IsCallable
- *(nbexec)* make f16 conversions core-friendly; doc-list indent
- *(builtins)* IsCallable(callbackfn) guard for Array/TypedArray iteration methods
- *(builtins)* abrupt-propagating coercions in TypedArray/Array index methods
- *(super)* arrows inherit the enclosing method's super binding
- *(assignment)* TypeError (not Unsupported) writing a member of null/undefined
- *(object)* clear attribute flags on property delete
- *(class)* reserve per-id side-table slots before evaluating members
- *(mem)* cap Array.from/TypedArray.from array-like length; ulimit guard in runner
- *(with)* hoist var declarations out of a with body
- *(functions)* name/length as real own properties
- *(destructuring)* NamedEvaluation for assignment-pattern defaults (VM)
- *(class)* propagate throws from computed static member keys
- *(functions)* bind arguments before parameter defaults
- *(class)* expose class .name on the bytecode path
- *(parser)* close parse-phase conformance gaps (templates, comments, private-in)
- *(no_std)* keep clz32/imul/split/trunc paths std-free; add trunc_toward_zero
- *(string)* replace/replaceAll/match pattern args ToString fallibly
- *(string)* split limit ToUint32, undefined separator, empty-string cases
- *(string)* String.raw handles array-like template.raw with ToLength + ToString
- *(string)* search/normalize fallible arg ToString, normalize.length=0
- *(string)* fallible ToString/ToInteger coercion for needle/index/pad args
- *(string)* fallible index coercion for slice/substring/substr/indexOf; fromCharCode/fromCodePoint ToNumber + RangeError
- *(string)* position arg ToIntegerOrInfinity + IsRegExp guard, primitive inherited methods
- *(string)* String.prototype methods coerce this (ToString), thisStringValue; new Object(v)
- *(date)* user-overridden proto methods, Date.parse human formats + TimeClip, now.length
- *(date)* generic toJSON, Date/Date.UTC length=7, setters coerce primary arg always
- *(date)* Date.UTC coercion, instance prototype chain, negative/expanded years, day rollover
- *(date)* NaN getters return NaN, correct method .length values
- *(date)* real Date.prototype, set* arg coercion order, TimeClip, constructor coercion
- *(math)* constants as own props, toStringTag, clz32/imul ToUint32/ToInt32, max/min/hypot full coercion, round large ints, add f16round
- *(number)* prototype chain, constants as own props, toString/toFixed/toExponential/toPrecision
- *(bigint)* spec-accurate BigInt() coercion, asIntN/asUintN, toString radix, thisBigIntValue, IsConstructor
- *(parser)* four for-head / labelled-function over-rejection regressions
- *(builtins)* TypedArray array-like ctor + 13 descriptor regressions
- *(destructuring)* lazy iterator + coercion for assignment patterns
- *(iteration)* drive built-in for-of/spread via a proper iterables path
- *(parser)* catch-lexical, private object keys, yield-star ASI, static-block await
- *(parser)* delete-identifier, arrow ASI, with-body, and accessor-arity early errors
- *(parser)* object-literal, parameter, and catch early errors
- *(parser)* for-head early errors (bound-name overlap/dup, labelled-fn body, async-of)
- *(lexer,parser)* early errors for legacy-octal / non-octal-decimal numeric literals
- *(parser)* close 5 parser regressions (vertical-tilde, static-block arguments, using-for-of)
- *(object)* spec-conformant ValidateAndApplyPropertyDescriptor + call/super fixes
- *(parser/lexer)* close parse-phase gaps for import(), escapes, yield/await, using
- *(descriptors)* merge omitted attributes on redefine; callable length/name own
- *(regex)* propagate captures from positive lookahead/lookbehind
- *(parser)* allow Annex B sloppy duplicate FunctionDeclarations in block/switch
- *(parser)* parse `export <decl>` as a StatementListItem so `export let` works
- *(parser)* `let` in single-statement position is an ExpressionStatement (sloppy)
- *(builtins)* precise function descriptors + accessor this-validation

### Other

- collapse short let-chain to one line (newer rustfmt)
- cargo fmt + clippy (collapse if-lets, wrap long lines)
- README — pass count 41,146/44,189
- *(test262)* re-bless ledger (3046 -> 3043 known failures)
- README — pass count 41,143/44,189
- *(test262)* re-bless ledger (3050 -> 3046 known failures)
- README — pass count 41,139/44,189
- *(test262)* re-bless ledger (3074 -> 3050 known failures)
- README — pass count 41,115/44,189
- *(test262)* re-bless ledger (3079 -> 3074 known failures)
- README — pass count 41,110/44,189
- *(test262)* re-bless ledger (3086 -> 3079 known failures)
- README — pass count 41,103/44,189
- *(test262)* re-bless ledger (3110 -> 3086 known failures)
- README — pass count 41,079/44,189
- *(test262)* re-bless ledger (3164 -> 3110 known failures)
- README — pass count 41,025/44,189
- *(test262)* re-bless ledger (3166 -> 3164 known failures)
- README — pass count 41,023/44,189
- *(test262)* re-bless ledger (3175 -> 3166 known failures)
- README — pass count 41,014/44,189
- *(test262)* re-bless ledger (3179 -> 3175 known failures)
- README — pass count 41,010/44,189
- *(test262)* re-bless ledger (3185 -> 3179 known failures)
- README — pass count 41,004/44,189
- *(test262)* re-bless ledger (3191 -> 3185 known failures)
- README — pass count 40,998/44,189
- *(test262)* re-bless ledger (3202 -> 3191 known failures)
- README — pass count 40,987/44,189
- *(test262)* re-bless ledger (3240 -> 3202 known failures)
- README — pass count 40,949/44,189
- *(test262)* re-bless ledger (3241 -> 3240 known failures)
- README — pass count 40,948/44,189 (latest bless 3241 fails)
- *(test262)* re-bless ledger (3260 -> 3241 known failures)
- README — pass count 40,929/44,189 (latest bless 3260 fails)
- *(test262)* re-bless ledger (3269 -> 3260 known failures)
- README — pass count 40,920/44,189 (latest bless 3269 fails)
- *(test262)* re-bless ledger (3290 -> 3269 known failures)
- README — pass count 40,899/44,189 (latest bless 3290 fails)
- *(test262)* re-bless ledger (3313 -> 3290 known failures)
- README — pass count 40,876/44,189 (latest bless 3313 fails)
- *(test262)* re-bless ledger (3335 -> 3313 known failures)
- README — pass count 40,854/44,189 (latest bless 3335 fails)
- *(test262)* re-bless ledger (3401 -> 3335 known failures)
- README — pass count 40,788/44,189 (latest bless 3401 fails)
- *(test262)* re-bless ledger (3447 -> 3401 known failures)
- README — pass count 40,742/44,189 (latest bless 3447 fails)
- *(test262)* re-bless ledger (3512 -> 3447 known failures)
- README — pass count 40,677/44,189 (latest bless 3512 fails)
- *(test262)* re-bless ledger (3547 -> 3512 known failures)
- README — pass count 40,626/44,189 (latest bless 3563 fails)
- *(test262)* re-bless ledger (3571 -> 3563 known failures)
- *(test262)* re-bless ledger (3575 -> 3571 known failures)
- README — pass count 40,614/44,189 (latest bless 3575 fails)
- *(test262)* re-bless ledger (3580 -> 3575 known failures)
- README — pass count 40,609/44,189 (≈92%, latest bless 3580 fails)
- *(test262)* re-bless ledger (3609 -> 3580 known failures)
- *(test262)* re-bless ledger (3646 -> 3609 known failures)
- README — pass count 40,558/44,189 (latest bless)
- *(test262)* re-bless ledger (3651 -> 3646 known failures)
- README — pass count 40,553/44,189 (latest bless)
- *(test262)* re-bless ledger (3678 -> 3651 known failures)
- README — pass count 40,527/44,189 (latest bless)
- *(test262)* re-bless ledger (3703 -> 3678 known failures)
- README — pass count 40,502/44,189 (latest bless)
- *(test262)* re-bless ledger (3727 -> 3703 known failures)
- README — pass count 40,478/44,189 (latest bless)
- *(test262)* re-bless ledger (3759 -> 3727 known failures)
- *(test262)* re-bless ledger (3782 -> 3759 known failures)
- README — pass count 40,423/44,189 (latest bless)
- *(test262)* re-bless ledger (3810 -> 3782 known failures)
- README — pass-rate figures 40,395/44,189 (latest bless)
- *(test262)* re-bless ledger (3964 -> 3810 known failures)
- README — pass-rate ≈91% (40,241/44,189) + recent conformance areas
- *(test262)* re-bless ledger (4042 -> 3964 known failures)
- *(test262)* re-bless ledger (4098 -> 4042 known failures)
- *(test262)* re-bless ledger (4274 -> 4098 known failures)
- drop accidentally-committed __pycache__ bytecode + gitignore it
- update curated tests for tightened conformance + enabled modules
- bump intl 0.4 -> 0.5
- *(test262)* re-bless ledger (4419 -> 4274 known failures)
- cargo fmt + fix rustdoc private-item links (CI gate)
- bless workflow rebases before pushing the ledger
- workflow_dispatch job to re-bless the Test262 ledger
- README — pass-rate ≈ 90%
- ROADMAP — mark ES modules + explicit resource management landed
- refresh ROADMAP + README to current engine state
- *(test262)* surface Test262Error message in typed throws
- *(test262)* refresh ledger after async/async-gen fix batch
- *(test262)* refresh ledger after batch 13 part 2 (Promise/JSON/Object + Array/RegExp/TypedArray)
- *(test262)* refresh ledger after batch 13 part 1 (AnnexB block-fns + class static super)
- *(test262)* refresh ledger after batch 12 part 2 (String/TypedArray/DataView)
- *(test262)* refresh ledger after batch 12 part 1 (async/await + array holes + Map/Set)
- *(test262)* refresh ledger after batch 11 (lazy generators, DisposableStack, parser)
- close parse-phase early-error gaps + fix regex-after-} lexing
- *(test262)* refresh ledger after prototype/branding model
- *(test262)* refresh ledger after batch 10 part 2 (language operators/caller)
- *(test262)* refresh ledger after batch 10 part 1 (RegExp engine + intl402)
- *(test262)* refresh ledger after batch 9 part 2 (Array/Function/arguments)
- *(test262)* refresh ledger after batch 9 part 1 (AnnexB + language)
- *(test262)* refresh ledger after batch 8 part 2 (Object/Iterator/Proxy/Reflect)
- *(test262)* batch 8 part 1 — language core + TypedArray/detach
- *(test262)* refresh ledger after RegExp.prototype sweep (batch 7 complete)
- *(test262)* refresh ledger after full Array/DataView/TypedArray sweep
- *(test262)* refresh ledger after batch 7 (language core + TypedArray/DataView)
- *(nbexec)* move the cfg(test) tests module into nbexec/tests.rs
- *(nbexec)* split the Interp impl into cohesive submodules
- *(nbexec)* git mv nbexec.rs to nbexec/mod.rs
- *(test262)* refresh ledger after batch 6 (language core, Promise/collections, parser)
- *(test262)* refresh ledger after batch 5 (builtin sweeps)
- remove test runner helper script
- cargo fmt
- *(test262)* refresh ledger after batch 4 (iterators, TypedArray ctor, parser)
- cargo fmt
- *(parser)* fix doc-comment formatting for clippy doc_lazy_continuation
- *(test262)* refresh ledger after batch 3 (Object descriptors, BigInt arrays, parser/lexer)
- *(test262)* refresh ledger after batch 2 (eval/Function, parser fixes, advanced regex)
- *(regex)* drop unused saw_named_group field
- *(parser)* cover `let` single-statement position and Annex B fn redeclaration
- *(test262)* refresh ledger after batch 1 (descriptors, regex \p, parser strictness)
- *(parser)* document jump/target/new.target/optchain rules in validate
- *(test262)* refresh ledger after %TypedArray% intrinsic
- *(test262)* conformant official-corpus runner + baseline + nightly CI
- *(test262)* vendor the official tc39/test262 corpus as a pinned submodule
- prune value-reachable GC side-tables as weak-key maps (H/GC)
- add ephemeron (weak-key) marking primitive + split sweep (H/GC)
- *(ic)* wire the inline cache into bytecode GetProp/SetProp (H1)
- *(ic)* add IC fast-path accessors for property read/write (H1)

## [0.0.4](https://github.com/KarpelesLab/kataan/compare/v0.0.3...v0.0.4) - 2026-06-14

### Fixed

- *(typed-array)* validate view/DataView bounds + checked offset math
- *(nbexec)* L1 — ArrayBuffer.prototype.transfer(n) caps length via validate_alloc_len
- *(nbexec)* C2 — guard tree-walk recursion depth to avoid native stack overflow
- *(arrays)* C1 — throw RangeError on oversized array element write / length set
- *(realm)* C1 cap dense-array growth; M3 skip frozen-set probes

### Other

- *(typed-array)* bulk byte ops for fill/copyWithin/set + alloc-free encode
- *(regex)* build the scalar adapter program lazily (RE-P2)
- *(regex)* cache the compiled Regex on the RegExp cell (RE-P1)
- *(string)* P1/P3/P5/P6 — stop double-flattening strings, borrow leaf bytes, run UTF-16 counts
- *(wasm)* cache decoded Module + cut per-call memory copies at JS↔wasm boundary
- *(nbvm)* M4 — borrow array backing for pure-scan builtins; tests
- *(nbvm)* H2 — gate regexp probe on key in GetProp hot path
- *(realm)* P4 byte-exact string comparison; add tests for C1/H3/P4/P2
- *(realm)* H3 truthy tests emptiness without materializing the string
- *(rope/realm)* P2 zero-copy leaf-byte borrow fast path
- *(bytecode)* verifier rejects LoadConst with a raw heap handle (M3)

## [0.0.3](https://github.com/KarpelesLab/kataan/compare/v0.0.2...v0.0.3) - 2026-06-14

### Added

- *(strings)* B5 — surrogate-aware case + normalize ([#12](https://github.com/KarpelesLab/kataan/pull/12))
- *(strings)* B4 — wire regex builtins to the UTF-16 code-unit engine ([#12](https://github.com/KarpelesLab/kataan/pull/12))
- *(strings)* B2/B3 — surrogate-correct string evaluation and UTF-16 ops
- *(strings)* B2 — carry string-literal cooked values as WTF-8 bytes
- *(ffi)* embedder buffer-creation API — owned + external (A6, #11)
- *(wasm)* WebAssembly.Memory shares one byte store with JS (A5, #11)
- *(wtf8)* WTF-8 string-storage foundation (B1)
- *(cell)* add Cell::Bytes byte store + TypedArray view variant (A1)

### Fixed

- *(typedarrays)* stable/shared .buffer object + restore wasm_bytes over byte stores

### Other

- *(regex)* fix broken intra-doc links in module docs
- *(regex)* match over UTF-16 code units with JS u-flag semantics [B4-engine]
- *(wtf8)* fix broken intra-doc links in rope/atom/wtf8 module docs
- *(typedarrays)* real byte-backed ArrayBuffer/TypedArray/DataView (A2/A3/A4)

## [0.0.2](https://github.com/KarpelesLab/kataan/compare/v0.0.1...v0.0.2) - 2026-06-14

### Added

- *(object)* dictionary-mode objects to bound shape-tree growth (MEM-3)
- *(limits)* add configurable Limits; migrate caps to read from Realm
- *(math)* drop fixed Math.random seed; entropy-mix the fallback
- *(math)* seed Math.random from purecrypto's OS CSPRNG
- *(math)* back Math.random with xorshift128+ (was xorshift64)

### Fixed

- *(wasm)* configurable limits, fuel metering, multi-byte blocktype (WASM-6/7/9)
- *(nbvm)* collect subject chars once in regex match/split loops (RE-7)
- *(regex)* collect subject chars once per match/replace/split loop (RE-7)
- *(rng)* don't let OsRng panic abort Interp::new; fall back to entropy mix (RNG-1)
- *(bigint)* cap asUintN/asIntN/** bit-size to prevent allocation bomb (MEM-6)
- *(regex)* tighten backtracking budget + share it across starts; add char-slice match API (RE-8, RE-7)
- *(regex)* avoid exponential compile-time blowup for nested unbounded quantifiers (RE-9)
- *(nbvm)* cap repeat + guard compiler register overflow (NBVM-1/NBVM-2)
- *(flatbc)* read untrusted program via fs::read, drop mmap (VM-9)
- *(bytecode)* verify untrusted load paths, bound allocs, drop mmap (VM-7/8/9)
- *(snapshot)* SNAP-3 read snapshot file instead of mmap (SIGBUS hazard)
- *(wasm)* use checked_add for element-segment slot index (WASM-8)
- *(heap)* clamp free-path generation below compaction range (MEM-4)
- *(parser)* guard recursion in parse_new/parse_exponent/parse_class (PARSE-2/3/4)
- *(date)* parse_iso_date panics on malformed/non-ASCII input (EXEC-7)
- *(lint)* satisfy clippy + rustdoc + rustfmt gates

### Other

- *(limits)* end-to-end config-override regression test
- *(snapshot)* validate restored func_id against a recorded bound (SNAP-2)
