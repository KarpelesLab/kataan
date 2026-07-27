# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.8](https://github.com/KarpelesLab/kataan/compare/v0.0.7...v0.0.8) - 2026-07-27

### Added

- *(temporal)* astronomical Chinese/Dangi calendars (+5, unbounds the range)
- *(intl)* CLDR alias tables for BCP-47 extension subtags (+3)
- *(regex)* Unicode 17 property tables, emoji string properties, spec \s (+~45)
- *(engine)* per-iteration let, switch TDZ, for-of arguments, tagged-template, optional-chaining, #x in (+17)
- *(engine)* yield/await in computed property keys (object literals + class members) (+8)
- *(engine)* class super()/this-binding cell + Function/GeneratorFunction subclassing (+17)
- *(engine)* Temporal DST duration rounding + IANA links, Array live flatten/map/filter observable order (+21)
- *(regex)* reverse-lookbehind captures, lossless surrogate ToString, AnnexB legacy parse (+20)
- *(intl)* wire intl-0.5.1 dangi/umalqura calendars + DateTimeFormat interval & non-Gregorian formatting + getCanonicalLocales aliases (+97)
- *(intl)* BigInt-exact NumberFormat via puremp::Decimal significant-digits (+4)
- *(engine)* String-exotic [[DefineOwnProperty]] + curated fixture corrections (+1)
- *(engine)* lexical-env separation for eval/global declaration-instantiation (+17)
- *(engine)* Array.fromAsync iterator-driven, RegExp cross-realm brand, ArrayIterator/Proxy-set, destructuring/for-in order (+42)
- *(engine)* Temporal lunisolar arithmetic, Array sparse mutators, NumberFormat compact, class extends-null/heritage-TDZ (+70)
- *(engine)* formatRange field-level parts, Date.UTC fp, RegExp lastIndex physical (+12)
- *(intl)* DateTimeFormat IANA timezone support via the Temporal tz data (+14)
- *(realm,module)* Gen/AsyncGen proto-from-ctor-realm-prototype + ShadowRealm importValue (+3)
- *(module)* single-module top-level-await via suspendable coroutine (+3)
- *(shadowrealm)* genuine realm isolation on the realm-global-swap foundation (+8)
- *(realm)* swap global_this/global_scope at realm boundaries (+7)
- *(realm)* cross-realm proto-from-ctor for dynamic Function (+1)
- *(module)* %AbstractModuleSource% intrinsic (+8)
- *(engine)* URI encode/decode, matchAll iterator, class constructor order (+31)
- *(tdz)* for-of/for-in head lexical TDZ (+12)
- *(temporal)* Duration relativeTo rounding — built-ins/Temporal now 0 fails (+10)
- *(engine)* Proxy Set/Get receiver preservation + AsyncFunction tag + Object (+25)
- *(temporal)* DST cross-midnight, IANA canon, NudgeToCalendarUnit, hang fix (+40)
- *(temporal)* DST/timezone arithmetic — disambiguation, transitions, DST-aware diff (+26)
- *(intl)* DTF hourCycle/ZDT-toLocaleString + NF exact-decimal + RTF numbering (+39)
- *(intl)* Islamic/Persian dateStyle for Temporal toLocaleString (+3)
- *(language)* 3rd contained grind — assignment order, for-of protocol, using decls (+18)
- *(array)* precise [[Get]]/[[Set]]/[[Delete]] for hole+accessor mutators (+16)
- *(array)* species-create through Proxy, prototype.length, weak-collection tag (+12)
- *(realm)* proto-from-ctor-realm — per-realm Intl protos + GetPrototypeFromConstructor (+15)
- *(regex)* v-flag property-of-strings (Emoji_Keycap_Sequence) + astral group names (+26)
- *(intl)* smaller-services algorithmic — ListFormat, Collator, localeCompare, numbering (+36)
- *(intl)* Temporal→DateTimeFormat protocol + toLocaleString + DTF algorithmic (+93)
- *(intl)* NumberFormat algorithmic — decimal rounding, BigInt/coercion, accounting (+52)
- *(intl)* re-wire Locale/Intl to intl-0.5 CLDR data + bounded aliases (+22)
- *(engine)* Promise finally/species, static-block scope, WTF-8 surrogates (+18)
- *(engine)* scattered code-tail — class extends, live collections, private-on-proxy (+25)
- *(realm)* cross-realm error identity + ArraySpeciesCreate lookup (+19)
- *(atomics)* virtual-clock cooperative scheduling for wait/waitAsync (+39)
- *(function)* source-text .toString() for classes + computed methods (+7)
- *(typedarray)* resizable-buffer OOB + length-tracking view completion (+31)
- *(async-gen)* lazy yield* delegation + for-await over sync iterators (+31)
- *(language)* diffuse tail — regex-literal LT, strict/global bindings, directives (+30)
- *(class)* diffuse grind — direct-eval private names, private-elem invariants (+42)
- *(language)* 2nd diffuse grind — lazy IteratorClose destructuring +47 (+48)
- *(builtins)* diffuse tail — JSON parse-with-source +9, Set +4, unary-wrapper +2 (+18)
- *(typedarray,temporal)* diffuse grind — TypedArray +19, Temporal ISO +27 (+48)
- *(object)* diffuse conformance grind — built-ins/Object 67→15 (+52)
- *(language)* diffuse conformance grind — statements/expressions (+34, 46 fixed)
- *(temporal,array)* diffuse conformance grind — Temporal ISO +92, Array +53 (+142)
- *(temporal)* widen Duration to i128 + correctly-rounded total/BigInt→Number (+14)
- *(temporal)* per-calendar edge fixes — BC round-trip, era validation, month-until overshoot (+181)
- *(temporal)* harden lunisolar leap-month semantics in the calendar layer (+67)
- *(temporal)* calendar-aware arithmetic for PlainDateTime/ZonedDateTime/YearMonth (+286)
- *(temporal)* calendar-aware arithmetic + PlainMonthDay wiring (+152)
- *(temporal)* non-ISO calendar wiring for ZonedDateTime + PlainYearMonth (+410)
- *(temporal)* non-ISO calendar wiring for PlainDateTime (+230)
- *(temporal)* non-ISO calendar layer + PlainDate wiring (+238)
- *(async-gen)* reify method-call arguments in the step machine (+3)
- *(async-gen)* spec-shaped request queue + per-await microtask suspension (+7)
- *(conformance)* implement Iterator.zip / Iterator.zipKeyed proposal (+33)
- *(conformance)* resizable/growable ArrayBuffer + length-tracking views (§3.8)
- *(conformance)* GetPrototypeFromConstructor uses newTarget's realm (proto-from-ctor-realm, +45)
- *(conformance)* implement mapped arguments object ([[ParameterMap]]) (26 → 2, +24)
- *(jit)* direct generic→generic calls + inline ArrayLen; retire JIT_DESIGN.md
- *(jit)* direct generic JIT→JIT calls + inline ArrayLen (§2.1 / Pass 6)
- *(jit)* inline property-SET + array-element GET/SET fast paths (§2.1 Pass 6)
- *(jit)* inline monomorphic property-GET fast path (§2.1 Pass 6)
- *(jit)* generic-tier array element access + .length — Pass 5
- *(jit)* generic-tier function calls (Op::Call / Op::CallNative) — Pass 4
- *(jit)* generic-tier control flow + comparisons + value arithmetic — Pass 3
- *(jit)* generic-tier property access (GetProp/SetProp) — Pass 2
- *(jit)* generic (NanBox) tier substrate — Pass 1, end-to-end generic Add
- *(host)* §4.3 web platform globals + fix no_std build (FloatExt) (+59 tests)
- *(host)* §4.5 node-compat builtins — Buffer, path, os, util, querystring, process (+23 tests)
- *(wasm)* §2.2 reference types + bulk-table + JS-boundary tables (+9 tests)
- *(host)* §4.1 event loop + timers (setTimeout/Interval/Immediate, nextTick, AbortController)
- *(conformance)* implement proper tail calls (PTC) on both tiers — un-skip +34
- *(test262)* $262.agent cooperative scheduler + Atomics.waitAsync
- *(conformance)* implement $262.createRealm (cross-realm) — un-skip +77
- *(conformance)* import attributes + assertions + JSON/text modules — un-skip +94
- *(conformance)* implement class decorators + accessor auto-accessors — un-skip +24
- *(conformance)* implement IsHTMLDDA (document.all) exotic — un-skip +34
- *(temporal)* implement Duration relativeTo calendar arithmetic (+124)
- *(temporal)* Temporal instances are ordinary extensible objects (+31)
- *(temporal)* PlainDate.toPlainDateTime accepts a time-like (ToTemporalTime)
- *(temporal)* implement toZonedDateTime on PlainDate + PlainDateTime (+26)
- *(temporal)* implement Temporal.Now — Temporal fully un-skipped, 3920 pass
- *(temporal)* implement Temporal.ZonedDateTime — 844/901, corpus 3003→3875 pass
- *(temporal)* Date.prototype.toTemporalInstant → Temporal.Instant (+8)
- *(temporal)* subclassing, subclass statics, and valueOf via ToPrimitive (+19)
- *(temporal)* implement 7 Temporal types — 2984 test262 tests now pass (was 0)
- *(temporal)* scaffold Temporal (ROADMAP §4) — ISO core + brand + registration + dispatch
- *(sab)* make SharedArrayBuffer subclassable (class extends SharedArrayBuffer)
- *(sab)* brand-check SharedArrayBuffer.prototype.slice/grow receiver
- *(sab)* brand-check SharedArrayBuffer prototype getters + grow receiver
- *(sab)* validate SharedArrayBuffer length + maxByteLength via ToIndex; un-skip
- *(atomics)* wait requires a shared buffer — TypeError before arg coercion
- *(atomics)* BigInt64Array/BigUint64Array read-modify-write
- *(atomics)* throw TypeError for writing ops on an immutable ArrayBuffer
- *(atomics)* implement single-agent notify/wait + fix arity/proto; un-skip Atomics
- *(tdz)* hoist block/script/function let/const/class to TDZ at scope entry
- async-generator yield Awaits its operand (AsyncGeneratorYield) (§async-generators)
- @@asyncDispose then-chains PromiseResolve(return()) → undefined (§async-generators)
- %AsyncIteratorPrototype%[Symbol.asyncDispose] (§async-generators)
- generator instance prototype chain (g.prototype inherits %GeneratorPrototype%) (§generators)
- %GeneratorFunction% / %AsyncGeneratorFunction% constructor objects (§generators)
- distinct %AsyncGeneratorFunction.prototype% / %AsyncGeneratorPrototype% / %AsyncIteratorPrototype% (§async-generators)
- distinct %GeneratorFunction.prototype% / %GeneratorPrototype% for sync generators (§generators)
- Intl.Locale Info API — getCalendars/getCollations/getHourCycles/getNumberingSystems/getTimeZones/getTextInfo/getWeekInfo (§intl402)
- Atomics.pause([N]) — spin-loop hint (§built-ins)
- %IteratorPrototype%[Symbol.toStringTag] get/set accessor (§built-ins)
- *(intl)* Intl.Locale can be subclassed (§3.9)
- *(intl)* subclassing for ListFormat/RelativeTimeFormat/Segmenter + fix instance proto (§3.9)
- *(intl)* Intl.PluralRules can be subclassed (§3.9)
- *(intl)* Intl.Locale firstDayOfWeek option + accessor (ES2024, §3.9)
- *(intl)* validate Intl.Locale language/script/region option shapes (§3.9)
- *(intl)* validate Intl.Locale calendar/collation/numberingSystem options (§3.9)
- *(intl)* Intl.Locale variants option (§3.9)
- *(intl)* DateTimeFormat defaults to a numeric date when no options given (§3.9)
- *(intl)* Collator ignorePunctuation locale default (Thai → true) (§3.9)
- *(intl)* DateTimeFormat resolvedOptions reports default hourCycle/hour12 (§3.9)
- *(intl)* complete supportedValuesOf('numberingSystem') list (78 systems) (§3.9)
- *(intl)* complete the numbering-system digit table (77 systems + hanidec) (§3.9)
- *(intl)* numbering system in DateTimeFormat (digits + locale default) (§3.9)
- *(intl)* resolve locale's CLDR default numbering system in NumberFormat (§3.9)
- *(intl)* apply numbering-system digits to NumberFormat.formatToParts too (§3.9)
- *(intl)* numbering-system digit substitution in NumberFormat (§3.9 CLDR)
- *(sab)* SharedArrayBuffer grow() + slice() — deterministic core complete (§3.9)
- *(sab)* SharedArrayBuffer core — construct + accessors + typed-array/Atomics backing (§3.9)
- *(atomics)* single-agent Atomics over integer typed arrays (§3.9)
- *(ffi)* C ABI value layer — KtValue + pure constructors/inspectors (§4.0)
- *(embed)* Ctx::array_set completes the host array API (§4.0)
- *(intl)* Intl.Segmenter segments.containing(index) (§3.9)
- *(embed)* host-backed native state + Drop finalizers (§4.0)
- *(embed)* deferred promises — async host continuation (§4.0)
- *(embed)* persistent handles (§4.0 handle scope)
- *(typedarray)* implement Float16Array (ES2025) (§3.6)
- *(embed)* trap host-function panics at the boundary as a JS Error (§4.0)
- *(embed)* host constructors — register_constructor + new HostCtor() (§4.0 M4)
- *(embed)* Ctx set_property — full [[Set]] for host functions (§4.0)
- *(embed)* Ctx construct + is_constructor for host functions (§4.0)
- *(embed)* Ctx promise creation for async host functions (§4.0)
- *(embed)* Ctx property API — has / has_own / delete / own_keys (§4.0)
- *(embed)* Ctx value-inspection + array access for host functions (§4.0)
- *(intl)* NumberFormat/DateTimeFormat formatRange + formatRangeToParts (§3.9)
- *(regex)* Annex B legacy octal escapes (§3.9 web-compat)
- *(lexer)* Annex B B.1.2 legacy octal string escapes (§3.9 web-compat)
- *(lexer)* Annex B B.1.3 HTML-like comments (§3.9 web-compat)
- *(typedarray)* live iteration observes resize / element writes (§3.6)
- *(collections)* live Set/Map iteration (mutation mid-iteration observed) (§3.6)

### Fixed

- *(intl)* canonicalize a `-t-` extension's tlang through the CLDR aliases
- *(regexp)* escape lone surrogates in RegExp.escape
- *(temporal)* PlainMonthDay year-range check and month-without-monthCode TypeError
- *(temporal)* floor the Islamic leap-day term for pre-epoch years (+6)
- *(temporal)* propagate the calendar through PlainDate's toPlain* conversions
- *(temporal)* constrain any Chinese/Dangi leap month code onto its base month
- *(temporal)* use the 33-year arithmetic rule for the Persian calendar (+17)
- *(intl)* gate the CLDR alias corpus behind the `intl` feature
- *(array)* bound ArraySetLength's non-configurable scan by real indices (+1)
- *(module)* dynamic import() in strict/module tail position (+2)
- *(intl)* signDisplay/-0/NaN sign logic + accounting -0 + unicode-ext yes→true (+12)
- *(conformance)* instance_proto throwing-getter propagation + aux symbol enumeration (+10)
- *(conformance)* preserve key insertion position on data->accessor redefine
- *(conformance)* Intl structural pass — option/descriptor/validation (+63)
- *(conformance)* second Temporal tractable pass (328 -> 154, +139, ~93%->96.7%)
- *(conformance)* super(...args) into a native constructor no longer drops the args
- *(conformance)* drive down tractable Temporal failures (453 → 328, +125, ~90%→93%)
- *(conformance)* Object.values/entries skip array holes on the nbexec tier
- *(conformance)* mop up small builtin clusters (+23)
- *(conformance)* mop up language statements/expressions tail (+61)
- *(conformance)* drive down built-ins/Promise cluster (16 → 5 fail, +11)
- *(conformance)* drive down super cluster (18 → 1 fail, +17, +2 class)
- *(conformance)* drive down generators cluster (56 → 24 fail, +32)
- *(conformance)* drive down language/eval-code cluster (26 → 1 fail, +25)
- *(conformance)* drive down class cluster (statements 105→85, expressions 49→31, +38)
- *(conformance)* drive down language/module-code cluster (31 → 13 fail, +18, +2 dynamic-import)
- *(conformance)* drive down built-ins/Iterator cluster (49 → 36 fail, +13)
- *(conformance)* drive down assignment cluster (33 → 22 fail, +11, +2 Object, +7 targettype)
- *(conformance)* drive down language/statements/for-of cluster (37 → 27 fail, +10)
- *(conformance)* materialize function/class `prototype` own property (+~35)
- *(conformance)* drive down built-ins/String cluster (37 → 10 fail, +27)
- *(conformance)* drive down built-ins/Function cluster (44 → 25 fail, +19)
- *(conformance)* drive down built-ins/Proxy cluster (51 → 18 fail, +33)
- *(conformance)* drive down TypedArray (+52) and RegExp (+26) clusters
- *(conformance)* drive down built-ins/Array cluster (220 → 170 fail, +50)
- *(conformance)* drive down built-ins/Object cluster (84 → 47 fail, +37)
- *(temporal)* ISO parser tail — >9-digit RangeError, offset ranges, date-sep consistency (+14)
- *(temporal)* reject U+2212 minus sign in ISO strings (+12)
- *(temporal)* toPlainDateTime rejects non-string primitives with TypeError
- *(temporal)* bind statics to their type (from/compare ignore this) (+11)
- *(proxy)* pass real Symbol keys to traps, not the internal sentinel string
- *(super)* super.x = v routes through the strict-aware member assignment
- *(super)* resolve super.prop/super.method() over the real prototype chain
- *(iterator)* Iterator.zip/zipKeyed close still-open iterators in reverse order
- *(array)* Array.from iterable path iterates lazily and IteratorCloses on mapFn throw
- *(promise)* all/allSettled/race/any iterate lazily and IteratorClose on abrupt
- *(error)* Error/NativeError/AggregateError coerce message via ToString; AggregateError IterableToList propagates
- *(collections)* Map/Set/WeakMap/WeakSet constructor uses spec AddEntriesFromIterable
- *(arraybuffer)* new ArrayBuffer(length) does ToIndex (runs valueOf, propagates)
- *(symbol)* Symbol(description) does ToString (runs user toString, propagates)
- *(json)* JSON.parse(text) does ToString first (runs user toString, propagates)
- *(symbol)* Symbol.for(key) does ToString (runs user toString, propagates)
- *(typedarray)* from/of coerce each element via ToNumber/ToBigInt
- *(global)* parseInt/parseFloat/URI functions coerce args via ToString/ToNumber
- *(global)* isNaN/isFinite coerce via ToNumber (runs valueOf, propagates)
- *(update)* member key ToPropertyKey deferred past the null-base check
- *(update)* with-binding update op resolves the reference once (self-mutating getter)
- *(assign)* read_target on an unresolvable identifier throws ReferenceError
- *(logical-assign)* computed-member target evaluates base+key once (lhs-before-rhs)
- *(logical-assign)* NamedEvaluation for x &&= / ||= / ??= anonymous RHS
- *(string)* replaceAll/matchAll use spec IsRegExp ([[Get]] @@match, propagates)
- *(json)* stringify replacer PropertyList built via [[Get]] (proxy-aware, propagates)
- *(json)* reviver object walk is proxy-aware ([[Delete]]/CreateDataProperty + target keys)
- *(json)* reviver IsArray unwraps proxies + ToLength coercion propagates
- *(json)* reviver array walk uses proxy-aware [[Delete]] / CreateDataProperty
- *(regex)* unicode-mode restricted identity/control escapes in classes + unterminated backslash-u-brace
- *(arraybuffer)* resize() ToIndex's newLength (negative → RangeError, not clamp)
- *(arraybuffer)* ToIndex + allocation-cap validation for maxByteLength option
- *(typedarray)* length-tracking view over a resizable buffer floors, not throws
- *(generators)* cache yield* [[NextMethod]] once instead of re-reading each step
- *(atomics)* store return value + shell-host wait model — Atomics non-agent 100%
- *(sab)* split length ToIndex from allocation limit — SAB single-agent 100%
- *(fn+sab)* honor non-object fn.prototype assignment; SAB newTarget proto via [[Get]]
- *(sab)* SharedArrayBuffer maxByteLength option runs its getter (poisoned propagates)
- *(sab)* non-object newTarget.prototype falls back to %SharedArrayBuffer.prototype%
- *(arraybuffer)* slice SpeciesConstructor rejects a non-object constructor
- *(sab)* SharedArrayBuffer.prototype.grow ToIndex's its newLength arg
- *(atomics)* isLockFree coerces its arg via ToIntegerOrInfinity
- *(nbvm)* mirror array non-writable-length mutator throw in the bytecode tier
- *(delete)* throw TypeError on delete of a nullish plain-member base
- *(array)* throw TypeError from push/pop/shift/unshift on non-writable length
- %TypedArray%.prototype.toLocaleString invokes each element's own toLocaleString (§built-ins)
- isSealed/isFrozen (and seal/freeze) work on aux-backed cells (functions, Dates) (§built-ins)
- Object.preventExtensions works on aux-backed cells (functions, Dates, …) (§built-ins)
- ++/-- on a member evaluates the reference (base + computed key) exactly once (§expressions)
- Object.freeze on an array freezes its named props + reports elements non-writable (§built-ins)
- NBVM regex .groups is null-proto + duplicate-name participating-capture-wins (§regexp)
- duplicate named groups — a participating capture wins in .groups/.indices.groups (§regexp)
- Iterator.zipKeyed longest-mode padding is read per key, not iterated (§iterator-helpers)
- iterator helpers throw TypeError on reentrant next() (GeneratorValidate) (§iterator-helpers)
- yield* over an iterator with a non-callable next throws a TypeError (§generators)
- named function expression name is a soft immutable binding (§functions)
- Object.seal/freeze work on aux-backed cells (functions, Dates) (§built-ins)
- constructor .prototype is non-writable and non-configurable (§built-ins)
- Set/Map.prototype.forEach iterates live (observes mid-callback mutation) (§built-ins)
- Intl.Locale getWeekInfo — {firstDay,weekend} only, firstDay from firstDayOfWeek (§intl402)
- AggregateError constructor has length 2 (§built-ins)
- String.prototype.padStart/padEnd have length 1, not 2 (§built-ins)
- correct length for collection constructors + several methods (§built-ins)
- correct length for Map/WeakMap.set (2), Map.groupBy/Object.groupBy (2) (§built-ins)
- anonymous functions materialize name "" as an own property (§language)
- well-known symbols are own data properties of %Symbol% (§built-ins)
- JSON.stringify.length is 3, JSON.parse.length is 2 (§built-ins)
- ToPrimitive on a wrapper runs OrdinaryToPrimitive (honors user valueOf/toString) (§built-ins)
- GetPrototypeFromConstructor uses default proto when newTarget.prototype isn't an object (§built-ins)
- %ThrowTypeError% has own non-configurable name ("")/length (0) (§built-ins)
- Function.prototype has own name ("") and length (0) (§built-ins)
- install name/length on ArrayBuffer.isView + Function.prototype[Symbol.hasInstance] (§built-ins)
- concise methods / accessors / class methods are not constructors (§language)
- *(proxy)* Object.getOwnPropertyNames forwards to target for a trap-less proxy (§built-ins)
- *(iter)* array destructuring/for-of honor an overridden Array Symbol.iterator (§language)
- *(intl)* NumberFormat/DateTimeFormat/Collator/Segmenter constructor length is 0 (§3.9)
- *(intl)* reject Symbol locale-list elements + DisplayNames uses CanonicalizeLocaleList (§3.9)
- *(intl)* DisplayNames length=2 + localeMatcher validated first (§3.9)
- *(intl)* Collator.resolvedOptions returns Collator keys, not NumberFormat (§3.9)
- *(intl)* DateTimeFormat resolvedOptions property order (§3.9)
- *(typedarray)* default sort places -0 before +0 (§3.6)
- *(intl)* NumberFormat.formatToParts decomposes unit/compact suffixes (§3.9)
- *(intl)* segments.containing coerces index via ToIntegerOrInfinity (§3.9)
- *(intl)* Date.prototype.toLocale{,Date,Time}String apply DateTimeFormat options (§3.9)
- *(intl)* Number.prototype.toLocaleString applies all NumberFormat options (§3.9)
- *(intl)* String.prototype.localeCompare honors numeric/sensitivity options (§3.9)
- *(intl)* Intl.Segmenter.resolvedOptions reports locale + granularity (§3.9)
- *(module)* re-exporting an imported binding follows the import (§3.1)
- *(module)* named imports resolve inside functions called cross-module (§3.1)
- *(module)* a module namespace object reports Object.isSealed true (§3.1)
- *(intl)* accounting parens are en-family only; de-DE keeps the minus (§3.9)
- *(intl)* accounting currency post-processes the locale-correct output (de-DE) (§3.9)
- *(array)* C.fromAsync populates a subclass result via CreateDataPropertyOrThrow (§3.7)
- *(intl)* DisplayNames.of honors fallback for a not-found name (§3.9)
- *(intl)* DisplayNames resolvedOptions + style/fallback/languageDisplay defaults (§3.9)
- *(intl)* signDisplay "negative" excludes negative zero (§3.9)
- *(intl)* currencySign "accounting" parenthesizes negative currency (§3.9)
- *(intl)* apply roundingIncrement (round to nearest step) (§3.9)
- *(intl)* trailingZeroDisplay "stripIfInteger" drops trailing zeros (§3.9)
- *(intl)* a lone significant-digit option defaults the other (§3.9)
- *(intl)* limit native-super resolution to NumberFormat/DateTimeFormat (§3.9)
- *(intl)* DisplayNames requires a valid `type` option (§3.9)
- *(intl)* signDisplay "always" on -0 is "-0", not "-+0" (§3.9)
- *(lexer)* HTML-like comments are script-only, not in modules (§3.9)
- *(intl)* NumberFormat / DateTimeFormat subclassing (§3.9)
- *(intl)* all service constructors are non-enumerable properties of Intl (§3.9)
- *(regex)* \b inside a character class is a backspace (§3.6)
- *(class)* a static method captures the class scope (self-name visible) (§3.6)
- *(class)* the class-name inner binding is an immutable const (§3.6)
- *(class)* static private members are not inherited by a subclass (§3.6)
- *(class)* private methods install after super(), not at allocation (§3.6)
- *(string)* match/replace/etc. read @@method only for an Object arg (§3.6)
- *(array)* C.from populates a subclass result via CreateDataPropertyOrThrow (§3.6)
- *(class)* AggregateError subclass errors + Symbol/BigInt non-constructor super (§3.6)
- *(class)* WeakRef / FinalizationRegistry subclassing (§3.6 subclass builtins)
- *(error)* a native-error subclass resolves .constructor to the subclass (§3.6)
- *(string)* for-in enumerates a String object's index keys (§3.6)

### Other

- *(test262)* re-bless ledger (293 -> 287 known failures)
- refresh the Test262 headline to the gated 99.45% (51,603/51,890)
- fix a broken intra-doc link in the trig helpers
- *(temporal)* memoize the lunisolar sui and year resolutions
- *(roadmap)* mark §3.10 non-ISO calendar arithmetic landed
- *(test262)* re-bless ledger (295 -> 292 known failures)
- refresh the Test262 headline to the gated 99.44% (51,598/51,890)
- *(intl)* drop the redundant 'static lifetimes on the generated tables
- *(test262)* re-bless ledger (325 -> 295 known failures)
- *(roadmap)* record why a plain rope memo is the wrong fix
- *(collections)* hash-index Map/Set lookups (O(n) -> O(1) per operation)
- *(roadmap)* add the three open items found this session, refresh headline
- *(readme)* update the Test262 pass-rate to the measured 99.4% (51,565/51,890)
- *(test262)* re-bless ledger (378 -> 325 known failures)
- *(array)* serve push/pop without snapshotting the whole array
- *(regex)* make global match/replace/split linear in the subject length
- *(engine)* O(1) string type test (string `+=` quadratic -> linear)
- *(engine)* flatten per-iteration `let` environments (quadratic -> linear)
- intl 0.5.0 → 0.5.1 (chinese range 1800-2200 + islamic/persian data) (+11)
- bump timezone-data 0.1→0.2, enable puremp rational+decimal
- *(test262)* un-ledger 3 detached-copyWithin tests that pass in release (+3)
- *(ci)* fix clippy (--all-features --all-targets) and rustdoc warnings
- *(array)* lock in first-class prototype-method behavior (already implemented)
- *(test262)* re-bless ledger (3509 -> 3428 known failures)
- *(test262)* re-bless ledger (3774 -> 3676 known failures)
- *(test262)* re-bless ledger (3897 -> 3896 known failures)
- *(test262)* re-bless ledger (4267 -> 3981 known failures)
- *(test262)* re-bless ledger (2518 -> 4391 known failures)
- *(jit)* record Pass 6 inline fast paths landed (property + element)
- *(jit)* repr(C)/repr(u8) on the property-GET hot types (Pass 6 prep)
- *(jit)* record generic tier (Passes 1-5) landed in ROADMAP §2.1 + design status
- *(jit)* design + pass spine for JIT completion (§2.1)
- *(roadmap)* mark §4.1 event loop, §4.3 web globals, §4.5 node builtins landed
- *(host)* src/host module for the §4 runtime (web/node/timers stubs)
- *(roadmap)* mark tail-call optimization implemented — §3.9 skipped-subsystems complete
- *(roadmap)* mark Atomics $262.agent cooperative scheduler implemented (§3.8)
- *(roadmap)* mark cross-realm implemented (identity bulk, §3.9)
- *(roadmap)* mark IsHTMLDDA/decorators/import-attributes implemented (§3.9)
- *(roadmap)* mark Temporal implemented (~90%, un-skipped) in §3.9
- *(temporal)* add timezone-data dep for Temporal.ZonedDateTime
- *(temporal)* implementation guide for the per-type fan-out
- de-ledger 5 more $262.evalScript tests flipped by the indirect-eval fix
- $262.evalScript is indirect eval (global scope) not direct
- de-ledger 8 postfix/prefix tests flipped by the read_ident_ref fix
- Revert "fix(string): replaceAll/matchAll use spec IsRegExp ([[Get]] @@match, propagates)"
- skip $262.agent tests + ledger newly-surfaced SAB-feature failures
- *(test262)* bless 5 flips (gate 0-reg), session -868
- *(test262)* bless 20 flips TDZ+array-mutators (gate 0-reg), session -863
- *(test262)* bless 9 flips (gate 0-reg), session -843
- *(test262)* bless 6 flips (gate 0-reg), session -834
- *(test262)* bless 8 flips (gate 0-reg), session -828
- *(test262)* bless 10 flips (gate 0-reg), session -820
- *(test262)* bless 7 flips (gate 0-reg), session -810
- *(test262)* bless 23 flips (gate 0-reg), session -803
- *(test262)* bless 36 flips (gate 0-reg), session -780
- *(test262)* bless 38 flips (gate 0-reg), session -744
- *(test262)* bless 27 flips (gate 0-reg), session -706
- *(test262)* bless 1 flips (gate 0-reg), session -679
- *(test262)* bless 8 flips (gate 0-reg), session -678
- *(test262)* bless 14 flips (gate 0-reg), session -670
- *(test262)* bless 16 flips (gate 0-reg), session -656
- *(test262)* bless 24 flips (gate 0-reg), session -640
- *(test262)* bless 3 flips (gate 0-reg), session -616
- *(test262)* bless 19 flips (prototype-descriptor/freeze-seal, gate 0-reg), session -613
- collapse freeze_object aux if-let to a let-chain (clippy)
- *(test262)* bless 11 flips (gate 0-reg), session -594
- *(test262)* bless 30 flips (gate 0-reg), session -583
- *(test262)* bless 5 flips (Atomics.pause/Locale Info, gate 0-reg), session -553
- *(test262)* bless 1 flips (AggregateError etc., gate 0-reg), session -548
- *(test262)* bless 2 more arity flips (padStart/padEnd/AggregateError), session -547
- *(test262)* bless 14 arity flips (Map/Set/etc. constructor+method lengths, gate 0-reg, 94.35%)
- *(test262)* ledger 3 generator/async prototype not-callable tests (proto conflation)
- *(test262)* bless 8 flips (anon-function name materialization, gate 0-reg), session -534
- *(test262)* bless 23 flips (Iterator toStringTag + well-known symbols, gate 0-reg), session -526
- fn_name_unset uses is_none_or (clippy)
- *(test262)* bless 8 flips (ToPrimitive wrapper coercion + JSON, gate 0-reg), session -503
- *(test262)* bless 24 flips (concise-method/name-length/GetPrototypeFromConstructor, gate 0-reg), session -495
- *(test262)* re-bless ledger (2598 -> 2601 known failures)
- *(test262)* bless 5 more flips (gate-confirmed 0 regressions), session -461
- *(test262)* bless 45 array-iterator destructuring flips (iter-val-array-prototype)
- Revert empty ledger — bless script bug (empty capture → awk treated ledger as skip file)
- *(test262)* bless remaining 2632 array-iterator flips (gate-confirmed 0 regressions, 94.16%)
- *(test262)* bless 40 flips (array-iterator ~110 + intl402, gate-confirmed 0 regressions)
- *(test262)* bless 11 flips (Intl subclassing + Locale, gate-confirmed 0 regressions)
- *(test262)* bless 5 Locale flips (language/script/region validation + firstDayOfWeek)
- *(test262)* remove 6 now-passing Intl.Locale tests (variants + option validation)
- *(test262)* remove 3 now-passing DateTimeFormat resolvedOptions tests
- Revert "Collator resolvedOptions" changes — locale -u- canonicalization gap caused regressions
- *(test262)* remove 3 now-passing DateTimeFormat resolvedOptions order tests
- *(test262)* remove 2 now-passing entries (numbering-systems + supportedValuesOf)
- *(intl)* fix hanidec assertion after completing the numbering table
- *(test262)* add KATAAN_TEST262_FILTER for local subset runs; note Atomics/SAB status
- *(test262)* remove 1 now-passing entry (Atomics/pause — namespace now present)
- *(roadmap)* record the single-agent Atomics/SAB deterministic core as done (§3.6)
- *(test262)* remove 2 now-passing entries (formatToParts unit/compact + TA sort -0/+0)
- *(test262)* remove 2 now-passing ledger entries (Segmenter containing index-coercion)
- *(roadmap)* refresh headline (≈93.9%) + record this cycle's converted clusters
- *(test262)* remove 4 now-passing ledger entries (Number/Date toLocaleString)
- *(test262)* remove 12 now-passing ledger entries (Segmenter + module)
- *(intl)* regression tests for this cycle's Segmenter/localeCompare/toLocaleString fixes
- *(test262)* remove 2 now-passing ledger entries (module import-closure fix)
- *(embed)* demo native state (napi_wrap) in the host-fn example
- *(roadmap)* record §4.0 native state + finalizers landed; Rust API feature-complete
- *(roadmap)* record §4.0 async continuation (deferred promises) landed
- *(roadmap)* record §4.0 handle scope (persistent handles) landed
- *(roadmap)* record §4.0 host-boundary panic-trapping landed
- *(roadmap)* record §4.0 host constructors landed
- *(test262)* remove 15 now-passing ledger entries (DisplayNames resolvedOptions + fromAsync)
- *(embed)* expand the host-fn example with the §4.0 M2 introspection API
- *(roadmap)* record §4.0 M2 Ctx expansion (inspection/property/promise API)
- *(test262)* remove 8 now-passing ledger entries (NumberFormat signDisplay/formatting)
- *(roadmap)* refresh headline pass-rate (≈93.8%) + record converted clusters
- fix fromAsync dense-storage test — observe via microtask drain
- *(test262)* remove 38 now-passing ledger entries (formatRange + DisplayNames + Intl structural)
- fix roundingIncrement test — inline JS (format! braces broke compilation)
- *(test262)* remove 15 now-passing ledger entries (Intl structural + regex/octal)
- *(test262)* remove 31 now-passing ledger entries (live typed-array iter + AnnexB octal/comments)
- *(test262)* remove 35 now-passing ledger entries (class-cluster fixes)
- *(test262)* remove 15 now-passing ledger entries (live Set/Map iteration + regression cleared)
- *(test262)* remove 40 now-passing ledger entries (String-exotic, BigInt, subclass, etc.)

## [0.0.7](https://github.com/KarpelesLab/kataan/compare/v0.0.6...v0.0.7) - 2026-07-04

### Added

- *(promise)* Promise[Symbol.species] getter → subclass then/catch chaining (§3.6)
- *(promise)* Promise subclassing — class P extends Promise (§3.6)

### Fixed

- *(string)* String-exotic getOwnPropertyDescriptor + getOwnPropertyNames (§3.6)
- *(string)* String index properties are enumerable (propertyIsEnumerable) (§3.6)
- *(string)* String exotic objects expose own index/length to has_own (§3.6)
- *(bigint)* a BigInt primitive's [[Prototype]] is %BigInt.prototype% (§3.6)
- *(typedarray)* array-like constructor uses ToLength (empty, not RangeError) (§3.6)
- *(array)* toSpliced coerces start/deleteCount via ToIntegerOrInfinity (§3.6)
- *(array)* slice coerces start/end via ToIntegerOrInfinity (§3.6)
- *(array)* fill/flat/copyWithin coerce index args via ToIntegerOrInfinity (§3.6)
- *(array)* splice coerces start/deleteCount via ToIntegerOrInfinity (§3.6)
- *(array)* copyWithin runs spec-precise algorithm for hole/accessor arrays (§3.6)
- *(array)* reverse runs spec-precise algorithm for hole/accessor arrays (§3.6)
- *(array)* sort runs spec-precise SortIndexedProperties for hole/accessor arrays (§3.6)
- *(arraybuffer)* slice allocates via SpeciesConstructor (§3.6)
- *(arraybuffer)* install ArrayBuffer[Symbol.species] getter (§3.6)
- *(proxy)* Reflect.set trapless chain to an ordinary getter-only accessor → false (§3.6)
- *(string)* split ToStrings separator before limit-0; nbvm faults object sep (§3.6)
- *(string)* String.raw validates template/raw with throwing ToObject (§3.6)
- *(string)* replace/replaceAll ToString the replacement value (§3.6)
- *(functions)* Function constructor ToString's its arguments (§3.6)
- *(proxy)* computed [[Set]] runs parent.[[Set]] through a proxy/setter proto (§3.6)
- *(proxy)* Reflect.set on a proxy target returns the real [[Set]] boolean (§3.6)
- *(with)* with-statement HasBinding consults the proxy has trap (§3.5)
- *(array)* flat/flatMap use ArraySpeciesCreate + CreateDataPropertyOrThrow (§3.6)
- *(array)* flat/flatMap skip absent array-like indices (HasProperty) (§3.6)
- *(array)* array holes are absent to `in`/HasProperty; flat skips holes (§3.6)
- *(array)* Array.of honors a constructor this (§3.7)
- *(proxy)* Reflect.set with a proxy receiver writes via [[DefineOwnProperty]] (§3.6)
- *(proxy)* a proxy write target runs its own [[Set]], not the cell gate (§3.6)
- *(proxy)* Reflect.ownKeys forwards trapless [[OwnPropertyKeys]] to target (§3.7)
- *(proxy)* JSON.stringify + getOwnPropertyDescriptors enumerate proxies (§3.7)
- *(proxy)* object-rest patterns run proxy-aware CopyDataProperties (§3.7)
- *(proxy)* object spread enumerates a proxy source via its traps (§3.7)
- *(proxy)* route HasProperty through the proxy trap in array-like iteration (§3.7)

### Other

- record the puremp BigInt-backend migration (CHANGELOG + ROADMAP §8)
- *(bignum)* back BigInt with puremp::Int (§8 reused-crate)
- fix string_exotic_own test — `in` on a string primitive throws
- *(test262)* remove 33 now-passing ledger entries (array precise mutators + ArrayBuffer species)
- *(test262)* remove 30 now-passing ledger entries (Promise subclassing + species)
- fix public->private intra-doc link in promise_state
- *(roadmap)* note puremp as the BigInt-layer backend candidate (§8)
- lower MSRV to Rust 1.88
- *(test262)* remove 31 more now-passing ledger entries (49 total this batch)
- *(test262)* remove 18 now-passing ledger entries (proxy/Array.of/JSON)
- *(roadmap)* record proxy-conformance progress + refresh pass-rate (§3.7)

### Added

- *(promise)* `class P extends Promise {}` subclassing — `super(executor)` now
  runs the executor and backs the (ordinary-object) subclass instance with a
  hidden `[[PromiseState]]` cell that `promise_state` follows, so construction,
  the static combinators (`P.resolve`/`all`/`race`/`allSettled`/…), `then`/`catch`,
  `await`, and microtask delivery all work on a subclass instance (a non-callable
  executor is a TypeError).
- *(promise)* `Promise[Symbol.species]` — Promise now carries the shared species
  getter (returns `this`), so `Promise[Symbol.species] === Promise`, a subclass
  inherits it (`class P extends Promise {}` → `P[Symbol.species] === P`), and
  `then`/`catch` produce subclass instances via SpeciesConstructor (an explicit
  `@@species` override is still honored) (ROADMAP §3.6 Promise).

### Changed

- *(bignum)* `BigInt` is now backed by [`puremp`](https://crates.io/crates/puremp)
  (the Karpelès Lab pure-Rust multi-precision maths crate) instead of the
  in-house arbitrary-precision integer: `src/bignum.rs` is a thin wrapper
  (`pub struct BigInt(puremp::Int)`) preserving the prior API, so `Cell::BigInt`
  and every call site are unchanged. Added `default-features = false,
  features = ["int"]` (no_std + `alloc`, MSRV 1.88). Truncated `div_rem`,
  two's-complement bit ops, `mod_2k` (BigInt64Array wrapping), and `write_radix`
  map directly; verified against the full bignum test module and the BigInt
  Test262 semantics (ROADMAP §8).

### Fixed

- *(array)* `Array.prototype.sort` runs the spec-precise `SortIndexedProperties`
  when the array has hole or accessor-override indices: it collects present
  elements via `[[Get]]` (index getters fire), sorts them, writes the sorted
  values back via `[[Set]]` (setters fire), and `DeletePropertyOrThrow`s the
  trailing indices. Dense arrays and typed arrays keep the fast in-memory sort
  (ROADMAP §3.6).
- *(arraybuffer)* `ArrayBuffer[Symbol.species]` getter is installed (returns
  `this`, so a subclass inherits it), and `ArrayBuffer.prototype.slice` allocates
  its result via `SpeciesConstructor(O, %ArrayBuffer%)` — a subclass instance for
  a subclass receiver, a non-constructor species is a TypeError, and the result
  is validated as a distinct ArrayBuffer of sufficient length (ROADMAP §3.6).
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
