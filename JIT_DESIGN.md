# JIT completion — design & pass spine (ROADMAP §2.1)

Extending the baseline numeric JIT (`src/jit.rs`) to object/string/array/call code.
Grounded in a full read of `gc.rs`/`heap.rs`/`nanbox.rs`/`ic.rs`/`shape.rs`/`nbvm.rs`.

## Key facts (measured, not assumed)

- **GC is compacting** (`Heap::compact_to`, `gc::compact_with`) **but never triggered
  mid-execution**: allocation sites do no threshold check and never call `collect`;
  GC is host/test/snapshot-driven, run only when no JS frames are live. **Therefore a
  JIT body that allocates/re-enters is GC-safe by construction today** — no live JIT
  frame is ever exposed to a moving collection.
- **NanBox**: a `u64`. Heap ref = `is_handle()` (not-a-number ∧ sign bit); payload =
  low 48 bits → `Handle::from_raw` (32-bit slot + 16-bit generation). Numbers/immediates
  self-describing. A stack map need only mark "slot holds a NanBox" — relocation tests
  `is_handle()` itself.
- **JIT ABI today**: `extern "C" fn(i64…)->i64` (Int) / `fn(f64…)->f64` (Float), no
  `&mut Realm`. `call_guarded` unboxes args → deopts (`None`) on a non-numeric arg;
  bail = "don't take the branch", no in-flight native state to unwind.
- **Tier-up**: `ctx.tiers[id]` counter → `TIER_UP_THRESHOLD` optimizes bytecode; JIT
  attempted when `optimized ∧ !is_async ∧ rest_from.is_none() ∧ n_captures==0`;
  `ensure_jit` memoizes `Rc<JitProto>` + a `registry: BTreeMap<funcid,codeaddr>` so
  JIT→JIT static calls are native.
- **Reusable runtime**: `PropertyCache` (monomorphic shape-guard→slot, `ic.rs`) +
  `Object::cached_get/cached_set` (`object.rs`); calls via `call_with`; throw =
  `VmError::Thrown(NanBox)` on a per-frame handler stack.
- **Safepoint/stackmap infra**: greenfield (zero hits). `host_persistent` is the only
  pin table; it's forwarded across compaction.

## The substrate (Pass 1 — the gating foundation)

Introduce a **generic value tier** operating on `NanBox` (u64) values, able to re-enter
the interpreter for anything non-numeric.

1. **Calling convention.** New entry ABI:
   `extern "C" fn(ctx: *mut Ctx, args: *const u64, n: usize) -> u64`.
   `ctx` is the live `nbvm::Ctx` (realm + caches). Result is a NanBox (u64). A reserved
   NanBox sentinel `TAG_JIT_DEOPT` (or a separate out-param flag) signals "bail to
   interpreter"; a second sentinel path signals "pending exception — deopt so the
   interpreter re-runs the op and throws with correct semantics."
2. **Register discipline.** Pin `ctx` in a callee-saved register (e.g. `r15`) across the
   body; the value/register file (NanBoxes) lives in rbp-relative spill slots so the
   whole frame is describable by a (future) stack map — one bit per slot: "is-NanBox".
3. **Runtime-helper ABI.** A table of `extern "C"` helpers, each
   `fn(ctx: *mut Ctx, a: u64, b: u64, …) -> u64` that reconstructs `&mut Ctx`, does the
   heap work, and returns a NanBox (or sets `ctx.jit_pending` + returns the deopt
   sentinel on throw). The assembler emits a System-V call: save caller-saved live
   NanBox temps to spill slots (they're roots-in-waiting), load args into `rdi/rsi/…`,
   `call`, check the sentinel, continue or exit-deopt.
4. **Emitting a helper call** (`X64Assembler` additions): `mov rdi, r15` (ctx); args from
   spill slots; `mov rax, <helper_addr>; call rax`; compare `rax` to the deopt sentinel
   → `je deopt_exit`; else store `rax`.
5. **GC-safety, stated honestly.** Because GC is not mid-execution-triggered, a
   helper-call safepoint cannot collect today, so no rooting is required for correctness
   *now*. Forward-insurance: a `Ctx::jit_shadow: Vec<u64>` root that a helper-call
   sequence spills live NanBox temps into before `call` and reloads after; register it
   in the realm root set. Land the hook; wiring it to an allocation-triggered GC is a
   later phase. **Do not claim GC-safety under concurrent collection until the shadow
   stack is proven.**
6. **First end-to-end op**: generic `Add` — inline the both-numbers fast path (guard
   `is_number` on both, `addsd`/int-add, box), else `call jit_helper_add(ctx,a,b)` which
   runs the interpreter's `+` (ToPrimitive/string-concat/throw). Deopt on the throw
   sentinel. Differentially verify: a hot function doing `a+b` over mixed types matches
   the interpreter exactly.

**Exit criterion for Pass 1**: a hot function containing one non-numeric op runs as
native code, returns via the helper, deopts correctly on the exception path, and is
differentially identical to the interpreter under the existing JIT test harness +
`KATAAN_TEST262_FILTER` smoke, with the JIT forced on.

## Lowering passes (2–5, parallel once Pass 1 lands)

Each: add helper(s) + emit an inline fast path + a helper-call slow path, extend
`lower_nbvm_generic`, differentially verify, no interpreter regression.

- **Pass 2 — property access**: `Op::GetProp`/`SetProp` → inline monomorphic
  shape-guard (`Rc::ptr_eq` on the shape, load `slots[slot]`) with the `PropertyCache`
  reused via a per-site IC embedded in the code; miss → `jit_helper_get_prop`. Deopt on
  accessor/proxy/missing.
- **Pass 3 — array elements**: `arr[i]` load/store fast path (dense element-kind guard,
  bounds check) → helper on hole/oob/typed/accessor.
- **Pass 4 — string/rope**: length, char access, concat fast paths → helper.
- **Pass 5 — calls**: `Op::Call` to a JIT callee = native call via `registry`; to a
  non-JIT/native callee = `jit_helper_call(ctx, fn, this, argv)`; argument marshaling;
  exception propagation via the deopt sentinel.

## Pass 6 — tiering/deopt polish + backend

Generalize deopt (guard-failure exit restoring interpreter regs), OSR for hot loops,
and (large, separate) a Cranelift-style shared backend + aarch64 — out of scope for the
first build-out; leave the x86-64 template assembler in place.

## Verification discipline (every pass)

- Differential: same function through JIT-forced vs interpreter → identical result/throw.
- `cargo test --lib` green; `KATAAN_TEST262_FILTER` smoke on affected areas green.
- `cargo build --no-default-features --features alloc` still compiles (JIT is
  `feature="jit"` + linux/x86_64 gated; the generic tier must stay behind those cfgs).

---

## STATUS (Passes 1-5 landed, 2026-07-08)

Generic value tier compiles: `Arg`/`Const`/`Move`/`Return`, `+`/`-`/`*`/`/`/`%`,
comparisons (`<`/`==`/`===`/`!=`/`!==`/`**`/bitwise/shifts), `Not`, control flow
(`Jump`/`JumpIfFalse`, forward+backward → loops), `GetProp`/`SetProp`, `Call`/
`CallNative`, `GetKey`/`SetKey`/`ArrayLen`. Value ops keep an inline both-numbers fast
path; the rest take a shared `vm_*` interpreter helper (so tiers can't diverge). 36
differential tests. lib 866 / jit 948 green; no_std compiles; corpus green.

Still bails whole-function: numeric-narrowed `Op::Add` (Int/Float tiers own it), `Neg`/
`BitNot`/`TypeOf`/`InstanceOf`/`HasProp`/`DeleteProp`, iterator/enum ops, object/array/
class construction, accessor definition, spread, `CallCtor`/`CallValue`/tail, generators/
async, closures with captures, `rest_from`, `n_regs>64`/`n_params>32`.

**Pass 6 (the remainder):** inline machine-code shape/dense-element guards (blocked on a
`repr(C)` heap/`Object` layout — every fast path is helper-routed until then), generalized
guard-failure deopt + OSR, an ABI-tagged registry for direct generic JIT→JIT calls, and
the Cranelift-style shared backend + aarch64. These are the "make it fast + portable"
layer; op-coverage + correctness are done.

## STATUS UPDATE — Pass 6 inline fast paths landed (2026-07-09)

Inline machine-code fast paths now emitted for `GetProp`/`SetProp`/`GetElem`/`SetElem`
(commits ac1de8e repr → 4807a67 GET → ba3e12d SET+element). Safety bedrock: a
`compute_jit_layout()` probe harness derives every offset/discriminant by pointer
arithmetic on real instances (NEVER hand-baked) + gate tests (`jit_layout_matches_safe_reads`,
`jit_set_and_array_layout_matches_safe_reads`) prove them against safe reads incl.
post-arena-realloc. The harness caught a real trap: `Vec`'s data ptr is at +8 (cap at +0)
on this toolchain. Verified offsets (this build): slot_stride=200, slot_gen@2, cell_tag@8,
od_tag@16, shape_rc@24, slots_ptr@40, slots_len@48, array_ptr@24, array_len@32,
Object.frozen@176, readonly_len@144, PropertyCache shape/slot@0/8.
Soundness: SET inline only when val is non-handle (skips the generational write barrier
safely) AND frozen=0 AND readonly empty (== cached_set precondition); array store only when
unrestricted (not frozen/sealed, no per-index override, dense). Any miss → the existing
correct helper. Heap-churn differentials (2000 iters, arena reallocating) JIT===interpreter.
lib 866 / jit 958 green. Remaining helper-only: `ArrayLen`, `Call`, `CallNative`; plus
generalized deopt/OSR + shared backend + aarch64.
