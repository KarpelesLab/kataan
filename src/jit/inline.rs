//! Call inlining for the generic tier (`ROADMAP.md` §2.1): a call site whose
//! callee the interpreter (or an earlier compilation) saw always be one small
//! VM function gets that function's body spliced into the caller's bytecode,
//! behind a guard, so the call costs no call sequence and the callee's code is
//! optimized together with its caller's (the receiver cache, the numeric
//! types, the promoted loop registers all see through it). The sites are
//! method calls, calls of a value, property reads that call an inherited
//! getter, and `new` of a small base constructor (whose guard also
//! allocates `this`, as the site's `call_seq::ConstructFast` record says,
//! into the callee's `this` register; see [`construct_ok`]).
//!
//! The splice happens on the tier's **own copy** of the caller's bytecode,
//! before lowering, and uses only ordinary ops besides two of its own
//! ([`Op::InlineGuard`], [`Op::InlineLeave`]), so every analysis of the
//! emitter applies to the inlined code unchanged. A call `dst = f(args)` at
//! pc `P` becomes:
//!
//! ```text
//! G     InlineGuard { dst: t, call: F, key, … }  -- the callee is `key`'s
//!                                               -- function: activation
//!                                               -- pending, captures loaded
//! G+1   JumpIfFalse t → F
//!       parameters, `this` and every other callee register initialized
//!       exactly as a call's register window is (a parameter or `this` the
//!       body never writes is not copied: the body reads the caller's
//!       register, which nothing writes while it runs: see `copied_param`)
//!       the callee's body, registers moved up into a sub-window above the
//!       caller's (`return r` becomes `dst = r` and a jump to L)
//!       dst = undefined                          -- falling off its end
//! L     InlineLeave                              -- activation ended
//! L+1   Jump E
//! F     the call op itself                       -- any other callee
//! E     (the op after the call)
//! ```
//!
//! The inlined activation is exact. The guard identifies the callee as the
//! call sequence does and checks the same limits (a call at the depth limit
//! throws its `RangeError` from the call op). The native-call header and
//! depth counts the call sequence would push are pushed when Rust code is
//! about to run inside the inlined body — at its first `call` — which is
//! the only code that can observe them (`f.caller`, `f.arguments`, the depth
//! of a nested call, a collection's audit); see
//! `call_seq::emit_inline_guard`. A throw out of the inlined body ends the
//! activation at the body's throw exit. Only small callees are inlined — no
//! `try`, `arguments`, `new.target`, `super`, eval or suspension in them —
//! so nothing can leave the compiled code inside one. A callee may loop
//! (`loop_ok`): an on-stack exit is at a loop header, and an inlined
//! loop's back-edge polls (interrupts, agents, the GC safepoint, all of
//! which run Rust code, so it first pushes the pending activation as a call
//! would) but never leaves there (`jit_tier::wants_to_leave` refuses a
//! header that is not the function's own, and the header is never an
//! on-stack-replacement entry). An inlined body may call: a call op in it (or a getter, a
//! `valueOf`, … that a property access or an operator calls) first pushes
//! the pending activation, like any Rust code run inside it, and a throw
//! out of the callee ends the inlined activation as any other throw does.
//! (A native call sequence instead pushes the activations still pending
//! with its own header and pops them with it, so they stay pending: see
//! `call_seq::InlineCtx`.)
//!
//! An inlined callee's own call sites may be inlined into its spliced body
//! in turn ([`Candidate::nested`], one more level): such an inner
//! activation runs inside the outer one, which may still be pending when the
//! inner guard hits, and the inner stub then pushes the outer activation
//! first (`call_seq::NestedSlots`). A `Vec.add` that returns `new Vec(…)`
//! inlines both the method and the constructor this way.
//!
//! The spliced bytecode numbers its ops differently from the function's own:
//! the body keeps both maps (`orig_of`, `new_of`), and every pc that crosses
//! between it and the interpreter — an on-stack-replacement entry, the header
//! a recompilation exit resumes at, a handler's catch pc — is the function's
//! own.

use crate::ic::WarmEntry;
use crate::nanbox::NanBox;
use crate::nbvm::{FnProto, Op, Reg};
use alloc::vec::Vec;

/// The largest callee body inlined (in ops).
const MAX_CALLEE_OPS: usize = 40;
/// The most callee registers inlined.
const MAX_CALLEE_REGS: usize = 48;
/// The largest callee body inlined once it has inlined calls of its own
/// (in ops; the callee itself is at most [`MAX_CALLEE_OPS`]).
const MAX_SPLICED_OPS: usize = 100;
/// The most registers of such a body.
const MAX_SPLICED_REGS: usize = 96;

/// The op and register limits of an inlined body (`spliced`: one that
/// inlined calls of its own).
const fn limits(spliced: bool) -> (usize, usize) {
    if spliced {
        (MAX_SPLICED_OPS, MAX_SPLICED_REGS)
    } else {
        (MAX_CALLEE_OPS, MAX_CALLEE_REGS)
    }
}

/// The most ops inlining may add to one caller.
const MAX_ADDED_OPS: usize = 600;
/// The most callees one polymorphic site inlines (`Op::InlineGuard::poly`).
const MAX_WAYS: usize = 3;

/// A callee to inline at a call site, as the caller's compiler is told it.
pub struct Candidate<'a> {
    /// `function id | capture count << 32` (the call sequence's callee key).
    pub key: u64,
    /// The callee's function.
    pub proto: &'a FnProto,
    /// The callee's warm inline caches, by its own pc.
    pub warm: Vec<Option<WarmEntry>>,
    /// The callee's own call sites to inline into its spliced body, by its
    /// own pc (one level: their candidates' `nested` are ignored — but at a
    /// `super(…)` site, whose callee's own `super(…)` sites inline one more
    /// level, see [`super_nested`]).
    pub nested: Vec<(usize, Candidate<'a>)>,
}

/// A site's callee as spliced: its function, or (when it inlines calls of
/// its own, see [`Candidate::nested`]) a copy of it with those calls
/// spliced in, and its warm caches by that body's pc.
struct Site<'a> {
    key: u64,
    proto: alloc::borrow::Cow<'a, FnProto>,
    warm: Vec<Option<WarmEntry>>,
}

/// The third inlining level (`Candidate::nested`): the `super(…)` sites of
/// `n`, a derived constructor inlined at a `super(…)` site itself, with
/// their candidates (and none deeper). A chain of derived class
/// constructors is inlined whole that way, `new C` through `super(…)` to
/// the base constructor that allocates.
fn super_nested<'a>(n: &Candidate<'a>) -> Vec<(usize, Candidate<'a>)> {
    if !n.proto.derived {
        return Vec::new();
    }
    n.nested
        .iter()
        .filter(|(p, _)| n.proto.ops.get(*p).and_then(super_parts).is_some())
        .map(|(p, m)| {
            (
                *p,
                Candidate {
                    key: m.key,
                    proto: m.proto,
                    warm: m.warm.clone(),
                    nested: Vec::new(),
                },
            )
        })
        .collect()
}

/// `c`'s body as spliced (see [`Site`]).
fn site_body(c: Candidate<'_>) -> Site<'_> {
    let Candidate {
        key,
        proto,
        warm,
        nested,
    } = c;
    let inner = (!nested.is_empty())
        .then(|| {
            expand(proto, &warm, &|pc, _| {
                nested
                    .iter()
                    .filter(|(p, _)| *p == pc)
                    .map(|(p, n)| Candidate {
                        key: n.key,
                        proto: n.proto,
                        warm: n.warm.clone(),
                        nested: if super_parts(&proto.ops[*p]).is_some() {
                            super_nested(n)
                        } else {
                            Vec::new()
                        },
                    })
                    .collect()
            })
        })
        .flatten();
    match inner {
        Some(e) => {
            let mut p = proto.clone();
            p.ops = e.ops;
            p.n_regs = e.n_regs;
            Site {
                key,
                proto: alloc::borrow::Cow::Owned(p),
                warm: e.warm,
            }
        }
        None => Site {
            key,
            proto: alloc::borrow::Cow::Borrowed(proto),
            warm,
        },
    }
}

/// A caller's bytecode with its inlined calls spliced in.
pub struct Expanded {
    /// The ops.
    pub ops: Vec<Op>,
    /// The register count (the caller's plus the inlined sub-window).
    pub n_regs: usize,
    /// The warm inline caches, by new pc.
    pub warm: Vec<Option<WarmEntry>>,
    /// Per new pc (and one past the end): the function's own pc, or
    /// `u32::MAX` inside an inlined body.
    pub orig_of: Vec<u32>,
    /// Per own pc (and one past the end): its new pc.
    pub new_of: Vec<u32>,
}

/// Whether `proto` can be inlined: small, a plain call target, and every op
/// one the splice can renumber and that cannot leave the activation other
/// than by its return or a throw.
#[must_use]
pub fn callee_ok(proto: &FnProto) -> bool {
    callee_ok_sized(proto, false)
}

/// [`callee_ok`], for a callee body that inlined calls of its own when
/// `spliced` (see [`Candidate::nested`]; the function itself was small
/// enough, and its inlined calls may add up to [`MAX_SPLICED_OPS`]).
fn callee_ok_sized(proto: &FnProto, spliced: bool) -> bool {
    let (max_ops, max_regs) = limits(spliced);
    if proto.class_ctor
        || proto.is_async
        || proto.is_generator
        || proto.rest_from.is_some()
        || proto.ops.is_empty()
        || proto.ops.len() > max_ops
        || proto.n_regs > max_regs
        || proto.n_params + proto.n_captures >= proto.n_regs
    {
        return false;
    }
    let n = proto.ops.len();
    proto.ops.iter().enumerate().all(|(pc, op)| match op {
        Op::Jump { target } => loop_ok(proto, pc, *target),
        Op::JumpIfFalse { target, .. } => *target > pc && *target <= n,
        // (Only a derived constructor's own, inlined at a construct site.)
        Op::SuperCtor { .. }
        | Op::InitFields { .. }
        | Op::DerivedResult { .. }
        | Op::SuperCallArgs { .. } => false,
        _ => {
            let mut o = op.clone();
            remap(&mut o, &|r| r, &|t| t)
        }
    })
}

/// Whether callee op `pc`, `Jump { target }`, is a branch the splice takes:
/// forward (to an op or the end), or a loop's back-edge — to an earlier op
/// no branch from outside the loop enters it past (a loop the body enters
/// only at its header). An inlined loop is never left at its header: its
/// back-edge polls as any other, but the activation it runs in is pending,
/// so the interpreter could not resume there (`jit_tier::wants_to_leave`
/// refuses).
fn loop_ok(proto: &FnProto, pc: usize, target: usize) -> bool {
    if target > pc {
        return target <= proto.ops.len();
    }
    proto.ops.iter().enumerate().all(|(i, o)| match o {
        Op::Jump { target: t } | Op::JumpIfFalse { target: t, .. } => {
            (target..=pc).contains(&i) || !(target < *t && *t <= pc)
        }
        _ => true,
    })
}

/// Whether `p` has a loop (a backward branch).
fn loops(p: &FnProto) -> bool {
    p.ops
        .iter()
        .enumerate()
        .any(|(pc, o)| matches!(o, Op::Jump { target } if *target <= pc))
}

/// Whether `proto` can be inlined at a `new` site (`Op::Construct`): a
/// small base constructor (a base class constructor or an ordinary
/// function) — the kinds a site's `call_seq::ConstructFast` record, which
/// the guard requires, serves with an allocated `this` — whose body never returns explicitly (falling off its
/// end yields `this`), reads `new.target` only through `Op::LoadNewTarget`
/// (the constructor itself, which such a `new` passes as `new.target`), and
/// is otherwise what [`callee_ok`] accepts.
#[must_use]
pub fn construct_ok(proto: &FnProto) -> bool {
    construct_ok_sized(proto, false)
}

/// [`construct_ok`] as [`callee_ok_sized`] is [`callee_ok`].
fn construct_ok_sized(proto: &FnProto, spliced: bool) -> bool {
    let (max_ops, max_regs) = limits(spliced);
    if proto.is_async
        || proto.is_generator
        || proto.rest_from.is_some()
        || proto.ops.is_empty()
        || proto.ops.len() > max_ops
        || proto.n_regs > max_regs
        || proto.n_params + proto.n_captures >= proto.n_regs
        || (proto.derived && !derived_ok(proto))
    {
        return false;
    }
    let n = proto.ops.len();
    proto.ops.iter().enumerate().all(|(pc, op)| match op {
        Op::Jump { target } => loop_ok(proto, pc, *target),
        Op::JumpIfFalse { target, .. } => *target > pc && *target <= n,
        Op::LoadNewTarget { .. } => true,
        // (A derived constructor's own end, see `derived_ok`.)
        Op::Return { .. } => proto.derived && pc + 1 == n,
        _ => {
            let mut o = op.clone();
            remap(&mut o, &|r| r, &|t| t)
        }
    })
}

/// Whether the derived class constructor `p` has the one shape inlined at a
/// construct site (see [`construct_site`]): its `this` (the register after
/// the captures, which the guard leaves unbound) is written by exactly one
/// op, a `super(…)` with its arguments in registers (`Op::SuperCallArgs`)
/// that every run reaches once — no branch before it, none after it back
/// to it or before it — and the body ends with the implicit
/// `return undefined` (`LoadConst undefined`, `Op::DerivedResult` of it,
/// `return`), which nothing branches past the `LoadConst` into and no
/// other `return` precedes. Then a run that does not throw binds `this`
/// exactly once, to the object its super constructor made, and returns
/// it: `[[Construct]]`'s checks (an object result; `this` bound) hold, so
/// the inlined body's result is that `this`.
fn derived_ok(p: &FnProto) -> bool {
    let n = p.ops.len();
    let this = (p.n_params + p.n_captures) as Reg;
    let [
        ..,
        Op::LoadConst { dst: u, value },
        Op::DerivedResult {
            dst: d,
            src,
            this: t,
        },
        Op::Return { src: r },
    ] = &p.ops[..]
    else {
        return false;
    };
    if value.to_bits() != NanBox::undefined().to_bits() || src != u || *t != this || r != d {
        return false;
    }
    // The `super(…)`: the one call op writing `this`. (In a body that
    // inlined it, `Candidate::nested`, its guards' blocks precede it, from
    // `g`, each binding `this` after its leave, and branch only within
    // them or to just past the call.)
    let Some(s) = p.ops.iter().position(
        |o| matches!(o, Op::SuperCallArgs { this: st, dst, .. } if *st == this && *dst != this),
    ) else {
        return false;
    };
    let g = p
        .ops
        .iter()
        .position(|o| matches!(o, Op::InlineGuard { call, .. } if *call == s))
        .unwrap_or(s);
    let binds = |pc: usize| {
        (g..s).contains(&pc)
            && pc > 0
            && matches!(p.ops[pc - 1], Op::InlineLeave)
            && matches!(p.ops[pc], Op::Move { dst, .. } if dst == this)
    };
    if p.ops
        .iter()
        .enumerate()
        .any(|(pc, o)| pc != s && !binds(pc) && writes_reg(o, this))
    {
        return false;
    }
    p.ops.iter().enumerate().all(|(pc, o)| match o {
        Op::Jump { target } | Op::JumpIfFalse { target, .. } => {
            if (g..=s).contains(&pc) {
                *target > g && *target <= s + 1
            } else {
                pc > s && *target > s && *target <= n - 3
            }
        }
        Op::Return { .. } => pc + 1 == n,
        // (An inlined super constructor's own is its body's.)
        Op::DerivedResult { this: t, .. } => *t != this || pc + 2 == n,
        // (`super(...spread)`, `eval`, `arguments`: not this shape.)
        Op::SuperCall { .. } => false,
        _ => true,
    })
}

/// Whether `op` is a construct site an inlined body allocates or leaves
/// unbound `this` for: a `new` (whose `new.target` is the constructor) or
/// a `super(…)` with its arguments in registers (`new.target` a register;
/// the result also binds the caller's `this`).
fn construct_site(op: &Op) -> bool {
    matches!(op, Op::Construct { .. } | Op::SuperCallArgs { .. })
}

/// Whether every call op in `p` (a callee body, maybe with calls of its own
/// spliced in) is the fallback of one of its inlined calls' guards: run as
/// guarded, it makes no call.
fn calls_only_inlined(p: &FnProto) -> bool {
    let fallbacks: Vec<usize> = p
        .ops
        .iter()
        .filter_map(|o| match o {
            Op::InlineGuard { call, .. } => Some(*call),
            _ => None,
        })
        .collect();
    p.ops.iter().enumerate().all(|(pc, o)| {
        !matches!(
            o,
            Op::CallMethod { .. }
                | Op::CallValue { .. }
                | Op::CallWithMethod { .. }
                | Op::CallKey { .. }
                | Op::Construct { .. }
                | Op::SuperCallArgs { .. }
                | Op::SuperCall { .. }
        ) || fallbacks.contains(&pc)
    })
}

/// Whether every `super(…)` in `p` (a callee body, maybe with calls of its
/// own spliced in) is the fallback of one of its inlined calls' guards.
fn super_inlined(p: &FnProto) -> bool {
    p.ops.iter().enumerate().all(|(pc, o)| {
        !matches!(o, Op::SuperCallArgs { .. } | Op::SuperCall { .. })
            || p.ops
                .iter()
                .any(|g| matches!(g, Op::InlineGuard { call, .. } if *call == pc))
    })
}

/// Renumbers `op`'s registers by `reg` and its branch targets by `target`;
/// `false` (leaving `op` partly changed) for an op the splice does not take.
fn remap(op: &mut Op, reg: &dyn Fn(Reg) -> Reg, target: &dyn Fn(usize) -> usize) -> bool {
    match op {
        Op::LoadConst { dst, .. }
        | Op::LoadGlobal { dst, .. }
        | Op::TypeofGlobal { dst, .. }
        | Op::NewObject { dst }
        | Op::NewBox { dst }
        | Op::NewString { dst, .. }
        | Op::NewArray { dst, .. } => *dst = reg(*dst),
        Op::Add { dst, a, b }
        | Op::Sub { dst, a, b }
        | Op::Mul { dst, a, b }
        | Op::Div { dst, a, b }
        | Op::Mod { dst, a, b }
        | Op::ValueBin { dst, a, b, .. }
        | Op::Compare { dst, a, b, .. }
        | Op::Lt { dst, a, b }
        | Op::AddValue { dst, a, b }
        | Op::StrictEq { dst, a, b }
        | Op::Relational { dst, a, b, .. } => {
            *dst = reg(*dst);
            *a = reg(*a);
            *b = reg(*b);
        }
        Op::TypeOf { dst, a } | Op::BitNot { dst, a } | Op::Neg { dst, a } | Op::Not { dst, a } => {
            *dst = reg(*dst);
            *a = reg(*a);
        }
        Op::Move { dst, src }
        | Op::ToStr { dst, src }
        | Op::ToStrLazy { dst, src }
        | Op::ToNumeric { dst, src }
        | Op::Step { dst, src, .. }
        | Op::ToKey { dst, src }
        | Op::CellGet { dst, cell: src }
        | Op::ArrayLen { dst, arr: src }
        | Op::GetProp { dst, obj: src, .. } => {
            *dst = reg(*dst);
            *src = reg(*src);
        }
        Op::GetElem {
            dst,
            arr: a,
            index: b,
        }
        | Op::GetKey {
            dst,
            obj: a,
            key: b,
        } => {
            *dst = reg(*dst);
            *a = reg(*a);
            *b = reg(*b);
        }
        Op::SetElem {
            arr: a,
            index: b,
            src,
        }
        | Op::SetKey {
            obj: a,
            key: b,
            src,
        }
        | Op::SetKeyStrict {
            obj: a,
            key: b,
            src,
        } => {
            *a = reg(*a);
            *b = reg(*b);
            *src = reg(*src);
        }
        Op::SetProp { obj, src, .. }
        | Op::SetPropStrict { obj, src, .. }
        | Op::CellSet { cell: obj, src } => {
            *obj = reg(*obj);
            *src = reg(*src);
        }
        Op::Concat { dst, parts }
        | Op::NewArrayFrom { dst, elems: parts }
        | Op::PlainPrims { dst, regs: parts } => {
            *dst = reg(*dst);
            for p in parts.iter_mut() {
                *p = reg(*p);
            }
        }
        Op::StoreGlobal { src, resolved, .. } => {
            *src = reg(*src);
            if let Some(r) = resolved {
                *r = reg(*r);
            }
        }
        Op::RequireObjectCoercible { src, .. }
        | Op::CheckTdz { src }
        | Op::Throw { src }
        | Op::Return { src }
        | Op::BindThis { this: src } => *src = reg(*src),
        Op::ThrowTypeError { .. } | Op::ThrowReferenceError { .. } => {}
        Op::MakeClosure { dst, captures, .. } => {
            *dst = reg(*dst);
            for a in captures.iter_mut() {
                *a = reg(*a);
            }
        }
        Op::LoadFunc { dst, .. } => *dst = reg(*dst),
        Op::InitProp { obj, src, .. } => {
            *obj = reg(*obj);
            *src = reg(*src);
        }
        Op::NewObjectLit { dst, lit } => {
            *dst = reg(*dst);
            for s in lit.srcs.iter_mut() {
                *s = reg(*s);
            }
        }
        Op::HasProp { dst, key, obj } => {
            *dst = reg(*dst);
            *key = reg(*key);
            *obj = reg(*obj);
        }
        Op::InstanceOf { dst, obj, .. } => {
            *dst = reg(*dst);
            *obj = reg(*obj);
        }
        // A call the callee's body inlined itself (see `Candidate::nested`).
        Op::InlineGuard {
            dst,
            call,
            callee,
            args,
            caps,
            ..
        } => {
            *dst = reg(*dst);
            *call = target(*call);
            *callee = reg(*callee);
            *caps = reg(*caps);
            for a in args.iter_mut() {
                *a = reg(*a);
            }
        }
        Op::InlineLeave => {}
        // A derived constructor's own steps (`derived_ok`).
        Op::SuperCtor { dst, f } => {
            *dst = reg(*dst);
            *f = reg(*f);
        }
        Op::InitFields { ctor, this } => {
            *ctor = reg(*ctor);
            *this = reg(*this);
        }
        Op::DerivedResult { dst, src, this } => {
            *dst = reg(*dst);
            *src = reg(*src);
            *this = reg(*this);
        }
        Op::SuperCallArgs {
            dst,
            ctor,
            new_target,
            args,
            this,
        } => {
            *dst = reg(*dst);
            *ctor = reg(*ctor);
            *new_target = reg(*new_target);
            *this = reg(*this);
            for a in args.iter_mut() {
                *a = reg(*a);
            }
        }
        // Calls: an inlined body may call (see the module docs).
        Op::GetMethod { dst, obj, .. } => {
            *dst = reg(*dst);
            *obj = reg(*obj);
        }
        Op::CallMethod {
            dst, recv: c, args, ..
        }
        | Op::CallValue {
            dst,
            callee: c,
            args,
            ..
        }
        | Op::Construct { dst, ctor: c, args } => {
            *dst = reg(*dst);
            *c = reg(*c);
            for a in args.iter_mut() {
                *a = reg(*a);
            }
        }
        Op::CallWithMethod {
            dst,
            callee: c,
            recv: k,
            args,
            ..
        }
        | Op::CallKey {
            dst,
            recv: c,
            key: k,
            args,
        } => {
            *dst = reg(*dst);
            *c = reg(*c);
            *k = reg(*k);
            for a in args.iter_mut() {
                *a = reg(*a);
            }
        }
        Op::Jump { target: t } => *t = target(*t),
        Op::JumpIfFalse { cond, target: t } => {
            *cond = reg(*cond);
            *t = target(*t);
        }
        _ => return false,
    }
    true
}

/// The call op at a site, as `(dst, callee or receiver, args, this)`: a
/// method call (`this` its receiver), a call of a value (`this` undefined),
/// a call of a method read before its arguments, a property read that
/// calls an inherited getter, or a property write that calls an inherited
/// setter (whose result is dropped: `dst` is [`Reg::MAX`], see
/// [`discards`]).
fn call_parts(op: &Op) -> Option<(Reg, Reg, &[Reg], Option<Reg>)> {
    match op {
        Op::GetProp { dst, obj, .. } => Some((*dst, *obj, &[], Some(*obj))),
        Op::SetProp { obj, src, .. } | Op::SetPropStrict { obj, src, .. } => {
            Some((Reg::MAX, *obj, core::slice::from_ref(src), Some(*obj)))
        }
        // (`this` is the object the guard allocates.)
        Op::Construct { dst, ctor, args } => Some((*dst, *ctor, args, None)),
        // (Likewise; `new.target` and the `this` the result binds are
        // the op's own, see `super_parts`.)
        Op::SuperCallArgs {
            dst, ctor, args, ..
        } => Some((*dst, *ctor, args, None)),
        Op::CallMethod {
            dst,
            recv,
            key,
            args,
        } if key != "caller" => Some((*dst, *recv, args, Some(*recv))),
        Op::CallValue {
            dst, callee, args, ..
        } => Some((*dst, *callee, args, None)),
        Op::CallWithMethod {
            dst,
            callee,
            recv,
            args,
            ..
        } => Some((*dst, *callee, args, Some(*recv))),
        _ => None,
    }
}

/// Whether the call at a site has no destination (a property write's
/// setter, whose result is dropped): its inlined body's result goes to a
/// scratch register above every way's window instead.
fn discards(op: &Op) -> bool {
    matches!(op, Op::SetProp { .. } | Op::SetPropStrict { .. })
}

/// A computed-key call `recv[key](args)` (`Op::CallKey`) is inlined as its
/// two halves: the element read into the scratch register `m` (above the
/// callee's window), then the call of `m` with `this` the receiver —
/// `Op::CallWithMethod` with no key, the plain call, exactly `Op::CallKey`'s
/// call half (the arguments are registers the read cannot change). `None`
/// for any other op.
fn split_call_key(op: &Op, m: Reg) -> Option<(Op, Op)> {
    let Op::CallKey {
        dst,
        recv,
        key,
        args,
    } = op
    else {
        return None;
    };
    Some((
        Op::GetKey {
            dst: m,
            obj: *recv,
            key: *key,
        },
        Op::CallWithMethod {
            dst: *dst,
            callee: m,
            recv: *recv,
            key: alloc::string::String::new(),
            args: args.clone(),
        },
    ))
}

/// Splices the callees `pick` names into `proto`'s bytecode (see the module
/// docs); `warm` is the caller's warm inline caches by its own pc. `None` when
/// no call was inlined.
#[must_use]
pub fn expand<'a>(
    proto: &FnProto,
    warm: &[Option<WarmEntry>],
    pick: &dyn Fn(usize, &Op) -> Vec<Candidate<'a>>,
) -> Option<Expanded> {
    let base = proto.n_regs;
    let mut sites: Vec<(usize, Vec<Site<'a>>)> = Vec::new();
    let mut added = 0usize;
    let mut window = 0usize;
    for (pc, op) in proto.ops.iter().enumerate() {
        // (A computed-key call's callee is its read's scratch register, above
        // the caller's: its receiver and key must be the caller's.)
        let split = split_call_key(op, Reg::MAX);
        let parts = match (&split, op) {
            (Some((_, call)), Op::CallKey { key, .. }) if (*key as usize) < base => {
                call_parts(call).map(|(d, _, a, t)| (d, 0, a, t))
            }
            (Some(_), _) => None,
            (None, _) => call_parts(op),
        };
        let Some((dst, callee, args, this)) = parts else {
            continue;
        };
        if args.len() > 32
            || ((dst as usize) >= base && !discards(op))
            || (callee as usize) >= base
            || this.is_some_and(|t| (t as usize) >= base)
            || args.iter().any(|&a| (a as usize) >= base)
            || super_parts(op).is_some_and(|(nt, t)| (nt as usize) >= base || (t as usize) >= base)
        {
            continue;
        }
        let cands = pick(pc, op);
        // A polymorphic site's ways: a call of a method or of a value
        // (whose later guards reuse the callee the first found, or read
        // its register again).
        let poly_ok = matches!(
            op,
            Op::CallMethod { .. }
                | Op::CallValue { .. }
                | Op::CallWithMethod { .. }
                | Op::CallKey { .. }
        );
        if cands.len() > 1 && !poly_ok {
            continue;
        }
        let mut ways: Vec<Site<'a>> = Vec::new();
        let poly = cands.len() > 1;
        for c in cands.into_iter().take(MAX_WAYS) {
            let c = site_body(c);
            // A polymorphic site's way inlines only a body that calls
            // nothing but through its own inlined calls' guards (an inner
            // call's activation pushes cost about what the call saved). So
            // does one that loops: its calls would push the activation on
            // every iteration.
            if (poly || loops(&c.proto)) && !calls_only_inlined(&c.proto) {
                continue;
            }
            // A derived constructor only with its own `super(…)` inlined
            // too: else that call would push the inlined activations on
            // every construct, costing about what inlining saved.
            if c.proto.derived && !super_inlined(&c.proto) {
                continue;
            }
            let spliced = matches!(c.proto, alloc::borrow::Cow::Owned(_));
            let ok = if construct_site(op) {
                construct_ok_sized(&c.proto, spliced)
            } else {
                callee_ok_sized(&c.proto, spliced)
            };
            if !ok
                || (c.key >> 32) as usize != c.proto.n_captures
                || ways.iter().any(|w| w.key == c.key)
            {
                continue;
            }
            let cost = c.proto.ops.len() + c.proto.n_regs + 6;
            if added + cost > MAX_ADDED_OPS {
                continue;
            }
            added += cost;
            window = window.max(c.proto.n_regs + 1 + usize::from(split.is_some() || discards(op)));
            ways.push(c);
        }
        if !ways.is_empty() {
            sites.push((pc, ways));
        }
    }
    if sites.is_empty() || base + window > 0xffff {
        return None;
    }
    let n_regs = base + window;
    let undefined = NanBox::undefined();
    let mut ops: Vec<Op> = Vec::with_capacity(proto.ops.len() + added);
    let mut nwarm: Vec<Option<WarmEntry>> = Vec::with_capacity(proto.ops.len() + added);
    let mut orig_of: Vec<u32> = Vec::with_capacity(proto.ops.len() + added + 1);
    let mut new_of: Vec<u32> = Vec::with_capacity(proto.ops.len() + 1);
    // First pass: each own pc's new pc (an inlined site's own pc is its
    // guard's, so a branch to the call enters the guard).
    let mut next = 0usize;
    let mut si = 0usize;
    for pc in 0..proto.ops.len() {
        new_of.push(u32::try_from(next).ok()?);
        if si < sites.len() && sites[si].0 == pc {
            next += sites[si]
                .1
                .iter()
                .map(|c| block_len(&c.proto, &proto.ops[pc]))
                .sum::<usize>()
                + 1;
            if matches!(proto.ops[pc], Op::CallKey { .. }) {
                next += 1;
            }
            si += 1;
        } else {
            next += 1;
        }
    }
    new_of.push(u32::try_from(next).ok()?);
    let own = |t: usize| new_of[t.min(proto.ops.len())] as usize;
    let mut si = 0usize;
    for (pc, op) in proto.ops.iter().enumerate() {
        let w = warm.get(pc).cloned().flatten();
        if si < sites.len() && sites[si].0 == pc {
            let site_ways = &sites[si].1;
            si += 1;
            // A `new`: the guard also allocates `this`, into the register
            // after the captures (the callee's `this` slot).
            let construct = construct_site(op);
            // A `super(…)`: its `new.target` register, and the caller's
            // `this` its result binds (after the inlined body leaves).
            let sup = super_parts(op);
            let rb = |r: Reg| -> Reg { (base + r as usize) as Reg };
            // (A computed-key call's method register is above every way's
            // window.)
            let top = site_ways.iter().map(|c| c.proto.n_regs).max().unwrap_or(0);
            let split = split_call_key(op, rb((top + 1) as Reg));
            let mut w = w;
            let op = match split {
                Some((read, call)) => {
                    // The read first (a branch to the call enters it); a
                    // throw from it is the call's, as `Op::CallKey`'s is.
                    ops.push(read);
                    nwarm.push(None);
                    orig_of.push(u32::MAX);
                    w = None;
                    alloc::borrow::Cow::Owned(call)
                }
                None => alloc::borrow::Cow::Borrowed(op),
            };
            let op = &*op;
            let (dst, callee, args, this) = call_parts(op)?;
            // (A dropped result: the scratch register above every way's
            // window.)
            let dst = if discards(op) {
                rb((top + 1) as Reg)
            } else {
                dst
            };
            // The ways' blocks in a row, then the call op itself.
            let shape = (args.len(), this.is_some());
            let fallback = ops.len()
                + site_ways
                    .iter()
                    .map(|c| block_len(&c.proto, op))
                    .sum::<usize>();
            let end = fallback + 1;
            let n_ways = site_ways.len();
            for (way, c) in site_ways.iter().enumerate() {
                let p: &FnProto = &c.proto;
                let g = ops.len();
                let tmp = rb(p.n_regs as Reg);
                let next_way = g + block_len(p, op);
                let leave = next_way - 2 - usize::from(sup.is_some());
                macro_rules! emit {
                    ($o:expr, $w:expr, $ops:expr $(,)?) => {{
                        $ops.push($o);
                        nwarm.push($w);
                        orig_of.push(u32::MAX);
                    }};
                }
                emit!(
                    Op::InlineGuard {
                        dst: tmp,
                        call: fallback,
                        key: c.key,
                        callee,
                        args: args.to_vec(),
                        caps: rb(p.n_params as Reg),
                        n_caps: (p.n_captures + usize::from(construct)) as u16,
                        poly: if n_ways > 1 {
                            (way | n_ways << 8) as u16
                        } else {
                            0
                        },
                    },
                    None,
                    ops
                );
                // The first guard's own pc is the call's (an entry, a branch to
                // it).
                if way == 0 {
                    *orig_of.last_mut()? = u32::try_from(pc).ok()?;
                }
                emit!(
                    Op::JumpIfFalse {
                        cond: tmp,
                        target: next_way,
                    },
                    None,
                    ops
                );
                // The register window, as `call_with_inner` lays it out — the
                // registers the body may read before writing them (the others'
                // values are never seen).
                let this_slot = p.n_params + p.n_captures;
                // The body's view of its registers: above the caller's, but
                // for those it shares with the caller.
                let shared: Vec<Option<Reg>> = (0..p.n_regs)
                    .map(|r| {
                        if !copied_param(p, r, construct, shape) {
                            None
                        } else if r < p.n_params {
                            args.get(r).copied()
                        } else {
                            this
                        }
                    })
                    .collect();
                let view = |r: Reg| -> Reg {
                    shared
                        .get(r as usize)
                        .copied()
                        .flatten()
                        .unwrap_or_else(|| rb(r))
                };
                for r in window_regs(p, construct) {
                    if (construct && r == this_slot) || shared[r].is_some() {
                        continue;
                    }
                    let o = if r < p.n_params {
                        match args.get(r) {
                            Some(&a) => Op::Move {
                                dst: rb(r as Reg),
                                src: a,
                            },
                            None => Op::LoadConst {
                                dst: rb(r as Reg),
                                value: undefined,
                            },
                        }
                    } else if let Some(t) = this.filter(|_| r == this_slot) {
                        Op::Move {
                            dst: rb(r as Reg),
                            src: t,
                        }
                    } else {
                        Op::LoadConst {
                            dst: rb(r as Reg),
                            value: undefined,
                        }
                    };
                    emit!(o, None, ops);
                }
                // The body: callee pc `j` at `body + body_pc(p, j)`.
                let body = ops.len();
                let tgt = |t: usize| body + body_pc(p, t, construct);
                for (j, bop) in p.ops.iter().enumerate() {
                    let bw = c.warm.get(j).cloned().flatten();
                    match width(p, j, construct) {
                        0 => continue,
                        w if matches!(bop, Op::Return { .. }) => {
                            let Op::Return { src } = bop else {
                                return None;
                            };
                            // `return r`: the call's result, then (but from the
                            // last op) to the end.
                            emit!(
                                Op::Move {
                                    dst,
                                    src: view(*src)
                                },
                                None,
                                ops,
                            );
                            if w == 2 {
                                emit!(Op::Jump { target: leave }, None, ops);
                            }
                        }
                        // `new.target`: the constructor (see `construct_ok`),
                        // or a `super(…)`'s own.
                        _ if construct && let Op::LoadNewTarget { dst: d } = bop => {
                            emit!(
                                Op::Move {
                                    dst: rb(*d),
                                    src: sup.map_or(callee, |(nt, _)| nt),
                                },
                                None,
                                ops
                            );
                        }
                        _ => {
                            let mut o = bop.clone();
                            if !remap(&mut o, &view, &tgt) {
                                return None;
                            }
                            emit!(o, bw, ops);
                        }
                    }
                }
                if falls_off(p) {
                    // A constructor's end yields `this`, a function's `undefined`.
                    let o = if construct {
                        Op::Move {
                            dst,
                            src: rb(this_slot as Reg),
                        }
                    } else {
                        Op::LoadConst {
                            dst,
                            value: undefined,
                        }
                    };
                    emit!(o, None, ops);
                }
                debug_assert_eq!(ops.len(), leave);
                emit!(Op::InlineLeave, None, ops);
                // `super(…)` binds the caller's `this` to its result (the
                // guard saw it unbound, and nothing but the caller's own
                // code writes its registers).
                if let Some((_, t)) = sup {
                    emit!(Op::Move { dst: t, src: dst }, None, ops);
                }
                emit!(Op::Jump { target: end }, None, ops);
                debug_assert_eq!(ops.len(), next_way);
            }
            debug_assert_eq!(ops.len(), fallback);
            // The call itself, with its warm cache.
            ops.push(op.clone());
            nwarm.push(w);
            orig_of.push(u32::MAX);
            continue;
        }
        let mut o = op.clone();
        match &mut o {
            Op::Jump { target }
            | Op::JumpIfFalse { target, .. }
            | Op::PushHandler { target, .. } => {
                *target = own(*target);
            }
            Op::InlineGuard { .. } | Op::InlineLeave => return None,
            _ => {}
        }
        ops.push(o);
        nwarm.push(w);
        orig_of.push(u32::try_from(pc).ok()?);
    }
    orig_of.push(u32::try_from(proto.ops.len()).ok()?);
    debug_assert_eq!(ops.len() + 1, orig_of.len());
    Some(Expanded {
        ops,
        n_regs,
        warm: nwarm,
        orig_of,
        new_of,
    })
}

/// How many ops callee op `j` takes in the spliced body: a `return` is a
/// move of the result and (but for the last op) a jump to the end; a sloppy
/// function's `this` binding is dropped when nothing reads `this` (as the
/// lowering drops it in the function's own code: computing the bound value
/// has no effect) or, at a `new` site (`construct`), when it binds the
/// guard's freshly allocated object (`OrdinaryCallBindThis` of an object is
/// that object); anything else is itself.
fn width(p: &FnProto, j: usize, construct: bool) -> usize {
    match &p.ops[j] {
        Op::Return { .. } => {
            if j + 1 == p.ops.len() && !falls_off(p) {
                1
            } else {
                2
            }
        }
        Op::BindThis { this } if construct && *this as usize == p.n_params + p.n_captures => 0,
        Op::BindThis { this }
            if !p
                .ops
                .iter()
                .enumerate()
                .any(|(i, o)| i != j && crate::bytecode::op_mentions_reg(o, *this)) =>
        {
            0
        }
        _ => 1,
    }
}

/// Whether control can fall off the callee's end (its last op is not a
/// `return`, or a branch goes to its end): the spliced body then ends with
/// `dst = undefined`.
fn falls_off(p: &FnProto) -> bool {
    let n = p.ops.len();
    !matches!(p.ops.last(), Some(Op::Return { .. }))
        || p.ops.iter().any(
            |o| matches!(o, Op::Jump { target } | Op::JumpIfFalse { target, .. } if *target == n),
        )
}

/// The offset, in the spliced body, of callee pc `t`; `t == len` is its end
/// (the fall-off op, if any).
fn body_pc(p: &FnProto, t: usize, construct: bool) -> usize {
    (0..t.min(p.ops.len()))
        .map(|j| width(p, j, construct))
        .sum()
}

/// The ops one inlined site (the call op `site`) takes before its call op:
/// guard, branch, window, body, fall-off, leave, a `super(…)`'s binding of
/// `this`, jump. (A construct site's guard writes `this`, see
/// [`construct_site`]; the site's shape is [`copied_param`]'s.)
fn block_len(p: &FnProto, site: &Op) -> usize {
    let (construct, shape) = (construct_site(site), site_shape(site));
    let this_slot = p.n_params + p.n_captures;
    let window = window_regs(p, construct)
        .into_iter()
        .filter(|&r| !(construct && r == this_slot) && !copied_param(p, r, construct, shape))
        .count();
    2 + window
        + body_pc(p, p.ops.len(), construct)
        + usize::from(falls_off(p))
        + 2
        + usize::from(super_parts(site).is_some())
}

/// A `super(…)` site's `(new.target, this)` registers (`Op::SuperCallArgs`:
/// the `this` its result binds); `None` for any other op.
fn super_parts(op: &Op) -> Option<(Reg, Reg)> {
    match op {
        Op::SuperCallArgs {
            new_target, this, ..
        } => Some((*new_target, *this)),
        _ => None,
    }
}

/// A call op's `(arguments, has a receiver)`, as [`copied_param`] asks them.
fn site_shape(op: &Op) -> (usize, bool) {
    // (A computed-key call inlines its call half, with the same parts.)
    if let Op::CallKey { args, .. } = op {
        return (args.len(), true);
    }
    call_parts(op).map_or((0, false), |(_, _, args, this)| {
        (args.len(), this.is_some())
    })
}

/// Whether callee register `r` is a parameter with an argument, or `this` with
/// a receiver (`shape`: the site's argument count, and whether it has one), that
/// the body never writes: it then needs no copy, and the body reads the
/// caller's register directly. Nothing can change that register while the
/// body runs: the caller is suspended at the call, the body writes only its
/// own registers and (at its returns, just before leaving) the call's
/// destination, and a captured variable lives in a box, not a register.
fn copied_param(p: &FnProto, r: usize, construct: bool, (argc, this): (usize, bool)) -> bool {
    let is_arg = r < p.n_params && r < argc;
    let is_this = this && !construct && r == p.n_params + p.n_captures;
    (is_arg || is_this) && !p.ops.iter().any(|o| writes_reg(o, r as Reg))
}

/// Whether `op`, one the splice takes ([`remap`]), may write register `r`.
fn writes_reg(op: &Op, r: Reg) -> bool {
    match op {
        Op::SetProp { .. }
        | Op::SetPropStrict { .. }
        | Op::CellSet { .. }
        | Op::SetElem { .. }
        | Op::SetKey { .. }
        | Op::SetKeyStrict { .. }
        | Op::InitProp { .. }
        | Op::StoreGlobal { .. }
        | Op::RequireObjectCoercible { .. }
        | Op::CheckTdz { .. }
        | Op::Throw { .. }
        | Op::Return { .. }
        | Op::ThrowTypeError { .. }
        | Op::ThrowReferenceError { .. }
        | Op::Jump { .. }
        | Op::JumpIfFalse { .. }
        | Op::InlineLeave
        | Op::InitFields { .. } => false,
        Op::BindThis { this } => *this == r,
        Op::SuperCallArgs { dst, this, .. } => *dst == r || *this == r,
        Op::SuperCtor { dst, .. } | Op::DerivedResult { dst, .. } => *dst == r,
        Op::InlineGuard {
            dst, caps, n_caps, ..
        } => *dst == r || (*caps..caps.saturating_add(*n_caps)).contains(&r),
        Op::LoadConst { dst, .. }
        | Op::LoadGlobal { dst, .. }
        | Op::TypeofGlobal { dst, .. }
        | Op::NewObject { dst }
        | Op::NewBox { dst }
        | Op::NewString { dst, .. }
        | Op::NewArray { dst, .. }
        | Op::Add { dst, .. }
        | Op::Sub { dst, .. }
        | Op::Mul { dst, .. }
        | Op::Div { dst, .. }
        | Op::Mod { dst, .. }
        | Op::ValueBin { dst, .. }
        | Op::Compare { dst, .. }
        | Op::Lt { dst, .. }
        | Op::AddValue { dst, .. }
        | Op::StrictEq { dst, .. }
        | Op::Relational { dst, .. }
        | Op::TypeOf { dst, .. }
        | Op::BitNot { dst, .. }
        | Op::Neg { dst, .. }
        | Op::Not { dst, .. }
        | Op::Move { dst, .. }
        | Op::ToStr { dst, .. }
        | Op::ToStrLazy { dst, .. }
        | Op::ToNumeric { dst, .. }
        | Op::Step { dst, .. }
        | Op::ToKey { dst, .. }
        | Op::CellGet { dst, .. }
        | Op::ArrayLen { dst, .. }
        | Op::GetProp { dst, .. }
        | Op::GetElem { dst, .. }
        | Op::GetKey { dst, .. }
        | Op::Concat { dst, .. }
        | Op::NewArrayFrom { dst, .. }
        | Op::PlainPrims { dst, .. }
        | Op::MakeClosure { dst, .. }
        | Op::LoadFunc { dst, .. }
        | Op::NewObjectLit { dst, .. }
        | Op::HasProp { dst, .. }
        | Op::InstanceOf { dst, .. }
        | Op::GetMethod { dst, .. }
        | Op::CallMethod { dst, .. }
        | Op::CallValue { dst, .. }
        | Op::Construct { dst, .. }
        | Op::CallWithMethod { dst, .. }
        | Op::CallKey { dst, .. }
        | Op::LoadNewTarget { dst } => *dst == r,
        // (Not one the splice takes: assume the worst.)
        _ => true,
    }
}

/// The callee registers an inlined site initializes as a call's register
/// window would be (its parameters, `this`, every other register
/// `undefined`; the captures are the guard's): those the spliced body may
/// read before it writes them, ascending (`construct`: as for [`width`]).
fn window_regs(p: &FnProto, construct: bool) -> Vec<usize> {
    let n = p.ops.len();
    let nr = p.n_regs;
    // Backward liveness over the spliced body: `live[j]`, the registers live
    // into callee op `j` (one pass when every branch goes forward; a loop's
    // back-edge takes more, until nothing changes).
    let mut live: Vec<Vec<bool>> = alloc::vec![alloc::vec![false; nr]; n + 1];
    let (mut defs, mut uses) = (Vec::new(), Vec::new());
    let mut changed = true;
    while core::mem::take(&mut changed) {
        for j in (0..n).rev() {
            live_step(
                p,
                j,
                construct,
                &mut live,
                &mut defs,
                &mut uses,
                &mut changed,
            );
        }
    }
    let caps = p.n_params..p.n_params + p.n_captures;
    (0..nr)
        .filter(|r| !caps.contains(r) && live[0][*r])
        .collect()
}

/// One step of [`window_regs`]' liveness: `live[j]` from its successors'
/// (`changed` set when it grows).
fn live_step(
    p: &FnProto,
    j: usize,
    construct: bool,
    live: &mut [Vec<bool>],
    defs: &mut Vec<Reg>,
    uses: &mut Vec<Reg>,
    changed: &mut bool,
) {
    let n = p.ops.len();
    let nr = p.n_regs;
    {
        let op = &p.ops[j];
        let succ: [Option<usize>; 2] = match op {
            Op::Jump { target } => [Some(*target), None],
            Op::JumpIfFalse { target, .. } => [Some(j + 1), Some(*target)],
            Op::Return { .. }
            | Op::Throw { .. }
            | Op::ThrowTypeError { .. }
            | Op::ThrowReferenceError { .. } => [None, None],
            _ => [Some(j + 1), None],
        };
        let mut out = alloc::vec![false; nr];
        for s in succ.into_iter().flatten() {
            for (o, l) in out.iter_mut().zip(&live[s.min(n)]) {
                *o |= *l;
            }
        }
        if width(p, j, construct) > 0 {
            super::numeric::def_use(op, defs, uses);
            for &d in defs.iter() {
                if let Some(x) = out.get_mut(d as usize) {
                    *x = false;
                }
            }
            for &u in uses.iter() {
                if let Some(x) = out.get_mut(u as usize) {
                    *x = true;
                }
            }
        }
        if live[j] != out {
            live[j] = out;
            *changed = true;
        }
    }
}
