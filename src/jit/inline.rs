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
//!       exactly as a call's register window is
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
//! activation at the body's throw exit. Only small straight-line callees are
//! inlined — no loop, `try`, `arguments`, `new.target`, `super`, eval or
//! suspension in them — so nothing can leave the compiled code inside one
//! (an on-stack exit is at a loop header, which an inlined body has none
//! of). An inlined body may call: a call op in it (or a getter, a
//! `valueOf`, … that a property access or an operator calls) first pushes
//! the pending activation, like any Rust code run inside it, and a throw
//! out of the callee ends the inlined activation as any other throw does.
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
/// The most ops inlining may add to one caller.
const MAX_ADDED_OPS: usize = 600;

/// A callee to inline at a call site, as the caller's compiler is told it.
pub struct Candidate<'a> {
    /// `function id | capture count << 32` (the call sequence's callee key).
    pub key: u64,
    /// The callee's function.
    pub proto: &'a FnProto,
    /// The callee's warm inline caches, by its own pc.
    pub warm: Vec<Option<WarmEntry>>,
    /// The callee's own call sites to inline into its spliced body, by its
    /// own pc (one level: their candidates' `nested` are ignored).
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
                let (_, n) = nested.iter().find(|(p, _)| *p == pc)?;
                Some(Candidate {
                    key: n.key,
                    proto: n.proto,
                    warm: n.warm.clone(),
                    nested: Vec::new(),
                })
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
    if proto.class_ctor
        || proto.is_async
        || proto.is_generator
        || proto.rest_from.is_some()
        || proto.ops.is_empty()
        || proto.ops.len() > MAX_CALLEE_OPS
        || proto.n_regs > MAX_CALLEE_REGS
        || proto.n_params + proto.n_captures >= proto.n_regs
    {
        return false;
    }
    let n = proto.ops.len();
    proto.ops.iter().enumerate().all(|(pc, op)| match op {
        Op::Jump { target } | Op::JumpIfFalse { target, .. } => *target > pc && *target <= n,
        _ => {
            let mut o = op.clone();
            remap(&mut o, &|r| r, &|t| t)
        }
    })
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
    if proto.derived
        || proto.is_async
        || proto.is_generator
        || proto.rest_from.is_some()
        || proto.ops.is_empty()
        || proto.ops.len() > MAX_CALLEE_OPS
        || proto.n_regs > MAX_CALLEE_REGS
        || proto.n_params + proto.n_captures >= proto.n_regs
    {
        return false;
    }
    let n = proto.ops.len();
    proto.ops.iter().enumerate().all(|(pc, op)| match op {
        Op::Jump { target } | Op::JumpIfFalse { target, .. } => *target > pc && *target <= n,
        Op::LoadNewTarget { .. } => true,
        Op::Return { .. } => false,
        _ => {
            let mut o = op.clone();
            remap(&mut o, &|r| r, &|t| t)
        }
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
        Op::Concat { dst, parts } | Op::NewArrayFrom { dst, elems: parts } => {
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
/// a call of a method read before its arguments, or a property read that
/// calls an inherited getter.
fn call_parts(op: &Op) -> Option<(Reg, Reg, &[Reg], Option<Reg>)> {
    match op {
        Op::GetProp { dst, obj, .. } => Some((*dst, *obj, &[], Some(*obj))),
        // (`this` is the object the guard allocates.)
        Op::Construct { dst, ctor, args } => Some((*dst, *ctor, args, None)),
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
    pick: &dyn Fn(usize, &Op) -> Option<Candidate<'a>>,
) -> Option<Expanded> {
    let base = proto.n_regs;
    let mut sites: Vec<(usize, Site<'a>)> = Vec::new();
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
            || (dst as usize) >= base
            || (callee as usize) >= base
            || this.is_some_and(|t| (t as usize) >= base)
            || args.iter().any(|&a| (a as usize) >= base)
        {
            continue;
        }
        let Some(c) = pick(pc, op) else { continue };
        let c = site_body(c);
        let ok = if matches!(op, Op::Construct { .. }) {
            construct_ok(&c.proto)
        } else {
            callee_ok(&c.proto)
        };
        if !ok || (c.key >> 32) as usize != c.proto.n_captures {
            continue;
        }
        let cost = c.proto.ops.len() + c.proto.n_regs + 6;
        if added + cost > MAX_ADDED_OPS {
            continue;
        }
        added += cost;
        window = window.max(c.proto.n_regs + 1 + usize::from(split.is_some()));
        sites.push((pc, c));
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
            next += block_len(
                &sites[si].1.proto,
                matches!(proto.ops[pc], Op::Construct { .. }),
            ) + 1;
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
            let c = &sites[si].1;
            si += 1;
            // A `new`: the guard also allocates `this`, into the register
            // after the captures (the callee's `this` slot).
            let construct = matches!(op, Op::Construct { .. });
            let p: &FnProto = &c.proto;
            let rb = |r: Reg| -> Reg { (base + r as usize) as Reg };
            let split = split_call_key(op, rb((p.n_regs + 1) as Reg));
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
            let g = ops.len();
            let tmp = rb(p.n_regs as Reg);
            let fallback = g + block_len(p, construct);
            let end = fallback + 1;
            let leave = fallback - 2;
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
                },
                None,
                ops
            );
            // The guard's own pc is the call's (an entry, a branch to it).
            *orig_of.last_mut()? = u32::try_from(pc).ok()?;
            emit!(
                Op::JumpIfFalse {
                    cond: tmp,
                    target: fallback,
                },
                None,
                ops
            );
            // The register window, as `call_with_inner` lays it out — the
            // registers the body may read before writing them (the others'
            // values are never seen).
            let this_slot = p.n_params + p.n_captures;
            for r in window_regs(p, construct) {
                if construct && r == this_slot {
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
                        emit!(Op::Move { dst, src: rb(*src) }, None, ops,);
                        if w == 2 {
                            emit!(Op::Jump { target: leave }, None, ops);
                        }
                    }
                    // `new.target`: the constructor (see `construct_ok`).
                    _ if construct && let Op::LoadNewTarget { dst: d } = bop => {
                        emit!(
                            Op::Move {
                                dst: rb(*d),
                                src: callee,
                            },
                            None,
                            ops
                        );
                    }
                    _ => {
                        let mut o = bop.clone();
                        if !remap(&mut o, &rb, &tgt) {
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
            emit!(Op::Jump { target: end }, None, ops);
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

/// The ops one inlined site takes before its call op: guard, branch, window,
/// body, fall-off, leave, jump.
/// (`construct`: a `new` site, whose `this` the guard writes.)
fn block_len(p: &FnProto, construct: bool) -> usize {
    let this_slot = p.n_params + p.n_captures;
    let window = window_regs(p, construct)
        .into_iter()
        .filter(|&r| !(construct && r == this_slot))
        .count();
    2 + window + body_pc(p, p.ops.len(), construct) + usize::from(falls_off(p)) + 2
}

/// The callee registers an inlined site initializes as a call's register
/// window would be (its parameters, `this`, every other register
/// `undefined`; the captures are the guard's): those the spliced body may
/// read before it writes them, ascending (`construct`: as for [`width`]).
fn window_regs(p: &FnProto, construct: bool) -> Vec<usize> {
    let n = p.ops.len();
    let nr = p.n_regs;
    // Backward liveness over the spliced body (its branches go forward, so
    // one pass): `live[j]`, the registers live into callee op `j`.
    let mut live: Vec<Vec<bool>> = alloc::vec![alloc::vec![false; nr]; n + 1];
    let (mut defs, mut uses) = (Vec::new(), Vec::new());
    for j in (0..n).rev() {
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
            super::numeric::def_use(op, &mut defs, &mut uses);
            for &d in &defs {
                if let Some(x) = out.get_mut(d as usize) {
                    *x = false;
                }
            }
            for &u in &uses {
                if let Some(x) = out.get_mut(u as usize) {
                    *x = true;
                }
            }
        }
        live[j] = out;
    }
    let caps = p.n_params..p.n_params + p.n_captures;
    (0..nr)
        .filter(|r| !caps.contains(r) && live[0][*r])
        .collect()
}
