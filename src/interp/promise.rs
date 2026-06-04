//! A `Promise` implementation over the interpreter's microtask queue.
//!
//! This covers the settlement core — `new Promise(executor)`,
//! `then`/`catch`/`finally` chaining (with thenable adoption), and the
//! `Promise.resolve`/`reject` statics. Reaction handlers run as microtasks,
//! drained after the main script (see [`Interp::run`]).
//!
//! Two structural notes on fitting Promises onto a tree-walker whose native
//! functions don't receive the evaluator:
//! - The scheduling side (`resolve`/`then`) only touches state and the shared
//!   queue, so it runs fine from plain native callbacks; only the reaction
//!   *jobs* need the evaluator, and they get it when the queue is drained.
//! - The `executor` passed to `new Promise` must call back into JS, so instead
//!   of running it inside the constructor native it is itself enqueued as a
//!   microtask (a one-tick delay that is unobservable, since a promise's state
//!   can't be read synchronously).
//!
//! Timer scheduling and `async`/`await` are out of scope here — they need the
//! host event loop and suspendable frames (see `ROADMAP.md`).

use super::eval::{Interp, Microtask, MicrotaskQueue};
use super::value::{NativeFn, Obj, Value};
use alloc::boxed::Box;
use alloc::rc::Rc;
use alloc::vec::Vec;
use core::cell::RefCell;

/// A promise's settlement status.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Status {
    Pending,
    Fulfilled,
    Rejected,
}

/// The internal state of a `Promise`.
pub struct PromiseState<'a> {
    status: Status,
    value: Value<'a>,
    reactions: Vec<Reaction<'a>>,
}

/// A registered `then` reaction: the handlers to run on settlement and the
/// resolve/reject functions of the dependent promise.
struct Reaction<'a> {
    on_fulfilled: Value<'a>,
    on_rejected: Value<'a>,
    resolve: Value<'a>,
    reject: Value<'a>,
}

type State<'a> = Rc<RefCell<PromiseState<'a>>>;

impl<'a> Interp<'a> {
    /// Installs the `Promise` constructor and its statics.
    pub(super) fn install_promise(&self) {
        let proto = Obj::object();
        let queue = self.microtask_queue();

        // The constructor object (callable via `new`).
        let ctor = Obj::object();
        {
            let queue = queue.clone();
            let proto = Rc::clone(&proto);
            ctor.set_callable(native("Promise", move |a| {
                let (obj, state) = new_promise(&queue, &proto);
                let executor = a.first().cloned().unwrap_or(Value::Undefined);
                if executor.is_callable() {
                    let (resolve, reject) = make_resolvers(&state, &queue);
                    // Run the executor on the next tick (see the module note).
                    let q = queue.clone();
                    queue
                        .borrow_mut()
                        .push_back(Box::new(move |interp: &mut Interp<'a>| {
                            let reject_on_throw = reject.clone();
                            if let Err(error) = interp.call_with_this(
                                executor,
                                Value::Undefined,
                                alloc::vec![resolve, reject],
                            ) {
                                settle(&state, Status::Rejected, error, &q);
                                let _ = reject_on_throw; // keep the resolver alive
                            }
                        }));
                }
                Ok(Value::Object(obj))
            }));
        }

        // `Promise.resolve` / `Promise.reject`.
        let q = queue.clone();
        let p = Rc::clone(&proto);
        ctor.set(
            "resolve",
            native("resolve", move |a| {
                let (obj, state) = new_promise(&q, &p);
                do_resolve(&state, &q, a.first().cloned().unwrap_or(Value::Undefined));
                Ok(Value::Object(obj))
            }),
        );
        let q = queue.clone();
        let p = Rc::clone(&proto);
        ctor.set(
            "reject",
            native("reject", move |a| {
                let (obj, state) = new_promise(&q, &p);
                settle(
                    &state,
                    Status::Rejected,
                    a.first().cloned().unwrap_or(Value::Undefined),
                    &q,
                );
                Ok(Value::Object(obj))
            }),
        );

        ctor.set("prototype", Value::Object(proto));
        self.define_global("Promise", Value::Object(ctor));
    }
}

/// Builds a native function value.
fn native<'a>(
    name: &'static str,
    f: impl Fn(&[Value<'a>]) -> super::Completion<'a, Value<'a>> + 'a,
) -> Value<'a> {
    Value::Native(Rc::new(NativeFn {
        name,
        call: Box::new(f),
    }))
}

/// Creates a fresh pending promise object (with `then`/`catch`/`finally`
/// capturing its state) plus its state cell.
fn new_promise<'a>(queue: &MicrotaskQueue<'a>, proto: &Rc<Obj<'a>>) -> (Rc<Obj<'a>>, State<'a>) {
    let obj = Obj::with_proto(Rc::clone(proto));
    let state = Rc::new(RefCell::new(PromiseState {
        status: Status::Pending,
        value: Value::Undefined,
        reactions: Vec::new(),
    }));
    obj.set_promise_state(Rc::clone(&state));

    // `then(onFulfilled, onRejected)` — registers a reaction and returns the
    // dependent promise.
    let then = {
        let state = Rc::clone(&state);
        let queue = queue.clone();
        let proto = Rc::clone(proto);
        native("then", move |a| {
            let on_f = a.first().cloned().unwrap_or(Value::Undefined);
            let on_r = a.get(1).cloned().unwrap_or(Value::Undefined);
            let (p2, p2_state) = new_promise(&queue, &proto);
            let (resolve, reject) = make_resolvers(&p2_state, &queue);
            register_reaction(
                &state,
                &queue,
                Reaction {
                    on_fulfilled: on_f,
                    on_rejected: on_r,
                    resolve,
                    reject,
                },
            );
            Ok(Value::Object(p2))
        })
    };
    obj.set("then", then.clone());

    // `catch(onRejected)` == `then(undefined, onRejected)`.
    let catch = {
        let then = then.clone();
        native("catch", move |a| {
            invoke(
                &then,
                &[
                    Value::Undefined,
                    a.first().cloned().unwrap_or(Value::Undefined),
                ],
            )
        })
    };
    obj.set("catch", catch);

    // `finally(onFinally)` — runs `onFinally` on both settlement paths.
    // (Simplification: the dependent promise resolves with `onFinally`'s result
    // rather than passing the original value through.)
    let fin = {
        let then = then.clone();
        native("finally", move |a| {
            let on_finally = a.first().cloned().unwrap_or(Value::Undefined);
            invoke(&then, &[on_finally.clone(), on_finally])
        })
    };
    obj.set("finally", fin);

    (obj, state)
}

/// Calls a native value directly (used to delegate `catch`/`finally` to
/// `then`).
fn invoke<'a>(f: &Value<'a>, args: &[Value<'a>]) -> super::Completion<'a, Value<'a>> {
    match f {
        Value::Native(n) => (n.call)(args),
        _ => Ok(Value::Undefined),
    }
}

/// Builds the resolve/reject functions for a promise state.
fn make_resolvers<'a>(state: &State<'a>, queue: &MicrotaskQueue<'a>) -> (Value<'a>, Value<'a>) {
    let s1 = Rc::clone(state);
    let q1 = queue.clone();
    let resolve = native("resolve", move |a| {
        do_resolve(&s1, &q1, a.first().cloned().unwrap_or(Value::Undefined));
        Ok(Value::Undefined)
    });
    let s2 = Rc::clone(state);
    let q2 = queue.clone();
    let reject = native("reject", move |a| {
        settle(
            &s2,
            Status::Rejected,
            a.first().cloned().unwrap_or(Value::Undefined),
            &q2,
        );
        Ok(Value::Undefined)
    });
    (resolve, reject)
}

/// Resolves `state` with `value`, adopting `value` if it is itself a thenable
/// promise.
fn do_resolve<'a>(state: &State<'a>, queue: &MicrotaskQueue<'a>, value: Value<'a>) {
    if let Value::Object(o) = &value
        && let Some(inner) = o.promise_state()
    {
        // Adopt: follow the inner promise's eventual settlement.
        let (resolve, reject) = make_resolvers(state, queue);
        register_reaction(
            &inner,
            queue,
            Reaction {
                on_fulfilled: Value::Undefined,
                on_rejected: Value::Undefined,
                resolve,
                reject,
            },
        );
        return;
    }
    settle(state, Status::Fulfilled, value, queue);
}

/// Settles `state` and schedules its pending reactions.
fn settle<'a>(state: &State<'a>, status: Status, value: Value<'a>, queue: &MicrotaskQueue<'a>) {
    let reactions = {
        let mut s = state.borrow_mut();
        if s.status != Status::Pending {
            return; // already settled
        }
        s.status = status;
        s.value = value.clone();
        core::mem::take(&mut s.reactions)
    };
    for reaction in reactions {
        enqueue_reaction(queue, reaction, status, value.clone());
    }
}

/// Registers a reaction; if the promise is already settled, schedules it now.
fn register_reaction<'a>(state: &State<'a>, queue: &MicrotaskQueue<'a>, reaction: Reaction<'a>) {
    let settled = {
        let s = state.borrow();
        match s.status {
            Status::Pending => None,
            status => Some((status, s.value.clone())),
        }
    };
    match settled {
        None => state.borrow_mut().reactions.push(reaction),
        Some((status, value)) => enqueue_reaction(queue, reaction, status, value),
    }
}

/// Pushes a microtask that runs `reaction` against a settled `(status, value)`.
fn enqueue_reaction<'a>(
    queue: &MicrotaskQueue<'a>,
    reaction: Reaction<'a>,
    status: Status,
    value: Value<'a>,
) {
    let job: Microtask<'a> = Box::new(move |interp: &mut Interp<'a>| {
        let Reaction {
            on_fulfilled,
            on_rejected,
            resolve,
            reject,
        } = reaction;
        let fulfilled = status == Status::Fulfilled;
        let handler = if fulfilled { on_fulfilled } else { on_rejected };
        if handler.is_callable() {
            match interp.call_with_this(handler, Value::Undefined, alloc::vec![value]) {
                Ok(result) => {
                    let _ = interp.call_with_this(resolve, Value::Undefined, alloc::vec![result]);
                }
                Err(error) => {
                    let _ = interp.call_with_this(reject, Value::Undefined, alloc::vec![error]);
                }
            }
        } else {
            // Passthrough: forward the value to the dependent promise.
            let settler = if fulfilled { resolve } else { reject };
            let _ = interp.call_with_this(settler, Value::Undefined, alloc::vec![value]);
        }
    });
    queue.borrow_mut().push_back(job);
}
