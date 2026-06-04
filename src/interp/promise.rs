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
    /// An internal (Rust) reaction used by the async driver: invoked with
    /// `(interp, fulfilled, value)` when the promise settles, bypassing the
    /// JS-callable handlers. Lets `await` resume a suspended async frame.
    internal: Option<InternalReaction<'a>>,
}

/// A Rust closure run when an awaited promise settles (the async driver).
type InternalReaction<'a> = alloc::boxed::Box<dyn FnOnce(&mut Interp<'a>, bool, Value<'a>) + 'a>;

type State<'a> = Rc<RefCell<PromiseState<'a>>>;

impl<'a> Interp<'a> {
    /// Installs the `Promise` constructor and its statics.
    /// Creates a promise already settled with `value` (fulfilled, or rejected
    /// when `rejected`). Used to wrap the result of an `async` function.
    pub(super) fn settled_promise(&self, value: Value<'a>, rejected: bool) -> Value<'a> {
        let proto = match self.global().get("Promise") {
            Some(Value::Object(ctor)) => match ctor.get("prototype") {
                Value::Object(p) => p,
                _ => Obj::object(),
            },
            _ => Obj::object(),
        };
        let queue = self.microtask_queue();
        let (obj, state) = new_promise(&queue, &proto);
        let status = if rejected {
            Status::Rejected
        } else {
            Status::Fulfilled
        };
        settle(&state, status, value, &queue);
        Value::Object(obj)
    }

    /// Drives an `async` function: starts pumping its generator-style frame and
    /// returns the promise that settles with the function's result.
    pub(super) fn drive_async(
        &mut self,
        genstate: Rc<RefCell<super::vm::GeneratorState<'a>>>,
    ) -> Value<'a> {
        let proto = self.promise_prototype();
        let queue = self.microtask_queue();
        let (obj, state) = new_promise(&queue, &proto);
        let (resolve, reject) = make_resolvers(&state, &queue);
        self.async_step(genstate, resolve, reject, Value::Undefined, false);
        Value::Object(obj)
    }

    /// One step of the async driver: resume the frame (with `send`, or injecting
    /// a throw), then settle the result promise on return / chain on `await`.
    fn async_step(
        &mut self,
        genstate: Rc<RefCell<super::vm::GeneratorState<'a>>>,
        resolve: Value<'a>,
        reject: Value<'a>,
        send: Value<'a>,
        is_throw: bool,
    ) {
        match self.generator_step(&genstate, send, is_throw) {
            // The async function returned: fulfil with its value (adopting an
            // inner promise if it returned one).
            Ok((value, true)) => {
                let _ = self.call_with_this(resolve, Value::Undefined, alloc::vec![value]);
            }
            // An `await`: continue when the awaited value settles.
            Ok((awaited, false)) => {
                self.await_settlement(
                    awaited,
                    Box::new(move |interp, fulfilled, v| {
                        interp.async_step(genstate, resolve, reject, v, !fulfilled);
                    }),
                );
            }
            // The async function threw: reject.
            Err(thrown) => {
                let _ = self.call_with_this(reject, Value::Undefined, alloc::vec![thrown]);
            }
        }
    }

    /// Registers `on_settle` to run when `value` settles (a promise), or on the
    /// next tick treating a plain value as already fulfilled.
    fn await_settlement(&mut self, value: Value<'a>, on_settle: InternalReaction<'a>) {
        let queue = self.microtask_queue();
        if let Value::Object(o) = &value
            && let Some(state) = o.promise_state()
        {
            register_reaction(
                &state,
                &queue,
                Reaction {
                    on_fulfilled: noop(),
                    on_rejected: noop(),
                    resolve: noop(),
                    reject: noop(),
                    internal: Some(on_settle),
                },
            );
        } else {
            let v = value.clone();
            queue
                .borrow_mut()
                .push_back(Box::new(move |interp: &mut Interp<'a>| {
                    on_settle(interp, true, v)
                }));
        }
    }

    /// The shared `Promise.prototype` object.
    fn promise_prototype(&self) -> Rc<Obj<'a>> {
        match self.global().get("Promise") {
            Some(Value::Object(ctor)) => match ctor.get("prototype") {
                Value::Object(p) => p,
                _ => Obj::object(),
            },
            _ => Obj::object(),
        }
    }

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

        // `Promise.all(iterable)` — resolves to an array of results, or rejects
        // with the first rejection.
        let q = queue.clone();
        let p = Rc::clone(&proto);
        ctor.set(
            "all",
            native("all", move |a| {
                let (obj, state) = new_promise(&q, &p);
                let (all_resolve, all_reject) = make_resolvers(&state, &q);
                let mut elements = Vec::new();
                super::builtins::iterate_into(
                    &a.first().cloned().unwrap_or(Value::Undefined),
                    &mut elements,
                );
                let n = elements.len();
                if n == 0 {
                    let _ = invoke(&all_resolve, &[Value::Object(Obj::array(Vec::new()))]);
                    return Ok(Value::Object(obj));
                }
                let results = Rc::new(RefCell::new(alloc::vec![Value::Undefined; n]));
                let remaining = Rc::new(RefCell::new(n));
                for (i, element) in elements.into_iter().enumerate() {
                    let results_c = Rc::clone(&results);
                    let remaining_c = Rc::clone(&remaining);
                    let resolve_all = all_resolve.clone();
                    let on_fulfilled = native("", move |b| {
                        results_c.borrow_mut()[i] = b.first().cloned().unwrap_or(Value::Undefined);
                        *remaining_c.borrow_mut() -= 1;
                        if *remaining_c.borrow() == 0 {
                            let arr = Obj::array(results_c.borrow().clone());
                            let _ = invoke(&resolve_all, &[Value::Object(arr)]);
                        }
                        Ok(Value::Undefined)
                    });
                    settle_element(&element, on_fulfilled, all_reject.clone(), &q);
                }
                Ok(Value::Object(obj))
            }),
        );

        // `Promise.race(iterable)` — settles as the first element settles.
        let q = queue.clone();
        let p = Rc::clone(&proto);
        ctor.set(
            "race",
            native("race", move |a| {
                let (obj, state) = new_promise(&q, &p);
                let (resolve, reject) = make_resolvers(&state, &q);
                let mut elements = Vec::new();
                super::builtins::iterate_into(
                    &a.first().cloned().unwrap_or(Value::Undefined),
                    &mut elements,
                );
                for element in elements {
                    settle_element(&element, resolve.clone(), reject.clone(), &q);
                }
                Ok(Value::Object(obj))
            }),
        );

        // `Promise.allSettled(iterable)` — never rejects; resolves to an array
        // of `{ status, value }` / `{ status, reason }` descriptors.
        let q = queue.clone();
        let p = Rc::clone(&proto);
        ctor.set(
            "allSettled",
            native("allSettled", move |a| {
                let (obj, state) = new_promise(&q, &p);
                let (resolve, _) = make_resolvers(&state, &q);
                let mut elements = Vec::new();
                super::builtins::iterate_into(
                    &a.first().cloned().unwrap_or(Value::Undefined),
                    &mut elements,
                );
                let n = elements.len();
                if n == 0 {
                    let _ = invoke(&resolve, &[Value::Object(Obj::array(Vec::new()))]);
                    return Ok(Value::Object(obj));
                }
                let results = Rc::new(RefCell::new(alloc::vec![Value::Undefined; n]));
                let remaining = Rc::new(RefCell::new(n));
                for (i, element) in elements.into_iter().enumerate() {
                    // A descriptor-recording handler for each settlement path.
                    let mk = |status: &'static str, key: &'static str| {
                        let results_c = Rc::clone(&results);
                        let remaining_c = Rc::clone(&remaining);
                        let resolve_c = resolve.clone();
                        native("", move |b| {
                            let desc = Obj::object();
                            desc.set("status", Value::str(status));
                            desc.set(key, b.first().cloned().unwrap_or(Value::Undefined));
                            results_c.borrow_mut()[i] = Value::Object(desc);
                            *remaining_c.borrow_mut() -= 1;
                            if *remaining_c.borrow() == 0 {
                                let arr = Obj::array(results_c.borrow().clone());
                                let _ = invoke(&resolve_c, &[Value::Object(arr)]);
                            }
                            Ok(Value::Undefined)
                        })
                    };
                    settle_element(
                        &element,
                        mk("fulfilled", "value"),
                        mk("rejected", "reason"),
                        &q,
                    );
                }
                Ok(Value::Object(obj))
            }),
        );

        // `queueMicrotask(callback)` — schedule a callback on the microtask
        // queue.
        let q = queue.clone();
        self.define_global(
            "queueMicrotask",
            native("queueMicrotask", move |a| {
                let callback = a.first().cloned().unwrap_or(Value::Undefined);
                if callback.is_callable() {
                    q.borrow_mut()
                        .push_back(Box::new(move |interp: &mut Interp<'a>| {
                            let _ = interp.call_with_this(callback, Value::Undefined, Vec::new());
                        }));
                }
                Ok(Value::Undefined)
            }),
        );

        ctor.set("prototype", Value::Object(proto));
        self.define_global("Promise", Value::Object(ctor));
    }
}

/// Routes one combinator element to `on_fulfilled`/`on_rejected`: registers a
/// reaction if it is a thenable, otherwise fulfills immediately.
fn settle_element<'a>(
    element: &Value<'a>,
    on_fulfilled: Value<'a>,
    on_rejected: Value<'a>,
    queue: &MicrotaskQueue<'a>,
) {
    if let Value::Object(o) = element
        && let Some(state) = o.promise_state()
    {
        register_reaction(
            &state,
            queue,
            Reaction {
                on_fulfilled,
                on_rejected,
                resolve: noop(),
                reject: noop(),
                internal: None,
            },
        );
        return;
    }
    // A plain value behaves like an already-resolved promise: its reaction is
    // deferred one tick, so element order (not promise-vs-plain) decides
    // ordering in `race`/`all`.
    let element = element.clone();
    queue
        .borrow_mut()
        .push_back(Box::new(move |interp: &mut Interp<'a>| {
            let _ = interp.call_with_this(on_fulfilled, Value::Undefined, alloc::vec![element]);
        }));
}

/// A native that ignores its arguments and returns `undefined`.
fn noop<'a>() -> Value<'a> {
    native("", |_| Ok(Value::Undefined))
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
                    internal: None,
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
                internal: None,
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
            internal,
        } = reaction;
        let fulfilled = status == Status::Fulfilled;
        // An async-driver reaction runs its Rust closure and is done.
        if let Some(run) = internal {
            run(interp, fulfilled, value);
            return;
        }
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
