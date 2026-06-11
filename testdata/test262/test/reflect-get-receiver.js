/*---
description: Reflect.get(target, key, receiver) runs a getter with receiver as this
esid: sec-reflect.get
---*/
var base = { get x() { return this.v; } };

// An own getter runs with the explicit receiver as `this`.
assert.sameValue(Reflect.get(base, "x", { v: 99 }), 99, "own getter receiver");

// An inherited getter too; without a receiver, `this` is the target.
var obj = Object.create(base);
obj.v = 42;
assert.sameValue(Reflect.get(obj, "x"), 42, "no receiver -> target this");
assert.sameValue(Reflect.get(obj, "x", { v: 7 }), 7, "inherited getter receiver");

// A data property ignores the receiver.
assert.sameValue(Reflect.get({ p: 5 }, "p", { p: 99 }), 5, "data property ignores receiver");

// A primitive receiver works (the getter reads its length).
var sg = { get len() { return this.length; } };
assert.sameValue(Reflect.get(sg, "len", "hello"), 5, "primitive receiver");

// A getter that mutates the receiver mutates the receiver, not the target.
var counter = { get next() { return this.n++; } };
var state = { n: 10 };
assert.sameValue(Reflect.get(counter, "next", state), 10, "first");
assert.sameValue(Reflect.get(counter, "next", state), 11, "second");
assert.sameValue(state.n, 12, "receiver mutated");

// A missing property is undefined; an accessor with no getter is undefined.
assert.sameValue(Reflect.get({}, "nope", {}), undefined, "missing");
var so = { set y(v) {} };
assert.sameValue(Reflect.get(so, "y", {}), undefined, "setter-only -> undefined");
