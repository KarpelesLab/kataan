/*---
description: object destructuring requires a coercible value and reads via [[Get]]
esid: sec-destructuring-binding-patterns
---*/
function throws(fn) { try { fn(); return false; } catch (e) { return e instanceof TypeError; } }

// Destructuring null/undefined as an object throws a TypeError (RequireObjectCoercible).
assert.sameValue(throws(function () { var { z } = null; }), true, "null");
assert.sameValue(throws(function () { var { z } = undefined; }), true, "undefined");
assert.sameValue(throws(function () { (function ({ a }) {})(null); }), true, "null parameter");
assert.sameValue(throws(function () { var { a: { b } } = { a: null }; }), true, "nested null");
assert.sameValue(throws(function () { for (var { a } of [null]) {} }), true, "for-of null element");

// A primitive number/string is coercible (no throw).
assert.sameValue(throws(function () { var { x } = 5; }), false, "number is coercible");

// Properties are read through [[Get]]: accessors fire, inherited and length resolve.
var calls = 0;
var { x } = { get x() { calls++; return 42; } };
assert.sameValue(x, 42, "getter value");
assert.sameValue(calls, 1, "getter invoked");

var base = { inh: 1 };
var { inh } = Object.create(base);
assert.sameValue(inh, 1, "inherited property");

var { length } = "abc";
assert.sameValue(length, 3, "string length");
var { length: al } = [1, 2, 3];
assert.sameValue(al, 3, "array length");

// A getter returning undefined lets the default apply.
var { y = 5 } = { get y() { return undefined; } };
assert.sameValue(y, 5, "default after undefined getter");

// Rest still copies only own enumerable properties.
var { a, ...rest } = { a: 1, b: 2, c: 3 };
assert.sameValue(JSON.stringify(rest), '{"b":2,"c":3}', "rest own enumerable");
