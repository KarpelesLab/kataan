/*---
description: array destructuring of a non-iterable (null/undefined/number/object) throws TypeError
esid: sec-runtime-semantics-bindinginitialization
---*/
// Binding form.
assert.throws(TypeError, function () { var [a] = null; }, "var [a] = null");
assert.throws(TypeError, function () { var [a] = undefined; }, "var [a] = undefined");
assert.throws(TypeError, function () { var [a] = 5; }, "var [a] = number");
assert.throws(TypeError, function () { var [a] = {}; }, "var [a] = plain object");
// Assignment form.
assert.throws(TypeError, function () { var a; [a] = null; }, "[a] = null");
// Destructuring parameter.
assert.throws(TypeError, function () { (function ([a]) { return a; })(null); }, "param [a] from null");

// Iterables and defaults still destructure fine.
var [p, q] = [1, 2];
assert.sameValue(p + q, 3, "array");
var [r, s] = "hi";
assert.sameValue(r + s, "hi", "string is iterable");
var [u, v] = new Set([3, 4]);
assert.sameValue(u + v, 7, "Set is iterable");
var [w = 9] = [];
assert.sameValue(w, 9, "default for missing element");
function f([a, b]) { return a + b; }
assert.sameValue(f([10, 20]), 30, "destructuring parameter");
function g([a] = [7]) { return a; }
assert.sameValue(g(), 7, "parameter default avoids the throw");
