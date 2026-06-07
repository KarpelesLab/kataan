/*---
description: Function.prototype.length is the parameter count before defaults/rest
esid: sec-function-instances-length
---*/
function f(a, b, c) {}
assert.sameValue(f.length, 3, "three params");
assert.sameValue(((x, y) => x).length, 2, "arrow");
assert.sameValue((function () {}).length, 0, "no params");

// A default value or a rest parameter ends the count.
function d(a, b = 1, c) {}
assert.sameValue(d.length, 1, "stops at first default");
function r(a, ...rest) {}
assert.sameValue(r.length, 1, "rest not counted");
function dr(a, b, ...rest) {}
assert.sameValue(dr.length, 2, "fixed params before rest");

// Methods.
var o = { m(a, b) {} };
assert.sameValue(o.m.length, 2, "method length");
class C { method(a, b, c) {} }
assert.sameValue(new C().method.length, 3, "class method length");

// length does not shadow array/string length.
assert.sameValue([1, 2, 3].length, 3, "array length");
assert.sameValue("hello".length, 5, "string length");
