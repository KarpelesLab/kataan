/*---
description: Function names from binding/key, Function.prototype.toString, bound length
esid: sec-function-instances-name
---*/
// Anonymous function/arrow takes the binding name.
var myFn = function () {};
assert.sameValue(myFn.name, "myFn", "var-assigned function name");
var arrow = () => 1;
assert.sameValue(arrow.name, "arrow", "var-assigned arrow name");
const c = function () {};
assert.sameValue(c.name, "c", "const-assigned name");
// A named function expression keeps its own name.
var x = function inner() {};
assert.sameValue(x.name, "inner", "named expression keeps its name");
// Object method / shorthand names.
var o = { method() {}, arrowProp: () => 1, fn: function () {} };
assert.sameValue(o.method.name, "method", "method shorthand name");
assert.sameValue(o.arrowProp.name, "arrowProp", "arrow property name");
assert.sameValue(o.fn.name, "fn", "function property name");
// Function.prototype.toString returns a string mentioning "function".
assert.sameValue(typeof (function f() {}).toString(), "string", "toString is a string");
assert.sameValue((function f() {}).toString().indexOf("function") >= 0, true, "mentions function");
assert.sameValue(typeof (class C {}).toString(), "string", "class toString");
// Bound function length and name.
function target(a, b, c) {}
assert.sameValue(target.bind(null).length, 3, "bound length (no args)");
assert.sameValue(target.bind(null, 1).length, 2, "bound length minus one arg");
assert.sameValue(target.bind(null, 1, 2, 3, 4).length, 0, "bound length floored at 0");
assert.sameValue(target.bind(null).name, "bound target", "bound name");
