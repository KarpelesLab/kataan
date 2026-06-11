/*---
description: new Function (dynamic code) throws a catchable TypeError, like Function()
esid: sec-function-constructor
---*/
// Dynamic code generation is unsupported here, but it must fail with a catchable
// TypeError — not abort the program — so feature-detection in a try/catch works.
function caught(fn) { try { fn(); return null; } catch (e) { return e.constructor.name; } }

assert.sameValue(caught(function () { new Function("a", "b", "return a + b"); }), "TypeError", "new Function throws TypeError");
assert.sameValue(caught(function () { Function("return 1"); }), "TypeError", "Function() throws TypeError");

// The Function constructor itself exists (typeof is function).
assert.sameValue(typeof Function, "function", "Function is a function");

// A feature-detect block completes without aborting.
var supported = (function () {
  try { new Function("return 1"); return true; } catch (e) { return false; }
})();
assert.sameValue(supported, false, "feature-detect returns false, not a crash");

// Other constructors are unaffected by this path.
assert.sameValue(new Map([["a", 1]]).get("a"), 1, "new Map");
assert.sameValue(new Set([1, 2, 3]).size, 3, "new Set");
assert.sameValue(new Date(0).getTime(), 0, "new Date");
assert.sameValue(new Array(3).length, 3, "new Array");
assert.sameValue(new Number(5).valueOf(), 5, "new Number");
