/*---
description: new Function (dynamic code) builds an anonymous function in the global scope; a bad body throws a catchable SyntaxError
esid: sec-function-constructor
---*/
// `new Function(args…, body)` compiles runtime source into a callable.
var add = new Function("a", "b", "return a + b");
assert.sameValue(typeof add, "function", "new Function returns a function");
assert.sameValue(add(2, 3), 5, "new Function-built function runs");
assert.sameValue(add(40, 2), 42, "again");

// Calling the constructor as a plain function behaves identically.
var twice = Function("x", "return x * 2");
assert.sameValue(twice(21), 42, "Function() (no new) builds a function too");

// The built function is anonymous with the right length, and is a Function instance.
assert.sameValue(add.name, "anonymous", "name is 'anonymous'");
assert.sameValue(add.length, 2, "length is the parameter count");
assert.sameValue(add instanceof Function, true, "result instanceof Function");
assert.sameValue(typeof add.prototype, "object", "has a prototype object");

// The function is created in the GLOBAL scope: it does not close over local
// variables of the call site.
function makeReader() {
  var local = 123;
  return new Function("return typeof local");
}
assert.sameValue(makeReader()(), "undefined", "Function() closes over globals only");

// A syntactically invalid body throws a catchable SyntaxError (not an abort).
function caught(fn) { try { fn(); return null; } catch (e) { return e.constructor.name; } }
assert.sameValue(caught(function () { new Function("return )("); }), "SyntaxError", "bad body throws SyntaxError");
assert.sameValue(caught(function () { Function("a", "b b", "return 1"); }), "SyntaxError", "bad params throw SyntaxError");

// A feature-detect block completes without aborting.
var supported = (function () {
  try { new Function("return 1"); return true; } catch (e) { return false; }
})();
assert.sameValue(supported, true, "feature-detect returns true");

// Other constructors are unaffected by this path.
assert.sameValue(new Map([["a", 1]]).get("a"), 1, "new Map");
assert.sameValue(new Set([1, 2, 3]).size, 3, "new Set");
assert.sameValue(new Date(0).getTime(), 0, "new Date");
assert.sameValue(new Array(3).length, 3, "new Array");
assert.sameValue(new Number(5).valueOf(), 5, "new Number");
