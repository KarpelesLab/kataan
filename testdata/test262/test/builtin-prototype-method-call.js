/*---
description: String/Number/Boolean/Function.prototype methods are first-class (call/apply)
esid: sec-properties-of-the-string-prototype-object
---*/
// String.prototype.X.call on a string.
assert.sameValue(String.prototype.slice.call("hello", 1, 3), "el", "String slice.call");
assert.sameValue(String.prototype.split.call("a,b,c", ",").join("|"), "a|b|c", "String split.call");
assert.sameValue(String.prototype.toUpperCase.call("abc"), "ABC", "String toUpperCase.call");
assert.sameValue(String.prototype.charAt.call("abc", 1), "b", "String charAt.call");

// Number.prototype.X.call.
assert.sameValue(Number.prototype.toFixed.call(3.14159, 2), "3.14", "Number toFixed.call");
assert.sameValue(Number.prototype.toString.call(255, 16), "ff", "Number toString.call radix");

// Boolean primitive methods (also via Boolean.prototype).
assert.sameValue((true).toString(), "true", "boolean toString");
assert.sameValue((false).valueOf(), false, "boolean valueOf");
assert.sameValue(Boolean.prototype.toString.call(true), "true", "Boolean toString.call");

// Function.prototype.call/apply/bind as first-class values.
function add(a, b) { return a + b + this.n; }
var bound = Function.prototype.bind.call(add, { n: 100 }, 5);
assert.sameValue(bound(10), 115, "Function bind.call");
assert.sameValue(Function.prototype.call.call(function (x) { return x * this.f; }, { f: 3 }, 7), 21, "Function call.call");

// Prototypes exist and link to their constructors; ordinary calls unaffected.
assert.sameValue(typeof String.prototype, "object", "String.prototype object");
assert.sameValue(String.prototype.constructor, String, "String.prototype.constructor");
assert.sameValue(Number.prototype.constructor, Number, "Number.prototype.constructor");
assert.sameValue("hi".toUpperCase(), "HI", "normal string method");
assert.sameValue((3.14).toFixed(1), "3.1", "normal number method");
