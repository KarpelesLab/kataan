/*---
description: implicit conversion of a Symbol to string/number throws a TypeError
esid: sec-symbol-objects
features: [Symbol]
---*/
var s = Symbol("s");

// String concatenation / numeric `+`.
assert.throws(TypeError, function () { return s + ""; }, "symbol + string");
assert.throws(TypeError, function () { return "x" + s; }, "string + symbol");
assert.throws(TypeError, function () { return s + 1; }, "symbol + number");

// Other arithmetic / relational operators.
assert.throws(TypeError, function () { return s - 1; }, "symbol - number");
assert.throws(TypeError, function () { return s * 2; }, "symbol * number");
assert.throws(TypeError, function () { return s < 1; }, "symbol < number");

// Template literal (implicit ToString).
assert.throws(TypeError, function () { return `${s}`; }, "template literal");

// Unary numeric operators (ToNumber).
assert.throws(TypeError, function () { return +s; }, "unary +");
assert.throws(TypeError, function () { return -s; }, "unary -");
assert.throws(TypeError, function () { return ~s; }, "bitwise not");

// EXPLICIT conversions and non-coercing operators are unaffected.
assert.sameValue(String(s), "Symbol(s)", "String() is explicit");
assert.sameValue(s.toString(), "Symbol(s)", "toString is explicit");
assert.sameValue(typeof s, "symbol", "typeof");
assert.sameValue(!s, false, "logical not (symbols are truthy)");
assert.sameValue(Boolean(s), true, "Boolean()");
assert.sameValue(s === s, true, "strict equality");
assert.sameValue(s === Symbol("s"), false, "distinct symbols");
