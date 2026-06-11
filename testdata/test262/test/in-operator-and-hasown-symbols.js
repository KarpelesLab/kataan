/*---
description: the in operator requires an object RHS; Object.hasOwn handles symbol keys
esid: sec-relational-operators-runtime-semantics-evaluation
---*/
function throwsType(fn) { try { fn(); return false; } catch (e) { return e instanceof TypeError; } }

// `in` with a non-object right operand is a TypeError.
assert.sameValue(throwsType(function () { return "x" in 5; }), true, "number RHS");
assert.sameValue(throwsType(function () { return "x" in null; }), true, "null RHS");
assert.sameValue(throwsType(function () { return "x" in undefined; }), true, "undefined RHS");
assert.sameValue(throwsType(function () { return "x" in "str"; }), true, "string primitive RHS");
assert.sameValue(throwsType(function () { return "x" in true; }), true, "boolean RHS");
assert.sameValue(throwsType(function () { return "x" in Symbol(); }), true, "symbol RHS");

// `in` with an object RHS works (own, inherited, array index, proxy has trap).
assert.sameValue("a" in { a: 1 }, true, "own property");
assert.sameValue("b" in { a: 1 }, false, "missing property");
assert.sameValue(0 in [1, 2], true, "array index");
assert.sameValue("toString" in {}, true, "inherited from Object.prototype");
var p = new Proxy({}, { has: function (t, k) { return k === "x"; } });
assert.sameValue("x" in p, true, "proxy has trap true");
assert.sameValue("y" in p, false, "proxy has trap false");

// Object.hasOwn resolves symbol keys (own, including non-enumerable).
var sym = Symbol("s");
var o = { [sym]: 1, normal: 2 };
assert.sameValue(Object.hasOwn(o, sym), true, "own symbol key");
assert.sameValue(Object.hasOwn(o, Symbol("other")), false, "a different symbol");
assert.sameValue(Object.hasOwn(o, "normal"), true, "string key");
var sym2 = Symbol("s2");
Object.defineProperty(o, sym2, { value: 3, enumerable: false });
assert.sameValue(Object.hasOwn(o, sym2), true, "non-enumerable symbol key");

// Object.hasOwn on array indices and length.
assert.sameValue(Object.hasOwn([1, 2, 3], "0"), true, "array index 0");
assert.sameValue(Object.hasOwn([1, 2, 3], "5"), false, "out-of-range index");
assert.sameValue(Object.hasOwn([1, 2, 3], "length"), true, "array length");
