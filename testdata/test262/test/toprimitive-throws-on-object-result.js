/*---
description: ToPrimitive throws TypeError when no primitive can be produced
esid: sec-ordinarytoprimitive
---*/
// Both valueOf and toString return objects -> TypeError (in any coercion).
var g = { valueOf: function () { return {}; }, toString: function () { return {}; } };
assert.throws(TypeError, function () { return g + ""; }, "string concat");
assert.throws(TypeError, function () { return g * 2; }, "arithmetic");
assert.throws(TypeError, function () { return `${g}`; }, "template");

// Symbol.toPrimitive returning an object -> TypeError.
var s = { [Symbol.toPrimitive]: function () { return {}; } };
assert.throws(TypeError, function () { return +s; }, "number coercion");
assert.throws(TypeError, function () { return s + ""; }, "string coercion");

// A null-prototype object has no toString/valueOf -> TypeError on coercion.
assert.throws(TypeError, function () { return Object.create(null) + ""; }, "null-proto coercion");

// valueOf returning an object falls back to a primitive toString (no throw).
var ok = { valueOf: function () { return {}; }, toString: function () { return "OK"; } };
assert.sameValue(ok + "", "OK", "falls back to toString");

// Symbol.toPrimitive returning a primitive works, honoring the hint.
var good = { [Symbol.toPrimitive]: function (h) { return h === "number" ? 42 : "str"; } };
assert.sameValue(+good, 42, "number hint");
assert.sameValue(`${good}`, "str", "string hint");

// Ordinary objects/arrays still coerce without throwing.
assert.sameValue(({}) + "", "[object Object]", "plain object");
assert.sameValue([1, 2] + "", "1,2", "array");
assert.sameValue(({ valueOf: function () { return 5; } }) + 1, 6, "valueOf");
