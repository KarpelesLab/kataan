/*---
description: typeof and instanceof comprehensive
esid: sec-typeof-operator
---*/
assert.sameValue(typeof undefined, "undefined");
assert.sameValue(typeof null, "object");
assert.sameValue(typeof 0, "number");
assert.sameValue(typeof NaN, "number");
assert.sameValue(typeof "", "string");
assert.sameValue(typeof true, "boolean");
assert.sameValue(typeof 0n, "bigint");
assert.sameValue(typeof Symbol(), "symbol");
assert.sameValue(typeof function () {}, "function");
assert.sameValue(typeof (() => {}), "function");
assert.sameValue(typeof [], "object");
assert.sameValue(typeof {}, "object");
assert.sameValue(typeof new Date(), "object");
assert.sameValue(typeof /x/, "object");
assert.sameValue([] instanceof Array, true);
assert.sameValue([] instanceof Object, true);
assert.sameValue({} instanceof Object, true);
assert.sameValue(new Date() instanceof Date, true);
assert.sameValue(/x/ instanceof RegExp, true);
assert.sameValue(new Error() instanceof Error, true);
assert.sameValue(new Map() instanceof Map, true);
assert.sameValue(new Set() instanceof Set, true);
assert.sameValue(typeof (function(){}), "function");
class A {} class B extends A {}
assert.sameValue(new B() instanceof A, true);
assert.sameValue(new B() instanceof B, true);
assert.sameValue(new A() instanceof B, false);
