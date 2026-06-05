/*---
description: typeof on undeclared variables and various values
esid: sec-typeof-operator
---*/
assert.sameValue(typeof undeclaredVariable, "undefined", "typeof undeclared is safe");
assert.sameValue(typeof undefined, "undefined");
assert.sameValue(typeof null, "object");
assert.sameValue(typeof 42, "number");
assert.sameValue(typeof 42n, "bigint");
assert.sameValue(typeof "s", "string");
assert.sameValue(typeof true, "boolean");
assert.sameValue(typeof Symbol(), "symbol");
assert.sameValue(typeof function () {}, "function");
assert.sameValue(typeof [], "object");
assert.sameValue(typeof {}, "object");
assert.sameValue(typeof (() => {}), "function", "arrow is a function");
