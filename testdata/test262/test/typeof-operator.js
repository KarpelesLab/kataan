/*---
description: typeof across value kinds
esid: sec-typeof-operator
---*/
assert.sameValue(typeof 42, "number");
assert.sameValue(typeof "s", "string");
assert.sameValue(typeof true, "boolean");
assert.sameValue(typeof undefined, "undefined");
assert.sameValue(typeof function () {}, "function");
assert.sameValue(typeof {}, "object");
assert.sameValue(typeof [], "object");
assert.sameValue(typeof null, "object");
assert.sameValue(typeof notDeclared, "undefined", "typeof on an undeclared name does not throw");
