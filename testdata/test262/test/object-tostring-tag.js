/*---
description: Object/String default toString/valueOf and Symbol.toStringTag
esid: sec-object.prototype.tostring
---*/
// Plain object toString / valueOf.
assert.sameValue(({}).toString(), "[object Object]", "plain object toString");
assert.sameValue(({ a: 1 }).valueOf().a, 1, "valueOf returns the object");
assert.sameValue("abc".toString(), "abc", "string toString");
assert.sameValue("abc".valueOf(), "abc", "string valueOf");
// A user-defined toString takes precedence.
assert.sameValue(({ toString() { return "custom"; } }).toString(), "custom", "user toString wins");
// An inherited toString is used.
var base = { toString() { return "base"; } };
assert.sameValue(Object.create(base).toString(), "base", "inherited toString");
// Symbol.toStringTag customizes the tag.
assert.sameValue(({ [Symbol.toStringTag]: "Widget" }).toString(), "[object Widget]", "toStringTag");
// Symbol.toStringTag is a real well-known symbol.
assert.sameValue(typeof Symbol.toStringTag, "symbol", "Symbol.toStringTag is a symbol");
assert.sameValue(Symbol.toStringTag, Symbol.toStringTag, "stable identity");
var o = { [Symbol.toStringTag]: "X" };
assert.sameValue(Object.getOwnPropertySymbols(o).length, 1, "stored as a symbol key");
assert.sameValue(Object.getOwnPropertySymbols(o)[0], Symbol.toStringTag, "the toStringTag symbol");
// Other well-known symbols now exist.
assert.sameValue(typeof Symbol.species, "symbol", "Symbol.species");
assert.sameValue(typeof Symbol.replace, "symbol", "Symbol.replace");
assert.sameValue(typeof Symbol.hasInstance, "symbol", "Symbol.hasInstance");
// Array/Date still use their own toString.
assert.sameValue([1, 2, 3].toString(), "1,2,3", "array toString");
assert.sameValue(new Error("x").toString(), "Error: x", "error toString unaffected");
