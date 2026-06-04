/*---
description: Symbol primitives — uniqueness, description, typeof, and the registry
esid: sec-symbol-constructor
---*/
var s = Symbol("desc");
assert.sameValue(typeof s, "symbol");
assert.sameValue(s.toString(), "Symbol(desc)");
assert.sameValue(s.description, "desc");
assert.sameValue(s === s, true);
assert.sameValue(Symbol("x") === Symbol("x"), false, "each Symbol() is unique");

var a = Symbol.for("shared");
var b = Symbol.for("shared");
assert.sameValue(a === b, true, "Symbol.for returns the registered symbol");
assert.sameValue(Symbol.keyFor(a), "shared");
assert.sameValue(Symbol.keyFor(Symbol("local")), undefined);

assert.sameValue(typeof Symbol.iterator, "symbol");
assert.sameValue(Symbol.iterator === Symbol.iterator, true, "well-known symbols are stable");
