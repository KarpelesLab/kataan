/*---
description: Object.getOwnPropertySymbols and Reflect.ownKeys with symbol keys
esid: sec-object.getownpropertysymbols
---*/
var s1 = Symbol("first");
var s2 = Symbol("second");
var o = { a: 1, b: 2 };
o[s1] = "x";
o[s2] = "y";
var symbols = Object.getOwnPropertySymbols(o);
assert.sameValue(symbols.length, 2, "two symbol keys");
assert.sameValue(symbols[0].description, "first");
assert.sameValue(symbols[1].description, "second");
assert.sameValue(symbols[0], s1, "same symbol identity");
assert.sameValue(o[symbols[0]], "x", "value reachable via the returned symbol");
assert.sameValue(Object.getOwnPropertyNames(o).join(","), "a,b", "names exclude symbols");
assert.sameValue(Object.getOwnPropertySymbols({}).length, 0, "no symbols");
var keys = Reflect.ownKeys(o);
assert.sameValue(keys.length, 4, "ownKeys: strings + symbols");
assert.sameValue(keys[0], "a", "string keys first");
assert.sameValue(keys[1], "b");
assert.sameValue(keys[2], s1, "then symbol keys");
assert.sameValue(keys[3], s2);
assert.sameValue(Object.keys(o).length, 2, "Object.keys excludes symbols");
var only = {};
only[Symbol("z")] = 1;
assert.sameValue(Object.keys(only).length, 0, "symbol-only object has no string keys");
assert.sameValue(Object.getOwnPropertySymbols(only).length, 1);
