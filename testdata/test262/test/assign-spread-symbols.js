/*---
description: Object.assign and spread copy own enumerable symbol keys
esid: sec-object.assign
---*/
var sym = Symbol("k");
var src = { a: 1 };
src[sym] = "sym-value";
var assigned = Object.assign({}, src);
assert.sameValue(assigned.a, 1, "string key copied");
assert.sameValue(assigned[sym], "sym-value", "symbol key copied by assign");
var spread = { ...src };
assert.sameValue(spread.a, 1);
assert.sameValue(spread[sym], "sym-value", "symbol key copied by spread");
assert.sameValue(Object.keys(spread).join(","), "a", "symbol not enumerated by Object.keys");
var s1 = Symbol("a"), s2 = Symbol("b");
var multi = {};
multi[s1] = 1;
multi[s2] = 2;
var copy = Object.assign({}, multi);
assert.sameValue(copy[s1] + copy[s2], 3, "multiple symbol keys");
var getterSym = Symbol("g");
var withGetter = { get [getterSym]() { return 42; } };
assert.sameValue(Object.assign({}, withGetter)[getterSym], 42, "symbol getter invoked");
var merged = Object.assign({}, { a: 1 }, { b: 2 }, { a: 3 });
assert.sameValue(merged.a, 3, "later sources override");
assert.sameValue(merged.b, 2);
var distinctCopy = { ...src };
assert.sameValue(distinctCopy[Symbol("k")], undefined, "a different symbol is absent");
