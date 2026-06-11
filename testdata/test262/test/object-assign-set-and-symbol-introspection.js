/*---
description: Object.assign uses [[Set]] (own setters), copies string sources; symbol hasOwn/propertyIsEnumerable
esid: sec-object.assign
---*/
// Object.assign invokes the target's own setter (it uses [[Set]], not define).
var setlog = [];
var st = { set y(v) { setlog.push(v); } };
Object.assign(st, { y: 9 });
assert.sameValue(setlog.join(","), "9", "target setter fires");

// A primitive string source contributes its character indices.
assert.sameValue(JSON.stringify(Object.assign({}, "ab")), '{"0":"a","1":"b"}', "string source");

// Symbols are copied; getters on the source are invoked (value, not accessor, copied).
var s = Symbol("s");
var o = Object.assign({}, { a: 1, [s]: 2 });
assert.sameValue(o.a, 1, "string key");
assert.sameValue(o[s], 2, "symbol key");
assert.sameValue(Object.getOwnPropertySymbols(o).length, 1, "one symbol");
var calls = 0;
var g = Object.assign({}, { get x() { calls++; return 5; } });
assert.sameValue(g.x, 5, "getter value copied");
assert.sameValue(calls, 1, "getter invoked once");

// Non-enumerable skipped; later source wins; null/undefined sources ignored.
var ne = {};
Object.defineProperty(ne, "h", { value: 1, enumerable: false });
ne.v = 2;
assert.sameValue(JSON.stringify(Object.assign({}, ne)), '{"v":2}', "non-enumerable skipped");
assert.sameValue(JSON.stringify(Object.assign({ a: 1 }, { b: 2 }, { a: 3 })), '{"a":3,"b":2}', "later wins");
assert.sameValue(JSON.stringify(Object.assign({ a: 1 }, null, undefined, { b: 2 })), '{"a":1,"b":2}', "nullish skipped");

// hasOwnProperty / propertyIsEnumerable work with symbol keys.
var key = Symbol("k");
var obj = {};
obj[key] = 1;
assert.sameValue(obj.hasOwnProperty(key), true, "hasOwnProperty symbol");
assert.sameValue(obj.propertyIsEnumerable(key), true, "propertyIsEnumerable symbol");
assert.sameValue(obj.hasOwnProperty(Symbol("other")), false, "different symbol");
var key2 = Symbol("k2");
Object.defineProperty(obj, key2, { value: 2, enumerable: false });
assert.sameValue(obj.propertyIsEnumerable(key2), false, "non-enumerable symbol");
