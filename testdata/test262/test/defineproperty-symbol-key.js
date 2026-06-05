/*---
description: Object.defineProperty / getOwnPropertyDescriptor / Reflect with symbol keys
esid: sec-object.defineproperty
---*/
var s = Symbol("k");
var o = {};
Object.defineProperty(o, s, { value: 42, enumerable: false, writable: true, configurable: true });
assert.sameValue(o[s], 42, "symbol-keyed value via defineProperty");
var d = Object.getOwnPropertyDescriptor(o, s);
assert.sameValue(d.value, 42, "getOwnPropertyDescriptor reads the symbol key");
assert.sameValue(d.enumerable, false, "descriptor enumerable flag");
assert.sameValue(Object.keys(o).length, 0, "non-enumerable symbol absent from keys");
assert.sameValue(Object.getOwnPropertySymbols(o).length, 1, "symbol appears in getOwnPropertySymbols");
assert.sameValue(Object.getOwnPropertySymbols(o)[0], s, "the same symbol");
// An accessor defined on a symbol key.
var s2 = Symbol("acc");
var backing = 0;
Object.defineProperty(o, s2, { get: function () { return backing; }, set: function (v) { backing = v; } });
o[s2] = 7;
assert.sameValue(o[s2], 7, "symbol-keyed accessor");
assert.sameValue(typeof Object.getOwnPropertyDescriptor(o, s2).get, "function", "accessor descriptor");
// Reflect mirrors.
var s3 = Symbol("r");
assert.sameValue(Reflect.defineProperty(o, s3, { value: 9 }), true, "Reflect.defineProperty symbol");
assert.sameValue(Reflect.getOwnPropertyDescriptor(o, s3).value, 9, "Reflect.getOwnPropertyDescriptor symbol");
assert.sameValue(o[Symbol.for("global")] === undefined, true, "absent registered symbol reads undefined");
