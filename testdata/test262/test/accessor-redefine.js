/*---
description: Redefining accessors and data properties
esid: sec-object.defineproperty
---*/
var o = {};
Object.defineProperty(o, "x", { get: function () { return 1; }, configurable: true });
assert.sameValue(o.x, 1);
Object.defineProperty(o, "x", { get: function () { return 2; }, configurable: true });
assert.sameValue(o.x, 2, "getter redefined");
Object.defineProperty(o, "x", { value: 42, configurable: true });
assert.sameValue(o.x, 42, "accessor to data");
var counter = { _n: 0 };
Object.defineProperty(counter, "n", {
  get: function () { return this._n; },
  set: function (v) { this._n = v * 2; },
  configurable: true
});
counter.n = 5;
assert.sameValue(counter.n, 10, "setter doubles");
var obj = { a: 1 };
Object.defineProperty(obj, "b", { value: 2, enumerable: true });
Object.defineProperty(obj, "c", { value: 3, enumerable: false });
assert.sameValue(Object.keys(obj).join(","), "a,b", "c is non-enumerable");
assert.sameValue(obj.c, 3, "but still accessible");
var withGetter = {};
var calls = 0;
Object.defineProperty(withGetter, "tracked", { get: function () { calls++; return calls; } });
assert.sameValue(withGetter.tracked, 1);
assert.sameValue(withGetter.tracked, 2, "getter called each time");
