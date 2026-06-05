/*---
description: Paired getter/setter via defineProperty and literals
esid: sec-object.defineproperty
---*/
var store = {};
var o = {};
Object.defineProperty(o, "temp", {
  get: function () { return store.temp || 0; },
  set: function (v) { store.temp = v < -273 ? -273 : v; },
  enumerable: true
});
o.temp = 25;
assert.sameValue(o.temp, 25);
o.temp = -300;
assert.sameValue(o.temp, -273, "setter clamps");
var counter = {
  _n: 0,
  get next() { return ++this._n; }
};
assert.sameValue(counter.next, 1);
assert.sameValue(counter.next, 2, "getter has side effects");
var withBoth = {
  _val: "init",
  get val() { return this._val; },
  set val(v) { this._val = v.toUpperCase(); }
};
withBoth.val = "hello";
assert.sameValue(withBoth.val, "HELLO");
