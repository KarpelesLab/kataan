/*---
description: Getters/setters via literal and Object.defineProperty
esid: sec-object.defineproperty
---*/
var temp = {
  _c: 0,
  get celsius() { return this._c; },
  set celsius(v) { this._c = v; },
  get fahrenheit() { return this._c * 9 / 5 + 32; }
};
temp.celsius = 25;
assert.sameValue(temp.celsius, 25);
assert.sameValue(temp.fahrenheit, 77);

var o = {};
Object.defineProperty(o, "x", { get: function () { return 42; } });
assert.sameValue(o.x, 42, "defineProperty getter");
var store = 0;
Object.defineProperty(o, "y", {
  get: function () { return store; },
  set: function (v) { store = v * 2; }
});
o.y = 10;
assert.sameValue(o.y, 20, "defineProperty setter");
