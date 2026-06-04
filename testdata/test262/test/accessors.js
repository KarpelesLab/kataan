/*---
description: Object getters and setters
esid: sec-method-definitions
---*/
var temp = {
  _c: 0,
  get celsius() { return this._c; },
  set celsius(v) { this._c = v; },
  get fahrenheit() { return this._c * 9 / 5 + 32; }
};
temp.celsius = 100;
assert.sameValue(temp.celsius, 100);
assert.sameValue(temp.fahrenheit, 212);
