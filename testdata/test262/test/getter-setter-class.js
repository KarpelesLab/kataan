/*---
description: Class accessors (get/set) and computed access
esid: sec-class-definitions
---*/
class Temp {
  constructor(c) { this._c = c; }
  get celsius() { return this._c; }
  set celsius(v) { this._c = v; }
  get fahrenheit() { return this._c * 9 / 5 + 32; }
  set fahrenheit(f) { this._c = (f - 32) * 5 / 9; }
}
var t = new Temp(0);
assert.sameValue(t.fahrenheit, 32);
t.celsius = 100;
assert.sameValue(t.fahrenheit, 212);
t.fahrenheit = 32;
assert.sameValue(t.celsius, 0);
