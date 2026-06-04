/*---
description: Class getters/setters, static members, and private fields
esid: sec-class-definitions
---*/
class Temperature {
  #celsius = 0;
  get celsius() { return this.#celsius; }
  set celsius(v) { this.#celsius = v; }
  get fahrenheit() { return this.#celsius * 9 / 5 + 32; }
  static fromFahrenheit(f) { var t = new Temperature(); t.celsius = (f - 32) * 5 / 9; return t; }
}
var t = new Temperature();
t.celsius = 100;
assert.sameValue(t.celsius, 100);
assert.sameValue(t.fahrenheit, 212);
var t2 = Temperature.fromFahrenheit(32);
assert.sameValue(t2.celsius, 0, "static factory");
