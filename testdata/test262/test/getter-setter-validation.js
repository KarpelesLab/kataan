/*---
description: Getter/setter with validation logic
esid: sec-property-accessors
---*/
var temp = {
  _celsius: 0,
  get celsius() { return this._celsius; },
  set celsius(v) { if (v < -273.15) throw new RangeError("below absolute zero"); this._celsius = v; },
  get fahrenheit() { return this._celsius * 9 / 5 + 32; },
  set fahrenheit(v) { this.celsius = (v - 32) * 5 / 9; }
};
temp.celsius = 25;
assert.sameValue(temp.celsius, 25);
assert.sameValue(temp.fahrenheit, 77, "computed from celsius");
temp.fahrenheit = 32;
assert.sameValue(temp.celsius, 0, "setter chains to another setter");
var threw = false;
try { temp.celsius = -300; } catch (e) { threw = e instanceof RangeError; }
assert.sameValue(threw, true, "validation throws");
assert.sameValue(temp.celsius, 0, "unchanged after throw");
var person = {
  _name: "",
  get name() { return this._name; },
  set name(v) { this._name = String(v).trim(); }
};
person.name = "  Alice  ";
assert.sameValue(person.name, "Alice", "setter transforms");
var clamped = {
  _v: 0,
  get v() { return this._v; },
  set v(x) { this._v = Math.max(0, Math.min(100, x)); }
};
clamped.v = 150;
assert.sameValue(clamped.v, 100, "clamped high");
clamped.v = -50;
assert.sameValue(clamped.v, 0, "clamped low");
