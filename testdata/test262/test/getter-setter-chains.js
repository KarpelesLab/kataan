/*---
description: Getter and setter inheritance and overriding through prototypes
esid: sec-property-accessors
---*/
var temp = {
  _celsius: 0,
  get celsius() { return this._celsius; },
  set celsius(v) { this._celsius = v; },
  get fahrenheit() { return this._celsius * 9 / 5 + 32; },
  set fahrenheit(v) { this._celsius = (v - 32) * 5 / 9; }
};
temp.celsius = 25;
assert.sameValue(temp.fahrenheit, 77, "computed getter");
temp.fahrenheit = 212;
assert.sameValue(temp.celsius, 100, "setter computes back");
var derived = Object.create(temp);
derived._celsius = 10;
assert.sameValue(derived.fahrenheit, 50, "inherited getter uses own state");
var obj = {};
var log = [];
Object.defineProperty(obj, "tracked", {
  get: function () { log.push("get"); return 42; },
  set: function (v) { log.push("set:" + v); }
});
obj.tracked = 5;
var x = obj.tracked;
assert.sameValue(log.join(","), "set:5,get", "getter/setter side effects");
assert.sameValue(x, 42);
