/*---
description: Computed getter/setter names in objects and classes
esid: sec-object-initializer
---*/
var prop = "value";
var obj = {
  _v: 10,
  get [prop]() { return this._v; },
  set [prop](v) { this._v = v; }
};
assert.sameValue(obj.value, 10, "computed getter name");
obj.value = 20;
assert.sameValue(obj._v, 20, "computed setter name");
var prefix = "get";
var methods = {
  get [prefix + "X"]() { return 1; },
  get [prefix + "Y"]() { return 2; }
};
assert.sameValue(methods.getX, 1);
assert.sameValue(methods.getY, 2);
class Temperature {
  constructor() { this._c = 0; }
  get ["celsius"]() { return this._c; }
  set ["celsius"](v) { this._c = v; }
}
var t = new Temperature();
t.celsius = 25;
assert.sameValue(t.celsius, 25, "class computed accessor");
var key = "dynamic";
var o2 = { get [key]() { return "computed"; } };
assert.sameValue(o2.dynamic, "computed");
