/*---
description: Spread (call/array) and accessor properties (get/set)
features: [spread]
---*/
function sum3(a, b, c) { return a + b + c; }
var nums = [10, 20, 30];
assert.sameValue(sum3(...nums), 60, "spread into a call");
assert.sameValue([0, ...nums, 40].length, 5, "array spread length");
assert.sameValue([0, ...nums, 40][4], 40, "array spread tail");

// Template literals with embedded expressions.
var n = 5;
assert.sameValue(`v=${n * 2}!`, "v=10!", "template literal interpolation");

// Accessor get/set with `this`.
var o = {
  _v: 1,
  get v() { return this._v * 10; },
  set v(x) { this._v = x; },
};
assert.sameValue(o.v, 10, "getter reads through");
o.v = 4;
assert.sameValue(o.v, 40, "setter then getter");
