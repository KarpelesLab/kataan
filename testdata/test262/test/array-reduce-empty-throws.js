/*---
description: Array.prototype.reduce/reduceRight throw TypeError on an empty array with no initial value
esid: sec-array.prototype.reduce
---*/
var threw = false;
try { [].reduce(function (a, b) { return a + b; }); } catch (e) { threw = e instanceof TypeError; }
assert.sameValue(threw, true, "reduce of empty array with no initial value throws TypeError");
var threwR = false;
try { [].reduceRight(function (a, b) { return a + b; }); } catch (e) { threwR = e instanceof TypeError; }
assert.sameValue(threwR, true, "reduceRight of empty array with no initial value throws TypeError");
// With an initial value, an empty array returns that value (no throw).
assert.sameValue([].reduce(function (a, b) { return a + b; }, 99), 99, "empty reduce with seed returns the seed");
assert.sameValue([].reduceRight(function (a, b) { return a + b; }, 7), 7, "empty reduceRight with seed");
// A single-element array with no seed returns that element (no callback call).
var calls = 0;
assert.sameValue([42].reduce(function (a, b) { calls++; return a + b; }), 42, "single element, no seed");
assert.sameValue(calls, 0, "callback not called for a single element with no seed");
// Normal reductions still work.
assert.sameValue([1, 2, 3, 4].reduce(function (a, b) { return a + b; }), 10, "reduce sum");
assert.sameValue([1, 2, 3, 4].reduce(function (a, b) { return a + b; }, 100), 110, "reduce sum with seed");
