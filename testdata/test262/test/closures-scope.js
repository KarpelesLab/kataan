/*---
description: Closures capture variables; counters and loop-let binding
esid: sec-closure
---*/
function makeCounter() {
  var n = 0;
  return function () { return ++n; };
}
var c = makeCounter();
assert.sameValue(c(), 1);
assert.sameValue(c(), 2);
assert.sameValue(c(), 3);
var c2 = makeCounter();
assert.sameValue(c2(), 1, "independent closure state");

var fns = [];
for (let i = 0; i < 3; i++) fns.push(function () { return i; });
assert.sameValue(fns[0]() + "," + fns[1]() + "," + fns[2](), "0,1,2", "let is per-iteration");

var adder = (function (base) { return function (x) { return base + x; }; })(100);
assert.sameValue(adder(5), 105, "IIFE-captured base");
