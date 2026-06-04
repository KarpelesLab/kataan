/*---
description: A closure captures and mutates an enclosing binding
esid: sec-function-definitions
---*/
function makeCounter() {
  var n = 0;
  return function () { n += 1; return n; };
}
var c = makeCounter();
assert.sameValue(c(), 1);
assert.sameValue(c(), 2);
assert.sameValue(c(), 3);
var d = makeCounter();
assert.sameValue(d(), 1, "counters are independent");
