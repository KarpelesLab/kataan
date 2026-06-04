/*---
description: Comma operator, nested ternaries, and short-circuit evaluation
esid: sec-comma-operator
---*/
var x = (1, 2, 3);
assert.sameValue(x, 3, "comma operator yields the last value");
var grade = function (n) { return n >= 90 ? "A" : n >= 80 ? "B" : n >= 70 ? "C" : "F"; };
assert.sameValue(grade(95), "A");
assert.sameValue(grade(85), "B");
assert.sameValue(grade(50), "F");
var calls = 0;
function inc() { calls++; return true; }
false && inc();
assert.sameValue(calls, 0, "&& short-circuits");
true || inc();
assert.sameValue(calls, 0, "|| short-circuits");
