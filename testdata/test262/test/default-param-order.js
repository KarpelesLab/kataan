/*---
description: Default parameters evaluate left-to-right and can reference earlier params
esid: sec-function-definitions
---*/
function f(a, b = a + 1, c = b * 2) { return [a, b, c].join(","); }
assert.sameValue(f(1), "1,2,4");
assert.sameValue(f(1, 10), "1,10,20");
assert.sameValue(f(1, 10, 100), "1,10,100");
var calls = 0;
function side() { calls++; return 5; }
function g(a = side(), b = side()) { return a + b; }
g();
assert.sameValue(calls, 2, "each missing default is evaluated once");
