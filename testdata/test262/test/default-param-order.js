/*---
description: Default parameter evaluation order and TDZ-like references
esid: sec-function-definitions
---*/
function f(a, b = a * 2, c = a + b) { return [a, b, c].join(","); }
assert.sameValue(f(1), "1,2,3");
assert.sameValue(f(2, 5), "2,5,7");
assert.sameValue(f(2, 5, 10), "2,5,10");
var order = [];
function track(label, val) { order.push(label); return val; }
function g(a = track("a", 1), b = track("b", 2)) { return a + b; }
g();
assert.sameValue(order.join(","), "a,b", "left-to-right");
function h(x, y = x) { return y; }
assert.sameValue(h(5), 5, "default refers to earlier param");
assert.sameValue(h(5, 10), 10);
