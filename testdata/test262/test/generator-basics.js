/*---
description: Generator functions, yield, and iteration
esid: sec-generator-function-definitions
---*/
function* gen() { yield 1; yield 2; yield 3; }
var out = [];
for (var v of gen()) out.push(v);
assert.sameValue(out.join(","), "1,2,3");
assert.sameValue([...gen()].join("-"), "1-2-3");

function* range(a, b) { for (var i = a; i <= b; i++) yield i; }
assert.sameValue([...range(5, 8)].join(","), "5,6,7,8");

var it = gen();
assert.sameValue(it.next().value, 1);
assert.sameValue(it.next().value, 2);
assert.sameValue(it.next().done, false);
assert.sameValue(it.next().done, true);
