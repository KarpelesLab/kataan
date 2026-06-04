/*---
description: Generator functions yield a sequence and are iterable
esid: sec-generator-function-definitions
---*/
function* range(n) {
  for (var i = 0; i < n; i++) yield i;
}
var collected = [];
for (var v of range(4)) { collected.push(v); }
assert.sameValue(collected.join(","), "0,1,2,3");

var g = range(2);
assert.sameValue(g.next().value, 0);
assert.sameValue(g.next().value, 1);
assert.sameValue(g.next().done, true);

assert.sameValue([...range(3)].join(","), "0,1,2");

function* withDelegate() { yield 1; yield* range(2); yield 9; }
assert.sameValue([...withDelegate()].join(","), "1,0,1,9");
