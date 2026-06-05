/*---
description: Generator return values and early termination
esid: sec-generator-function-definitions
---*/
function* gen() {
  yield 1;
  yield 2;
  return 99;
  yield 3;
}
var g = gen();
assert.sameValue(g.next().value, 1);
assert.sameValue(g.next().value, 2);
var r = g.next();
assert.sameValue(r.value, 99, "return value");
assert.sameValue(r.done, true, "done after return");
assert.sameValue(g.next().done, true);
assert.sameValue([...gen()].join(","), "1,2", "spread excludes return value");
