/*---
description: Generator next, return, and manual iteration
esid: sec-generator-function-definitions
---*/
function* gen() { yield 1; yield 2; yield 3; }
var g = gen();
assert.sameValue(g.next().value, 1);
assert.sameValue(g.next().value, 2);
var returned = g.return(99);
assert.sameValue(returned.value, 99, "return value");
assert.sameValue(returned.done, true, "done after return");
assert.sameValue(g.next().done, true, "exhausted after return");
function* withValues() { var x = yield 1; }
var w = withValues();
assert.sameValue(w.next().value, 1);
assert.sameValue(w.next(42).done, true, "next with value completes");
