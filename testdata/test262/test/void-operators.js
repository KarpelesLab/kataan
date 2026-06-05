/*---
description: void, comma, and conditional operators
esid: sec-void-operator
---*/
assert.sameValue(void 0, undefined);
assert.sameValue(void "anything", undefined);
assert.sameValue((1, 2, 3), 3, "comma yields last");
var x = (5, 10);
assert.sameValue(x, 10);
assert.sameValue(true ? 1 : 2, 1);
assert.sameValue(false ? 1 : true ? 2 : 3, 2, "nested ternary");
var count = 0;
var r = (count++, count++, count);
assert.sameValue(r, 2);
assert.sameValue(count, 2);
assert.sameValue(!!"", false);
assert.sameValue(!!"x", true);
