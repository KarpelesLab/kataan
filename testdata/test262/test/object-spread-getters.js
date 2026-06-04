/*---
description: Object spread copies own enumerable props and invokes getters
esid: sec-object-initializer
---*/
var src = { a: 1, get b() { return 2; } };
var copy = { ...src, c: 3 };
assert.sameValue(copy.a, 1);
assert.sameValue(copy.b, 2, "getter is invoked during spread");
assert.sameValue(copy.c, 3);
var merged = { ...{ x: 1 }, ...{ y: 2 }, x: 9 };
assert.sameValue(merged.x, 9, "later keys win");
assert.sameValue(merged.y, 2);
