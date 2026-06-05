/*---
description: Reflect methods for meta-operations
esid: sec-reflect-object
---*/
var o = { a: 1, b: 2 };
assert.sameValue(Reflect.get(o, "a"), 1);
Reflect.set(o, "c", 3);
assert.sameValue(o.c, 3);
assert.sameValue(Reflect.has(o, "a"), true);
assert.sameValue(Reflect.has(o, "z"), false);
assert.sameValue(Reflect.ownKeys(o).join(","), "a,b,c");
Reflect.deleteProperty(o, "a");
assert.sameValue(Reflect.has(o, "a"), false);
function Point(x, y) { this.x = x; this.y = y; }
var p = Reflect.construct(Point, [3, 4]);
assert.sameValue(p.x, 3);
assert.sameValue(p.y, 4);
assert.sameValue(Reflect.apply(function (a, b) { return a + b; }, null, [5, 6]), 11);
var keys = Reflect.ownKeys({ x: 1, y: 2 });
assert.sameValue(keys.length, 2);
