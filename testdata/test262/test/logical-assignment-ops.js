/*---
description: &&=, ||=, ??= logical assignment operators
esid: sec-assignment-operators
---*/
var a = 1;
a &&= 2;
assert.sameValue(a, 2, "&&= assigns when truthy");
var b = 0;
b &&= 5;
assert.sameValue(b, 0, "&&= skips when falsy");
var c = 0;
c ||= 10;
assert.sameValue(c, 10, "||= assigns when falsy");
var d = 5;
d ||= 20;
assert.sameValue(d, 5, "||= skips when truthy");
var e = null;
e ??= "default";
assert.sameValue(e, "default", "??= assigns when nullish");
var f = "value";
f ??= "other";
assert.sameValue(f, "value", "??= skips when not nullish");
var g = 0;
g ??= 99;
assert.sameValue(g, 0, "??= keeps 0");
var obj = { x: null, y: 5 };
obj.x ??= "set";
obj.y ??= "skip";
assert.sameValue(obj.x, "set");
assert.sameValue(obj.y, 5);
var count = 0;
var h = 1;
h &&= (count++, 2);
assert.sameValue(count, 1, "&&= evaluates rhs when truthy");
var k = 0;
k &&= (count++, 2);
assert.sameValue(count, 1, "&&= skips rhs when falsy");
