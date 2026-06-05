/*---
description: Logical assignment operators &&= ||= ??=
esid: sec-assignment-operators
---*/
var a = 1; a &&= 2; assert.sameValue(a, 2, "&&= when truthy");
var b = 0; b &&= 5; assert.sameValue(b, 0, "&&= short-circuits when falsy");
var c = 0; c ||= 9; assert.sameValue(c, 9, "||= when falsy");
var d = 3; d ||= 9; assert.sameValue(d, 3, "||= short-circuits when truthy");
var e = null; e ??= "default"; assert.sameValue(e, "default", "??= when nullish");
var f = 0; f ??= 9; assert.sameValue(f, 0, "??= keeps 0");
var obj = { x: 0, y: null };
obj.x ||= 10;
obj.y ??= 20;
assert.sameValue(obj.x, 10);
assert.sameValue(obj.y, 20);
var calls = 0;
function side() { calls++; return 1; }
var g = 5; g ||= side();
assert.sameValue(calls, 0, "||= does not evaluate rhs when truthy");
