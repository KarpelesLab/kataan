/*---
description: Spread for shallow cloning and merging arrays/objects
esid: sec-object-initializer
---*/
var arr = [1, 2, 3];
var arrCopy = [...arr];
arrCopy.push(4);
assert.sameValue(arr.length, 3, "spread clones the array");
assert.sameValue(arrCopy.length, 4);
var obj = { a: 1, nested: { x: 1 } };
var objCopy = { ...obj };
objCopy.a = 99;
assert.sameValue(obj.a, 1, "spread clones top level");
objCopy.nested.x = 5;
assert.sameValue(obj.nested.x, 5, "spread is shallow");
var combined = { ...{ a: 1 }, ...{ b: 2 }, ...{ a: 3 } };
assert.sameValue(combined.a + "," + combined.b, "3,2");
