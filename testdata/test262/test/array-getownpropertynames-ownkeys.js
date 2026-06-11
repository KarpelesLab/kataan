/*---
description: getOwnPropertyNames / Reflect.ownKeys on an array list its indices, length, and named props
esid: sec-array-exotic-objects-ownpropertykeys
---*/
// Indices (ascending) then "length".
assert.sameValue(Object.getOwnPropertyNames([1, 2, 3]).join(","), "0,1,2,length", "getOwnPropertyNames");
assert.sameValue(JSON.stringify(Reflect.ownKeys([5, 6])), '["0","1","length"]', "Reflect.ownKeys");
assert.sameValue(Object.getOwnPropertyNames([]).join(","), "length", "empty array still has length");

// A custom named property comes after length (creation order).
var a = [10, 20];
a.custom = "x";
assert.sameValue(Object.getOwnPropertyNames(a).join(","), "0,1,length,custom", "custom after length");

// Non-enumerable named properties are included.
var ne = [1];
Object.defineProperty(ne, "hidden", { value: 9, enumerable: false });
assert.sameValue(Object.getOwnPropertyNames(ne).join(","), "0,length,hidden", "includes non-enumerable");

// Object.keys still lists only enumerable string keys (indices + enumerable customs).
assert.sameValue(Object.keys(a).join(","), "0,1,custom", "Object.keys");
assert.sameValue(Object.keys(ne).join(","), "0", "Object.keys excludes length and non-enumerable");

// Plain objects are unaffected.
assert.sameValue(Object.getOwnPropertyNames({ a: 1, b: 2 }).join(","), "a,b", "object names");
assert.sameValue(JSON.stringify(Reflect.ownKeys({ x: 1 })), '["x"]', "object ownKeys");
