/*---
description: Object.keys/values/entries enumerate an array's integer indices
esid: sec-object.keys
---*/
// An array's own enumerable keys are its indices, as strings, in order.
assert.sameValue(Object.keys([1, 2, 3]).join(","), "0,1,2", "keys are the indices");
assert.sameValue(Object.values([10, 20, 30]).join(","), "10,20,30", "values are the elements");

var entries = Object.entries(["a", "b"]);
assert.sameValue(entries.length, 2, "two entries");
assert.sameValue(entries[0][0], "0", "entry key is the index string");
assert.sameValue(entries[0][1], "a", "entry value is the element");
assert.sameValue(entries[1][0], "1", "second key");
assert.sameValue(entries[1][1], "b", "second value");

// Empty array → no keys.
assert.sameValue(Object.keys([]).length, 0, "empty array has no keys");

// Object.keys still works on plain objects.
assert.sameValue(Object.keys({ a: 1, b: 2 }).join(","), "a,b", "object keys unaffected");
assert.sameValue(Object.values({ a: 1, b: 2 }).join(","), "1,2", "object values unaffected");

// for-of over Object.entries pairs.
var s = "";
for (var pair of Object.entries([7, 8])) { s += pair[0] + "=" + pair[1] + ";"; }
assert.sameValue(s, "0=7;1=8;", "entries iterate as [index, value] pairs");
