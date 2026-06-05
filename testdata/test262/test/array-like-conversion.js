/*---
description: Converting array-like and iterable objects to arrays
esid: sec-array.from
---*/
function makeArrayLike() {
  return { length: 3, 0: "x", 1: "y", 2: "z" };
}
assert.sameValue(Array.from(makeArrayLike()).join(","), "x,y,z");
assert.sameValue(Array.from("abc").length, 3);
assert.sameValue(Array.from(new Set([1, 2, 2, 3])).length, 3);
var indexed = Array.from({ length: 3 }, function (_, i) { return i * 2; });
assert.sameValue(indexed.join(","), "0,2,4");
