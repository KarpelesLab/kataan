/*---
description: Array concat, flat behaviour, and spreading
esid: sec-array.prototype.concat
---*/
assert.sameValue([1, 2].concat([3, 4]).join(","), "1,2,3,4");
assert.sameValue([1].concat(2, [3, 4], 5).join(","), "1,2,3,4,5", "concat flattens one level");
assert.sameValue([1, 2].concat([[3, 4]]).length, 3, "nested array not flattened");
var a = [1, 2, 3];
var b = a.concat();
b.push(4);
assert.sameValue(a.length, 3, "concat with no args clones");
assert.sameValue([].concat([1], [], [2]).join(","), "1,2");
assert.sameValue(["a"].concat("b", "c").join(""), "abc");
