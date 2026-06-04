/*---
description: Array entries/keys/values iterators and Array.from of them
esid: sec-array.prototype.entries
---*/
assert.sameValue([...["a", "b"].keys()].join(","), "0,1");
assert.sameValue([...["a", "b"].values()].join(","), "a,b");
var pairs = [...["x", "y"].entries()];
assert.sameValue(pairs[0][0], 0);
assert.sameValue(pairs[0][1], "x");
assert.sameValue(pairs[1][0], 1);
