/*---
description: Chained array mutations and transformations
esid: sec-array.prototype
---*/
var a = [1, 2, 3, 4, 5];
a.splice(1, 2);
assert.sameValue(a.join(","), "1,4,5", "splice removes");
a.splice(1, 0, 10, 11);
assert.sameValue(a.join(","), "1,10,11,4,5", "splice inserts");
var b = [3, 1, 2];
b.sort().reverse();
assert.sameValue(b.join(","), "3,2,1", "sort then reverse");
var c = [1, 2, 3];
c.push(4, 5);
c.unshift(0);
assert.sameValue(c.join(","), "0,1,2,3,4,5");
assert.sameValue(c.pop(), 5);
assert.sameValue(c.shift(), 0);
assert.sameValue(c.join(","), "1,2,3,4");
var d = [1, 2, 3, 4, 5, 6];
var removed = d.splice(2, 2, "a");
assert.sameValue(removed.join(","), "3,4", "splice returns removed");
assert.sameValue(d.join(","), "1,2,a,5,6");
var e = [5, 3, 8, 1];
e.sort(function (x, y) { return x - y; });
e.length = 2;
assert.sameValue(e.join(","), "1,3", "truncate via length");
