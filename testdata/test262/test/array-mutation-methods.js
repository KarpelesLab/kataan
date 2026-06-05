/*---
description: push, pop, shift, unshift, reverse, splice mutate in place
esid: sec-array.prototype.push
---*/
var a = [1, 2, 3];
assert.sameValue(a.push(4), 4, "push returns new length");
assert.sameValue(a.join(","), "1,2,3,4");
assert.sameValue(a.pop(), 4, "pop returns removed");
assert.sameValue(a.join(","), "1,2,3");
assert.sameValue(a.shift(), 1, "shift returns first");
assert.sameValue(a.join(","), "2,3");
assert.sameValue(a.unshift(0), 3, "unshift returns new length");
assert.sameValue(a.join(","), "0,2,3");
var r = [1, 2, 3];
var rr = r.reverse();
assert.sameValue(r.join(","), "3,2,1", "reverse mutates in place");
assert.sameValue(r === rr, true, "reverse returns same array");
var s = [1, 2, 3, 4, 5];
var removed = s.splice(1, 2, "a", "b", "c");
assert.sameValue(removed.join(","), "2,3", "splice returns removed");
assert.sameValue(s.join(","), "1,a,b,c,4,5");
