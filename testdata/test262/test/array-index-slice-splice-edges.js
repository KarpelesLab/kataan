/*---
description: Array indexOf/includes fromIndex, slice/splice/copyWithin/fill with negative indices
esid: sec-array.prototype.copywithin
---*/
var a = [1, 2, 3, 2, 1];

// indexOf / lastIndexOf / includes with fromIndex (incl. negative).
assert.sameValue(a.indexOf(2, 2), 3, "indexOf fromIndex");
assert.sameValue(a.indexOf(2, -2), 3, "indexOf negative fromIndex");
assert.sameValue(a.lastIndexOf(2, 2), 1, "lastIndexOf fromIndex");
assert.sameValue(a.lastIndexOf(2, -3), 1, "lastIndexOf negative fromIndex");
assert.sameValue(a.includes(2, 2), true, "includes from middle");
assert.sameValue(a.includes(3, 3), false, "includes past the element");
assert.sameValue([NaN].indexOf(NaN), -1, "indexOf NaN is -1");
assert.sameValue([NaN].includes(NaN), true, "includes NaN");

// slice with negatives.
assert.sameValue(a.slice(-2).join(","), "2,1", "slice(-2)");
assert.sameValue(a.slice(1, -1).join(","), "2,3,2", "slice(1,-1)");
assert.sameValue(a.slice(-3, -1).join(","), "3,2", "slice(-3,-1)");
assert.sameValue(a.slice(3, 1).join(","), "", "slice with start>end");

// splice: replace, negative start, insert, truncate.
var b = [1, 2, 3, 4, 5];
assert.sameValue(b.splice(1, 2, "a", "b", "c").join(","), "2,3", "splice returns removed");
assert.sameValue(b.join(","), "1,a,b,c,4,5", "splice mutated");
var c = [1, 2, 3, 4, 5]; c.splice(-2, 1);
assert.sameValue(c.join(","), "1,2,3,5", "splice negative start");

// copyWithin with negative target/start/end (overlapping ranges).
assert.sameValue([1, 2, 3, 4, 5].copyWithin(-2, -3, -1).join(","), "1,2,3,3,4", "copyWithin all negative");
assert.sameValue([1, 2, 3, 4, 5].copyWithin(0, 3).join(","), "4,5,3,4,5", "copyWithin to start");
assert.sameValue([1, 2, 3, 4, 5].copyWithin(0, -2).join(","), "4,5,3,4,5", "copyWithin negative start");

// fill with negative bounds.
assert.sameValue([1, 2, 3, 4, 5].fill(0, -3, -1).join(","), "1,2,0,0,5", "fill negative range");

// at, findLastIndex, reduce on empty.
assert.sameValue(a.at(-1), 1, "at(-1)");
assert.sameValue(a.at(-10), undefined, "at out of range");
assert.sameValue([5, 12, 8, 130, 44].findLastIndex(function (x) { return x > 10; }), 4, "findLastIndex");
var threw = false;
try { [].reduce(function (x, y) { return x + y; }); } catch (e) { threw = e instanceof TypeError; }
assert.sameValue(threw, true, "reduce of [] with no initial throws");
assert.sameValue([].reduce(function (x, y) { return x + y; }, 0), 0, "reduce with initial");
