/*---
description: Map/Set use SameValueZero (NaN keys); ES2025 Set composition methods
esid: sec-set-objects
---*/
// SameValueZero: NaN is a valid, matchable key.
var m = new Map();
m.set(NaN, "yes");
assert.sameValue(m.get(NaN), "yes", "NaN as a Map key");
assert.sameValue(m.has(NaN), true, "Map.has(NaN)");
var s = new Set([NaN, NaN, 1]);
assert.sameValue(s.size, 2, "Set dedupes NaN");
assert.sameValue(s.has(NaN), true, "Set.has(NaN)");
// -0 and +0 are the same key.
var z = new Map();
z.set(-0, "neg");
assert.sameValue(z.get(0), "neg", "+0 and -0 are the same key");
// ES2025 Set methods.
var a = new Set([1, 2, 3]);
var b = new Set([3, 4, 5]);
assert.sameValue([...a.union(b)].join(","), "1,2,3,4,5", "union");
assert.sameValue([...a.intersection(b)].join(","), "3", "intersection");
assert.sameValue([...a.difference(b)].join(","), "1,2", "difference");
assert.sameValue([...a.symmetricDifference(b)].join(","), "1,2,4,5", "symmetricDifference");
assert.sameValue(new Set([1, 2]).isSubsetOf(a), true, "isSubsetOf true");
assert.sameValue(new Set([1, 9]).isSubsetOf(a), false, "isSubsetOf false");
assert.sameValue(a.isSupersetOf(new Set([1, 2])), true, "isSupersetOf");
assert.sameValue(a.isDisjointFrom(new Set([8, 9])), true, "isDisjointFrom true");
assert.sameValue(a.isDisjointFrom(b), false, "isDisjointFrom false");
// The argument may be any iterable.
assert.sameValue([...a.intersection([2, 3, 99])].join(","), "2,3", "intersection with an array argument");
