/*---
description: ES2023+ array methods — at, findLast, flat/flatMap, change-by-copy
features: [Array.prototype.findLast, array-find-from-last, change-array-by-copy]
---*/
// Relative indexing.
assert.sameValue([1, 2, 3].at(-1), 3, "at(-1)");
assert.sameValue([1, 2, 3].at(-5), undefined, "at out of range");
assert.sameValue("abc".at(-1), "c", "String.prototype.at");

// Find from the end.
assert.sameValue([1, 2, 3, 4].findLast(x => x % 2), 3, "findLast");
assert.sameValue([1, 2, 3, 4].findLastIndex(x => x % 2), 2, "findLastIndex");
assert.sameValue([].findLast(() => true), undefined, "findLast on empty");

// Flattening.
assert.sameValue([1, [2, [3, [4]]]].flat(2).join(","), "1,2,3,4", "flat(depth)");
assert.sameValue([1, 2].flatMap(x => [x, x * 2]).join(","), "1,2,2,4", "flatMap");

// In-place fill / copyWithin.
assert.sameValue([1, 2, 3, 4].fill(9, 1, 3).join(","), "1,9,9,4", "fill range");
assert.sameValue([1, 2, 3, 4, 5].copyWithin(0, 3).join(","), "4,5,3,4,5", "copyWithin");

// includes uses SameValueZero (NaN matches).
assert.sameValue([1, NaN, 3].includes(NaN), true, "includes NaN");

// Change-by-copy: the original is never mutated.
var a = [3, 1, 2];
assert.sameValue(a.toSorted().join(","), "1,2,3", "toSorted result");
assert.sameValue(a.join(","), "3,1,2", "toSorted leaves original");
assert.sameValue([1, 2, 3].toReversed().join(","), "3,2,1", "toReversed");
assert.sameValue([1, 2, 3].with(1, 9).join(","), "1,9,3", "with");
assert.sameValue([1, 2, 3, 4].toSpliced(1, 2, 9).join(","), "1,9,4", "toSpliced");

// Misc.
assert.sameValue([1, 2, 3].reduceRight((acc, x) => acc + "-" + x), "3-2-1", "reduceRight");
assert.sameValue(Array.of(7).length, 1, "Array.of(7) is a one-element array");
