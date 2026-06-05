/*---
description: ES2023 immutable array methods toSorted/toReversed/with/toSpliced
esid: sec-array.prototype.tosorted
---*/
var a = [3, 1, 2];
var sorted = a.toSorted();
assert.sameValue(sorted.join(","), "1,2,3", "toSorted result");
assert.sameValue(a.join(","), "3,1,2", "toSorted does not mutate");
var sortedDesc = [1, 2, 3].toSorted(function (x, y) { return y - x; });
assert.sameValue(sortedDesc.join(","), "3,2,1", "toSorted with comparator");
var reversed = [1, 2, 3].toReversed();
assert.sameValue(reversed.join(","), "3,2,1");
var b = [1, 2, 3];
assert.sameValue(b.toReversed().join(",") + "|" + b.join(","), "3,2,1|1,2,3", "toReversed immutable");
var withVal = [1, 2, 3].with(1, 99);
assert.sameValue(withVal.join(","), "1,99,3");
assert.sameValue([1, 2, 3].with(-1, 9).join(","), "1,2,9", "with negative index");
var spliced = [1, 2, 3, 4, 5].toSpliced(1, 2, "a", "b");
assert.sameValue(spliced.join(","), "1,a,b,4,5");
assert.sameValue([1, 2, 3, 4].toSpliced(2).join(","), "1,2", "toSpliced delete to end");
var c = [1, 2, 3];
assert.sameValue(c.toSpliced(0, 0, 0).join(",") + "|" + c.join(","), "0,1,2,3|1,2,3", "toSpliced immutable");
var thrown = false;
try { [1, 2, 3].with(10, 0); } catch (e) { thrown = e instanceof RangeError; }
assert.sameValue(thrown, true, "with out-of-range throws RangeError");
