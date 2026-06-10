/*---
description: Array.prototype.sort moves undefined to the end without calling the comparator
esid: sec-array.prototype.sort
---*/
// With a comparator, undefined is not passed to it — it sorts to the end.
assert.sameValue([3, undefined, 1].sort(function (a, b) { return a - b; }).join(","), "1,3,", "comparator + undefined");
assert.sameValue([3, undefined, 1].sort(function (a, b) { return b - a; }).join(","), "3,1,", "desc comparator + undefined");

// Default (no comparator): undefined also sorts last, even past strings.
assert.sameValue([3, undefined, 1, undefined, 2].sort().join(","), "1,2,3,,", "default + undefined");
assert.sameValue(["zzz", undefined, "aaa"].sort().join(","), "aaa,zzz,", "default with strings");

// All-undefined stays length-stable.
assert.sameValue([undefined, undefined].sort(function (a, b) { return a - b; }).length, 2, "all undefined");

// The comparator only ever sees defined values (so it never produces NaN here).
var sawUndefined = false;
[2, undefined, 1, undefined, 3].sort(function (a, b) {
  if (a === undefined || b === undefined) sawUndefined = true;
  return a - b;
});
assert.sameValue(sawUndefined, false, "comparator never receives undefined");

// Ordinary sorts are unaffected.
assert.sameValue([3, 1, 2].sort(function (a, b) { return a - b; }).join(","), "1,2,3", "numeric");
assert.sameValue([10, 1, 2].sort().join(","), "1,10,2", "default lexicographic");
