/*---
description: Generators with parameters, nested loops, and yield*
esid: sec-generator-function-definitions
---*/
function* range(start, end, step) {
  for (var i = start; i < end; i += step) yield i;
}
assert.sameValue([...range(0, 10, 2)].join(","), "0,2,4,6,8");
function* zip(a, b) {
  var n = Math.min(a.length, b.length);
  for (var i = 0; i < n; i++) yield [a[i], b[i]];
}
var zipped = [...zip([1, 2, 3], ["a", "b", "c"])];
assert.sameValue(zipped.length, 3);
assert.sameValue(zipped[0].join(""), "1a");
function* flatten(arr) {
  for (var x of arr) { if (Array.isArray(x)) { yield* flatten(x); } else { yield x; } }
}
assert.sameValue([...flatten([1, [2, [3, 4]], 5])].join(","), "1,2,3,4,5", "recursive yield*");
