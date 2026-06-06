/*---
description: A `let` loop binding is captured per-iteration by closures
esid: sec-forbodyevaluation
---*/
var fns = [];
for (let i = 0; i < 3; i = i + 1) {
  fns.push(function () { return i; });
}
assert.sameValue(fns[0](), 0, "first closure sees its own i");
assert.sameValue(fns[1](), 1, "second closure sees its own i");
assert.sameValue(fns[2](), 2, "third closure sees its own i");

// A `var` binding, by contrast, is shared across iterations.
var vfns = [];
for (var j = 0; j < 3; j = j + 1) {
  vfns.push(function () { return j; });
}
assert.sameValue(vfns[0](), 3, "var binding is shared (final value)");
assert.sameValue(vfns[2](), 3, "var binding is shared (final value)");
