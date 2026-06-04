/*---
description: Labeled continue skips to the outer loop iteration
esid: sec-continue-statement
---*/
var visited = "";
outer:
for (var i = 0; i < 3; i++) {
  for (var j = 0; j < 3; j++) {
    if (j === 1) continue outer;
    visited += i + "" + j + " ";
  }
}
assert.sameValue(visited.trim(), "00 10 20");
