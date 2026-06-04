/*---
description: Labeled break exits an outer loop
esid: sec-labelled-statements
---*/
var pairs = 0;
outer:
for (var i = 0; i < 3; i++) {
  for (var j = 0; j < 3; j++) {
    if (i + j === 3) break outer;
    pairs++;
  }
}
assert.sameValue(pairs, 5);
