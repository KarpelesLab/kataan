/*---
description: Labeled break and continue across nested loops
esid: sec-labelled-statements
---*/
var hits = [];
outer:
for (var i = 0; i < 3; i++) {
  for (var j = 0; j < 3; j++) {
    if (i === 1 && j === 1) continue outer;
    if (i === 2 && j === 0) break outer;
    hits.push(i + "" + j);
  }
}
assert.sameValue(hits.join(","), "00,01,02,10");
