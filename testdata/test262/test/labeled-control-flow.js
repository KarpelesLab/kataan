/*---
description: Labeled statements with break and continue in various loops
esid: sec-labelled-statements
---*/
var sum = 0;
outer: for (var i = 0; i < 5; i++) {
  inner: for (var j = 0; j < 5; j++) {
    if (j === 3) continue outer;
    if (i === 3) break outer;
    sum += 1;
  }
}
assert.sameValue(sum, 9, "labeled break/continue");
var count = 0;
loop: while (count < 100) {
  count++;
  if (count >= 10) break loop;
}
assert.sameValue(count, 10);
var collected = [];
rows: for (var r = 0; r < 3; r++) {
  for (var c = 0; c < 3; c++) {
    if (c > r) continue rows;
    collected.push(r + "," + c);
  }
}
assert.sameValue(collected.join(";"), "0,0;1,0;1,1;2,0;2,1;2,2");
var doCount = 0;
search: do {
  for (var k = 0; k < 5; k++) { if (k === 2) continue search; doCount++; }
  break;
} while (doCount < 100);
assert.sameValue(doCount > 0, true);
