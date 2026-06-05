/*---
description: Labeled statements with break and continue
esid: sec-labelled-statements
---*/
var result = [];
outer: for (var i = 0; i < 3; i++) {
  for (var j = 0; j < 3; j++) {
    if (j === 2) continue outer;
    if (i === 2) break outer;
    result.push(i + "" + j);
  }
}
assert.sameValue(result.join(","), "00,01,10,11", "labeled continue/break");
var found = null;
search: for (var a = 0; a < 5; a++) {
  for (var b = 0; b < 5; b++) {
    if (a * b === 6) { found = a + "x" + b; break search; }
  }
}
assert.sameValue(found, "2x3", "break out of nested");
var sum = 0;
loop: for (var k = 0; k < 10; k++) {
  if (k % 2 === 0) continue loop;
  if (k > 7) break loop;
  sum += k;
}
assert.sameValue(sum, 1 + 3 + 5 + 7);
var count = 0;
block: { count = 1; break block; count = 99; }
assert.sameValue(count, 1, "break from labeled block");
var matrix = [[1, 2], [3, 4]];
var total = 0;
rows: for (var r = 0; r < matrix.length; r++) {
  for (var c = 0; c < matrix[r].length; c++) {
    total += matrix[r][c];
    if (total > 5) break rows;
  }
}
assert.sameValue(total, 6, "break with accumulator");
