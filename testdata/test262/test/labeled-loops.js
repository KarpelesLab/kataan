/*---
description: Labeled break and continue in nested loops
esid: sec-labelled-statements
---*/
var pairs = [];
outer: for (var i = 0; i < 3; i++) {
  for (var j = 0; j < 3; j++) {
    if (i === j) continue outer;
    pairs.push(i + "" + j);
  }
}
assert.sameValue(pairs.join(","), "10,20,21", "continue to outer label");
var found = null;
search: for (var a = 0; a < 5; a++) {
  for (var b = 0; b < 5; b++) {
    if (a * b === 6) { found = a + "x" + b; break search; }
  }
}
assert.sameValue(found, "2x3", "break to outer label");
var count = 0;
loop: for (var k = 0; k < 10; k++) {
  if (k % 2 === 0) continue loop;
  count++;
}
assert.sameValue(count, 5, "labeled continue same loop");
var block = 0;
done: { block = 1; if (block === 1) break done; block = 2; }
assert.sameValue(block, 1, "break from labeled block");
