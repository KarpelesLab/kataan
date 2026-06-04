/*---
description: do-while loops and continue in loops
esid: sec-do-while-statement
---*/
var i = 0, sum = 0;
do { sum += i; i++; } while (i < 5);
assert.sameValue(sum, 10);

var evens = 0;
for (var k = 0; k < 10; k++) {
  if (k % 2 !== 0) continue;
  evens++;
}
assert.sameValue(evens, 5);
