/*---
description: do-while loops execute the body at least once
esid: sec-do-while-statement
---*/
var n = 0;
do { n++; } while (n < 5);
assert.sameValue(n, 5);
var runs = 0;
do { runs++; } while (false);
assert.sameValue(runs, 1, "body runs at least once");
var sum = 0, i = 0;
do { sum += i; i++; } while (i < 4);
assert.sameValue(sum, 0 + 1 + 2 + 3);
var collected = [];
var k = 10;
do { collected.push(k); k -= 2; } while (k > 0);
assert.sameValue(collected.join(","), "10,8,6,4,2");
var breakCount = 0;
do { breakCount++; if (breakCount === 3) break; } while (true);
assert.sameValue(breakCount, 3, "break exits do-while");
var continued = [];
var j = 0;
do { j++; if (j % 2 === 0) continue; continued.push(j); } while (j < 6);
assert.sameValue(continued.join(","), "1,3,5", "continue in do-while");
