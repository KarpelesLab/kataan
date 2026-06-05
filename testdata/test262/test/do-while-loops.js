/*---
description: do-while loop semantics and edge cases
esid: sec-do-while-statement
---*/
var count = 0;
do { count++; } while (count < 5);
assert.sameValue(count, 5);
var runs = 0;
do { runs++; } while (false);
assert.sameValue(runs, 1, "body runs at least once");
var sum = 0, i = 0;
do { sum += i; i++; } while (i < 4);
assert.sameValue(sum, 0 + 1 + 2 + 3);
var result = [];
var n = 3;
do { result.push(n); n--; } while (n > 0);
assert.sameValue(result.join(","), "3,2,1");
var x = 10;
do { x = x - 2; if (x === 4) break; } while (x > 0);
assert.sameValue(x, 4, "break in do-while");
var collected = [];
var j = 0;
do { j++; if (j % 2 === 0) continue; collected.push(j); } while (j < 6);
assert.sameValue(collected.join(","), "1,3,5", "continue in do-while");
