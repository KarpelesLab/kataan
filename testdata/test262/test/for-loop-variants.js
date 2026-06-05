/*---
description: for loop with multiple init/update, empty sections, comma operator
esid: sec-for-statement
---*/
var sum = 0;
for (var i = 0, j = 10; i < j; i++, j--) sum += i;
assert.sameValue(sum, 0 + 1 + 2 + 3 + 4, "two-variable for");
var count = 0;
for (;;) { count++; if (count >= 5) break; }
assert.sameValue(count, 5, "infinite for with break");
var product = 1;
for (var k = 1; k <= 4; ) { product *= k; k++; }
assert.sameValue(product, 24, "empty update section");
var collected = [];
for (var x = 0; x < 3; x++) for (var y = 0; y < 2; y++) collected.push(x + "" + y);
assert.sameValue(collected.join(","), "00,01,10,11,20,21", "nested for");
