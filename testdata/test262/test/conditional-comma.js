/*---
description: Conditional (ternary) and comma operators
esid: sec-conditional-operator
---*/
assert.sameValue(true ? "yes" : "no", "yes");
assert.sameValue(false ? "yes" : "no", "no");
assert.sameValue(5 > 3 ? 5 > 4 ? "a" : "b" : "c", "a", "nested ternary");
var x = 5;
assert.sameValue(x > 0 ? "positive" : x < 0 ? "negative" : "zero", "positive");
assert.sameValue((1, 2, 3), 3, "comma yields last");
var result = (x++, x++, x);
assert.sameValue(result, 7);
assert.sameValue(x, 7);
var chained = 1 ? 2 ? 3 : 4 : 5;
assert.sameValue(chained, 3);
var arr = [1, 2, 3];
assert.sameValue(arr.length > 0 ? arr[0] : -1, 1);
var fn = function (a, b) { return a, b; };
assert.sameValue(fn(1, 2), 2, "comma in return");
var count = 0;
var val = (count = 1, count = 2, count + 10);
assert.sameValue(val, 12);
