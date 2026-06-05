/*---
description: Pre and post increment/decrement semantics
esid: sec-postfix-increment-operator
---*/
var a = 5;
assert.sameValue(a++, 5, "post-increment returns old value");
assert.sameValue(a, 6);
assert.sameValue(++a, 7, "pre-increment returns new value");
var b = 10;
assert.sameValue(b--, 10);
assert.sameValue(--b, 8);
var arr = [0, 0, 0];
var i = 0;
arr[i++] = 1;
arr[i++] = 2;
assert.sameValue(arr.join(","), "1,2,0", "post-increment as index");
var obj = { count: 0 };
obj.count++;
obj.count++;
assert.sameValue(obj.count, 2);
var c = 3;
var result = c++ + ++c;
assert.sameValue(result, 3 + 5, "mixed inc in expression");
assert.sameValue(c, 5);
