/*---
description: All compound assignment operators
esid: sec-assignment-operators
---*/
var x = 10;
x += 5; assert.sameValue(x, 15);
x -= 3; assert.sameValue(x, 12);
x *= 2; assert.sameValue(x, 24);
x /= 4; assert.sameValue(x, 6);
x %= 4; assert.sameValue(x, 2);
x **= 3; assert.sameValue(x, 8);
x <<= 2; assert.sameValue(x, 32);
x >>= 1; assert.sameValue(x, 16);
x &= 12; assert.sameValue(x, 0); // 16 & 12 = 0
x |= 3; assert.sameValue(x, 3);
x ^= 5; assert.sameValue(x, 6);
var s = "a";
s += "b"; s += "c";
assert.sameValue(s, "abc");
var obj = { n: 5 };
obj.n += 10;
assert.sameValue(obj.n, 15, "compound on member");
var arr = [1, 2, 3];
arr[1] *= 10;
assert.sameValue(arr[1], 20, "compound on element");
