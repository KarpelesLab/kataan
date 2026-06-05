/*---
description: Array.from with iterables, array-likes, and mapping functions
esid: sec-array.from
---*/
assert.sameValue(Array.from([1, 2, 3], function (x) { return x * 10; }).join(","), "10,20,30");
assert.sameValue(Array.from("abc", function (c) { return c.toUpperCase(); }).join(""), "ABC");
assert.sameValue(Array.from(new Set([1, 1, 2, 3])).join(","), "1,2,3");
assert.sameValue(Array.from({ length: 3 }, function (_, i) { return i * i; }).join(","), "0,1,4");
assert.sameValue(Array.from([1, 2, 3].keys()).join(","), "0,1,2", "array keys");
assert.sameValue(Array.from(new Map([["a", 1]]).entries()).length, 1);
assert.sameValue(Array.from({ length: 0 }).length, 0);
