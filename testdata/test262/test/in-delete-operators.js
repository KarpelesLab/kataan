/*---
description: in operator, delete, and property existence
esid: sec-relational-operators
---*/
var o = { a: 1, b: undefined };
assert.sameValue("a" in o, true);
assert.sameValue("b" in o, true, "undefined value still has the key");
assert.sameValue("c" in o, false);
assert.sameValue("toString" in o || true, true);
delete o.a;
assert.sameValue("a" in o, false, "delete removes the key");
var arr = [1, 2, 3];
assert.sameValue(0 in arr, true);
assert.sameValue(5 in arr, false);
delete arr[1];
assert.sameValue(arr[1], undefined, "delete clears the element");
assert.sameValue(arr.length, 3, "delete does not change length");
assert.sameValue("length" in arr, true);
