/*---
description: void, typeof, and delete unary operators
esid: sec-unary-operators
---*/
assert.sameValue(void 0, undefined);
assert.sameValue(void "hello", undefined);
assert.sameValue(void (2 + 3), undefined);
assert.sameValue(typeof undefined, "undefined");
assert.sameValue(typeof 42, "number");
assert.sameValue(typeof "s", "string");
assert.sameValue(typeof true, "boolean");
assert.sameValue(typeof {}, "object");
assert.sameValue(typeof [], "object");
assert.sameValue(typeof function () {}, "function");
assert.sameValue(typeof Symbol(), "symbol");
assert.sameValue(typeof 10n, "bigint");
var obj = { a: 1, b: 2 };
assert.sameValue(delete obj.a, true, "delete returns true");
assert.sameValue("a" in obj, false);
assert.sameValue("b" in obj, true);
assert.sameValue(delete obj.nonexistent, true, "delete missing returns true");
var arr = [1, 2, 3];
delete arr[1];
assert.sameValue(arr.length, 3, "delete does not change length");
assert.sameValue(typeof typeof 5, "string", "typeof always returns string");
assert.sameValue(!typeof undefined, false, "typeof result is truthy string");
