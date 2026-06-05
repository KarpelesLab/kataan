/*---
description: String conversion of various types
esid: sec-tostring
---*/
assert.sameValue(String(123), "123");
assert.sameValue(String(true), "true");
assert.sameValue(String(false), "false");
assert.sameValue(String(null), "null");
assert.sameValue(String(undefined), "undefined");
assert.sameValue(String([1, 2, 3]), "1,2,3");
assert.sameValue(String([]), "");
assert.sameValue(String([null, undefined]), ",", "null/undefined render empty in join");
assert.sameValue(String({}), "[object Object]");
assert.sameValue(String(NaN), "NaN");
assert.sameValue(String(Infinity), "Infinity");
assert.sameValue(String(-0), "0");
assert.sameValue(String(1.5), "1.5");
assert.sameValue(String(0.5), "0.5");
assert.sameValue(`${[1, 2]}`, "1,2", "template coerces array");
assert.sameValue("" + 42, "42");
assert.sameValue("" + null, "null");
assert.sameValue("" + [1, [2, 3]], "1,2,3", "nested array");
