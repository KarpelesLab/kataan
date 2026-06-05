/*---
description: new Number/String/Boolean wrapper objects box a primitive
esid: sec-number-constructor
---*/
// Number wrapper.
var n = new Number(5);
assert.sameValue(typeof n, "object", "new Number is an object");
assert.sameValue(n.valueOf(), 5, "Number valueOf");
assert.sameValue(n + 3, 8, "Number wrapper coerces in arithmetic");
assert.sameValue(n > 3, true, "Number wrapper compares");
assert.sameValue(new Number(255).toString(16), "ff", "Number method delegates (radix)");
assert.sameValue(n instanceof Number, true, "instanceof Number");
// String wrapper.
var s = new String("hello");
assert.sameValue(typeof s, "object", "new String is an object");
assert.sameValue(s.valueOf(), "hello", "String valueOf");
assert.sameValue(s.length, 5, "String wrapper length");
assert.sameValue(s[1], "e", "String wrapper indexing");
assert.sameValue(new String("abc")[0], "a", "index 0");
assert.sameValue(new String("HELLO").toLowerCase(), "hello", "String method delegates");
assert.sameValue(new String("a") + "b", "ab", "String wrapper concatenation");
assert.sameValue(s instanceof String, true, "instanceof String");
// Boolean wrapper.
var b = new Boolean(false);
assert.sameValue(typeof b, "object", "new Boolean is an object");
assert.sameValue(b.valueOf(), false, "Boolean valueOf");
assert.sameValue(b ? "truthy" : "falsy", "truthy", "a Boolean object is always truthy");
assert.sameValue(b instanceof Boolean, true, "instanceof Boolean");
// Defaults.
assert.sameValue(new Number().valueOf(), 0, "new Number() defaults to 0");
assert.sameValue(new String().valueOf(), "", "new String() defaults to empty");
assert.sameValue(new Boolean().valueOf(), false, "new Boolean() defaults to false");
