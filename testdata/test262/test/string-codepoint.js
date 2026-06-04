/*---
description: String iteration, codePointAt, and template edge cases
esid: sec-properties-of-the-string-prototype-object
---*/
assert.sameValue("abc".codePointAt(0), 97);
assert.sameValue([..."hello"].length, 5);
assert.sameValue("a-b-c".split("-").reverse().join("/"), "c/b/a");
var n = 5;
assert.sameValue(`val=${n > 3 ? "big" : "small"}`, "val=big");
assert.sameValue("Hello".charAt(0) + "Hello".slice(1).toLowerCase(), "Hello");
