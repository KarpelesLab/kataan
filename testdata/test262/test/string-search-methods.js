/*---
description: String searching and comparison methods
esid: sec-string.prototype
---*/
assert.sameValue("hello world".indexOf("o"), 4);
assert.sameValue("hello world".indexOf("o", 5), 7, "fromIndex");
assert.sameValue("hello world".lastIndexOf("o"), 7);
assert.sameValue("hello".indexOf("z"), -1);
assert.sameValue("abcabc".lastIndexOf("bc"), 4);
assert.sameValue("Hello".localeCompare("Hello"), 0, "equal strings");
assert.sameValue("apple".localeCompare("banana") < 0, true, "a before b");
assert.sameValue("banana".localeCompare("apple") > 0, true);
assert.sameValue("café".normalize().length >= 4, true);
assert.sameValue("  hi  ".trim().length, 2);
assert.sameValue("aaa".replaceAll("a", "b"), "bbb");
assert.sameValue("test".startsWith("te"), true);
assert.sameValue("test".endsWith("st"), true);
