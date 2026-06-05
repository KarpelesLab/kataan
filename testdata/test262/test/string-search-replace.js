/*---
description: String search and replace methods
esid: sec-string.prototype.replace
---*/
assert.sameValue("hello world".replaceAll("o", "0"), "hell0 w0rld");
assert.sameValue("aaa".replaceAll("a", "b"), "bbb");
assert.sameValue("a.b.c".replaceAll(".", "-"), "a-b-c", "literal dots");
assert.sameValue("hello".replace("l", "L"), "heLlo", "replace first only");
assert.sameValue("test".search(/st/), 2, "search returns index");
assert.sameValue("test".search(/xyz/), -1, "search no match");
assert.sameValue("Hello World".replace(/o/g, "0"), "Hell0 W0rld");
assert.sameValue("2024-06-15".replace(/-/g, "/"), "2024/06/15");
assert.sameValue("camelCase".replace(/([A-Z])/g, "_$1").toLowerCase(), "camel_case");
assert.sameValue("  spaces  ".replaceAll(" ", "."), "..spaces..");
assert.sameValue("abcabc".replaceAll(/a/g, "X"), "XbcXbc", "replaceAll with global regex");
assert.sameValue("one two three".split(" ").join("_"), "one_two_three");
assert.sameValue("Hello".replace(/l/g, function (m) { return m.toUpperCase(); }), "HeLLo");
assert.sameValue("price: 100".replace(/(\d+)/, function (m, d) { return "$" + d; }), "price: $100");
