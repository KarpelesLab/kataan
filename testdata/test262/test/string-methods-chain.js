/*---
description: Chained string method calls
esid: sec-string.prototype
---*/
assert.sameValue("  Hello World  ".trim().toLowerCase(), "hello world");
assert.sameValue("hello".toUpperCase().split("").reverse().join(""), "OLLEH");
assert.sameValue("a,b,c".split(",").map(function (s) { return s.toUpperCase(); }).join("-"), "A-B-C");
assert.sameValue("hello world".replace("world", "there").toUpperCase(), "HELLO THERE");
assert.sameValue("  padded  ".trim().padStart(10, "*"), "****padded");
assert.sameValue("abcdef".slice(1, 4).toUpperCase(), "BCD");
assert.sameValue("Hello".charAt(0).toLowerCase() + "Hello".slice(1), "hello");
assert.sameValue("a-b-c-d".split("-").filter(function (s) { return s !== "b"; }).join(""), "acd");
assert.sameValue("The quick brown fox".split(" ").length, 4);
assert.sameValue("hello".repeat(3).length, 15);
assert.sameValue("HELLO".toLowerCase().split("").map(function (c) { return c.charCodeAt(0); }).reduce(function (a, b) { return a + b; }, 0) > 0, true);
assert.sameValue("x".padStart(5).trim(), "x");
assert.sameValue("Hello World Foo".split(" ").map(function (w) { return w[0]; }).join(""), "HWF");
