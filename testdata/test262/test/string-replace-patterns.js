/*---
description: String replace with special replacement patterns and replaceAll
esid: sec-string.prototype.replace
---*/
assert.sameValue("hello".replace("l", "L"), "heLlo", "first only");
assert.sameValue("hello".replaceAll("l", "L"), "heLLo");
assert.sameValue("a-b-c".replace(/-/g, "_"), "a_b_c");
assert.sameValue("2024".replace(/(\d)(\d)/, "$2$1"), "0224", "swap captures");
assert.sameValue("test".replace(/t/g, "[$&]"), "[t]es[t]", "$& is the match");
assert.sameValue("price: 5".replace(/\d+/, function (m) { return m * 2; }), "price: 10", "function replacement");
assert.sameValue("aaa".split("").length, 3);
