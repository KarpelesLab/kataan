/*---
description: String.prototype.replace with a function replacement
esid: sec-string.prototype.replace
---*/
assert.sameValue("a1b2c3".replace(/[0-9]/g, function (m) { return "[" + m + "]"; }), "a[1]b[2]c[3]");
assert.sameValue(
  "2024-01-15".replace(/(\d+)-(\d+)-(\d+)/, function (_, y, mo, d) { return d + "/" + mo + "/" + y; }),
  "15/01/2024"
);
assert.sameValue("hello".replace(/l/, function (m) { return m.toUpperCase(); }), "heLlo");
assert.sameValue("abc".replace(/x/, "y"), "abc");
