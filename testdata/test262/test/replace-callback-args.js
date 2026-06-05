/*---
description: String replace callback receives match, captures, offset, string
esid: sec-string.prototype.replace
---*/
var calls = [];
"abc".replace(/(b)/, function (match, p1, offset, str) {
  calls.push(match + "," + p1 + "," + offset + "," + str);
  return "X";
});
assert.sameValue(calls[0], "b,b,1,abc", "callback args");
assert.sameValue("a1b2c3".replace(/(\d)/g, function (m, d) { return "[" + d + "]"; }), "a[1]b[2]c[3]");
assert.sameValue("2024-06".replace(/(\d+)-(\d+)/, function (m, y, mo) { return mo + "/" + y; }), "06/2024");
assert.sameValue("hello".replace(/l/g, function () { return "L"; }), "heLLo");
