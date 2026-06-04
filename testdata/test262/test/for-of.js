/*---
description: for-of iterates arrays and strings
esid: sec-for-in-and-for-of-statements
---*/
var total = 0;
for (var v of [10, 20, 30]) { total += v; }
assert.sameValue(total, 60);

var chars = "";
for (var ch of "abc") { chars += ch + "."; }
assert.sameValue(chars, "a.b.c.");
