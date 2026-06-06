/*---
description: for-in enumerates own keys in insertion order; do-while runs once first
esid: sec-for-in-and-for-of-statements
---*/
var o = { a: 1, b: 2, c: 3 };
var keys = "";
for (var k in o) { keys += k; }
assert.sameValue(keys, "abc", "for-in enumerates own enumerable keys in order");

// Inherited enumerable properties are also visited (after own ones here).
var proto = { inherited: 9 };
var child = Object.create(proto);
child.own = 1;
var seen = [];
for (var p in child) { seen.push(p); }
assert.sameValue(seen.indexOf("own") >= 0, true, "for-in sees own key");
assert.sameValue(seen.indexOf("inherited") >= 0, true, "for-in sees inherited key");

// do-while always executes the body at least once.
var ran = 0;
do { ran++; } while (false);
assert.sameValue(ran, 1, "do-while body runs once even when the test is false");

var d = 0, acc = 0;
do { acc += d; d++; } while (d < 5);
assert.sameValue(acc, 10, "0+1+2+3+4");
