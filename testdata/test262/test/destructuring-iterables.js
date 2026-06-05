/*---
description: Destructuring from iterables (strings, sets, generators)
esid: sec-destructuring-assignment
---*/
var [a, b, c] = "xyz";
assert.sameValue(a + b + c, "xyz", "destructure string");
var [first, ...rest] = new Set([1, 2, 3, 4]);
assert.sameValue(first, 1);
assert.sameValue(rest.join(","), "2,3,4");
function* gen() { yield 10; yield 20; yield 30; }
var [g1, g2] = gen();
assert.sameValue(g1 + g2, 30, "destructure generator");
var [[x1, y1], [x2, y2]] = [[1, 2], [3, 4]];
assert.sameValue(x1 + y1 + x2 + y2, 10);
var map = new Map([["a", 1], ["b", 2]]);
var [[k1, v1]] = map;
assert.sameValue(k1, "a");
assert.sameValue(v1, 1);
var [p, q = 100] = [5];
assert.sameValue(p, 5);
assert.sameValue(q, 100, "default in iterable destructure");
var str = "hello";
var [h, e] = str;
assert.sameValue(h + e, "he");
