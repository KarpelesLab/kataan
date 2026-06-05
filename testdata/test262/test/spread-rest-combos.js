/*---
description: Spread and rest combinations
esid: sec-spread
---*/
function f(first, ...rest) { return first + ":" + rest.length; }
assert.sameValue(f(1, 2, 3, 4), "1:3");
assert.sameValue(f(1), "1:0");
function g(...all) { return all.reduce(function (a, b) { return a + b; }, 0); }
assert.sameValue(g(...[1, 2], ...[3, 4]), 10);
var combined = [...[1, 2], ...[3, 4], ...[5]];
assert.sameValue(combined.join(","), "1,2,3,4,5");
var obj = { ...{ a: 1 }, ...{ b: 2 }, c: 3 };
assert.sameValue(obj.a + obj.b + obj.c, 6);
var { x, ...others } = { x: 1, y: 2, z: 3 };
assert.sameValue(x, 1);
assert.sameValue(Object.keys(others).join(","), "y,z");
var [head, ...tail] = [1, 2, 3, 4];
assert.sameValue(head, 1);
assert.sameValue(tail.join(","), "2,3,4");
function h(a, b, ...c) { return a + b + c.join(""); }
assert.sameValue(h(1, 2, 3, 4, 5), 1 + 2 + "345");
assert.sameValue(Math.max(...[3, 1, 4, 1, 5, 9]), 9);
var merged = { ...{ a: 1, b: 2 }, b: 20, ...{ c: 3 } };
assert.sameValue(merged.b, 20, "later overrides");
assert.sameValue([...new Set([1, 2, 2, 3, 3, 3])].join(","), "1,2,3");
var nested = [...[1, [2, 3]], ...[[4]]];
assert.sameValue(nested.length, 3);
