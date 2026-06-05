/*---
description: Spread operator edge cases
esid: sec-spread
---*/
assert.sameValue([...[]].length, 0, "spread empty array");
assert.sameValue([...[1], ...[2], ...[3]].join(","), "1,2,3");
assert.sameValue([0, ...[1, 2], 3].join(","), "0,1,2,3");
var obj = { ...{ a: 1 }, ...{ b: 2 } };
assert.sameValue(obj.a + obj.b, 3);
assert.sameValue({ ...{ a: 1 }, a: 2 }.a, 2, "later overrides spread");
assert.sameValue({ a: 1, ...{ a: 2 } }.a, 2, "spread overrides earlier");
function f(...args) { return args.length; }
assert.sameValue(f(...[1, 2], ...[3]), 3);
assert.sameValue([..."abc"].join("-"), "a-b-c", "spread string");
assert.sameValue([...new Set([1, 1, 2])].length, 2);
assert.sameValue([...new Map([["a", 1]]).keys()].join(""), "a");
var copy = [...[1, 2, 3]];
copy.push(4);
assert.sameValue([1, 2, 3].length, 3, "spread creates a new array");
var merged = { ...{ x: 1 }, ...{ y: 2 }, ...{ z: 3 } };
assert.sameValue(Object.keys(merged).join(""), "xyz");
