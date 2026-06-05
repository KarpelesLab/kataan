/*---
description: Spread and rest in assignment and object construction
esid: sec-object-initializer
---*/
var base = { a: 1, b: 2, c: 3 };
var { a, ...rest } = base;
assert.sameValue(a, 1);
assert.sameValue(rest.b + rest.c, 5);
assert.sameValue(Object.keys(rest).join(","), "b,c", "rest collects remaining");
var arr = [1, 2, 3, 4, 5];
var [first, second, ...others] = arr;
assert.sameValue(first + second, 3);
assert.sameValue(others.join(","), "3,4,5");
var merged = { ...{ x: 1 }, ...{ y: 2 }, z: 3 };
assert.sameValue(merged.x + merged.y + merged.z, 6);
var clone = { ...base };
clone.a = 99;
assert.sameValue(base.a, 1, "spread clones");
function collect(...args) { return args.length; }
assert.sameValue(collect(...arr), 5, "spread into rest");
