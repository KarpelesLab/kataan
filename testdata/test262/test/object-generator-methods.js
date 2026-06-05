/*---
description: Generator methods in object literals (including computed keys)
esid: sec-object-initializer
---*/
var counter = {
  *count() { yield 1; yield 2; yield 3; }
};
assert.sameValue([...counter.count()].join(","), "1,2,3", "named generator method");
var iterable = {
  *[Symbol.iterator]() { yield "a"; yield "b"; }
};
assert.sameValue([...iterable].join(","), "a,b", "computed [Symbol.iterator] generator method");
var collected = [];
for (var x of iterable) collected.push(x);
assert.sameValue(collected.join(","), "a,b", "for-of over object generator iterator");
var key = "gen";
var dynamic = {
  *[key]() { yield 10; yield 20; }
};
assert.sameValue([...dynamic.gen()].join(","), "10,20", "computed-key generator method");
var pairs = {
  *[Symbol.iterator]() { yield ["a", 1]; yield ["b", 2]; }
};
assert.sameValue(new Map(pairs).get("b"), 2, "Map from an object iterator");
var [first, second] = { *[Symbol.iterator]() { yield 100; yield 200; } };
assert.sameValue(first + "," + second, "100,200", "destructuring an object iterator");
var mixed = {
  value: 42,
  *items() { yield this.value; yield this.value * 2; }
};
assert.sameValue([...mixed.items()].join(","), "42,84", "generator method reads this");
