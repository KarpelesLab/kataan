/*---
description: Generators implement the iterator protocol (self-iterator, spread, return)
esid: sec-generator-objects
---*/
function* g() { yield 1; yield 2; yield 3; }
var it = g();
assert.sameValue(it[Symbol.iterator]() === it, true, "a generator is its own iterator");
assert.sameValue(it[Symbol.iterator]().next().value, 1, "iterating drains the same generator");
assert.sameValue(it.next().value, 2, "continues from where it left off");
assert.sameValue([...g()].join(","), "1,2,3", "spread");
function* delegated() { yield* [1, 2]; yield* "ab"; }
assert.sameValue([...delegated()].join(","), "1,2,a,b", "yield* over array and string");
function* nested() { yield 0; yield* g(); yield 4; }
assert.sameValue([...nested()].join(","), "0,1,2,3,4", "yield* over a generator");
var r = g();
r.next();
var ret = r.return(99);
assert.sameValue(ret.value, 99, "return value");
assert.sameValue(ret.done, true, "done after return");
assert.sameValue(r.next().done, true, "exhausted after return");
assert.sameValue(Array.from(g(), function (v) { return v * 10; }).join(","), "10,20,30", "Array.from with map");
var m = new Map((function* () { yield ["a", 1]; yield ["b", 2]; })());
assert.sameValue(m.get("b"), 2, "Map from a generator");
var [first, second] = g();
assert.sameValue(first + "," + second, "1,2", "destructuring a generator");
