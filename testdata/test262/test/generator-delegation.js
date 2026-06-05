/*---
description: Generator delegation (yield*) over arrays and other generators
esid: sec-generator-function-definitions
---*/
function* inner() { yield 2; yield 3; }
function* outer() { yield 1; yield* inner(); yield 4; }
assert.sameValue([...outer()].join(","), "1,2,3,4", "yield* delegates");
function* fromArray() { yield* [10, 20, 30]; }
assert.sameValue([...fromArray()].join(","), "10,20,30", "yield* over an array");
function* combined() { yield* [1, 2]; yield* [3, 4]; }
assert.sameValue([...combined()].join(","), "1,2,3,4");
