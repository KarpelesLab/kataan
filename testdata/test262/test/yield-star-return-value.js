/*---
description: yield* evaluates to the delegated iterator's return value
features: [generators]
---*/
// A delegated generator's `return` value becomes the value of `yield*`.
function* inner() { yield 1; yield 2; return 99; }
function* outer() { var r = yield* inner(); yield "r=" + r; }
assert.sameValue([...outer()].join(","), "1,2,r=99", "yield* surfaces the delegate's return");

// Delegating to a plain iterable (no return value) yields `undefined`.
function* overArray() { var r = yield* [7, 8]; yield "r=" + r; }
assert.sameValue([...overArray()].join(","), "7,8,r=undefined", "array delegation returns undefined");

// Nested delegation threads return values through each level.
function* a() { yield 1; return 2; }
function* b() { var x = yield* a(); return x + 10; }
function* c() { var y = yield* b(); yield "y=" + y; }
assert.sameValue([...c()].join(","), "1,y=12", "nested yield* return values thread through");
