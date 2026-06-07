/*---
description: Generator delegation with yield* (generators, arrays, strings, return values)
esid: sec-generator-function-definitions
features: [generators]
---*/
function* inner() { yield 1; yield 2; yield 3; }
function* outer() { yield 0; yield* inner(); yield 4; }
assert.sameValue([...outer()].join(","), "0,1,2,3,4", "yield* delegation");

function* letters() { yield* ["a", "b", "c"]; }
assert.sameValue([...letters()].join(""), "abc", "yield* an array");

function* nested() { yield* inner(); yield* inner(); }
assert.sameValue([...nested()].length, 6, "double delegation");

function* withString() { yield* "xy"; yield "z"; }
assert.sameValue([...withString()].join(""), "xyz", "yield* a string");

function* combined() {
  yield 1;
  yield* [2, 3];
  yield* (function* () { yield 4; yield 5; })();
}
assert.sameValue([...combined()].join(","), "1,2,3,4,5", "mixed delegation");

// A return halts iteration (its value is not yielded), and yield* evaluates to
// the delegated generator's return value.
function* withReturn() { yield 1; return 99; yield 2; }
assert.sameValue([...withReturn()].join(","), "1", "return stops the generator");
function* d1() { yield 1; return 5; }
function* d2() { var r = yield* d1(); yield "r=" + r; }
assert.sameValue([...d2()].join(","), "1,r=5", "yield* yields the delegate's return");

// A generator object is its own iterator; for-of/destructuring/Array.from consume it.
var g = inner();
assert.sameValue(g[Symbol.iterator]() === g, true, "generator is self-iterable");
function* doubled() { for (var x of inner()) yield x * 10; }
assert.sameValue([...doubled()].join(","), "10,20,30", "for-of over a generator inside a generator");
var [a, b, ...rest] = (function* () { yield 1; yield 2; yield 3; yield 4; })();
assert.sameValue(a + "," + b + "," + rest.join(","), "1,2,3,4", "destructuring with rest");
assert.sameValue(Array.from(inner()).join(","), "1,2,3", "Array.from a generator");
