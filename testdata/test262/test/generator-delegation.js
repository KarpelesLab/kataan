/*---
description: Generator delegation with yield*
esid: sec-generator-function-definitions
---*/
function* inner() { yield 1; yield 2; yield 3; }
function* outer() { yield 0; yield* inner(); yield 4; }
assert.sameValue([...outer()].join(","), "0,1,2,3,4", "yield* delegation");
function* letters() { yield* ["a", "b", "c"]; }
assert.sameValue([...letters()].join(""), "abc", "yield* an array");
function* nested() {
  yield* inner();
  yield* inner();
}
assert.sameValue([...nested()].length, 6, "double delegation");
function* withString() { yield* "xy"; yield "z"; }
assert.sameValue([...withString()].join(""), "xyz", "yield* a string");
function* combined() {
  yield 1;
  yield* [2, 3];
  yield* (function* () { yield 4; yield 5; })();
}
assert.sameValue([...combined()].join(","), "1,2,3,4,5");
