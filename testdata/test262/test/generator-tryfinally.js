/*---
description: Generators with try/finally and cleanup
esid: sec-generator-function-definitions
---*/
function* withCleanup() {
  try { yield 1; yield 2; yield 3; }
  finally { /* cleanup */ }
}
assert.sameValue([...withCleanup()].join(","), "1,2,3");
function* counter() {
  var i = 0;
  try { while (i < 5) yield i++; }
  finally { i = -1; }
}
assert.sameValue([...counter()].join(","), "0,1,2,3,4");
function* nested() {
  yield* (function* () { yield "a"; yield "b"; })();
  yield "c";
}
assert.sameValue([...nested()].join(""), "abc");
function* range(start, end) {
  for (var i = start; i < end; i++) yield i;
}
var doubled = [...range(1, 4)].map(function (x) { return x * 2; });
assert.sameValue(doubled.join(","), "2,4,6");
function* fibGen(n) {
  var a = 0, b = 1;
  for (var i = 0; i < n; i++) { yield a; var t = a + b; a = b; b = t; }
}
assert.sameValue([...fibGen(8)].join(","), "0,1,1,2,3,5,8,13");
