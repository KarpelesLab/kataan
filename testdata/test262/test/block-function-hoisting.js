/*---
description: Annex B block-level function declarations hoist to the function scope
esid: sec-block-level-function-declarations-web-legacy-compatibility-semantics
flags: [noStrict]
---*/
// A function declared in a block is visible after the block (sloppy-mode Annex B).
(function () {
  { function g() { return 1; } }
  assert.sameValue(typeof g, "function", "block function is var-hoisted");
  assert.sameValue(g(), 1, "block function is callable after the block");
})();
// Inside an `if`.
(function () {
  if (true) { function h() { return 5; } }
  assert.sameValue(typeof h, "function", "function in an if-block");
  assert.sameValue(h(), 5);
})();
// Nested blocks.
(function () {
  { { function deep() { return 9; } } }
  assert.sameValue(typeof deep, "function", "deeply nested block function");
})();
// A later block declaration wins (function-scoped).
(function () {
  function f() { return "outer"; }
  { function f() { return "inner"; } }
  assert.sameValue(f(), "inner", "block declaration overrides the outer one");
})();
// Inside a loop body.
(function () {
  for (var i = 0; i < 1; i++) { function looped() { return "L"; } }
  assert.sameValue(typeof looped, "function", "function in a loop body");
})();
// Regular top-level function hoisting still works.
(function () {
  assert.sameValue(early(), "hoisted", "top-level hoisting");
  function early() { return "hoisted"; }
})();
