/*---
description: Labeled (non-loop) statements, named class expressions, class .name
esid: sec-labelled-statements
---*/
// break to a label on a block (previously crashed the bytecode VM).
(function () {
  var r = [];
  blk: { r.push(1); if (true) break blk; r.push(2); }
  r.push(3);
  assert.sameValue(r.join(","), "1,3", "break out of a labeled block");
})();
// Nested labeled blocks.
(function () {
  var hit = "no";
  a: { b: { break a; } hit = "inner"; }
  assert.sameValue(hit, "no", "break a skips past block a");
})();
// continue label still works on a loop.
(function () {
  var r = [];
  outer: for (var i = 0; i < 3; i++) {
    for (var j = 0; j < 3; j++) { if (j === 1) continue outer; r.push(i + "," + j); }
  }
  assert.sameValue(r.join(";"), "0,0;1,0;2,0", "continue to a loop label");
})();
// A named class expression can reference itself.
var C = class Named {
  who() { return Named === C; }
  named() { return Named.name; }
};
assert.sameValue(new C().who(), true, "named class self-reference");
assert.sameValue(new C().named(), "Named", "self-reference has the class name");
// Class .name.
class Declared {}
assert.sameValue(Declared.name, "Declared", "declared class name");
assert.sameValue((class Anon {}).name, "Anon", "class expression name");
