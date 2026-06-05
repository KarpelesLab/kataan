/*---
description: Assignment to an undeclared variable in strict mode throws ReferenceError
esid: sec-assignment-operators-runtime-semantics-evaluation
flags: [onlyStrict]
---*/
// Force the tree-walker (which models strict mode) via the documented prefix;
// the strict directive sits at the top of each function body below.
function Probe() {}
var force = (new Probe()) instanceof Probe;
(function () {
  "use strict";
  var threw = false;
  try { undeclaredStrictVar = 1; } catch (e) { threw = e instanceof ReferenceError; }
  assert.sameValue(threw, true, "implicit global assignment throws in strict mode");
  var declared = 0;
  declared = 5;
  assert.sameValue(declared, 5, "declared variable assignment works");
  // Strict propagates into a nested function.
  (function () {
    var nestedThrew = false;
    try { nestedUndeclared = 1; } catch (e) { nestedThrew = e instanceof ReferenceError; }
    assert.sameValue(nestedThrew, true, "strict propagates to nested functions");
  })();
})();
// Sloppy code outside the strict function still creates an implicit global.
(function () {
  sloppyImplicit = 7;
  assert.sameValue(typeof sloppyImplicit, "number", "sloppy mode still allows implicit globals");
})();
