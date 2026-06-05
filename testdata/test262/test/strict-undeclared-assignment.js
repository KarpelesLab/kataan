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
  // A write to a non-writable property throws (silently ignored in sloppy mode).
  var ro = {};
  Object.defineProperty(ro, "x", { value: 1, writable: false });
  var roThrew = false;
  try { ro.x = 2; } catch (e) { roThrew = e instanceof TypeError; }
  assert.sameValue(roThrew, true, "write to read-only property throws in strict mode");
  assert.sameValue(ro.x, 1, "value unchanged");
  // A frozen object rejects writes.
  var frozen = Object.freeze({ a: 1 });
  var fThrew = false;
  try { frozen.a = 9; } catch (e) { fThrew = e instanceof TypeError; }
  assert.sameValue(fThrew, true, "write to a frozen object throws in strict mode");
  // A writable property is fine.
  var ok = {};
  ok.y = 5;
  assert.sameValue(ok.y, 5, "writable property assignment works");
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
  var so = {};
  Object.defineProperty(so, "x", { value: 1, writable: false });
  so.x = 2;
  assert.sameValue(so.x, 1, "sloppy read-only write is silently ignored");
})();
