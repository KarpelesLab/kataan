/*---
description: eval is a global function; non-string input passes through, string input throws (no dynamic code)
esid: sec-eval-x
---*/
// eval exists as a function, on both the binding and globalThis (same identity).
assert.sameValue(typeof eval, "function", "typeof eval");
assert.sameValue(typeof globalThis.eval, "function", "typeof globalThis.eval");
assert.sameValue(globalThis.eval, eval, "same eval identity");

// A non-string argument is returned unchanged (per spec step 2).
assert.sameValue(eval(42), 42, "number passthrough");
assert.sameValue(eval(true), true, "boolean passthrough");
assert.sameValue(eval(undefined), undefined, "undefined passthrough");
assert.sameValue(eval(null), null, "null passthrough");
var obj = { a: 1 };
assert.sameValue(eval(obj), obj, "object identity passthrough");
var arr = [1, 2, 3];
assert.sameValue(eval(arr), arr, "array identity passthrough");

// A string argument would require compiling source at runtime, which this engine does not
// support — it throws a catchable EvalError rather than leaving eval undefined.
function evalString(s) {
  try { eval(s); return null; } catch (e) { return e; }
}
var err = evalString("1 + 1");
assert.sameValue(err instanceof EvalError, true, "string eval throws EvalError");
assert.sameValue(err instanceof Error, true, "EvalError is an Error");

// Indirect eval behaves the same.
var indirect = eval;
var ierr = (function () { try { indirect("x"); return null; } catch (e) { return e; } })();
assert.sameValue(ierr instanceof EvalError, true, "indirect eval throws too");

// The throw is recoverable — surrounding code keeps running.
var reached = false;
try { eval("nope"); } catch (e) { reached = true; }
assert.sameValue(reached, true, "catch runs");
