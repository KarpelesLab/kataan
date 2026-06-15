/*---
description: eval is a global function; non-string input passes through, string input is compiled and run (direct and indirect)
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

// A string argument is parsed and evaluated; the completion value is returned.
assert.sameValue(eval("1 + 1"), 2, "string eval returns completion value");
assert.sameValue(eval("'a' + 'b'"), "ab", "string concat eval");

// Direct eval can read and modify the surrounding (caller's) scope.
var captured = 10;
assert.sameValue(eval("captured + 5"), 15, "direct eval reads local");
eval("captured = 99");
assert.sameValue(captured, 99, "direct eval writes local");

// Sloppy direct eval hoists `var` into the surrounding variable environment.
eval("var hoisted = 7;");
assert.sameValue(hoisted, 7, "direct eval var hoists outward");

// Indirect eval (`(0, eval)` / a copy of eval) runs in the global scope with the
// global `this`.
var indirect = eval;
assert.sameValue(indirect("1 + 2"), 3, "indirect eval evaluates");
assert.sameValue((0, eval)("this") === globalThis, true, "indirect eval this is global");

// A parse error throws a catchable SyntaxError; surrounding code keeps running.
function evalString(s) {
  try { eval(s); return null; } catch (e) { return e; }
}
var err = evalString("1 +");
assert.sameValue(err instanceof SyntaxError, true, "parse error throws SyntaxError");
assert.sameValue(err instanceof Error, true, "SyntaxError is an Error");

// A runtime throw inside eval propagates to the caller's try/catch.
var thrown = null;
try { eval("throw 42"); } catch (e) { thrown = e; }
assert.sameValue(thrown, 42, "runtime throw propagates");

// The throw is recoverable — surrounding code keeps running.
var reached = false;
try { eval("("); } catch (e) { reached = true; }
assert.sameValue(reached, true, "catch runs");
