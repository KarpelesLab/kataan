/*---
description: Function.prototype.toString reproduces the source each function was defined in, across eval and the dynamic Function constructor
features: [Reflect]
---*/
// A function's AST spans are byte offsets into the source it was *written* in.
// Slicing them against whatever program happens to be executing yields a
// nonsense substring of unrelated text, so each of these calls the function
// after the defining source has finished evaluating.

// 1. A function returned directly by eval.
assert.sameValue(eval("(function ev(){ return 42; })").toString(),
  "function ev(){ return 42; }", "eval, immediate");

// 2. A function *nested* inside an eval'd function, called after eval returned.
var outer = eval("(function outer(){ return function inner () { return 1; }; })");
assert.sameValue(outer().toString(), "function inner () { return 1; }",
  "eval, nested and deferred");

// 3. CreateDynamicFunction step 20 sets [[SourceText]] to the assembled text,
//    so the result reproduces the exact `function anonymous(...) {...}` form.
assert.sameValue(new Function("a", "b", "return a + b;").toString(),
  "function anonymous(a,b\n) {\nreturn a + b;\n}", "Function constructor");
assert.sameValue(new Function().toString(), "function anonymous(\n) {\n\n}",
  "Function constructor, no arguments");

// 4. The assembled text keeps the wrapper's keyword, so it stays valid syntax.
var GeneratorFunction = Object.getPrototypeOf(function* () {}).constructor;
assert.sameValue(new GeneratorFunction("yield 10").toString(),
  "function* anonymous(\n) {\nyield 10\n}", "GeneratorFunction constructor");
var AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
assert.sameValue(new AsyncFunction("return 1").toString(),
  "async function anonymous(\n) {\nreturn 1\n}", "AsyncFunction constructor");

// 5. A function declared *inside* a dynamic body, called after it was built.
var dyn = new Function("return function dyn () { return 2; };")();
assert.sameValue(dyn.toString(), "function dyn () { return 2; }",
  "Function constructor, nested and deferred");

// 6. Whatever the text is, it must round-trip: re-evaluating it produces an
//    equivalent function rather than a fragment that fails to parse.
assert.sameValue(eval("(" + new Function("a", "return a * 2;").toString() + ")")(21), 42,
  "round-trips through eval");
