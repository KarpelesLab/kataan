/*---
description: a function declaration has a single stable identity, holds properties, works as a key
esid: sec-function-definitions-runtime-semantics-instantiatefunctionobject
---*/
function f() { return 1; }

// Stable identity across references.
assert.sameValue(f === f, true, "f === f");
var ref = f;
assert.sameValue(ref === f, true, "aliased reference is identical");

// Holds assigned properties (a memoization-style pattern).
f.cache = 42;
assert.sameValue(f.cache, 42, "assigned property persists");
ref.tag = "t";
assert.sameValue(f.tag, "t", "property set via an alias is visible");
assert.sameValue(Object.keys(f).sort().join(","), "cache,tag", "own enumerable props");

// Works as a Map/Set key.
var m = new Map();
m.set(f, "v");
assert.sameValue(m.get(f), "v", "function as a Map key");
assert.sameValue(new Set([f]).has(f), true, "function as a Set member");

// Calls, recursion, higher-order use, and shadowing are unaffected.
assert.sameValue(f(), 1, "direct call");
function fact(n) { return n <= 1 ? 1 : n * fact(n - 1); }
assert.sameValue(fact(5), 120, "recursion");
assert.sameValue([1, 2, 3].map(f).join(","), "1,1,1", "passed to a higher-order function");
function g() { return "outer"; }
{ let g = function () { return "inner"; }; assert.sameValue(g(), "inner", "shadowed"); }
assert.sameValue(g(), "outer", "unshadowed after the block");
