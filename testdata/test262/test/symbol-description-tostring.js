/*---
description: a no-argument Symbol renders as "Symbol()"; description is preserved
features: [Symbol]
---*/
// A no-argument Symbol has an undefined description and stringifies as "Symbol()".
var s = Symbol();
assert.sameValue(s.description, undefined, "no description");
assert.sameValue(s.toString(), "Symbol()", "toString of no-arg Symbol");
assert.sameValue(String(s), "Symbol()", "String() of no-arg Symbol");

// Symbol("") has an empty (but present) description; still renders "Symbol()".
var e = Symbol("");
assert.sameValue(e.description, "", "empty description");
assert.sameValue(e.toString(), "Symbol()", "toString of Symbol('')");

// A described Symbol preserves it.
assert.sameValue(Symbol("x").toString(), "Symbol(x)", "toString with description");
assert.sameValue(String(Symbol("hi")), "Symbol(hi)", "String() with description");
assert.sameValue(Symbol("desc").description, "desc", "description preserved");

// Well-known symbols render their canonical names.
assert.sameValue(Symbol.iterator.toString(), "Symbol(Symbol.iterator)", "well-known symbol");
