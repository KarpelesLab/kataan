/*---
description: typeof resolves the Intl/WebAssembly/structuredClone globals on every engine
---*/
assert.sameValue(typeof Intl, "object", "typeof Intl");
assert.sameValue(typeof WebAssembly, "object", "typeof WebAssembly");
assert.sameValue(typeof structuredClone, "function", "typeof structuredClone");
// And the other well-known namespaces stay correct.
assert.sameValue(typeof Math, "object", "typeof Math");
assert.sameValue(typeof JSON, "object", "typeof JSON");
assert.sameValue(typeof Reflect, "object", "typeof Reflect");
// An undeclared identifier is still undefined (typeof doesn't throw).
assert.sameValue(typeof someUndeclaredGlobal, "undefined", "undeclared -> undefined");

// The namespaces are usable, not just type-detectable.
assert.sameValue(typeof Intl.NumberFormat, "function", "Intl.NumberFormat");
assert.sameValue(new Intl.NumberFormat("en-US").format(1234.5), "1,234.5", "NumberFormat works");
assert.sameValue(typeof WebAssembly.validate, "function", "WebAssembly.validate");
assert.sameValue(structuredClone({ a: 1 }).a, 1, "structuredClone works");
