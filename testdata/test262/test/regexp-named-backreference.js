/*---
description: \k<name> named backreferences
features: [regexp-named-groups]
---*/
// A named backreference matches the text the named group captured.
assert.sameValue(/(?<q>["']).*?\k<q>/.test('"hi"'), true, "matched quotes");
assert.sameValue(/(?<q>["']).*?\k<q>/.test('"hi\''), false, "mismatched quotes");

// Duplicate-word detection, with the captured value available via groups.
var m = "hello hello".match(/(?<w>\w+) \k<w>/);
assert.sameValue(m !== null, true, "duplicate matched");
assert.sameValue(m.groups.w, "hello", "captured word");
assert.sameValue(/(?<w>\w+) \k<w>/.test("hello world"), false, "distinct words don't match");

// Numeric backreferences still work alongside named ones.
assert.sameValue(/(\w+) \1/.test("hi hi"), true, "numeric backref");
assert.sameValue(/(\w+) \1/.test("hi ho"), false, "numeric backref mismatch");

// Multiple named groups and references.
assert.sameValue(/(?<a>x)(?<b>y)\k<a>\k<b>/.test("xyxy"), true, "two named backrefs");

// $<name> replacement keeps working.
assert.sameValue("2024-01".replace(/(?<y>\d+)-(?<m>\d+)/, "$<m>/$<y>"), "01/2024", "named replacement");
