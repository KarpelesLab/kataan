/*---
description: an invalid escape makes a tagged template's cooked value undefined; untagged it is a SyntaxError
esid: sec-template-literal-lexical-components
features: [template-literal-revision]
---*/
// In a TAGGED template an invalid escape yields no cooked value (undefined), while .raw
// still preserves the literal text (ES2018 template-literal revision).
function tag(strings) { return strings; }

var s1 = tag`\unicode`;
assert.sameValue(s1[0], undefined, "invalid \\u escape -> undefined cooked");
assert.sameValue(s1.raw[0], "\\unicode", "raw preserves the escape");

var s2 = tag`\xZZ`;
assert.sameValue(s2[0], undefined, "invalid \\x escape -> undefined cooked");
assert.sameValue(s2.raw[0], "\\xZZ", "raw preserved");

// A valid escape still cooks normally.
var s3 = tag`a\nb`;
assert.sameValue(s3[0], "a\nb", "valid escape cooks");
assert.sameValue(s3.raw[0], "a\\nb", "raw of valid escape");

// A non-special backslash (\w) is not an invalid escape; it cooks to the bare char.
var s4 = tag`a\wb`;
assert.sameValue(s4[0], "awb", "\\w cooks to w");

// Mixed: a valid leading quasi, then an invalid one.
var s5 = tag`ok${1}\u{110000}`;
assert.sameValue(s5[0], "ok", "first quasi cooks");
assert.sameValue(s5[1], undefined, "out-of-range code point -> undefined cooked");

// In an UNTAGGED template, an invalid escape is a SyntaxError (raised on evaluation here).
function throwsSyntax(fn) { try { fn(); return false; } catch (e) { return e instanceof SyntaxError; } }
assert.sameValue(throwsSyntax(function () { return `\unicode`; }), true, "untagged invalid escape -> SyntaxError");

// A valid untagged template is unaffected.
var x = 5;
assert.sameValue(`a\nb=${x}`, "a\nb=5", "valid untagged template");
