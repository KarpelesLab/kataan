/*---
description: new RegExp throws SyntaxError on an invalid pattern
esid: sec-regexp-pattern-flags
---*/
function bad(p) {
  try { new RegExp(p); return "no-throw"; }
  catch (e) { return e instanceof SyntaxError ? "SyntaxError" : e.constructor.name; }
}
assert.sameValue(bad("("), "SyntaxError", "unbalanced open paren");
assert.sameValue(bad(")"), "SyntaxError", "unbalanced close paren");
assert.sameValue(bad("["), "SyntaxError", "unterminated character class");

// Valid patterns construct and match.
assert.sameValue(new RegExp("[a-z]+").source, "[a-z]+", "character class source");
assert.sameValue("hello123".match(new RegExp("[a-z]+"))[0], "hello", "class match");
assert.sameValue("hello123".match(new RegExp("\\d+"))[0], "123", "digit match");
assert.sameValue(new RegExp("(ab)+", "g").flags, "g", "valid group with flags");
