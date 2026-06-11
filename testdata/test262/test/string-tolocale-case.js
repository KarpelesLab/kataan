/*---
description: String.prototype.toLocaleUpperCase / toLocaleLowerCase exist and convert case
esid: sec-string.prototype.tolocaleuppercase
---*/
// They are callable functions.
assert.sameValue(typeof "x".toLocaleUpperCase, "function", "toLocaleUpperCase is a function");
assert.sameValue(typeof "x".toLocaleLowerCase, "function", "toLocaleLowerCase is a function");

// Basic conversion (no locale-specific tailoring in this engine — same as the
// locale-independent forms).
assert.sameValue("abc".toLocaleUpperCase(), "ABC", "upper");
assert.sameValue("ABC".toLocaleLowerCase(), "abc", "lower");
assert.sameValue("café".toLocaleUpperCase(), "CAFÉ", "accented upper");
assert.sameValue("ΑΒΓ".toLocaleLowerCase(), "αβγ", "greek lower");

// They agree with the non-locale variants.
assert.sameValue("Hello World".toLocaleUpperCase(), "Hello World".toUpperCase(), "matches toUpperCase");
assert.sameValue("Hello World".toLocaleLowerCase(), "Hello World".toLowerCase(), "matches toLowerCase");

// Special case mapping (ß -> SS) works.
assert.sameValue("ß".toLocaleUpperCase(), "SS", "eszett uppercases to SS");

// Generic application via call.
assert.sameValue(String.prototype.toLocaleUpperCase.call("hi"), "HI", "generic call");

// Empty string and chaining.
assert.sameValue("".toLocaleUpperCase(), "", "empty");
assert.sameValue("Hello".toLocaleLowerCase().toLocaleUpperCase(), "HELLO", "chained");
