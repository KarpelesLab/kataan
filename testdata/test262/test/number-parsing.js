/*---
description: Number parsing, special forms, and conversions
esid: sec-number-objects
---*/
assert.sameValue(parseInt("42px"), 42);
assert.sameValue(parseInt("0xFF"), 255, "hex auto-detected");
assert.sameValue(parseInt("  -17  "), -17);
assert.sameValue(parseInt("abc"), parseInt("abc"), "NaN compares to itself via parseInt");
assert.sameValue(Number.isNaN(parseInt("abc")), true);
assert.sameValue(parseFloat("3.14xyz"), 3.14);
assert.sameValue(parseFloat("1e3"), 1000, "scientific notation");
assert.sameValue(Number("0b101"), 5, "binary literal string");
assert.sameValue(Number("0o17"), 15, "octal literal string");
assert.sameValue((1234.5678).toFixed(2), "1234.57");
assert.sameValue((255).toString(16), "ff");
assert.sameValue(Number.parseInt("10", 2), 2, "Number.parseInt");
