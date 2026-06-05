/*---
description: Number conversions and special value handling
esid: sec-tonumber
---*/
assert.sameValue(Number("42"), 42);
assert.sameValue(Number("3.14"), 3.14);
assert.sameValue(Number("0x1F"), 31, "hex");
assert.sameValue(Number("0b101"), 5, "binary");
assert.sameValue(Number("0o17"), 15, "octal");
assert.sameValue(Number(""), 0);
assert.sameValue(Number("  10  "), 10, "trimmed");
assert.sameValue(Number.isNaN(Number("abc")), true);
assert.sameValue(Number(true), 1);
assert.sameValue(Number(null), 0);
assert.sameValue(Number.isNaN(Number(undefined)), true);
assert.sameValue(Number([]), 0, "empty array");
assert.sameValue(Number([5]), 5, "single element");
assert.sameValue(Number.isInteger(5.0), true);
assert.sameValue(Number.isInteger(5.5), false);
assert.sameValue(Number.isFinite(Infinity), false);
assert.sameValue(Number.isFinite(42), true);
assert.sameValue(parseInt("42px", 10), 42);
assert.sameValue(parseFloat("3.14abc"), 3.14);
assert.sameValue(Number("Infinity"), Infinity);
assert.sameValue(Number("-Infinity"), -Infinity);
