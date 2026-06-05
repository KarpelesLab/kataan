/*---
description: parseFloat recognizes Infinity (optionally signed, with leading whitespace)
esid: sec-parsefloat-string
---*/
assert.sameValue(parseFloat("Infinity"), Infinity, "Infinity");
assert.sameValue(parseFloat("-Infinity"), -Infinity, "negative Infinity");
assert.sameValue(parseFloat("+Infinity"), Infinity, "explicit positive Infinity");
assert.sameValue(parseFloat("  Infinity  "), Infinity, "leading/trailing whitespace");
assert.sameValue(parseFloat("Infinity and beyond"), Infinity, "Infinity then trailing text");
assert.sameValue(parseFloat("InfinityX"), Infinity, "Infinity immediately followed by text");
assert.sameValue(Number.parseFloat("Infinity"), Infinity, "Number.parseFloat mirror");
assert.sameValue(parseFloat("1.5e3"), 1500, "regular exponent still works");
assert.sameValue(parseFloat("3.14abc"), 3.14, "trailing text after a number");
assert.sameValue(Number.isNaN(parseFloat("Inf")), true, "a partial 'Inf' is not Infinity");
assert.sameValue(Number.isNaN(parseFloat("xyz")), true, "non-numeric is NaN");
assert.sameValue(parseFloat("-Infinity") < 0, true, "sign preserved");
