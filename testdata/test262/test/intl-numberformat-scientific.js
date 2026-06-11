/*---
description: Intl.NumberFormat notation:"scientific" and "engineering" (mantissa E exponent)
features: [Intl.NumberFormat]
---*/
function nf(o, v) { return new Intl.NumberFormat("en-US", o).format(v); }

// Scientific: mantissa in [1,10) times a power of ten.
assert.sameValue(nf({ notation: "scientific" }, 123456), "1.235E5", "123456 -> 1.235E5");
assert.sameValue(nf({ notation: "scientific" }, 1000), "1E3", "1000 -> 1E3");
assert.sameValue(nf({ notation: "scientific" }, 1500), "1.5E3", "1500 -> 1.5E3");
assert.sameValue(nf({ notation: "scientific" }, 0.0012), "1.2E-3", "0.0012 -> 1.2E-3 (negative exponent)");
assert.sameValue(nf({ notation: "scientific" }, 0), "0E0", "0 -> 0E0");
assert.sameValue(nf({ notation: "scientific" }, -123456), "-1.235E5", "negative");

// Engineering: exponent is a multiple of 3.
assert.sameValue(nf({ notation: "engineering" }, 123456), "123.456E3", "123456 -> 123.456E3");
assert.sameValue(nf({ notation: "engineering" }, 1234), "1.234E3", "1234 -> 1.234E3");
assert.sameValue(nf({ notation: "engineering" }, 12), "12E0", "12 -> 12E0");
assert.sameValue(nf({ notation: "engineering" }, 1000000), "1E6", "1e6 -> 1E6");
assert.sameValue(nf({ notation: "engineering" }, 5), "5E0", "5 -> 5E0");

// Composes with minimumFractionDigits and signDisplay; standard is unaffected.
assert.sameValue(nf({ notation: "scientific", minimumFractionDigits: 2 }, 5), "5.00E0", "min fraction digits");
assert.sameValue(nf({ notation: "scientific", signDisplay: "always" }, 123456), "+1.235E5", "scientific + always");
assert.sameValue(nf({ notation: "standard" }, 123456), "123,456", "standard notation groups normally");
