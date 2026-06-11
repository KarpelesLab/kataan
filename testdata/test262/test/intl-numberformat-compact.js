/*---
description: Intl.NumberFormat notation:"compact" (short) — K/M/B/T scaling with one fraction digit for single-digit mantissas
features: [Intl.NumberFormat]
---*/
function nf(o, v) { return new Intl.NumberFormat("en-US", o).format(v); }
function c(v) { return nf({ notation: "compact" }, v); }

// Millions/thousands with the short suffix; a single-digit mantissa keeps one fraction digit.
assert.sameValue(c(1234567), "1.2M", "1.2M");
assert.sameValue(c(1500000), "1.5M", "1.5M");
assert.sameValue(c(1000000), "1M", "exact million -> 1M (trailing zero trimmed)");
assert.sameValue(c(123456), "123K", "123K (>=10 mantissa -> no fraction)");
assert.sameValue(c(12345), "12K", "12K");
assert.sameValue(c(1234), "1.2K", "1.2K");
assert.sameValue(c(1500), "1.5K", "1.5K");
assert.sameValue(c(1000), "1K", "1K");

// Below 1000 there is no suffix.
assert.sameValue(c(999), "999", "999");
assert.sameValue(c(5), "5", "5");
assert.sameValue(c(0), "0", "0");

// Billions and trillions.
assert.sameValue(c(1234567890), "1.2B", "1.2B");
assert.sameValue(c(1500000000000), "1.5T", "1.5T");

// Negatives and signDisplay compose with the suffix.
assert.sameValue(c(-1234567), "-1.2M", "negative compact");
assert.sameValue(nf({ notation: "compact", signDisplay: "always" }, 1234567), "+1.2M", "compact + always");
