/*---
description: Number.prototype.toLocaleString honors style and fraction-digit options
esid: sec-number.prototype.tolocalestring
---*/
// No options: the grouped default (unchanged).
assert.sameValue((1234567).toLocaleString(), "1,234,567", "default grouping");
assert.sameValue((1234.56).toLocaleString(), "1,234.56", "default fraction");

// minimum/maximumFractionDigits.
assert.sameValue((1234.5).toLocaleString(undefined, { minimumFractionDigits: 2 }), "1,234.50", "min frac pads");
assert.sameValue((1.23456).toLocaleString(undefined, { maximumFractionDigits: 2 }), "1.23", "max frac rounds");
assert.sameValue((1).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 4 }), "1.00", "min/max");
assert.sameValue((1234567.891).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }), "1,234,567.89", "grouping + fraction");

// style: percent.
assert.sameValue((0.5).toLocaleString(undefined, { style: "percent" }), "50%", "percent");
assert.sameValue((0.1234).toLocaleString("en-US", { style: "percent" }), "12%", "percent rounds to 0 digits");
assert.sameValue((0.1234).toLocaleString(undefined, { style: "percent", minimumFractionDigits: 1 }), "12.3%", "percent with fraction");

// style: currency.
assert.sameValue((1234.5).toLocaleString("en-US", { style: "currency", currency: "USD" }), "$1,234.50", "USD");
assert.sameValue((99).toLocaleString(undefined, { style: "currency", currency: "EUR" }), "€99.00", "EUR");

// Negatives.
assert.sameValue((-1234.5).toLocaleString(undefined, { minimumFractionDigits: 2 }), "-1,234.50", "negative fraction");
assert.sameValue((-0.25).toLocaleString(undefined, { style: "percent" }), "-25%", "negative percent");

// A bare locale argument (no options) is still the default.
assert.sameValue((1234567).toLocaleString("en-US"), "1,234,567", "locale only");
