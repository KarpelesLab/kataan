/*---
description: Intl.NumberFormat signDisplay option and sign placement relative to currency/percent affixes
features: [Intl.NumberFormat]
---*/
function nf(o, v) { return new Intl.NumberFormat("en-US", o).format(v); }

// signDisplay: auto (default) shows a sign only for negatives.
assert.sameValue(nf({}, 5), "5", "auto positive");
assert.sameValue(nf({}, -5), "-5", "auto negative");
assert.sameValue(nf({}, 0), "0", "auto zero");

// always: a sign on every value, including zero.
assert.sameValue(nf({ signDisplay: "always" }, 5), "+5", "always positive");
assert.sameValue(nf({ signDisplay: "always" }, -5), "-5", "always negative");
assert.sameValue(nf({ signDisplay: "always" }, 0), "+0", "always zero");

// never: no sign, even for negatives.
assert.sameValue(nf({ signDisplay: "never" }, 5), "5", "never positive");
assert.sameValue(nf({ signDisplay: "never" }, -5), "5", "never negative");

// exceptZero: a sign on non-zero values, none on zero.
assert.sameValue(nf({ signDisplay: "exceptZero" }, 5), "+5", "exceptZero positive");
assert.sameValue(nf({ signDisplay: "exceptZero" }, -5), "-5", "exceptZero negative");
assert.sameValue(nf({ signDisplay: "exceptZero" }, 0), "0", "exceptZero zero");

// resolvedOptions reflects the chosen signDisplay.
assert.sameValue(new Intl.NumberFormat("en-US", { signDisplay: "always" }).resolvedOptions().signDisplay, "always", "resolvedOptions.signDisplay");

// The sign sits outside the currency symbol / percent sign.
assert.sameValue(nf({ style: "currency", currency: "USD" }, -1234.5), "-$1,234.50", "negative currency sign placement");
assert.sameValue(nf({ style: "currency", currency: "USD" }, 1234.5), "$1,234.50", "positive currency");
assert.sameValue(nf({ style: "currency", currency: "USD", signDisplay: "always" }, 5), "+$5.00", "currency + always");
assert.sameValue(nf({ style: "currency", currency: "EUR" }, -5), "-€5.00", "negative EUR");
assert.sameValue(nf({ style: "percent", signDisplay: "always" }, 0.25), "+25%", "percent + always");
assert.sameValue(nf({ style: "percent" }, -0.25), "-25%", "negative percent");
