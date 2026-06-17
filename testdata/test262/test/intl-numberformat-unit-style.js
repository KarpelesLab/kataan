/*---
description: Intl.NumberFormat style:"unit" appends the unit's short symbol (NBSP-separated; temperature attaches directly)
features: [Intl.NumberFormat]
---*/
function nf(o, v) { return new Intl.NumberFormat("en-US", o).format(v); }
var NBSP = " ";

// Common units render with a non-breaking space before the short symbol.
assert.sameValue(nf({ style: "unit", unit: "kilometer" }, 5), "5" + NBSP + "km", "kilometer");
assert.sameValue(nf({ style: "unit", unit: "meter" }, 3), "3" + NBSP + "m", "meter");
assert.sameValue(nf({ style: "unit", unit: "megabyte" }, 1.5), "1.5" + NBSP + "MB", "megabyte");
assert.sameValue(nf({ style: "unit", unit: "millisecond" }, 250), "250" + NBSP + "ms", "millisecond");

// A `x-per-y` compound joins the two short symbols with a slash.
assert.sameValue(nf({ style: "unit", unit: "kilometer-per-hour" }, 60), "60" + NBSP + "km/h", "km/h");
assert.sameValue(nf({ style: "unit", unit: "meter-per-second" }, 10), "10" + NBSP + "m/s", "m/s");

// Temperature units attach with no space.
assert.sameValue(nf({ style: "unit", unit: "celsius" }, 20), "20°C", "celsius (no space)");
assert.sameValue(nf({ style: "unit", unit: "fahrenheit" }, 68), "68°F", "fahrenheit (no space)");

// signDisplay and grouping compose with the unit affix.
assert.sameValue(nf({ style: "unit", unit: "kilogram", signDisplay: "always" }, 5), "+5" + NBSP + "kg", "unit + always");
assert.sameValue(nf({ style: "unit", unit: "meter" }, -3), "-3" + NBSP + "m", "negative unit");
assert.sameValue(nf({ style: "unit", unit: "meter" }, 1234567), "1,234,567" + NBSP + "m", "grouped unit");
assert.sameValue(nf({ style: "unit", unit: "liter", minimumFractionDigits: 2 }, 1.5), "1.50" + NBSP + "L", "fraction digits");

// A unit identifier outside the ECMA-402 sanctioned set is a RangeError
// (IsWellFormedUnitIdentifier).
assert.throws(RangeError, function () { nf({ style: "unit", unit: "furlong" }, 2); }, "unsanctioned unit");
