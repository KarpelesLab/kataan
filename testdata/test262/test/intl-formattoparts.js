/*---
description: Intl.NumberFormat.prototype.formatToParts splits the output into typed parts
features: [Intl]
---*/
function parts(p) { return p.map(function (x) { return x.type + ":" + x.value; }).join("|"); }

// Decimal with grouping and a fraction.
assert.sameValue(
  parts(new Intl.NumberFormat("en-US").formatToParts(1234.56)),
  "integer:1|group:,|integer:234|decimal:.|fraction:56",
  "decimal"
);

// An integer (one part).
assert.sameValue(parts(new Intl.NumberFormat("en-US").formatToParts(42)), "integer:42", "integer");

// Multiple groups.
assert.sameValue(
  parts(new Intl.NumberFormat("en-US").formatToParts(1234567)),
  "integer:1|group:,|integer:234|group:,|integer:567",
  "two groups"
);

// Currency prefix.
assert.sameValue(
  parts(new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" }).formatToParts(1234.5)),
  "currency:$|integer:1|group:,|integer:234|decimal:.|fraction:50",
  "currency"
);

// Percent suffix.
assert.sameValue(
  parts(new Intl.NumberFormat("en-US", { style: "percent" }).formatToParts(0.25)),
  "integer:25|percentSign:%",
  "percent"
);

// Negative numbers get a minusSign part first.
assert.sameValue(
  parts(new Intl.NumberFormat("en-US").formatToParts(-1234.5)),
  "minusSign:-|integer:1|group:,|integer:234|decimal:.|fraction:5",
  "negative"
);

// Non-finite values: NaN and Infinity (with affixes).
assert.sameValue(parts(new Intl.NumberFormat("en-US").formatToParts(NaN)), "nan:NaN", "NaN");
assert.sameValue(parts(new Intl.NumberFormat("en-US").formatToParts(Infinity)), "infinity:∞", "Infinity");
assert.sameValue(parts(new Intl.NumberFormat("en-US").formatToParts(-Infinity)), "minusSign:-|infinity:∞", "-Infinity");

// format() itself renders ∞ / NaN (not Rust's "inf").
assert.sameValue(new Intl.NumberFormat("en-US").format(Infinity), "∞", "format Infinity");
assert.sameValue(new Intl.NumberFormat("en-US").format(-Infinity), "-∞", "format -Infinity");
assert.sameValue(new Intl.NumberFormat("en-US").format(NaN), "NaN", "format NaN");

// Concatenating the part values reproduces format().
var nf = new Intl.NumberFormat("en-US", { style: "currency", currency: "EUR" });
assert.sameValue(nf.formatToParts(9999.99).map(function (p) { return p.value; }).join(""), nf.format(9999.99), "parts rejoin to format");
