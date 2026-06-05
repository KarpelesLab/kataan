/*---
description: Intl.NumberFormat and Intl.DateTimeFormat basic formatting
esid: sec-intl-numberformat-constructor
---*/
assert.sameValue(typeof Intl, "object", "Intl namespace exists");
assert.sameValue(typeof Intl.NumberFormat, "function", "Intl.NumberFormat");
assert.sameValue(new Intl.NumberFormat("en-US").format(1234.5), "1,234.5", "decimal grouping");
assert.sameValue(new Intl.NumberFormat("en-US").format(1000000), "1,000,000", "millions");
assert.sameValue(new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" }).format(1234.5), "$1,234.50", "USD currency");
assert.sameValue(new Intl.NumberFormat("en-US", { style: "currency", currency: "EUR" }).format(99), "€99.00", "EUR currency");
assert.sameValue(new Intl.NumberFormat("en-US", { style: "currency", currency: "JPY" }).format(1234), "¥1,234", "JPY has no fraction digits");
assert.sameValue(new Intl.NumberFormat("en-US", { style: "percent" }).format(0.25), "25%", "percent");
assert.sameValue(new Intl.NumberFormat("en-US", { minimumFractionDigits: 2 }).format(5), "5.00", "minimumFractionDigits");
assert.sameValue(new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 }).format(3.14159), "3.14", "maximumFractionDigits");
assert.sameValue(new Intl.NumberFormat("en-US", { useGrouping: false }).format(1234567), "1234567", "useGrouping false");
assert.sameValue(new Intl.NumberFormat("en-US").format(-1234.5), "-1,234.5", "negative");
// DateTimeFormat.
assert.sameValue(typeof Intl.DateTimeFormat, "function", "Intl.DateTimeFormat");
assert.sameValue(new Intl.DateTimeFormat("en-US").format(new Date(Date.UTC(2020, 5, 15))), "6/15/2020", "date format");
assert.sameValue(typeof new Intl.DateTimeFormat("en-US").format(new Date(0)), "string", "format returns a string");
// Reachable via globalThis.
assert.sameValue(globalThis.Intl.NumberFormat("en-US").format(42), "42", "via globalThis");
