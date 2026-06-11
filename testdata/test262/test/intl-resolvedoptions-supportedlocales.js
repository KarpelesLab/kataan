/*---
description: Intl resolvedOptions() and supportedLocalesOf() on NumberFormat/DateTimeFormat
features: [Intl]
---*/
// NumberFormat.resolvedOptions reports the resolved configuration.
var nf = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", minimumFractionDigits: 2 });
var ro = nf.resolvedOptions();
assert.sameValue(typeof ro, "object", "resolvedOptions returns an object");
assert.sameValue(ro.locale, "en-US", "locale");
assert.sameValue(ro.numberingSystem, "latn", "numberingSystem");
assert.sameValue(ro.style, "currency", "style");
assert.sameValue(ro.currency, "USD", "currency");
assert.sameValue(ro.minimumFractionDigits, 2, "minimumFractionDigits");
assert.sameValue(ro.useGrouping, true, "useGrouping default");

// Defaults for a plain decimal formatter.
var dr = new Intl.NumberFormat("fr").resolvedOptions();
assert.sameValue(dr.locale, "fr", "decimal locale");
assert.sameValue(dr.style, "decimal", "decimal style");
assert.sameValue(dr.maximumFractionDigits, 3, "default maximumFractionDigits");

// DateTimeFormat.resolvedOptions has its own shape.
var df = new Intl.DateTimeFormat("en-US").resolvedOptions();
assert.sameValue(df.locale, "en-US", "datetime locale");
assert.sameValue(df.calendar, "gregory", "calendar");
assert.sameValue(df.timeZone, "UTC", "timeZone");

// supportedLocalesOf is static on each constructor and returns the requested locales.
assert.sameValue(typeof Intl.NumberFormat.supportedLocalesOf, "function", "NumberFormat.supportedLocalesOf");
assert.sameValue(typeof Intl.DateTimeFormat.supportedLocalesOf, "function", "DateTimeFormat.supportedLocalesOf");
assert.sameValue(typeof Intl.Collator.supportedLocalesOf, "function", "Collator.supportedLocalesOf");
assert.sameValue(Intl.NumberFormat.supportedLocalesOf(["en-US", "fr-FR"]).join(","), "en-US,fr-FR", "array of locales");
assert.sameValue(Intl.DateTimeFormat.supportedLocalesOf("de").join(","), "de", "single locale string");
assert.sameValue(Intl.NumberFormat.supportedLocalesOf([]).length, 0, "empty request");

// format still works alongside.
assert.sameValue(nf.format(1234.5), "$1,234.50", "format unaffected");
