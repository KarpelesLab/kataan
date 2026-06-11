/*---
description: Intl.DisplayNames.of for language/region/currency types (en, common subset)
features: [Intl.DisplayNames]
---*/
function dn(t, c) { return new Intl.DisplayNames("en", { type: t }).of(c); }

assert.sameValue(typeof Intl.DisplayNames, "function", "Intl.DisplayNames exists");

// Language names (the primary subtag, case-insensitive).
assert.sameValue(dn("language", "fr"), "French", "fr");
assert.sameValue(dn("language", "en"), "English", "en");
assert.sameValue(dn("language", "de"), "German", "de");
assert.sameValue(dn("language", "ja"), "Japanese", "ja");
assert.sameValue(dn("language", "fr-FR"), "French", "language-region subtag");

// Region names (uppercased).
assert.sameValue(dn("region", "US"), "United States", "US");
assert.sameValue(dn("region", "GB"), "United Kingdom", "GB");
assert.sameValue(dn("region", "JP"), "Japan", "JP");
assert.sameValue(dn("region", "us"), "United States", "lowercase region normalized");

// Currency names.
assert.sameValue(dn("currency", "USD"), "US Dollar", "USD");
assert.sameValue(dn("currency", "EUR"), "Euro", "EUR");
assert.sameValue(dn("currency", "JPY"), "Japanese Yen", "JPY");
assert.sameValue(dn("currency", "usd"), "US Dollar", "lowercase currency normalized");

// Unrecognized codes fall back to themselves.
assert.sameValue(dn("region", "XX"), "XX", "unknown region -> code");
assert.sameValue(dn("currency", "ZZZ"), "ZZZ", "unknown currency -> code");
assert.sameValue(dn("language", "ZZ"), "ZZ", "unknown language -> code");

// of is readable and works without new.
assert.sameValue(typeof new Intl.DisplayNames("en", { type: "region" }).of, "function", "of is readable");
assert.sameValue(Intl.DisplayNames("en", { type: "region" }).of("FR"), "France", "callable without new");
