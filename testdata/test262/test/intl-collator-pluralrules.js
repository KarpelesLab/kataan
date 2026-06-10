/*---
description: Intl.Collator.compare and Intl.PluralRules.select (code-point/English fallback)
features: [Intl]
---*/
// Collator.compare orders strings and is usable as a sort comparator.
var c = new Intl.Collator("en");
assert.sameValue(c.compare("a", "b"), -1, "a < b");
assert.sameValue(c.compare("b", "a"), 1, "b > a");
assert.sameValue(c.compare("x", "x"), 0, "x === x");
assert.sameValue(["banana", "apple", "cherry"].sort(new Intl.Collator().compare).join(","), "apple,banana,cherry", "sort with collator");

// PluralRules.select returns the English category.
var p = new Intl.PluralRules("en-US");
assert.sameValue(p.select(1), "one", "1 -> one");
assert.sameValue(p.select(2), "other", "2 -> other");
assert.sameValue(p.select(0), "other", "0 -> other");

// Callable without `new`, and type-detectable.
assert.sameValue(Intl.Collator().compare("a", "b"), -1, "Collator() without new");
assert.sameValue(typeof Intl.Collator, "function", "Intl.Collator is a function");
assert.sameValue(typeof Intl.PluralRules, "function", "Intl.PluralRules is a function");
assert.sameValue(typeof new Intl.Collator().compare, "function", "compare is a function value");

// The existing NumberFormat is unaffected.
assert.sameValue(new Intl.NumberFormat("en-US").format(1234.5), "1,234.5", "NumberFormat still works");
