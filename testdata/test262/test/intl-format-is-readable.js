/*---
description: Intl NumberFormat/DateTimeFormat .format is a readable function usable as a callback
esid: sec-intl.numberformat.prototype.format
---*/
var nf = new Intl.NumberFormat("en-US");

// `.format` is a function value (not just a call-intercepted method).
assert.sameValue(typeof nf.format, "function", "format is a function");

// A member call formats with the instance's options (grouping here).
assert.sameValue(typeof nf.format(1234.5), "string", "format returns a string");
assert.sameValue(nf.format(1234.5).replace(/[^0-9.,]/g, ""), "1,234.5", "grouping applied");

// It can be read out and used as a callback (the common pattern).
assert.sameValue([1, 2, 3].map(nf.format).length, 3, "map(nf.format)");
var detached = nf.format;
assert.sameValue(typeof detached(99), "string", "detached call returns a string");

// DateTimeFormat.format is likewise a readable function.
var dtf = new Intl.DateTimeFormat("en-US");
assert.sameValue(typeof dtf.format, "function", "dtf.format is a function");
assert.sameValue(typeof dtf.format(new Date(0)), "string", "dtf.format returns a string");

// Collator.compare is still a readable function (used by Array.prototype.sort).
assert.sameValue(typeof new Intl.Collator().compare, "function", "compare is a function");
assert.sameValue(["b", "a", "c"].sort(new Intl.Collator().compare).join(""), "abc", "sort via collator");

// PluralRules.select too.
assert.sameValue(new Intl.PluralRules("en-US").select(1), "one", "plural one");
