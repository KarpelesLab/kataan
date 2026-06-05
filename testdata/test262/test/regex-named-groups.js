/*---
description: Regex named capture groups
esid: sec-regexp-pattern-semantics
---*/
var m = "2024-06-15".match(/(?<year>\d{4})-(?<month>\d{2})-(?<day>\d{2})/);
assert.sameValue(m.groups.year, "2024", "named group year");
assert.sameValue(m.groups.month, "06");
assert.sameValue(m.groups.day, "15");
assert.sameValue(m[1], "2024", "still positionally indexable");
var result = "John Smith".replace(/(?<first>\w+) (?<last>\w+)/, "$<last>, $<first>");
assert.sameValue(result, "Smith, John", "named backreference in replacement");
var color = "#ff0000".match(/#(?<r>[0-9a-f]{2})(?<g>[0-9a-f]{2})(?<b>[0-9a-f]{2})/);
assert.sameValue(color.groups.r, "ff");
assert.sameValue(color.groups.g, "00");
assert.sameValue(color.groups.b, "00");
var noMatch = "abc".match(/(?<digit>\d)/);
assert.sameValue(noMatch, null);
