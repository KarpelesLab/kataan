/*---
description: matchAll with named groups and multiple matches
esid: sec-string.prototype.matchall
---*/
var dates = [..."2024-01 2025-12".matchAll(/(?<year>\d{4})-(?<month>\d{2})/g)];
assert.sameValue(dates.length, 2);
assert.sameValue(dates[0].groups.year, "2024");
assert.sameValue(dates[0].groups.month, "01");
assert.sameValue(dates[1].groups.year, "2025");
assert.sameValue(dates[1].groups.month, "12");
var pairs = [..."a=1;b=2;c=3".matchAll(/(?<key>\w)=(?<val>\d)/g)];
assert.sameValue(pairs.length, 3);
assert.sameValue(pairs.map(function (m) { return m.groups.key + m.groups.val; }).join(","), "a1,b2,c3");
var words = [..."hello world foo".matchAll(/(?<w>\w+)/g)];
assert.sameValue(words.map(function (m) { return m.groups.w; }).join("-"), "hello-world-foo");
assert.sameValue(dates[0][1], "2024", "positional still works");
assert.sameValue(dates[0].index, 0, "match index");
assert.sameValue([..."abc".matchAll(/\d/g)].length, 0);
