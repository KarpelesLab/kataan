/*---
description: matchAll/replaceAll require a global RegExp; named-group template replace
esid: sec-string.prototype.matchall
---*/
var threw1 = false;
try { "aaa".replaceAll(/a/, "b"); } catch (e) { threw1 = e instanceof TypeError; }
assert.sameValue(threw1, true, "replaceAll with a non-global RegExp throws");
var threw2 = false;
try { [..."aaa".matchAll(/a/)]; } catch (e) { threw2 = e instanceof TypeError; }
assert.sameValue(threw2, true, "matchAll with a non-global RegExp throws");
assert.sameValue("aaa".replaceAll(/a/g, "b"), "bbb", "global replaceAll works");
assert.sameValue([..."a1b2".matchAll(/[a-z]\d/g)].length, 2, "global matchAll works");
assert.sameValue("2024-06".replace(/(?<y>\d+)-(?<m>\d+)/, "$<m>/$<y>"), "06/2024", "named group template");
assert.sameValue("John Smith".replace(/(\w+) (\w+)/, "$2 $1"), "Smith John", "numbered group template");
assert.sameValue("price".replace(/price/, "$$5"), "$5", "escaped dollar");
assert.sameValue(/abc/gi.flags, "gi", "flags property");
assert.sameValue(/abc/.source, "abc", "source property");
assert.sameValue(/abc/g.global, true);
assert.sameValue(/abc/i.ignoreCase, true);
assert.sameValue("xxabc".match(/abc/).index, 2, "match index");
