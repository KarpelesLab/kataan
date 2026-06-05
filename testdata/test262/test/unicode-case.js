/*---
description: Unicode case conversion
esid: sec-string.prototype.touppercase
features: [intl]
---*/
assert.sameValue("hello".toUpperCase(), "HELLO");
assert.sameValue("WORLD".toLowerCase(), "world");
assert.sameValue("MiXeD".toLowerCase(), "mixed");
assert.sameValue("café".toUpperCase(), "CAFÉ", "accented uppercase");
assert.sameValue("CAFÉ".toLowerCase(), "café");
assert.sameValue("ñ".toUpperCase(), "Ñ");
assert.sameValue("Ñ".toLowerCase(), "ñ");
assert.sameValue("ß".toUpperCase(), "SS", "German sharp s expands");
assert.sameValue("ΑΒΓ".toLowerCase(), "αβγ", "Greek");
assert.sameValue("αβγ".toUpperCase(), "ΑΒΓ");
assert.sameValue("héllo wörld".toUpperCase(), "HÉLLO WÖRLD");
assert.sameValue("Title Case".toLowerCase(), "title case");
