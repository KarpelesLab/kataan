/*---
description: Regex operations on multi-byte (non-ASCII BMP) strings do not panic
esid: sec-regexp-pattern-semantics
---*/
assert.sameValue("café".match(/é/)[0], "é", "match a multi-byte character");
assert.sameValue("café".match(/(.+)/)[1], "café", "capture group with accents");
assert.sameValue("café".replace(/é/, "e"), "cafe", "replace a multi-byte character");
assert.sameValue("naïve café".replace(/é/g, "e"), "naïve cafe", "global replace of an accent");
assert.sameValue("ünïçödé".match(/.+/)[0], "ünïçödé", "dot over accented text");
assert.sameValue("a→b→c".split(/→/).join("|"), "a|b|c", "split on a multi-byte separator");
assert.sameValue("über 123 straße".match(/\d+/)[0], "123", "digits among multi-byte text");
assert.sameValue("café".match(/(?<first>.)(?<rest>.+)/).groups.rest, "afé", "named groups over accents");
assert.sameValue([..."café déjà".matchAll(/é/g)].length, 2, "matchAll over accented text");
assert.sameValue("ré-do ré-mi".replace(/ré/g, "RE"), "RE-do RE-mi", "global replace, accented match");
assert.sameValue("naïveté".search(/eté/), 4, "search index in a multi-byte string");
assert.sameValue("héllo wörld".match(/.örld/)[0], "wörld", "dot before a multi-byte char");
