/*---
description: String methods delegate to a custom argument's Symbol.match/replace/search/split
features: [Symbol.match, Symbol.replace, Symbol.split]
---*/
// str.match(obj) -> obj[Symbol.match](str)
var matcher = { [Symbol.match](s) { return s.includes("x") ? ["found"] : null; } };
assert.sameValue(JSON.stringify("axb".match(matcher)), '["found"]', "Symbol.match hit");
assert.sameValue("abc".match(matcher), null, "Symbol.match miss");

// str.replace(obj, repl) -> obj[Symbol.replace](str, repl)
var replacer = { [Symbol.replace](s, r) { return s.toUpperCase() + ":" + r; } };
assert.sameValue("abc".replace(replacer, "X"), "ABC:X", "Symbol.replace receives string + replacement");

// str.search(obj) -> obj[Symbol.search](str)
var searcher = { [Symbol.search](s) { return s.indexOf("z"); } };
assert.sameValue("abzc".search(searcher), 2, "Symbol.search");

// str.split(obj) -> obj[Symbol.split](str)
var splitter = { [Symbol.split](s) { return s.split("").reverse(); } };
assert.sameValue("abc".split(splitter).join(","), "c,b,a", "Symbol.split");

// Real RegExp and string arguments are unaffected.
assert.sameValue("a1b2".replace(/\d/g, "#"), "a#b#", "regex replace still works");
assert.sameValue("a.b.c".replace(".", "-"), "a-b.c", "string replace still works");
assert.sameValue("a,b,c".split(",").join("|"), "a|b|c", "string split still works");
