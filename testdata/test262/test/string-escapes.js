/*---
description: String escape sequences in literals
esid: sec-literals-string-literals
---*/
assert.sameValue("a\tb".length, 3, "tab");
assert.sameValue("a\nb".length, 3, "newline");
assert.sameValue("a\\b".length, 3, "backslash");
assert.sameValue("a\"b".length, 3, "escaped quote");
assert.sameValue("A", "A", "unicode escape");
assert.sameValue("\u{1F600}".length, 2, "code point escape astral");
assert.sameValue("\x41", "A", "hex escape");
assert.sameValue("\0".charCodeAt(0), 0, "null character");
assert.sameValue("line1\nline2".split("\n").length, 2);
assert.sameValue("tab\tseparated".split("\t").length, 2);
assert.sameValue('single \'quote\'', "single 'quote'");
assert.sameValue("é", "é", "accented e");
assert.sameValue("a\rb".length, 3, "carriage return");
assert.sameValue("ABC", "ABC");
assert.sameValue("back\\slash".indexOf("\\"), 4);
