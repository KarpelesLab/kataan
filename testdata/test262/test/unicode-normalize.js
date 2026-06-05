/*---
description: String.prototype.normalize NFC/NFD/NFKC/NFKD
esid: sec-string.prototype.normalize
features: [intl]
---*/
var composed = "é";          // é as a single code point
var decomposed = "é";        // e + combining acute accent
assert.sameValue(composed.length, 1);
assert.sameValue(decomposed.length, 2);
assert.sameValue(composed.normalize("NFC"), decomposed.normalize("NFC"), "NFC composes");
assert.sameValue(decomposed.normalize("NFC").length, 1, "NFC of decomposed is one code point");
assert.sameValue(composed.normalize("NFD").length, 2, "NFD of composed decomposes");
assert.sameValue(composed.normalize(), composed.normalize("NFC"), "default form is NFC");
assert.sameValue("abc".normalize(), "abc", "ASCII unchanged");
var ligature = "ﬁ";            // ﬁ ligature
assert.sameValue(ligature.normalize("NFKC"), "fi", "NFKC expands compatibility ligature");
assert.sameValue(ligature.normalize("NFC"), ligature, "NFC keeps the ligature");
