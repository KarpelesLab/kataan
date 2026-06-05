/*---
description: Regex \p{...} general categories (groups and common subcategories)
esid: prod-CharacterClassEscape
---*/
// Group categories.
assert.sameValue("Hello World".match(/\p{Lu}/g).join(""), "HW", "uppercase letters");
assert.sameValue("Hello".match(/\p{Ll}/g).join(""), "ello", "lowercase letters");
assert.sameValue("abc123".match(/\p{N}/g).join(""), "123", "number group");
assert.sameValue("a.b!c".match(/\p{P}/g).join(""), ".!", "punctuation group (ASCII)");
assert.sameValue("中文字".match(/\p{Lo}/g).length, 3, "uncased letters (CJK)");
assert.sameValue("x y".match(/\p{Z}/)[0], " ", "separator (non-breaking space)");
// Long-form aliases parse and compile.
assert.sameValue("Hi".match(/\p{Uppercase_Letter}/)[0], "H", "long-form alias");
// The negated form.
assert.sameValue("a1b2".match(/\P{N}/g).join(""), "ab", "negated number");
// The full subcategory set at least compiles (matching needs Unicode tables).
var ok = true;
try {
  ["Lt", "Lm", "Mn", "Mc", "Me", "Nl", "No", "Pc", "Pd", "Ps", "Pe", "Sm", "Sc", "Sk", "So", "Zs", "Cf"].forEach(function (cat) {
    new RegExp("\\p{" + cat + "}");
  });
} catch (e) { ok = false; }
assert.sameValue(ok, true, "all general-category codes compile");
