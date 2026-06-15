/*---
description: Regex \p{...} Unicode property escapes (general categories, scripts, binary properties)
esid: prod-CharacterClassEscape
---*/
// Property escapes require the `u` flag (without it `\p` is the literal `p`).
// Group categories.
assert.sameValue("Hello World".match(/\p{Lu}/gu).join(""), "HW", "uppercase letters");
assert.sameValue("Hello".match(/\p{Ll}/gu).join(""), "ello", "lowercase letters");
assert.sameValue("abc123".match(/\p{N}/gu).join(""), "123", "number group");
assert.sameValue("a.b!c".match(/\p{P}/gu).join(""), ".!", "punctuation group (ASCII)");
assert.sameValue("中文字".match(/\p{Lo}/gu).length, 3, "uncased letters (CJK)");
assert.sameValue("x y".match(/\p{Z}/u)[0], " ", "separator");
// Long-form aliases parse and compile.
assert.sameValue("Hi".match(/\p{Uppercase_Letter}/u)[0], "H", "long-form alias");
// The negated form.
assert.sameValue("a1b2".match(/\P{N}/gu).join(""), "ab", "negated number");

// `Property=Value`: General_Category, Script, Script_Extensions (canonical
// names, the gc/sc/scx aliases, and ISO 15924 short codes).
assert.sameValue("abc".match(/\p{General_Category=Letter}/gu).join(""), "abc", "gc long");
assert.sameValue("abc".match(/\p{gc=L}/gu).join(""), "abc", "gc alias + short");
assert.sameValue("αβγ".match(/\p{Script=Greek}/gu).join(""), "αβγ", "Script long name");
assert.sameValue("abc".match(/\p{sc=Latn}/gu).join(""), "abc", "Script short code (sc=)");
assert.sameValue("abc".match(/\p{Script_Extensions=Latin}/gu).join(""), "abc", "scx long");
assert.sameValue("LC".match(/\p{LC}/gu).join(""), "LC", "Cased_Letter union (LC)");

// Binary properties (closed-form and table-backed).
assert.sameValue("aé".match(/\p{ASCII}/gu).join(""), "a", "ASCII");
assert.sameValue("0aF".match(/\p{ASCII_Hex_Digit}/gu).join(""), "0aF", "ASCII_Hex_Digit");
assert.sameValue("a b".match(/\p{White_Space}/gu)[0], " ", "White_Space");
assert.sameValue("aΩ1".match(/\p{Alphabetic}/gu).join(""), "aΩ", "Alphabetic");

// The full general-category subcategory set compiles under `u`; valid binary
// and script value names also parse even when no local data is available.
var ok = true;
try {
  ["Lt", "Lm", "Mn", "Mc", "Me", "Nl", "No", "Pc", "Pd", "Ps", "Pe", "Sm", "Sc", "Sk", "So", "Zs", "Cf", "Cs", "Co", "Cn"].forEach(function (cat) {
    new RegExp("\\p{" + cat + "}", "u");
  });
  ["Emoji", "ID_Start", "XID_Continue", "Cased", "Math", "Dash", "Any", "Assigned"].forEach(function (bin) {
    new RegExp("\\p{" + bin + "}", "u");
  });
  ["Script=Han", "sc=Cyrl", "scx=Hira"].forEach(function (pv) {
    new RegExp("\\p{" + pv + "}", "u");
  });
} catch (e) { ok = false; }
assert.sameValue(ok, true, "all valid property escapes compile under u");

// Invalid property escapes are a SyntaxError at construction.
["\\p{Nonsense}", "\\p{Script=Nonsense}", "\\p{General_Category}", "\\p{Script}",
 "\\p{ASCII=Invalid}", "\\p{Alphabetic=Yes}", "\\p{}", "\\p{=}", "\\p{=L}",
 "\\p{^L}", "\\p{ Lowercase }", "\\pL"].forEach(function (bad) {
  var threw = false;
  try { new RegExp(bad, "u"); } catch (e) { threw = e instanceof SyntaxError; }
  assert.sameValue(threw, true, "must throw SyntaxError: " + bad);
});

// Without the `u` flag, `\p` is the literal `p` (Annex B IdentityEscape).
assert.sameValue(/\p/.test("p"), true, "non-u \\p matches literal p");
assert.sameValue(/\p/.test("x"), false, "non-u \\p does not match x");
assert.sameValue(new RegExp("\\p{Nonsense}").source, "\\p{Nonsense}", "non-u never throws");
