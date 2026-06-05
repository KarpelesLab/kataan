/*---
description: String.raw and raw template strings
esid: sec-string.raw
---*/
assert.sameValue(String.raw`a\nb`, "a\\nb", "raw preserves backslash");
assert.sameValue(String.raw`line1\nline2`.length, 12, "no escape processing");
assert.sameValue(String.raw`${1}+${2}=${3}`, "1+2=3", "interpolation still works");
assert.sameValue(String.raw`no escapes`, "no escapes");
assert.sameValue(String.raw`tab\there`, "tab\\there");
function showRaw(strings) { return strings.raw[0]; }
assert.sameValue(showRaw`hello\nworld`, "hello\\nworld", "strings.raw in tag");
