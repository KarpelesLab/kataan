/*---
description: Tagged template with strings.raw and multiple substitutions
esid: sec-tagged-templates
---*/
function tag(strings) {
  var out = strings[0];
  for (var i = 1; i < arguments.length; i++) out += "[" + arguments[i] + "]" + strings[i];
  return out;
}
assert.sameValue(tag`a${1}b${2}c`, "a[1]b[2]c");
