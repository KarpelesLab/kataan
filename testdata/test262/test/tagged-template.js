/*---
description: Tagged template literals receive strings and substitutions
esid: sec-tagged-templates
---*/
function tag(strings, a, b) {
  return strings[0] + "|" + a + "|" + strings[1] + "|" + b + "|" + strings[2];
}
var r = tag`start ${1 + 1} mid ${"x"} end`;
assert.sameValue(r, "start |2| mid |x| end");
