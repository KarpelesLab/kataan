/*---
description: Tagged template literals
esid: sec-tagged-templates
---*/
function tag(strings, ...values) {
  var result = "";
  for (var i = 0; i < strings.length; i++) {
    result += strings[i];
    if (i < values.length) result += "[" + values[i] + "]";
  }
  return result;
}
var name = "World";
assert.sameValue(tag`Hello ${name}!`, "Hello [World]!");
assert.sameValue(tag`${1} plus ${2} is ${3}`, "[1] plus [2] is [3]");
function count(strings, ...values) { return strings.length + ":" + values.length; }
assert.sameValue(count`no interpolation`, "1:0");
assert.sameValue(count`${"a"}${"b"}`, "3:2");
function raw(strings) { return strings[0]; }
assert.sameValue(raw`plain`, "plain");
