/*---
description: switch statement fall-through, default, and break
esid: sec-switch-statement
---*/
function classify(n) {
  var r = "";
  switch (n) {
    case 1: r += "1";
    case 2: r += "2"; break;
    case 3: r += "3";
    default: r += "d";
  }
  return r;
}
assert.sameValue(classify(1), "12", "fall-through to break");
assert.sameValue(classify(2), "2", "single case");
assert.sameValue(classify(3), "3d", "fall to default");
assert.sameValue(classify(9), "d", "default only");
function typeOf(x) {
  switch (typeof x) {
    case "number": return "num";
    case "string": return "str";
    case "boolean": return "bool";
    default: return "other";
  }
}
assert.sameValue(typeOf(5), "num");
assert.sameValue(typeOf("a"), "str");
assert.sameValue(typeOf(true), "bool");
assert.sameValue(typeOf({}), "other");
function strictMatch(v) {
  switch (v) {
    case "1": return "string one";
    case 1: return "number one";
    default: return "none";
  }
}
assert.sameValue(strictMatch(1), "number one", "switch uses strict equality");
assert.sameValue(strictMatch("1"), "string one");
var count = 0;
for (var i = 0; i < 5; i++) {
  switch (i % 3) {
    case 0: count += 1; break;
    case 1: count += 10; break;
    default: count += 100;
  }
}
assert.sameValue(count, 1 + 10 + 100 + 1 + 10, "switch in loop");
