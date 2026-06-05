/*---
description: switch statement comprehensive cases
esid: sec-switch-statement
---*/
function describe(n) {
  switch (true) {
    case n < 0: return "negative";
    case n === 0: return "zero";
    case n < 10: return "small";
    default: return "large";
  }
}
assert.sameValue(describe(-5), "negative", "switch(true) pattern");
assert.sameValue(describe(0), "zero");
assert.sameValue(describe(5), "small");
assert.sameValue(describe(100), "large");
function typeOf(v) {
  switch (typeof v) {
    case "number": case "bigint": return "numeric";
    case "string": return "text";
    case "boolean": return "bool";
    default: return "other";
  }
}
assert.sameValue(typeOf(42), "numeric");
assert.sameValue(typeOf("hi"), "text");
assert.sameValue(typeOf(true), "bool");
assert.sameValue(typeOf({}), "other");
function dayType(day) {
  var result;
  switch (day) {
    case 6: case 0: result = "weekend"; break;
    default: result = "weekday";
  }
  return result;
}
assert.sameValue(dayType(0), "weekend");
assert.sameValue(dayType(6), "weekend");
assert.sameValue(dayType(3), "weekday");
