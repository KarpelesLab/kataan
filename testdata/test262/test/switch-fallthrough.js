/*---
description: switch fall-through, default position, and string discriminant
esid: sec-switch-statement
---*/
function f(x) {
  var out = [];
  switch (x) {
    case 1: out.push("one");
    case 2: out.push("two"); break;
    case 3: out.push("three"); break;
    default: out.push("other");
  }
  return out.join(",");
}
assert.sameValue(f(1), "one,two", "fall-through from case 1 to 2");
assert.sameValue(f(2), "two");
assert.sameValue(f(9), "other");
function g(s) { switch (s) { case "a": return 1; case "b": return 2; default: return 0; } }
assert.sameValue(g("b"), 2);
