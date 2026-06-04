/*---
description: Switch with expressions, strict matching, and default in the middle
esid: sec-switch-statement
---*/
function classify(x) {
  switch (true) {
    case x < 0: return "negative";
    case x === 0: return "zero";
    case x < 10: return "small";
    default: return "large";
  }
}
assert.sameValue(classify(-5), "negative");
assert.sameValue(classify(0), "zero");
assert.sameValue(classify(5), "small");
assert.sameValue(classify(100), "large");
function t(x) { switch (x) { case 1: case 2: return "low"; case 3: return "mid"; default: return "?"; } }
assert.sameValue(t(1), "low", "case grouping");
assert.sameValue(t(2), "low");
assert.sameValue(t(3), "mid");
