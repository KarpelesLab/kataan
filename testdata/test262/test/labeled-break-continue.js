/*---
description: Labeled break/continue target an outer loop; switch fallthrough
esid: sec-labelled-statements
---*/
// `break label` exits the labeled outer loop from an inner loop.
var hits = 0;
outer: for (var i = 0; i < 3; i++) {
  for (var j = 0; j < 3; j++) {
    if (i * 3 + j === 4) break outer;
    hits++;
  }
}
assert.sameValue(hits, 4, "break outer stops after 4 inner iterations");

// `continue label` resumes the labeled outer loop.
var sum = 0;
L: for (var a = 0; a < 3; a++) {
  for (var b = 0; b < 3; b++) {
    if (b === 1) continue L;
    sum += a * 10 + b;
  }
}
assert.sameValue(sum, 30, "continue L skips to the next outer iteration");

// switch fallthrough across empty cases, plus default.
function classify(x) {
  switch (x) {
    case 1:
    case 2:
      return "low";
    case 3:
      return "mid";
    default:
      return "high";
  }
}
assert.sameValue(classify(1), "low", "case 1 falls through to 2's body");
assert.sameValue(classify(2), "low", "case 2");
assert.sameValue(classify(3), "mid", "case 3");
assert.sameValue(classify(9), "high", "default");
