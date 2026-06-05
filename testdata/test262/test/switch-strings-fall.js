/*---
description: switch with string cases, fall-through groups, and default
esid: sec-switch-statement
---*/
function classify(s) {
  switch (s) {
    case "a":
    case "e":
    case "i":
    case "o":
    case "u": return "vowel";
    case " ": return "space";
    default: return "consonant";
  }
}
assert.sameValue(classify("a"), "vowel");
assert.sameValue(classify("e"), "vowel", "case grouping");
assert.sameValue(classify("b"), "consonant");
assert.sameValue(classify(" "), "space");
function priority(level) {
  var result = "";
  switch (level) {
    case 3: result += "high ";
    case 2: result += "medium ";
    case 1: result += "low"; break;
    default: result = "none";
  }
  return result;
}
assert.sameValue(priority(3), "high medium low", "fall-through accumulates");
assert.sameValue(priority(1), "low");
assert.sameValue(priority(0), "none");
