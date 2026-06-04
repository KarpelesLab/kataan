/*---
description: switch statement with fall-through and default
esid: sec-switch-statement
---*/
function classify(n) {
  var out = "";
  switch (n) {
    case 1:
    case 2:
      out = "small";
      break;
    case 3:
      out = "three";
      break;
    default:
      out = "big";
  }
  return out;
}
assert.sameValue(classify(1), "small");
assert.sameValue(classify(2), "small", "fall-through");
assert.sameValue(classify(3), "three");
assert.sameValue(classify(9), "big", "default");
