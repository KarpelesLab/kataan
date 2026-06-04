/*---
description: Default and rest parameters
esid: sec-function-definitions
---*/
function f(a, b = 5, ...more) {
  return a + b + more.length;
}
assert.sameValue(f(1), 6, "default used");
assert.sameValue(f(1, 2), 3, "default overridden");
assert.sameValue(f(1, 2, 9, 9, 9), 6, "rest collects extras");
