/*---
description: Optional catch binding and finally always running
esid: sec-try-statement
---*/
var ran = "";
function f() {
  try {
    throw new Error("x");
  } catch {
    ran += "caught";
  } finally {
    ran += "-finally";
  }
  return ran;
}
assert.sameValue(f(), "caught-finally");
