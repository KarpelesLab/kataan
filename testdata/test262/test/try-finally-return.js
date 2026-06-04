/*---
description: try/finally interaction with return and exceptions
esid: sec-try-statement
---*/
function a() {
  try { return "try"; }
  finally { /* runs but does not override */ }
}
assert.sameValue(a(), "try");

function b() {
  try { return "try"; }
  finally { return "finally"; }
}
assert.sameValue(b(), "finally", "finally return overrides try return");

function c() {
  var log = "";
  try { log += "t"; throw new Error("x"); }
  catch (e) { log += "c"; return log; }
  finally { log += "f"; }
}
assert.sameValue(c(), "tc");
