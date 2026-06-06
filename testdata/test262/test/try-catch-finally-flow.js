/*---
description: try/catch/finally control flow, including finally overriding return
esid: sec-try-statement
---*/
function risky(x) {
  try {
    if (x < 0) throw new Error("neg");
    return "ok:" + x;
  } catch (e) {
    return "caught:" + e.message;
  }
}
assert.sameValue(risky(5), "ok:5", "no throw");
assert.sameValue(risky(-1), "caught:neg", "throw is caught with message");

// `finally` runs on both normal and exceptional paths.
var log = "";
function withFinally(fail) {
  try {
    if (fail) throw new Error("x");
    return "try";
  } catch (e) {
    return "catch";
  } finally {
    log += "F";
  }
}
assert.sameValue(withFinally(false), "try", "normal path return");
assert.sameValue(withFinally(true), "catch", "exception path return");
assert.sameValue(log, "FF", "finally ran on both paths");

// A `finally` return overrides a `try` return.
function override() {
  try { return "a"; } finally { return "b"; }
}
assert.sameValue(override(), "b", "finally return wins");
