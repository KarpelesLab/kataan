/*---
description: Optional catch binding (catch without a parameter)
esid: sec-try-statement
---*/
var ran = false;
try { throw new Error("x"); } catch { ran = true; }
assert.sameValue(ran, true, "catch without binding still runs");
function safe(fn, fallback) {
  try { return fn(); } catch { return fallback; }
}
assert.sameValue(safe(function () { throw 1; }, "caught"), "caught");
assert.sameValue(safe(function () { return "ok"; }, "fb"), "ok");
