/*---
description: finally runs on (and can override) abrupt completion of try/catch
esid: sec-try-statement
---*/
// finally's return overrides the try's return.
function a() { try { return 1; } finally { return 2; } }
assert.sameValue(a(), 2, "finally return overrides");

// finally runs on a normal return.
function b() { var log = []; try { log.push("t"); return log; } finally { log.push("f"); } }
assert.sameValue(b().join(","), "t,f", "finally runs before returning");

// finally runs when break exits the try.
function d() {
  var r = "";
  for (var i = 0; i < 3; i++) {
    try { if (i === 1) break; r += "t" + i; } finally { r += "f" + i; }
  }
  return r;
}
assert.sameValue(d(), "t0f0f1", "finally runs on break");

// finally runs when continue exits the try.
function e() {
  var r = "";
  for (var i = 0; i < 3; i++) {
    try { if (i === 1) continue; r += "t" + i; } finally { r += "f" + i; }
  }
  return r;
}
assert.sameValue(e(), "t0f0f1t2f2", "finally runs on continue");

// finally's break overrides the try's return.
function f() {
  for (var i = 0; i < 3; i++) { try { return "ret"; } finally { break; } }
  return "broke";
}
assert.sameValue(f(), "broke", "finally break overrides return");

// A throw in catch propagates after finally runs.
function m() {
  var r = "";
  try { try { throw "1"; } catch (x) { throw "2"; } finally { r += "f"; } }
  catch (x) { r += "c" + x; }
  return r;
}
assert.sameValue(m(), "fc2", "finally runs, then the rethrow propagates");

// A plain finally (no abrupt exit) still works.
function g() { var r = ""; try { r += "t"; } finally { r += "f"; } return r; }
assert.sameValue(g(), "tf", "normal finally");
