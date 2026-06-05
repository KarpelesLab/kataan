/*---
description: try/finally interaction with control flow
esid: sec-try-statement
---*/
function withFinally() {
  try { return "try"; } finally { /* runs but does not override */ }
}
assert.sameValue(withFinally(), "try");
function finallyOverrides() {
  try { return "try"; } finally { return "finally"; }
}
assert.sameValue(finallyOverrides(), "finally", "finally return wins");
var log = [];
function order() {
  try { log.push("try"); return 1; }
  finally { log.push("finally"); }
}
order();
assert.sameValue(log.join(","), "try,finally", "finally runs after try return");
function loopFinally() {
  var results = [];
  for (var i = 0; i < 3; i++) {
    try { if (i === 1) continue; results.push("body" + i); }
    finally { results.push("fin" + i); }
  }
  return results.join(",");
}
assert.sameValue(loopFinally(), "body0,fin0,fin1,body2,fin2", "finally on continue");
function nestedFinally() {
  var out = [];
  try { try { throw new Error("x"); } finally { out.push("inner"); } }
  catch (e) { out.push("caught"); }
  finally { out.push("outer"); }
  return out.join(",");
}
assert.sameValue(nestedFinally(), "inner,caught,outer", "nested finally order");
function breakFinally() {
  var r = [];
  for (var i = 0; i < 5; i++) {
    try { if (i === 2) break; r.push(i); } finally { r.push("f"); }
  }
  return r.join(",");
}
assert.sameValue(breakFinally(), "0,f,1,f,f", "finally on break");
