/*---
description: try/catch/finally control flow interactions
esid: sec-try-statement
---*/
function f1() {
  try { return "try"; }
  finally { /* runs but does not override */ }
}
assert.sameValue(f1(), "try");
function f2() {
  try { return "try"; }
  finally { return "finally"; }
}
assert.sameValue(f2(), "finally", "finally return overrides");
var order = [];
function f3() {
  try { order.push("try"); throw new Error("x"); }
  catch (e) { order.push("catch"); }
  finally { order.push("finally"); }
}
f3();
assert.sameValue(order.join(","), "try,catch,finally");
function f4() {
  for (var i = 0; i < 5; i++) {
    try { if (i === 2) break; }
    finally { order.push("f" + i); }
  }
}
order = [];
f4();
assert.sameValue(order.join(","), "f0,f1,f2", "finally runs on break");
var cleanup = [];
function withResource() {
  try { throw new Error("fail"); }
  catch (e) { return "handled"; }
  finally { cleanup.push("cleaned"); }
}
assert.sameValue(withResource(), "handled");
assert.sameValue(cleanup.length, 1);
