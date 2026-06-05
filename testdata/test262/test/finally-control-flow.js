/*---
description: finally interacts with break, continue, and return
esid: sec-try-statement
---*/
function loopBreak() {
  var log = [];
  for (var i = 0; i < 5; i++) {
    try { if (i === 2) break; log.push(i); }
    finally { log.push("f" + i); }
  }
  return log.join(",");
}
assert.sameValue(loopBreak(), "0,f0,1,f1,f2", "finally runs on break");
function loopContinue() {
  var log = [];
  for (var i = 0; i < 3; i++) {
    try { if (i === 1) continue; log.push(i); }
    finally { log.push("f" + i); }
  }
  return log.join(",");
}
assert.sameValue(loopContinue(), "0,f0,f1,2,f2", "finally runs on continue");
function returnInFinally() {
  try { return "try"; }
  finally { return "finally"; }
}
assert.sameValue(returnInFinally(), "finally");
