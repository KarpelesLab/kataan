/*---
description: try/catch/finally control flow and binding
esid: sec-try-statement
---*/
var log = [];
function run() {
  try {
    log.push("try");
    throw new Error("boom");
  } catch (e) {
    log.push("catch:" + e.message);
    return "caught";
  } finally {
    log.push("finally");
  }
}
assert.sameValue(run(), "caught");
assert.sameValue(log.join(","), "try,catch:boom,finally");
