/*---
description: setTimeout / clearTimeout / queueMicrotask event-loop ordering
features: [host-timers]
---*/
// The whole script runs first; then microtasks; then macrotasks (setTimeout),
// ordered by delay. A drained log is asserted from the last (latest) timer.
var log = [];
setTimeout(function () { log.push("t1"); }, 0);
setTimeout(function () { log.push("t2"); }, 0);
queueMicrotask(function () { log.push("micro"); });
Promise.resolve().then(function () { log.push("promise"); });
log.push("sync");
setTimeout(function () { log.push("late"); }, 100);
setTimeout(function () { log.push("early"); }, 10);
setTimeout(function (a, b) { log.push("args:" + a + b); }, 0, "x", "y");
var cancelled = setTimeout(function () { log.push("CANCELLED"); }, 0);
clearTimeout(cancelled);
setTimeout(function () { log.push("outer"); setTimeout(function () { log.push("inner"); }, 0); }, 0);

setTimeout(function () {
  // Microtasks (sync-pushed then micro/promise) precede every macrotask.
  assert.sameValue(log[0], "sync", "synchronous code first");
  assert.sameValue(log.indexOf("micro") < log.indexOf("t1"), true, "microtasks before macrotasks");
  assert.sameValue(log.indexOf("promise") < log.indexOf("t1"), true, "promise reaction before timers");
  // Zero-delay timers run in insertion order.
  assert.sameValue(log.indexOf("t1") < log.indexOf("t2"), true, "t1 before t2");
  assert.sameValue(log.indexOf("args:xy") > log.indexOf("t2"), true, "args timer after t2");
  // A nested zero-delay timer (inner) still beats a 10ms one (early).
  assert.sameValue(log.indexOf("outer") < log.indexOf("inner"), true, "outer before its nested inner");
  assert.sameValue(log.indexOf("inner") < log.indexOf("early"), true, "nested zero-delay before 10ms");
  // Delay ordering, and the cancelled timer never ran.
  assert.sameValue(log.indexOf("early") < log.indexOf("late"), true, "10ms before 100ms");
  assert.sameValue(log.indexOf("CANCELLED"), -1, "clearTimeout cancelled the timer");
  // Argument forwarding worked.
  assert.sameValue(log.indexOf("args:xy") >= 0, true, "extra args forwarded");
}, 200);
