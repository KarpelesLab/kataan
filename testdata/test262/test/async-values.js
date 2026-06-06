/*---
description: async/await produces correct values, chaining, and error handling
features: [async-functions]
---*/
// Assertions run inside microtask callbacks; the harness drains the queue, and a
// throw in any callback fails the test. (This pins async *values*, not the
// eager-vs-deferred interleaving, which is a known limitation.)
async function val() { return 42; }
async function chain() { var a = await val(); var b = await Promise.resolve(a + 1); return a + b; }
async function caught() { try { await Promise.reject(new Error("x")); return "no"; } catch (e) { return "caught:" + e.message; } }
async function rethrow() { throw new Error("boom"); }

val().then(function (r) { assert.sameValue(r, 42, "async return value"); });
chain().then(function (r) { assert.sameValue(r, 85, "awaited chain"); });
caught().then(function (r) { assert.sameValue(r, "caught:x", "await rejection caught"); });
rethrow().then(
  function () { assert.sameValue(true, false, "rethrow should reject"); },
  function (e) { assert.sameValue(e.message, "boom", "throw becomes rejection"); }
);
Promise.all([val(), Promise.resolve(7)]).then(function (a) {
  assert.sameValue(a.join(","), "42,7", "Promise.all over async results");
});
assert.sameValue(val() instanceof Promise, true, "async function returns a Promise");
