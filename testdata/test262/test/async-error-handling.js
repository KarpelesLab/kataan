/*---
description: async/await error handling with try/catch
esid: sec-async-function-definitions
flags: [async]
---*/
async function main() {
  async function fails() { throw new Error("async failure"); }
  var caught = false;
  try { await fails(); } catch (e) { caught = e.message === "async failure"; }
  assert.sameValue(caught, true, "await rejection caught");
  async function rejects() { return Promise.reject("rejected value"); }
  var caught2 = false;
  try { await rejects(); } catch (e) { caught2 = e === "rejected value"; }
  assert.sameValue(caught2, true);
  var recovered = await rejects().catch(function (e) { return "recovered: " + e; });
  assert.sameValue(recovered, "recovered: rejected value");
  async function withFinally() {
    var order = [];
    try { order.push("try"); throw new Error("x"); }
    catch (e) { order.push("catch"); }
    finally { order.push("finally"); }
    return order.join(",");
  }
  assert.sameValue(await withFinally(), "try,catch,finally");
  async function chainCatch() {
    return await Promise.resolve(5).then(function (x) { return x * 2; }).then(function (x) { return x + 1; });
  }
  assert.sameValue(await chainCatch(), 11);
}
main();
