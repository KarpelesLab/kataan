/*---
description: Promise.prototype.finally passes the value through; Promise.resolve adopts thenables
esid: sec-promise.prototype.finally
flags: [async]
---*/
async function main() {
  // finally passes the fulfilled value through (callback return is ignored).
  assert.sameValue(await Promise.resolve(5).finally(function () { return 99; }), 5, "finally passes value");
  var sideEffect = 0;
  var v = await Promise.resolve(10).finally(function () { sideEffect = 1; });
  assert.sameValue(v, 10, "value after finally side-effect");
  assert.sameValue(sideEffect, 1, "finally callback ran");
  // finally re-throws a rejection.
  var caught = await Promise.reject("err").finally(function () {}).catch(function (e) { return "caught:" + e; });
  assert.sameValue(caught, "caught:err", "finally re-throws rejection");
  // A throw in finally overrides the original value.
  var overridden = await Promise.resolve(1).finally(function () { throw "boom"; }).catch(function (e) { return "o:" + e; });
  assert.sameValue(overridden, "o:boom", "finally throw overrides");
  // Promise.resolve adopts a thenable.
  assert.sameValue(await Promise.resolve({ then: function (res) { res(42); } }), 42, "resolve adopts a thenable");
  // A thenable that rejects.
  var rej = await Promise.resolve({ then: function (res, rej) { rej("bad"); } }).catch(function (e) { return "r:" + e; });
  assert.sameValue(rej, "r:bad", "thenable rejection adopted");
  // finally in a chain.
  assert.sameValue(await Promise.resolve(3).then(function (x) { return x + 1; }).finally(function () {}).then(function (x) { return x * 10; }), 40, "finally in a chain");
}
main();
