/*---
description: Promise.withResolvers returns { promise, resolve, reject }
esid: sec-promise.withresolvers
flags: [async]
---*/
async function main() {
  var d = Promise.withResolvers();
  assert.sameValue(typeof d.promise, "object", "has a promise");
  d.resolve(42);
  var v = await d.promise;
  assert.sameValue(v, 42, "resolved value");
  var d2 = Promise.withResolvers();
  d2.reject("failure");
  var caught = await d2.promise.catch(function (e) { return "caught:" + e; });
  assert.sameValue(caught, "caught:failure", "rejected value");
  var { promise, resolve } = Promise.withResolvers();
  resolve("destructured");
  assert.sameValue(await promise, "destructured", "destructured resolvers");
}
main();
