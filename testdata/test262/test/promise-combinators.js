/*---
description: Promise.all, race, resolve, reject in the eager model
esid: sec-promise.all
flags: [async]
---*/
async function main() {
  var all = await Promise.all([Promise.resolve(1), Promise.resolve(2), Promise.resolve(3)]);
  assert.sameValue(all.join(","), "1,2,3", "Promise.all collects in order");
  var first = await Promise.race([Promise.resolve("fast"), Promise.resolve("slow")]);
  assert.sameValue(first, "fast");
  var v = await Promise.resolve(42);
  assert.sameValue(v, 42);
  var chained = await Promise.resolve(10).then(function (x) { return x * 2; });
  assert.sameValue(chained, 20, "then transforms");
  var caught = await Promise.reject("err").catch(function (e) { return "caught:" + e; });
  assert.sameValue(caught, "caught:err");
}
main();
