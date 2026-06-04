/*---
description: Promise.all, Promise.race, and Promise.allSettled
esid: sec-promise.all
flags: [async]
---*/
async function main() {
  var all = await Promise.all([Promise.resolve(1), Promise.resolve(2), 3]);
  assert.sameValue(all.join(","), "1,2,3");

  var first = await Promise.race([Promise.resolve("a"), Promise.resolve("b")]);
  assert.sameValue(first, "a");

  var settled = await Promise.allSettled([Promise.resolve(10), Promise.reject("err")]);
  assert.sameValue(settled[0].status, "fulfilled");
  assert.sameValue(settled[0].value, 10);
  assert.sameValue(settled[1].status, "rejected");
  assert.sameValue(settled[1].reason, "err");

  var rejected = false;
  try {
    await Promise.all([Promise.resolve(1), Promise.reject("boom")]);
  } catch (e) {
    rejected = (e === "boom");
  }
  assert.sameValue(rejected, true, "Promise.all rejects on first rejection");
}
main();
