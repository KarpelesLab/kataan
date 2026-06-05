/*---
description: Promise.allSettled, Promise.any, and rejection handling
esid: sec-promise.allsettled
flags: [async]
---*/
async function main() {
  var settled = await Promise.allSettled([
    Promise.resolve(1),
    Promise.reject("err"),
    Promise.resolve(3)
  ]);
  assert.sameValue(settled.length, 3);
  assert.sameValue(settled[0].status, "fulfilled");
  assert.sameValue(settled[0].value, 1);
  assert.sameValue(settled[1].status, "rejected");
  assert.sameValue(settled[1].reason, "err");
  assert.sameValue(settled[2].value, 3);
  var any = await Promise.any([
    Promise.reject("a"),
    Promise.resolve("winner"),
    Promise.reject("b")
  ]);
  assert.sameValue(any, "winner", "Promise.any returns first fulfilled");
  var allResult = await Promise.all([Promise.resolve(1), Promise.resolve(2)]);
  assert.sameValue(allResult.join(","), "1,2");
}
main();
