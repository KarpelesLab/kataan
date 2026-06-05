/*---
description: async/await execution ordering
esid: sec-async-function-definitions
flags: [async]
---*/
async function main() {
  var order = [];
  async function step(label) { order.push(label); return label; }
  await step("a");
  await step("b");
  await step("c");
  assert.sameValue(order.join(","), "a,b,c", "sequential awaits");
  var results = await Promise.all([step("x"), step("y"), step("z")]);
  assert.sameValue(results.join(","), "x,y,z");
  async function compute(n) { return n * 2; }
  var doubled = await compute(21);
  assert.sameValue(doubled, 42);
  var chained = await compute(await compute(5));
  assert.sameValue(chained, 20, "nested await");
  var sum = 0;
  for (var i = 1; i <= 3; i++) sum += await compute(i);
  assert.sameValue(sum, 12, "await in loop");
}
main();
