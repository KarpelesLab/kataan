/*---
description: async/await chaining with multiple awaits and Promise.resolve
esid: sec-async-function-definitions
flags: [async]
---*/
async function double(x) { return x * 2; }
async function main() {
  var a = await double(5);
  assert.sameValue(a, 10, "await an async function");
  var b = await Promise.resolve(7);
  assert.sameValue(b, 7, "await a resolved promise");
  var c = (await double(a)) + (await double(b));
  assert.sameValue(c, 34, "sum of two awaited values");
}
main();
