/*---
description: async functions return promises and await resolves them
esid: sec-async-function-definitions
flags: [async]
---*/
async function getValue() { return 42; }
async function main() {
  var v = await getValue();
  assert.sameValue(v, 42);
}
main();
