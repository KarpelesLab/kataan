/*---
description: for-await-of over promises, values, and async generators
esid: sec-for-in-and-for-of-statements
flags: [async]
---*/
async function main() {
  // Awaits each promise in turn.
  var sum = 0;
  for await (const x of [Promise.resolve(1), Promise.resolve(2), Promise.resolve(3)]) sum += x;
  assert.sameValue(sum, 6, "for-await over promises");
  // Plain values pass through.
  var s2 = 0;
  for await (const x of [10, 20, 30]) s2 += x;
  assert.sameValue(s2, 60, "for-await over plain values");
  // Async generators.
  async function* gen() { yield 1; yield 2; yield 3; }
  var collected = [];
  for await (const x of gen()) collected.push(x);
  assert.sameValue(collected.join(","), "1,2,3", "for-await over an async generator");
  // await inside the generator body.
  async function* gen2() { yield await Promise.resolve(10); yield 20; }
  var c2 = [];
  for await (const x of gen2()) c2.push(x);
  assert.sameValue(c2.join(","), "10,20", "async generator with internal await");
  // A break works.
  var seen = [];
  for await (const x of [1, 2, 3, 4]) { if (x === 3) break; seen.push(x); }
  assert.sameValue(seen.join(","), "1,2", "break in for-await");
  // Symbol.asyncIterator exists.
  assert.sameValue(typeof Symbol.asyncIterator, "symbol", "Symbol.asyncIterator");
}
main();
