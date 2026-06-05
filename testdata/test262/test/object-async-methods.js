/*---
description: Async methods in object literals (and async-as-property-name)
esid: sec-object-initializer
flags: [async]
---*/
async function main() {
  var api = { async getValue() { return 42; } };
  assert.sameValue(await api.getValue(), 42, "async method returns a resolved value");
  var chained = { async compute() { var x = await Promise.resolve(10); return x * 2; } };
  assert.sameValue(await chained.compute(), 20, "await inside an async method");
  var counter = { n: 0, async inc() { this.n++; return this.n; } };
  assert.sameValue(await counter.inc(), 1, "async method reads/writes this");
  assert.sameValue(counter.n, 1);
  var key = "run";
  var dynamic = { async [key]() { return "dyn"; } };
  assert.sameValue(await dynamic.run(), "dyn", "computed-key async method");
  // `async` is still usable as an ordinary property name.
  var named = { async: 5, get: 6, set: 7 };
  assert.sameValue(named.async, 5, "async as a property name");
  assert.sameValue(named.get + named.set, 13);
  var asyncVal = 9;
  var shorthand = { asyncVal };
  assert.sameValue(shorthand.asyncVal, 9, "shorthand still works");
  var all = await Promise.all([api.getValue(), chained.compute()]);
  assert.sameValue(all.join(","), "42,20", "Promise.all over async methods");
}
main();
