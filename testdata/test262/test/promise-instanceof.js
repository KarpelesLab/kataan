/*---
description: Promise objects are instances of Promise
features: [Promise]
---*/
assert.sameValue(Promise.resolve(1) instanceof Promise, true, "Promise.resolve result");
assert.sameValue(new Promise(function (res) { res(1); }) instanceof Promise, true, "new Promise");
assert.sameValue(Promise.reject(1).catch(function () {}) instanceof Promise, true, "catch result");
assert.sameValue(Promise.all([]) instanceof Promise, true, "Promise.all result");
assert.sameValue(Promise.race([Promise.resolve(1)]) instanceof Promise, true, "Promise.race result");
assert.sameValue(Promise.resolve(1).then(function () {}) instanceof Promise, true, "then result");

// Non-promises are not instances.
assert.sameValue(({}) instanceof Promise, false, "plain object");
assert.sameValue([] instanceof Promise, false, "array");
assert.sameValue((function () {}) instanceof Promise, false, "function");
