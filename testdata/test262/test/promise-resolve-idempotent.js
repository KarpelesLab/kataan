/*---
description: Promise.resolve returns the argument unchanged when it is already a promise
esid: sec-promise.resolve
features: [Promise]
---*/
var p = Promise.resolve(7);
assert.sameValue(Promise.resolve(p), p, "Promise.resolve(promise) is the same promise");

// A non-promise value is wrapped in a fresh promise each call.
var w = Promise.resolve(5);
assert.sameValue(w instanceof Promise, true, "wraps a plain value");
assert.notSameValue(Promise.resolve(5), w, "distinct wrappers for plain values");

// A thenable (not a real promise) is NOT passed through; it is adopted into a new promise.
var thenable = { then(res) { res(99); } };
assert.notSameValue(Promise.resolve(thenable), thenable, "thenable is wrapped, not returned");
assert.sameValue(Promise.resolve(thenable) instanceof Promise, true, "thenable wrapped in a Promise");

// A rejected promise is also returned as-is.
var rej = Promise.reject("x");
rej.catch(function () {}); // mark handled
assert.sameValue(Promise.resolve(rej), rej, "Promise.resolve(rejectedPromise) is the same promise");
