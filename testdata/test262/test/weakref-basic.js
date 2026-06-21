/*---
description: WeakRef.deref and FinalizationRegistry (bounded, no mid-run GC)
esid: sec-weak-ref-objects
---*/
var target = { value: 42 };
var ref = new WeakRef(target);
assert.sameValue(ref.deref(), target, "deref returns the target");
assert.sameValue(ref.deref().value, 42);
assert.sameValue(typeof WeakRef, "function");
var arr = [1, 2, 3];
var arrRef = new WeakRef(arr);
assert.sameValue(arrRef.deref().length, 3, "deref an array target");
var registry = new FinalizationRegistry(function () {});
registry.register(target, "held-value");
assert.sameValue(registry.unregister({}), false, "nothing registered to unregister");
assert.sameValue(typeof FinalizationRegistry, "function");
var ref2 = new WeakRef(target);
assert.sameValue(ref.deref() === ref2.deref(), true, "two refs to the same target");
