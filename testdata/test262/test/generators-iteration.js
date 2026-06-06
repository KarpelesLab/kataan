/*---
description: Generator functions yield values consumed by for-of and spread
features: [generators]
---*/
function* gen() { yield 1; yield 2; yield 3; }

// for-of drives the generator to completion.
var out = "";
for (var v of gen()) { out += v; }
assert.sameValue(out, "123", "for-of over a generator");

// Spread collects a generator's yields into an array.
var arr = [...gen()];
assert.sameValue(arr.length, 3, "spread length");
assert.sameValue(arr[0] + arr[1] + arr[2], 6, "spread values");

// A generator with a return value terminates iteration (return is not yielded).
function* withReturn() { yield 10; return 99; yield 20; }
var seen = [];
for (var x of withReturn()) { seen.push(x); }
assert.sameValue(seen.length, 1, "values after return are not produced");
assert.sameValue(seen[0], 10, "first yield");

// Manual .next() exposes done/value.
var it = gen();
assert.sameValue(it.next().value, 1, "next 1");
assert.sameValue(it.next().value, 2, "next 2");
assert.sameValue(it.next().done, false, "third not done yet");
assert.sameValue(it.next().done, true, "exhausted");
