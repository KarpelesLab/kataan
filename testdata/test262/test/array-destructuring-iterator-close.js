/*---
description: array destructuring without a rest element pulls lazily and closes the iterator
esid: sec-runtime-semantics-iteratordestructuringassignmentevaluation
---*/
// An infinite iterator can be destructured: only the needed values are pulled, and
// the iterator is closed (return() called).
var closed = false;
var infinite = {
  [Symbol.iterator]() {
    var i = 0;
    return {
      next() { return { value: i++, done: false }; },
      return() { closed = true; return { done: true }; },
    };
  },
};
var [a, b] = infinite;
assert.sameValue(a, 0, "first");
assert.sameValue(b, 1, "second");
assert.sameValue(closed, true, "iterator closed via return()");

// A generator's finally runs when destructuring stops early.
var ran = false;
function* g() { try { yield 1; yield 2; yield 3; } finally { ran = true; } }
var [first] = g();
assert.sameValue(first, 1, "generator first value");
assert.sameValue(ran, true, "generator return() ran finally");

// An object whose Symbol.iterator is a generator method destructures correctly,
// reading `this` exactly once.
var mixed = { value: 42, *[Symbol.iterator]() { yield this.value; yield this.value * 2; } };
var [m1, m2] = mixed;
assert.sameValue(m1, 42, "generator method this #1");
assert.sameValue(m2, 84, "generator method this #2");

// Ordinary cases are unchanged.
var [x, y, z] = [10, 20, 30];
assert.sameValue(x + "," + y + "," + z, "10,20,30", "array");
var [p, q = 99] = [1];
assert.sameValue(p + "," + q, "1,99", "default for missing");
var [head, ...tail] = [1, 2, 3, 4];
assert.sameValue(head + ":" + tail.join(","), "1:2,3,4", "rest collects remainder");
var [c1, c2] = "hi";
assert.sameValue(c1 + c2, "hi", "string");
var [s1, s2] = new Set([7, 8, 9]);
assert.sameValue(s1 + "," + s2, "7,8", "set");
