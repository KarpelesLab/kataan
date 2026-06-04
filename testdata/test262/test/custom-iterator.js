/*---
description: A custom Symbol.iterator drives for-of and spread
esid: sec-for-in-and-for-of-statements
---*/
var range = {
  from: 1, to: 4,
  [Symbol.iterator]() {
    var cur = this.from, last = this.to;
    return {
      next() {
        return cur <= last ? { value: cur++, done: false } : { value: undefined, done: true };
      }
    };
  }
};
var out = [];
for (var n of range) out.push(n);
assert.sameValue(out.join(","), "1,2,3,4");
assert.sameValue([...range].join(","), "1,2,3,4");

// A class implementing the iterator protocol.
class Counter {
  constructor(n) { this.n = n; }
  [Symbol.iterator]() {
    var i = 0, max = this.n;
    return { next() { return i < max ? { value: i++, done: false } : { value: undefined, done: true }; } };
  }
}
assert.sameValue([...new Counter(3)].join(","), "0,1,2");
