/*---
description: Custom iterables via Symbol.iterator
esid: sec-symbol.iterator
---*/
var range = {
  from: 1, to: 5,
  [Symbol.iterator]() {
    var current = this.from, last = this.to;
    return {
      next() {
        return current <= last ? { value: current++, done: false } : { value: undefined, done: true };
      }
    };
  }
};
var collected = [];
for (var n of range) collected.push(n);
assert.sameValue(collected.join(","), "1,2,3,4,5", "for-of custom iterable");
assert.sameValue([...range].join(","), "1,2,3,4,5", "spread custom iterable");
assert.sameValue(Array.from(range).length, 5, "Array.from custom iterable");
var fib = {
  [Symbol.iterator]() {
    var a = 0, b = 1, count = 0;
    return {
      next() {
        if (count++ >= 6) return { done: true };
        var v = a; var t = a + b; a = b; b = t;
        return { value: v, done: false };
      }
    };
  }
};
assert.sameValue([...fib].join(","), "0,1,1,2,3,5");
