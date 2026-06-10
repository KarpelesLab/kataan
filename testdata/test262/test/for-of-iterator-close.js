/*---
description: for-of over a user iterator is lazy and runs IteratorClose (return) on early exit
features: [Symbol.iterator]
---*/
// break calls the iterator's return().
var closed = false;
var it = {
  [Symbol.iterator]() {
    var i = 0;
    return { next() { return i < 10 ? { value: i++, done: false } : { done: true }; },
             return() { closed = true; return { done: true }; } };
  },
};
for (var x of it) { if (x === 2) break; }
assert.sameValue(closed, true, "break runs return()");

// An infinite iterator can be cut short by break (lazy pull).
var inf = { [Symbol.iterator]() { var i = 0; return { next() { return { value: i++, done: false }; } }; } };
var sum = 0;
for (var y of inf) { if (y >= 5) break; sum += y; }
assert.sameValue(sum, 10, "0+1+2+3+4");

// throw inside the loop also runs return().
var tClosed = false;
var tit = { [Symbol.iterator]() { return { next() { return { value: 1, done: false }; },
                                           return() { tClosed = true; return { done: true }; } }; } };
try { for (var z of tit) { throw new Error("stop"); } } catch (e) {}
assert.sameValue(tClosed, true, "throw runs return()");

// Natural completion does NOT call return(); continue does not close.
var nClosed = false;
var nit = { [Symbol.iterator]() { var i = 0; return { next() { return i < 2 ? { value: i++, done: false } : { done: true }; },
                                                       return() { nClosed = true; return { done: true }; } }; } };
for (var q of nit) { continue; }
assert.sameValue(nClosed, false, "natural end / continue does not close");

// Built-in iterables and generators are unaffected.
assert.sameValue([1, 2, 3].join(","), "1,2,3", "array");
assert.sameValue([...(function* () { yield 1; yield 2; })()].join(","), "1,2", "generator value");
class R { *[Symbol.iterator]() { yield 10; yield 20; } }
assert.sameValue([...new R()].join(","), "10,20", "generator [Symbol.iterator]");
