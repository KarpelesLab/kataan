/*---
description: Iterator protocol with for-of and spread
esid: sec-iteration
---*/
var iterable = {
  [Symbol.iterator]() {
    var i = 0;
    return { next() { return i < 3 ? { value: i++, done: false } : { value: undefined, done: true }; } };
  }
};
assert.sameValue([...iterable].join(","), "0,1,2", "custom iterator spread");
var collected = [];
for (var x of iterable) collected.push(x);
assert.sameValue(collected.join(","), "0,1,2", "custom iterator for-of");
assert.sameValue(Array.from(iterable).length, 3);
var arrIter = [10, 20, 30][Symbol.iterator]();
assert.sameValue(arrIter.next().value, 10);
var strIter = "abc"[Symbol.iterator]();
assert.sameValue(strIter.next().value, "a");
var mapIter = new Map([["k", "v"]])[Symbol.iterator]();
var mfirst = mapIter.next().value;
assert.sameValue(mfirst[0], "k");
assert.sameValue(mfirst[1], "v");
var setIter = new Set([1, 2])[Symbol.iterator]();
assert.sameValue(setIter.next().value, 1);
assert.sameValue([...[1, 2, 3].entries()].length, 3);
assert.sameValue([...[1, 2, 3].keys()].join(","), "0,1,2");
