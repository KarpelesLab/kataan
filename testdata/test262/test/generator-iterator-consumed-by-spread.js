/*---
description: draining a generator iterator (spread/for-of/Array.from) consumes it; a later next() is done
esid: sec-generator.prototype.next
features: [generators]
---*/
function* g() { yield 1; yield 2; }

// After a full spread, the same iterator is exhausted.
var it = g();
var arr = [].concat(...[it].map(function (x) { return Array.prototype.slice.call(x); }));
// (use a simple drain via spread)
var it1 = g();
var a1 = [...it1];
assert.sameValue(a1.join(","), "1,2", "spread yields all values");
assert.sameValue(it1.next().done, true, "iterator is done after spread");

// for-of also consumes it.
var it2 = g();
for (var x of it2) { /* drain */ }
assert.sameValue(it2.next().done, true, "iterator is done after for-of");

// Array.from consumes it.
var it3 = g();
Array.from(it3);
assert.sameValue(it3.next().done, true, "iterator is done after Array.from");

// Spreading the same iterator twice gives the values once, then nothing.
var it4 = g();
assert.sameValue([...it4].join(","), "1,2", "first spread");
assert.sameValue([...it4].length, 0, "second spread is empty");

// A partial next() then spread respects the position, and is then done.
var it5 = g();
it5.next();
assert.sameValue([...it5].join(","), "2", "spread resumes after one next()");
assert.sameValue(it5.next().done, true, "done after resumed spread");

// next()-based draining and fresh generators are unaffected.
var it6 = g();
assert.sameValue(it6.next().value, 1, "next 1");
assert.sameValue(it6.next().value, 2, "next 2");
assert.sameValue(it6.next().done, true, "next done");
assert.sameValue([...g()].join(","), "1,2", "a fresh generator still yields all");
