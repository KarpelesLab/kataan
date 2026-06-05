/*---
description: ES2025 iterator helpers (map/filter/take/drop/toArray/reduce/etc.)
esid: sec-iterator-helper-objects
---*/
function gen() { return (function* () { yield 1; yield 2; yield 3; yield 4; })(); }
assert.sameValue([...gen().map(function (x) { return x * 10; })].join(","), "10,20,30,40", "map");
assert.sameValue([...gen().filter(function (x) { return x % 2 === 0; })].join(","), "2,4", "filter");
assert.sameValue([...gen().take(2)].join(","), "1,2", "take");
assert.sameValue([...gen().drop(2)].join(","), "3,4", "drop");
assert.sameValue(gen().toArray().join(","), "1,2,3,4", "toArray");
assert.sameValue(gen().reduce(function (a, b) { return a + b; }, 0), 10, "reduce with initial");
assert.sameValue(gen().reduce(function (a, b) { return a + b; }), 10, "reduce without initial");
assert.sameValue(gen().some(function (x) { return x > 3; }), true, "some");
assert.sameValue(gen().some(function (x) { return x > 9; }), false, "some false");
assert.sameValue(gen().every(function (x) { return x > 0; }), true, "every");
assert.sameValue(gen().every(function (x) { return x > 2; }), false, "every false");
assert.sameValue(gen().find(function (x) { return x > 2; }), 3, "find");
// Chaining helpers.
assert.sameValue([...gen().map(function (x) { return x * 2; }).filter(function (x) { return x > 4; })].join(","), "6,8", "map then filter");
assert.sameValue([...gen().drop(1).take(2)].join(","), "2,3", "drop then take");
// flatMap.
assert.sameValue([...gen().flatMap(function (x) { return [x, -x]; })].slice(0, 4).join(","), "1,-1,2,-2", "flatMap");
// forEach.
var sum = 0;
gen().forEach(function (x) { sum += x; });
assert.sameValue(sum, 10, "forEach");
// Resuming partway then mapping the rest.
var it = gen();
it.next();
assert.sameValue(it.map(function (x) { return x; }).toArray().join(","), "2,3,4", "helper over the remaining values");
