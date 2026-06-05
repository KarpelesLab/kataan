/*---
description: Destructuring in function parameters
esid: sec-function-definitions
---*/
function point({ x, y }) { return x + "," + y; }
assert.sameValue(point({ x: 1, y: 2 }), "1,2");
function first([a, b]) { return a + b; }
assert.sameValue(first([10, 20]), 30);
function withDefaults({ a = 1, b = 2 } = {}) { return a + b; }
assert.sameValue(withDefaults(), 3);
assert.sameValue(withDefaults({ a: 10 }), 12);
function mixed(x, { y, z }, ...rest) { return x + y + z + rest.length; }
assert.sameValue(mixed(1, { y: 2, z: 3 }, 4, 5), 1 + 2 + 3 + 2);
function nested({ a: { b } }) { return b; }
assert.sameValue(nested({ a: { b: 42 } }), 42);
function arrayInObj({ items: [first, second] }) { return first + second; }
assert.sameValue(arrayInObj({ items: [3, 4] }), 7);
function renamed({ x: localX, y: localY }) { return localX * localY; }
assert.sameValue(renamed({ x: 3, y: 4 }), 12);
