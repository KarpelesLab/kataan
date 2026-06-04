/*---
description: Destructuring and defaults in function parameters
esid: sec-function-definitions
---*/
function point({ x, y }) { return x + "," + y; }
assert.sameValue(point({ x: 1, y: 2 }), "1,2");
function withDefaults({ a = 10, b = 20 } = {}) { return a + b; }
assert.sameValue(withDefaults(), 30);
assert.sameValue(withDefaults({ a: 5 }), 25);
function arr([first, second, ...rest]) { return first + ":" + second + ":" + rest.length; }
assert.sameValue(arr([1, 2, 3, 4]), "1:2:2");
function mixed(a, { b, c = 9 }, ...d) { return a + b + c + d.length; }
assert.sameValue(mixed(1, { b: 2 }, 7, 8), 14);
