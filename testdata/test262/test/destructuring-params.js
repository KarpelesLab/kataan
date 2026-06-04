/*---
description: Destructuring in function parameters with defaults
esid: sec-function-definitions
---*/
function dist({ x = 0, y = 0 }) { return x + y; }
assert.sameValue(dist({ x: 3, y: 4 }), 7);
assert.sameValue(dist({ x: 5 }), 5, "param default");
function head([first, ...rest]) { return first + ":" + rest.length; }
assert.sameValue(head([10, 20, 30]), "10:2");
