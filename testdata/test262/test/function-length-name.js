/*---
description: Function length (params before default/rest) and name
esid: sec-function-instances-length
---*/
function add(a, b) { return a + b; }
assert.sameValue(add.length, 2);
assert.sameValue(add.name, "add");

function withDefault(a, b, c) {}
assert.sameValue(withDefault.length, 3);

function defStop(a, b, c) { return a; }
assert.sameValue(defStop.length, 3);

function rest(a, b) { return a; }
assert.sameValue(rest.length, 2);

var named = function namedExpr() {};
assert.sameValue(named.name, "namedExpr");

function noArgs() {}
assert.sameValue(noArgs.length, 0);
