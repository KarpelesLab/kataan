/*---
description: Function declaration hoisting and var hoisting
esid: sec-hoisting
---*/
assert.sameValue(hoisted(), "yes", "function declarations are hoisted");
function hoisted() { return "yes"; }

assert.sameValue(typeof laterVar, "undefined", "var is hoisted but undefined");
var laterVar = 5;
assert.sameValue(laterVar, 5);

function scope() {
  var inner = before;
  var before = "set";
  return inner;
}
assert.sameValue(scope(), undefined, "var read before assignment is undefined");
