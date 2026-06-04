/*---
description: The arguments object in non-arrow functions; arrows inherit it
esid: sec-arguments-exotic-objects
---*/
function sum() {
  var total = 0;
  for (var i = 0; i < arguments.length; i++) total += arguments[i];
  return total;
}
assert.sameValue(sum(1, 2, 3, 4), 10);
assert.sameValue(sum(), 0);

function first() { return arguments[0]; }
assert.sameValue(first("a", "b"), "a");
assert.sameValue(arguments_length_of(1, 2, 3), 3);
function arguments_length_of() { return arguments.length; }

// An arrow inside a function sees the enclosing arguments.
function outer() {
  var grab = function () { return arguments.length; };
  var arrow = () => arguments[0];
  return grab(9, 9) + ":" + arrow();
}
assert.sameValue(outer("X", "Y"), "2:X");
