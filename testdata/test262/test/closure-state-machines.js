/*---
description: Closures implementing state machines and generators
esid: sec-closure
---*/
function makeStack() {
  var items = [];
  return {
    push: function (x) { items.push(x); return items.length; },
    pop: function () { return items.pop(); },
    peek: function () { return items[items.length - 1]; },
    size: function () { return items.length; }
  };
}
var stack = makeStack();
stack.push(1);
stack.push(2);
stack.push(3);
assert.sameValue(stack.size(), 3);
assert.sameValue(stack.peek(), 3);
assert.sameValue(stack.pop(), 3);
assert.sameValue(stack.size(), 2);
function makeToggle() {
  var state = false;
  return function () { state = !state; return state; };
}
var toggle = makeToggle();
assert.sameValue(toggle(), true);
assert.sameValue(toggle(), false);
assert.sameValue(toggle(), true);
function makeSequence() {
  var n = 0;
  return function () { return n++; };
}
var seq = makeSequence();
assert.sameValue(seq() + "," + seq() + "," + seq(), "0,1,2");
function compose() {
  var fns = [...arguments];
  return function (x) { return fns.reduceRight(function (acc, fn) { return fn(acc); }, x); };
}
var addThenDouble = compose(function (x) { return x * 2; }, function (x) { return x + 1; });
assert.sameValue(addThenDouble(5), 12, "compose: (5+1)*2");
