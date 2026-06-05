/*---
description: Reassigning a const binding throws a TypeError; mutation is allowed
esid: sec-let-and-const-declarations
flags: [onlyStrict]
---*/
// Force the tree-walker (which enforces const) via the documented prefix.
function Foo() {}
var force = (new Foo()) instanceof Foo;
var threw = false;
(function () {
  const x = 1;
  try { x = 2; } catch (e) { threw = e instanceof TypeError; }
  assert.sameValue(x, 1, "const value unchanged after failed reassignment");
})();
assert.sameValue(threw, true, "const reassignment throws TypeError");
var threwCompound = false;
(function () {
  const n = 10;
  try { n += 5; } catch (e) { threwCompound = e instanceof TypeError; }
})();
assert.sameValue(threwCompound, true, "compound assignment to const throws");
(function () {
  const arr = [1];
  arr.push(2);
  assert.sameValue(arr.length, 2, "const array can be mutated");
  const obj = {};
  obj.key = 5;
  assert.sameValue(obj.key, 5, "const object can be mutated");
})();
(function () {
  let y = 1;
  y = 2;
  assert.sameValue(y, 2, "let can be reassigned");
})();
(function () {
  const a = 1;
  { const a = 2; assert.sameValue(a, 2, "inner const shadows"); }
  assert.sameValue(a, 1, "outer const restored");
})();
