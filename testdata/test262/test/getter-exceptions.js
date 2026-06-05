/*---
description: Exceptions thrown from getters propagate correctly
esid: sec-property-accessors
---*/
var obj = {
  get bad() { throw new Error("getter failed"); },
  get good() { return 42; }
};
assert.sameValue(obj.good, 42);
var caught = false;
try { var x = obj.bad; } catch (e) { caught = e.message === "getter failed"; }
assert.sameValue(caught, true, "getter exception propagates");
var counter = 0;
var sideEffect = {
  get value() { counter++; if (counter > 2) throw new Error("too many"); return counter; }
};
assert.sameValue(sideEffect.value, 1);
assert.sameValue(sideEffect.value, 2);
var threw = false;
try { sideEffect.value; } catch (e) { threw = true; }
assert.sameValue(threw, true, "third access throws");
var result;
try { result = { get x() { return undefined.foo; } }.x; }
catch (e) { result = "caught TypeError"; }
assert.sameValue(result, "caught TypeError");
