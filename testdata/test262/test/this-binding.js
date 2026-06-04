/*---
description: this binding in methods, arrows, and explicit call/apply
esid: sec-this-keyword
---*/
var obj = {
  val: 42,
  getVal: function () { return this.val; },
  getArrow: function () { var f = () => this.val; return f(); }
};
assert.sameValue(obj.getVal(), 42, "method this");
assert.sameValue(obj.getArrow(), 42, "arrow captures lexical this");
assert.sameValue(obj.getVal.call({ val: 7 }), 7, "call rebinds this");
assert.sameValue(obj.getVal.apply({ val: 9 }), 9, "apply rebinds this");
var bound = obj.getVal.bind({ val: 11 });
assert.sameValue(bound(), 11, "bind fixes this");
