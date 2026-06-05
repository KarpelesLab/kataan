/*---
description: Object method shorthand and this binding
esid: sec-method-definitions
---*/
var counter = {
  count: 0,
  increment() { this.count++; return this; },
  reset() { this.count = 0; return this.count; }
};
counter.increment().increment().increment();
assert.sameValue(counter.count, 3, "chained method calls");
assert.sameValue(counter.reset(), 0);
var calc = {
  value: 10,
  add(n) { this.value += n; return this; },
  multiply(n) { this.value *= n; return this; },
  result() { return this.value; }
};
assert.sameValue(calc.add(5).multiply(2).result(), 30, "fluent interface");
var obj = {
  name: "test",
  getName() { return this.name; },
  getNameArrow: function () { return (() => this.name)(); }
};
assert.sameValue(obj.getName(), "test");
assert.sameValue(obj.getNameArrow(), "test", "arrow inherits this");
