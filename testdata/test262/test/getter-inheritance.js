/*---
description: Getters and methods inherited through a class extends chain
esid: sec-runtime-semantics-classdefinitionevaluation
---*/
class Base {
  constructor(name) { this.name = name; }
  greet() { return "hello from " + this.name; }
  get shout() { return this.name.toUpperCase(); }
}
class Derived extends Base {
  constructor(name) { super(name); }
}
var d = new Derived("rex");
assert.sameValue(d.greet(), "hello from rex", "inherited method");
assert.sameValue(d.shout, "REX", "inherited getter");
assert.sameValue(d.name, "rex", "own property");
assert.sameValue(d instanceof Base, true);
