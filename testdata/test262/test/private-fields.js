/*---
description: Private class fields store, read, and are non-enumerable
esid: sec-class-definitions
---*/
class Box {
  #value = 0;
  set(v) { this.#value = v; return this.#value; }
  read() { return this.#value; }
}
var b = new Box();
assert.sameValue(b.set(42), 42);
assert.sameValue(b.read(), 42, "private field round-trips");

b.pub = 7;
// The private field never appears in enumeration.
assert.sameValue(Object.keys(b).join(","), "pub", "private field and methods are non-enumerable");
