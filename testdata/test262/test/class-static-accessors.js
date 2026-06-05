/*---
description: Static methods, fields, and getters in classes
esid: sec-class-definitions
---*/
class Counter {
  static count = 0;
  static increment() { return ++Counter.count; }
  static get current() { return Counter.count; }
}
assert.sameValue(Counter.count, 0);
assert.sameValue(Counter.increment(), 1);
assert.sameValue(Counter.increment(), 2);
assert.sameValue(Counter.current, 2, "static getter");
class MathUtils {
  static PI = 3.14159;
  static square(x) { return x * x; }
  static cube(x) { return x * x * x; }
}
assert.sameValue(MathUtils.PI, 3.14159);
assert.sameValue(MathUtils.square(4), 16);
assert.sameValue(MathUtils.cube(3), 27);
