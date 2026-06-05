/*---
description: Class static initialization blocks
esid: sec-class-static-block
---*/
class Config {
  static settings = {};
  static {
    Config.settings.version = 1;
    Config.settings.name = "app";
  }
}
assert.sameValue(Config.settings.version, 1, "static block ran");
assert.sameValue(Config.settings.name, "app");
class Counter {
  static count = 0;
  static { Counter.count = 10; }
  static { Counter.count += 5; }
}
assert.sameValue(Counter.count, 15, "multiple static blocks run in order");
class WithThis {
  static x = 1;
  static { this.y = this.x + 100; }
}
assert.sameValue(WithThis.y, 101, "this is the class in a static block");
class Computed {
  static data = [1, 2, 3];
  static sum;
  static { Computed.sum = Computed.data.reduce(function (a, b) { return a + b; }, 0); }
}
assert.sameValue(Computed.sum, 6, "static block computes from static field");
