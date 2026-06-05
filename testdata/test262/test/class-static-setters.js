/*---
description: Class static getters and setters together
esid: sec-class-definitions
---*/
class Temperature {
  static _celsius = 0;
  static get celsius() { return Temperature._celsius; }
  static set celsius(v) { Temperature._celsius = v; }
  static get fahrenheit() { return Temperature._celsius * 9 / 5 + 32; }
}
Temperature.celsius = 25;
assert.sameValue(Temperature.celsius, 25, "static setter then getter");
assert.sameValue(Temperature.fahrenheit, 77);
Temperature.celsius = 100;
assert.sameValue(Temperature.fahrenheit, 212);
class Registry {
  static _items = [];
  static add(item) { Registry._items.push(item); return Registry._items.length; }
  static get count() { return Registry._items.length; }
}
Registry.add("a");
Registry.add("b");
assert.sameValue(Registry.count, 2);
