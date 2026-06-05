/*---
description: Object shorthand properties, methods, and computed names
esid: sec-object-initializer
---*/
var x = 1, y = 2;
var point = { x, y };
assert.sameValue(point.x + point.y, 3, "shorthand properties");
var name = "Alice";
var person = { name, greet() { return "Hi " + this.name; } };
assert.sameValue(person.greet(), "Hi Alice", "shorthand method");
var key = "dynamic";
var obj = { [key]: "value", [key + "2"]: "value2" };
assert.sameValue(obj.dynamic, "value");
assert.sameValue(obj.dynamic2, "value2");
var base = { a: 1, b: 2 };
var extended = { ...base, c: 3 };
assert.sameValue(extended.a + extended.b + extended.c, 6, "spread plus new");
var override = { ...base, a: 10 };
assert.sameValue(override.a, 10, "spread then override");
var nested = { outer: { ...base } };
assert.sameValue(nested.outer.a, 1);
var calc = {
  value: 10,
  get double() { return this.value * 2; },
  set double(v) { this.value = v / 2; }
};
assert.sameValue(calc.double, 20, "getter shorthand");
calc.double = 30;
assert.sameValue(calc.value, 15, "setter shorthand");
var methods = { add(a, b) { return a + b; }, sub(a, b) { return a - b; } };
assert.sameValue(methods.add(5, 3) - methods.sub(5, 3), 6);
