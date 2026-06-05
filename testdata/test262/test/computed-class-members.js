/*---
description: Computed property names in classes (methods + fields) and object literals
esid: sec-object-initializer
---*/
var methodName = "greet";
var propKey = "dynamic";
class Dynamic {
  [methodName]() { return "hello"; }
  [propKey + "Value"] = 42;
  [propKey] = "set";
  get [methodName + "Upper"]() { return "HELLO"; }
}
var d = new Dynamic();
assert.sameValue(d.greet(), "hello", "computed method name");
assert.sameValue(d.dynamicValue, 42, "computed field name");
assert.sameValue(d.dynamic, "set");
assert.sameValue(d.greetUpper, "HELLO", "computed getter name");
var key1 = "a", key2 = "b";
var obj = {
  [key1]: 1,
  [key2]: 2,
  [key1 + key2]: 3
};
assert.sameValue(obj.a, 1);
assert.sameValue(obj.b, 2);
assert.sameValue(obj.ab, 3);
var prefix = "get";
var methods = {
  [prefix + "Name"]() { return "name"; },
  [prefix + "Age"]() { return 30; }
};
assert.sameValue(methods.getName(), "name", "computed method in object literal");
assert.sameValue(methods.getAge(), 30);

var sname = "build";
class Factory {
  static [sname]() { return "built"; }
  static [sname + "Count"] = 7;
  static get [sname + "Mode"]() { return "auto"; }
}
assert.sameValue(Factory.build(), "built", "static computed method");
assert.sameValue(Factory.buildCount, 7, "static computed field");
assert.sameValue(Factory.buildMode, "auto", "static computed getter");
