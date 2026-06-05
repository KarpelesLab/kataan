/*---
description: Computed property names in classes (fields) and object literals
esid: sec-object-initializer
---*/
var propKey = "dynamic";
class Dynamic {
  [propKey + "Value"] = 42;
  [propKey] = "set";
}
var d = new Dynamic();
assert.sameValue(d.dynamicValue, 42, "computed field name");
assert.sameValue(d.dynamic, "set");
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
