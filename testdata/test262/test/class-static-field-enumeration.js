/*---
description: a class's static fields are enumerable own keys; methods/accessors are not
features: [class, class-static-fields-public]
---*/
// A static getter keeps this exercising the same enumeration on both engines.
class G {
  static x = 1;
  static y = 2;
  static m() {}
  static get z() { return 3; }
}
assert.sameValue(Object.keys(G).join(","), "x,y", "static fields are own enumerable keys");
assert.sameValue(Object.values(G).join(","), "1,2", "static field values");
assert.sameValue(JSON.stringify(Object.entries(G)), '[["x",1],["y",2]]', "static field entries");

// A class with only methods/accessors has no enumerable own keys.
class M { static only() {} static get g() { return 1; } }
assert.sameValue(Object.keys(M).length, 0, "methods/accessors are non-enumerable");

// An empty class has none either.
class E {}
assert.sameValue(Object.keys(E).length, 0, "empty class");

// A static field is still readable as a property.
assert.sameValue(G.x, 1, "static field readable");
assert.sameValue(G.z, 3, "static getter readable");
