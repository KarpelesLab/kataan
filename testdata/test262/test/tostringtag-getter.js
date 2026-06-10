/*---
description: Object.prototype.toString consults a Symbol.toStringTag accessor (not just a data prop)
features: [Symbol.toStringTag]
---*/
// A class getter for Symbol.toStringTag (on the prototype) is invoked.
class N { get [Symbol.toStringTag]() { return "NThing"; } }
assert.sameValue(Object.prototype.toString.call(new N()), "[object NThing]", "class getter tag");

// Inherited through the prototype chain.
class Base { get [Symbol.toStringTag]() { return "Base"; } }
class Sub extends Base {}
assert.sameValue(Object.prototype.toString.call(new Sub()), "[object Base]", "inherited tag");

// A plain data property still works.
assert.sameValue(Object.prototype.toString.call({ [Symbol.toStringTag]: "Custom" }), "[object Custom]", "data tag");

// Built-in tags are unaffected.
assert.sameValue(Object.prototype.toString.call({}), "[object Object]", "plain object");
assert.sameValue(Object.prototype.toString.call([]), "[object Array]", "array");
assert.sameValue(Object.prototype.toString.call(null), "[object Null]", "null");
assert.sameValue(Object.prototype.toString.call(undefined), "[object Undefined]", "undefined");
