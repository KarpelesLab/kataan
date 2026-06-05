/*---
description: Symbol.toPrimitive customizes object coercion
esid: sec-symbol.toprimitive
---*/
var obj = {
  [Symbol.toPrimitive]: function (hint) {
    if (hint === "number") return 42;
    if (hint === "string") return "str";
    return "default";
  }
};
assert.sameValue(+obj, 42, "number hint");
assert.sameValue(`${obj}`, "str", "string hint");
assert.sameValue(obj + "", "default", "default hint");
var counter = { n: 0, [Symbol.toPrimitive]: function () { return ++this.n; } };
assert.sameValue(counter + counter, 3, "1 + 2");
