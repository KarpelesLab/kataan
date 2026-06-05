/*---
description: Symbol.toPrimitive controls object coercion by hint
esid: sec-symbol.toprimitive
---*/
var obj = {
  [Symbol.toPrimitive](hint) {
    if (hint === "number") return 42;
    if (hint === "string") return "str";
    return "default";
  }
};
assert.sameValue(+obj, 42, "number hint via unary plus");
assert.sameValue(obj - 0, 42, "number hint via subtraction");
assert.sameValue(obj * 1, 42, "number hint via multiplication");
assert.sameValue(`${obj}`, "str", "string hint via template");
assert.sameValue(String(obj), "str", "string hint via String()");
assert.sameValue(obj + "", "default", "default hint via +");
assert.sameValue(obj + 1, "default1", "default hint with number");
assert.sameValue("" + obj, "default");
var money = {
  amount: 100,
  [Symbol.toPrimitive](hint) { return hint === "string" ? "$" + this.amount : this.amount; }
};
assert.sameValue(money * 2, 200, "number hint reads this");
assert.sameValue(`${money}`, "$100", "string hint reads this");
assert.sameValue(money > 50, true, "number hint in comparison");
var sym = Symbol("key");
var holder = {};
holder[sym] = "value";
assert.sameValue(holder[sym], "value", "symbol-keyed property");
assert.sameValue(holder[Symbol("key")], undefined, "distinct symbols");
assert.sameValue(Symbol("d").description, "d", "symbol description");
assert.sameValue(Symbol("x").toString(), "Symbol(x)");
assert.sameValue(typeof Symbol.iterator, "symbol", "well-known symbol");
