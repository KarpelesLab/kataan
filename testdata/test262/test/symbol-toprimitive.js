/*---
description: Symbol.toPrimitive custom coercion
esid: sec-symbol.toprimitive
---*/
var temperature = {
  celsius: 25,
  [Symbol.toPrimitive](hint) {
    if (hint === "number") return this.celsius;
    if (hint === "string") return this.celsius + "°C";
    return "default:" + this.celsius;
  }
};
assert.sameValue(+temperature, 25, "number hint");
assert.sameValue(`${temperature}`, "25°C", "string hint");
assert.sameValue(temperature + "", "default:25", "default hint in concatenation");
assert.sameValue(temperature * 2, 50, "number hint in arithmetic");
var money = {
  amount: 100,
  [Symbol.toPrimitive](hint) { return hint === "string" ? "$" + this.amount : this.amount; }
};
assert.sameValue(money - 50, 50);
assert.sameValue(String(money), "$100");
assert.sameValue(Number(money), 100);
var obj = {
  valueOf() { return 10; },
  toString() { return "twenty"; }
};
assert.sameValue(obj + 5, 15, "valueOf for default hint");
assert.sameValue(`${obj}`, "twenty", "toString for string hint");
