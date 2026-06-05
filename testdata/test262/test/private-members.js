/*---
description: Class private fields, methods, static, and accessors
esid: sec-private-names
---*/
class Counter {
  #count = 0;
  #step = 1;
  increment() { this.#count += this.#step; return this.#count; }
  get value() { return this.#count; }
  #reset() { this.#count = 0; }
  resetCount() { this.#reset(); return this.#count; }
}
var c = new Counter();
assert.sameValue(c.increment(), 1);
assert.sameValue(c.increment(), 2);
assert.sameValue(c.value, 2, "private field via getter");
assert.sameValue(c.resetCount(), 0, "private method");
class BankAccount {
  #balance;
  static #count = 0;
  constructor(initial) { this.#balance = initial; BankAccount.#count++; }
  deposit(n) { this.#balance += n; return this; }
  get balance() { return this.#balance; }
  static get accountCount() { return BankAccount.#count; }
}
var a1 = new BankAccount(100);
var a2 = new BankAccount(200);
a1.deposit(50);
assert.sameValue(a1.balance, 150, "private state with chaining");
assert.sameValue(a2.balance, 200);
assert.sameValue(BankAccount.accountCount, 2, "private static counter");
class Temperature {
  #celsius = 0;
  get #fahrenheit() { return this.#celsius * 9 / 5 + 32; }
  describe() { return this.#celsius + "C = " + this.#fahrenheit + "F"; }
}
var t = new Temperature();
assert.sameValue(t.describe(), "0C = 32F", "private getter");
class Hidden {
  #secret = 42;
  reveal() { return this.#secret; }
}
assert.sameValue(new Hidden().reveal(), 42, "private field access");
