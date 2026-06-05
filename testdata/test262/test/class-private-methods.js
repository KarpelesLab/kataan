/*---
description: Private methods and fields in classes
esid: sec-class-definitions
---*/
class Counter {
  #count = 0;
  #step() { this.#count += 1; }
  increment() { this.#step(); return this.#count; }
  get value() { return this.#count; }
}
var c = new Counter();
assert.sameValue(c.increment(), 1);
assert.sameValue(c.increment(), 2);
assert.sameValue(c.value, 2);
assert.sameValue(c.count, undefined, "private field is not public");
class Bank {
  #balance = 100;
  #validate(amt) { return amt > 0 && amt <= this.#balance; }
  withdraw(amt) { if (this.#validate(amt)) { this.#balance -= amt; return true; } return false; }
  get balance() { return this.#balance; }
}
var b = new Bank();
assert.sameValue(b.withdraw(30), true);
assert.sameValue(b.balance, 70);
assert.sameValue(b.withdraw(1000), false);
