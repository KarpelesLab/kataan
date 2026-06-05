/*---
description: Class private fields and methods
esid: sec-class-definitions
---*/
class Counter {
  #count = 0;
  increment() { this.#count++; return this.#count; }
  get value() { return this.#count; }
  #double() { return this.#count * 2; }
  doubled() { return this.#double(); }
}
var c = new Counter();
assert.sameValue(c.increment(), 1);
assert.sameValue(c.increment(), 2);
assert.sameValue(c.value, 2, "private field via getter");
assert.sameValue(c.doubled(), 4, "private method");
class Account {
  #balance;
  constructor(initial) { this.#balance = initial; }
  deposit(n) { this.#balance += n; return this; }
  get balance() { return this.#balance; }
}
var a = new Account(100);
a.deposit(50).deposit(25);
assert.sameValue(a.balance, 175, "chained with private state");
