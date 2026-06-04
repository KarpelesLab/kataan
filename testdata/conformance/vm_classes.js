// Conformance fixture: classes (constructor, methods, fields, statics) compiled
// by the bytecode VM. Runs through both the tree-walker and the VM. (Inheritance
// via `extends`/`super` is exercised elsewhere and still falls back.)

class Rectangle {
  constructor(w, h) {
    this.w = w;
    this.h = h;
  }
  area() {
    return this.w * this.h;
  }
  perimeter() {
    return 2 * (this.w + this.h);
  }
  scale(factor) {
    return new Rectangle(this.w * factor, this.h * factor);
  }
}

const r = new Rectangle(3, 4);
assertEq(r.area(), 12);
assertEq(r.perimeter(), 14);
assert(r instanceof Rectangle, "instanceof");
const big = r.scale(2);
assertEq(big.area(), 48);
assertEq(r.area(), 12); // original unchanged

// --- instance fields with initializers ---
class Account {
  balance = 0;
  history = [];
  deposit(amount) {
    this.balance += amount;
    this.history.push(amount);
    return this.balance;
  }
  count() {
    return this.history.length;
  }
}
const acc = new Account();
assertEq(acc.balance, 0);
acc.deposit(100);
acc.deposit(50);
assertEq(acc.balance, 150);
assertEq(acc.count(), 2);
// Each instance gets its own field objects.
const acc2 = new Account();
assertEq(acc2.count(), 0);

// --- a method calling another method through `this` ---
class Greeter {
  constructor(name) {
    this.name = name;
  }
  greeting() {
    return "Hello, " + this.name;
  }
  shout() {
    return this.greeting().toUpperCase() + "!";
  }
}
assertEq(new Greeter("ada").shout(), "HELLO, ADA!");

// --- static methods and fields ---
class MathUtil {
  static PI = 3;
  static square(x) {
    return x * x;
  }
  static sumOfSquares(a, b) {
    return MathUtil.square(a) + MathUtil.square(b);
  }
}
assertEq(MathUtil.PI, 3);
assertEq(MathUtil.square(5), 25);
assertEq(MathUtil.sumOfSquares(3, 4), 25);

// --- a class expression ---
const Pair = class {
  constructor(a, b) {
    this.a = a;
    this.b = b;
  }
  sum() {
    return this.a + this.b;
  }
};
assertEq(new Pair(10, 20).sum(), 30);

// --- a stateful class used in a loop ---
class Accumulator {
  total = 0;
  add(n) {
    this.total += n;
    return this;
  }
}
const sum = new Accumulator();
for (const n of [1, 2, 3, 4, 5]) {
  sum.add(n);
}
assertEq(sum.total, 15);
