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

// --- inheritance: extends, super(), super.method(), override ---
class Shape {
  constructor(name) {
    this.name = name;
  }
  describe() {
    return "a " + this.name;
  }
  area() {
    return 0;
  }
}
class Square extends Shape {
  constructor(side) {
    super("square");
    this.side = side;
  }
  area() {
    return this.side * this.side;
  }
  describe() {
    return super.describe() + " with area " + this.area();
  }
}
const sq = new Square(5);
assertEq(sq.name, "square");
assertEq(sq.area(), 25);
assertEq(sq.describe(), "a square with area 25");
assert(sq instanceof Square, "instanceof Square");
assert(sq instanceof Shape, "instanceof Shape");

// A subclass with no explicit constructor forwards to super.
class ColoredSquare extends Square {
  hue() {
    return "red " + this.name;
  }
}
const cs = new ColoredSquare(3);
assertEq(cs.area(), 9);
assertEq(cs.hue(), "red square");
assert(cs instanceof Shape, "deep instanceof");

// --- getters / setters ---
class Celsius {
  constructor(t) {
    this._t = t;
  }
  get fahrenheit() {
    return this._t * 9 / 5 + 32;
  }
  set fahrenheit(f) {
    this._t = (f - 32) * 5 / 9;
  }
  get value() {
    return this._t;
  }
}
const temp = new Celsius(100);
assertEq(temp.fahrenheit, 212);
temp.fahrenheit = 32;
assertEq(temp.value, 0);
