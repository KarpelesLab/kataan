// Conformance fixture: functions, closures, and classes.

// --- functions & closures ---
function fib(n) {
  return n < 2 ? n : fib(n - 1) + fib(n - 2);
}
assertEq(fib(15), 610);

const adder = (x) => (y) => x + y;
assertEq(adder(3)(4), 7);

function makeCounter() {
  let c = 0;
  return () => ++c;
}
const next = makeCounter();
next();
next();
assertEq(next(), 3);

function withDefaults(a, b = 10, ...rest) {
  return a + b + rest.length;
}
assertEq(withDefaults(1), 11);
assertEq(withDefaults(1, 2, 3, 4), 5);

// --- destructuring ---
const [first, , third] = [1, 2, 3];
assertEq(first + third, 4);
const { x, y: yy = 99 } = { x: 7 };
assertEq(x + yy, 106);

// --- classes & inheritance ---
class Shape {
  constructor(name) {
    this.name = name;
  }
  describe() {
    return this.name + " with area " + this.area();
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
}

const sq = new Square(5);
assertEq(sq.area(), 25);
assertEq(sq.describe(), "square with area 25");
assert(sq instanceof Square, "instanceof self");
assert(sq instanceof Shape, "instanceof parent");

// --- getters/setters ---
class Temperature {
  constructor(c) {
    this._c = c;
  }
  get fahrenheit() {
    return (this._c * 9) / 5 + 32;
  }
  set fahrenheit(f) {
    this._c = ((f - 32) * 5) / 9;
  }
}
const t = new Temperature(100);
assertEq(t.fahrenheit, 212);
t.fahrenheit = 32;
assertEq(t._c, 0);

// --- static members & super.method ---
class Base {
  greet() {
    return "hi";
  }
  static kind() {
    return "base";
  }
}
class Derived extends Base {
  greet() {
    return super.greet() + "!";
  }
}
assertEq(new Derived().greet(), "hi!");
assertEq(Base.kind(), "base");
