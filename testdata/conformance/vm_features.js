// Conformance fixture: the imperative/OO feature set the bytecode compiler
// handles directly. Run through both the tree-walker and the VM (no construct
// here forces a fallback), so it pins down VM behavior across the board.

// --- functions, recursion, closures-of-globals ---
function fib(n) {
  if (n < 2) return n;
  return fib(n - 1) + fib(n - 2);
}
assertEq(fib(12), 144);

function fact(n) {
  let acc = 1;
  for (let i = 2; i <= n; i += 1) acc *= i;
  return acc;
}
assertEq(fact(6), 720);

const square = (x) => x * x;
const compose = (f, g) => (x) => f(g(x));
assertEq(square(7), 49);

// --- every operator ---
assertEq(1 + 2 * 3 - 4 / 2, 5);
assertEq(2 ** 10, 1024);
assertEq(17 % 5, 2);
assert(3 < 5 && 5 <= 5 && 9 > 2 && 2 >= 2, "comparisons");
assert(1 == "1" && 1 !== "1" && 2 === 2 && 3 != 4, "equality");
assertEq(5 & 3, 1);
assertEq(5 | 2, 7);
assertEq(5 ^ 1, 4);
assertEq(1 << 4, 16);
assertEq(-1 >>> 28, 15);
assertEq(true && "y", "y");
assertEq(false || "z", "z");
assertEq(null ?? "d", "d");
assertEq(0 ?? "d", 0);
assertEq(typeof 42, "number");
assertEq(typeof "x", "string");
assertEq(typeof fib, "function");
assertEq(typeof nothingHere, "undefined");
assertEq(-square(3), -9);
assertEq(+"42", 42);

// --- objects, arrays, methods ---
const data = { name: "kataan", tags: ["js", "rust"], meta: { stars: 3 } };
assertEq(data.name, "kataan");
assertEq(data.tags[1], "rust");
assertEq(data.meta.stars, 3);
data.meta.stars += 1;
assertEq(data.meta.stars, 4);
assert("name" in data, "in operator");
assert(!("missing" in data), "in operator negative");
delete data.name;
assert(!("name" in data), "delete");

const nums = [5, 2, 8, 1, 9, 3];
assertEq([...nums].sort((a, b) => a - b).join(","), "1,2,3,5,8,9");
assertEq(
  nums.filter((n) => n > 3).map((n) => n * 2).reduce((a, b) => a + b, 0),
  44,
);
assertEq(nums.find((n) => n > 7), 8);

// --- this in methods ---
const counter = {
  count: 0,
  inc() {
    this.count += 1;
    return this.count;
  },
};
counter.inc();
counter.inc();
assertEq(counter.inc(), 3);

// --- control flow: loops, switch, try/catch ---
let collatz = 27;
let steps = 0;
while (collatz !== 1) {
  collatz = collatz % 2 === 0 ? collatz / 2 : 3 * collatz + 1;
  steps += 1;
}
assertEq(steps, 111);

function classify(n) {
  switch (n) {
    case 0:
      return "zero";
    case 1:
    case 2:
    case 3:
      return "small";
    default:
      return "big";
  }
}
assertEq(classify(0), "zero");
assertEq(classify(2), "small");
assertEq(classify(99), "big");

function safeDiv(a, b) {
  try {
    if (b === 0) throw "division by zero";
    return a / b;
  } catch (e) {
    return "error: " + e;
  }
}
assertEq(safeDiv(10, 2), 5);
assertEq(safeDiv(10, 0), "error: division by zero");

// --- templates + new ---
const who = "world";
assertEq(`Hello, ${who}! ${1 + 1} times`, "Hello, world! 2 times");
const m = new Map([["a", 1]]);
m.set("b", 2);
assertEq(m.get("a") + m.get("b"), 3);
assertEq(new Error("boom").message, "boom");
