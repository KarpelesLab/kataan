// Conformance fixture: closures / upvalues, now compiled directly by the
// bytecode VM. Runs through both the tree-walker and the VM (dual-path runner).

// --- counter over a mutable captured local ---
function makeCounter() {
  let count = 0;
  return {
    inc: () => ++count,
    dec: () => --count,
    value: () => count,
  };
}
const c = makeCounter();
assertEq(c.inc(), 1);
assertEq(c.inc(), 2);
assertEq(c.inc(), 3);
assertEq(c.dec(), 2);
assertEq(c.value(), 2);

// --- two independent counters don't share state ---
const c2 = makeCounter();
c2.inc();
assertEq(c2.value(), 1);
assertEq(c.value(), 2);

// --- read-only capture in higher-order array methods ---
const factor = 3;
const tripled = [1, 2, 3, 4].map((n) => n * factor);
assertEq(tripled.join(","), "3,6,9,12");

const threshold = 5;
assertEq([2, 8, 4, 9, 1].filter((n) => n > threshold).join(","), "8,9");

const offset = 100;
assertEq([1, 2, 3].reduce((acc, n) => acc + n + offset, 0), 306);

// --- currying / partial application ---
const adder = (a) => (b) => (c) => a + b + c;
assertEq(adder(1)(2)(3), 6);

function multiplier(factor) {
  return function (x) {
    return x * factor;
  };
}
const double = multiplier(2);
const triple = multiplier(3);
assertEq(double(10), 20);
assertEq(triple(10), 30);

// --- closure capturing a shared object, with memoization ---
function memoize(fn) {
  const cache = {};
  return (n) => {
    if (n in cache) return cache[n];
    const result = fn(n);
    cache[n] = result;
    return result;
  };
}
let evalCount = 0;
const square = memoize((n) => {
  evalCount += 1;
  return n * n;
});
assertEq(square(4), 16);
assertEq(square(4), 16);
assertEq(square(5), 25);
assertEq(evalCount, 2); // 4 and 5 each computed once

// --- post-creation mutation is observed by the closure ---
let captured = "before";
const reader = () => captured;
captured = "after";
assertEq(reader(), "after");

// --- a factory producing closures that each capture their own argument ---
function tagger(tag) {
  return (msg) => tag + ":" + msg;
}
const fns = [tagger("a"), tagger("b"), tagger("c")];
assertEq(fns[0]("x") + "," + fns[1]("y") + "," + fns[2]("z"), "a:x,b:y,c:z");

// --- per-iteration `let` bindings: each closure captures its own `i` ---
const loopFns = [];
for (let i = 0; i < 3; i += 1) {
  loopFns.push(() => i);
}
assertEq(loopFns[0]() + "," + loopFns[1]() + "," + loopFns[2](), "0,1,2");
// `var`, by contrast, shares one binding.
const varFns = [];
for (var v = 0; v < 3; v += 1) {
  varFns.push(() => v);
}
assertEq(varFns[0]() + "," + varFns[1]() + "," + varFns[2](), "3,3,3");

// --- accumulator closure ---
function makeAccumulator() {
  let total = 0;
  return (amount) => {
    total += amount;
    return total;
  };
}
const acc = makeAccumulator();
acc(10);
acc(20);
assertEq(acc(5), 35);
