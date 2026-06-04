// Conformance fixture: core language semantics.
// The harness injects assert(cond, msg) and assertEq(actual, expected).

// --- operators & coercion ---
assertEq(1 + 2 * 3, 7);
assertEq(2 ** 3 ** 2, 512); // right-associative
assertEq(7 % 3, 1);
assertEq(-5 % 3, -2); // sign of the dividend
assertEq("a" + 1 + 2, "a12");
assertEq(1 + 2 + "a", "3a");
assertEq("5" * 2, 10);
assert(0.1 + 0.2 !== 0.3, "float rounding");
assertEq(typeof 1, "number");
assertEq(typeof "x", "string");
assertEq(typeof undefined, "undefined");
assertEq(typeof null, "object");
assertEq(typeof function () {}, "function");

// --- equality ---
assert(1 == "1", "loose number/string");
assert(1 !== "1", "strict number/string");
assert(null == undefined, "null == undefined");
assert(NaN !== NaN, "NaN inequality");
assert(0 === -0, "+0 === -0");

// --- short circuiting ---
assertEq(true && "yes", "yes");
assertEq(false || "no", "no");
assertEq(null ?? "default", "default");
assertEq(0 ?? "default", 0);

// --- control flow ---
let sum = 0;
for (let i = 1; i <= 5; i++) sum += i;
assertEq(sum, 15);

let count = 0;
outer: for (let i = 0; i < 3; i++) {
  for (let j = 0; j < 3; j++) {
    if (j === 1) continue outer;
    count++;
  }
}
assertEq(count, 3);

// --- try / catch / finally ---
let trace = "";
try {
  trace += "t";
  throw new Error("boom");
} catch (e) {
  trace += "c:" + e.message;
} finally {
  trace += ":f";
}
assertEq(trace, "tc:boom:f");
