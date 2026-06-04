// Conformance fixture: standard-library builtins.

// --- Math ---
assertEq(Math.max(1, 9, 4), 9);
assertEq(Math.min(3, -2, 8), -2);
assertEq(Math.floor(3.9), 3);
assertEq(Math.ceil(3.1), 4);
assertEq(Math.abs(-7), 7);
assertEq(Math.pow(2, 8), 256);
assertEq(Math.sqrt(81), 9);

// --- Number / parsing ---
assertEq(parseInt("42px"), 42);
assertEq(parseInt("ff", 16), 255);
assertEq(parseFloat("3.14xyz"), 3.14);
assert(isNaN(0 / 0), "isNaN");
assert(!isFinite(1 / 0), "isFinite");
assertEq((255).toString(16), "ff");
assertEq((3.14159).toFixed(2), "3.14");

// --- Array methods ---
const nums = [5, 3, 8, 1, 9, 2];
assertEq([...nums].sort((a, b) => a - b).join(","), "1,2,3,5,8,9");
assertEq(nums.filter((n) => n > 4).length, 3);
assertEq(nums.map((n) => n * 2).reduce((a, b) => a + b, 0), 56);
assertEq(nums.find((n) => n > 7), 8);
assertEq(nums.includes(9), true);
assertEq(nums.indexOf(8), 2);
assertEq([1, [2, 3], [4]].flat().length, 4);
assertEq([3, 1, 2].reverse().join(""), "213");
assertEq([1, 2].concat([3], 4).join(""), "1234");

// --- String methods ---
assertEq("Hello".toUpperCase(), "HELLO");
assertEq("WORLD".toLowerCase(), "world");
assertEq("  trim  ".trim(), "trim");
assertEq("a,b,c".split(",").length, 3);
assertEq("ab".repeat(3), "ababab");
assertEq("hello world".includes("world"), true);
assertEq("5".padStart(3, "0"), "005");
assertEq("a-b-c".replaceAll("-", "+"), "a+b+c");

// --- Object statics ---
const obj = { a: 1, b: 2, c: 3 };
assertEq(Object.keys(obj).join(","), "a,b,c");
assertEq(Object.values(obj).join(","), "1,2,3");
assertEq(Object.entries(obj).length, 3);
const merged = Object.assign({}, { a: 1 }, { b: 2 });
assertEq(merged.a + merged.b, 3);

// --- JSON ---
const data = { name: "kataan", nums: [1, 2, 3], nested: { ok: true } };
const round = JSON.parse(JSON.stringify(data));
assertEq(round.name, "kataan");
assertEq(round.nums[2], 3);
assertEq(round.nested.ok, true);
assertEq(JSON.stringify([1, "two", null, true]), '[1,"two",null,true]');

// --- Map / Set ---
const m = new Map([
  ["one", 1],
  ["two", 2],
]);
m.set("three", 3);
assertEq(m.get("two"), 2);
assertEq(m.size, 3);
assertEq(m.has("one"), true);

const s = new Set([1, 1, 2, 3, 3, 3]);
assertEq(s.size, 3);
assertEq([...s].join(","), "1,2,3");
let total = 0;
for (const v of s) total += v;
assertEq(total, 6);
