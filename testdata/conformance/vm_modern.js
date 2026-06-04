// Conformance fixture: modern destructuring / iteration / spread features that
// the bytecode compiler now handles directly. Runs through both the tree-walker
// and the VM (dual-path conformance runner), pinning down VM behavior.

// --- array destructuring ---
const [a, b, c] = [1, 2, 3];
assertEq(a + b + c, 6);
const [first, ...rest] = [10, 20, 30, 40];
assertEq(first, 10);
assertEq(rest.length, 3);
assertEq(rest.join(","), "20,30,40");
const [, second, , fourth] = [1, 2, 3, 4];
assertEq(second + fourth, 6);
const [p = 100, q = 200] = [5];
assertEq(p, 5);
assertEq(q, 200);

// --- object destructuring ---
const { x, y } = { x: 7, y: 8 };
assertEq(x * y, 56);
const { a: alpha, b: beta } = { a: 1, b: 2 };
assertEq(alpha - beta, -1);
const { found = "default" } = {};
assertEq(found, "default");

// --- nested destructuring ---
const {
  user: { name, roles: [primary] },
} = { user: { name: "ada", roles: ["admin", "dev"] } };
assertEq(name, "ada");
assertEq(primary, "admin");

// --- destructuring parameters + defaults ---
function dist([x1, y1], [x2, y2]) {
  return Math.abs(x2 - x1) + Math.abs(y2 - y1);
}
assertEq(dist([0, 0], [3, 4]), 7);

function makeRange(start, end = start + 5) {
  const out = [];
  for (let i = start; i < end; i += 1) out.push(i);
  return out;
}
assertEq(makeRange(1).join(","), "1,2,3,4,5");
assertEq(makeRange(0, 3).join(","), "0,1,2");

// --- for-of over many iterables ---
let sum = 0;
for (const n of [1, 2, 3, 4, 5]) sum += n;
assertEq(sum, 15);

let chars = "";
for (const ch of "kataan") chars = ch + chars;
assertEq(chars, "naatak");

let setSum = 0;
for (const v of new Set([2, 2, 3, 3, 4])) setSum += v;
assertEq(setSum, 9);

let mapOut = "";
for (const [k, v] of new Map([["a", 1], ["b", 2]])) mapOut += k + v;
assertEq(mapOut, "a1b2");

// --- for-in over object keys / array indices ---
const obj = { one: 1, two: 2, three: 3 };
let keys = "";
let valueSum = 0;
for (const key in obj) {
  keys += key;
  valueSum += obj[key];
}
assertEq(keys, "onetwothree");
assertEq(valueSum, 6);

let indices = "";
for (const i in [9, 8, 7]) indices += i;
assertEq(indices, "012");

// --- array spread ---
const merged = [...[1, 2], ...[3, 4], 5];
assertEq(merged.join(","), "1,2,3,4,5");
const fromString = [..."abc"];
assertEq(fromString.length, 3);
assertEq(fromString[2], "c");
const deduped = [...new Set([1, 1, 2, 2, 3])];
assertEq(deduped.join(","), "1,2,3");
const clone = [...rest];
assertEq(clone.join(","), rest.join(","));
