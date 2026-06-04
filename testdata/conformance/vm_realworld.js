// Conformance fixture: real-world patterns and coercion corners that regressed
// (or were missing) and are now fixed. Runs through both the tree-walker and
// the VM (dual-path runner), so the two paths must agree.

// --- compound + logical assignment ---
let bits = 0b1010;
bits &= 0b1100;
assertEq(bits, 0b1000);
bits |= 0b0001;
assertEq(bits, 0b1001);
bits <<= 1;
assertEq(bits, 0b10010);

const cfg = {};
cfg.timeout ??= 30;
cfg.timeout ??= 99; // already set; must not change
assertEq(cfg.timeout, 30);
let enabled = 0;
enabled ||= "on";
assertEq(enabled, "on");
let valid = "yes";
valid &&= "still-yes";
assertEq(valid, "still-yes");

// Short-circuit: the right-hand side must not run when not needed.
let evals = 0;
const bump = () => {
  evals += 1;
  return 1;
};
let present = 5;
present ??= bump();
assertEq(evals, 0);

// --- optional chaining + optional calls ---
const api = { user: { name: "ada", greet: () => "hi" } };
assertEq(api?.user?.name, "ada");
assertEq(api?.user?.greet?.(), "hi");
assertEq(String(api?.user?.missing?.()), "undefined");
assertEq(api?.admin?.name ?? "none", "none");
assertEq(api?.user?.roles?.[0] ?? "default", "default");

// --- destructuring assignment ---
let a = 1, b = 2;
[a, b] = [b, a];
assertEq(a + "," + b, "2,1");
let head, tail;
[head, ...tail] = [1, 2, 3, 4];
assertEq(head, 1);
assertEq(tail.join(","), "2,3,4");
const point = {};
({ x: point.px, y: point.py } = { x: 10, y: 20 });
assertEq(point.px + "," + point.py, "10,20");
let opts;
let rest;
({ opts, ...rest } = { opts: "d", other: 1 });
assertEq(opts, "d");
assertEq(rest.other, 1);

// --- error instanceof ---
try {
  null.field;
} catch (e) {
  assert(e instanceof TypeError, "TypeError");
  assert(e instanceof Error, "is Error");
  assert(!(e instanceof RangeError), "not RangeError");
}
try {
  undefinedIdentifierXYZ;
} catch (e) {
  assert(e instanceof ReferenceError, "ReferenceError");
}

// --- `+` does ToPrimitive (array/object → string) ---
assertEq("" + [1, 2, 3], "1,2,3");
assertEq(String([1, 2] + [3, 4]), "1,23,4");
assertEq(String([] + []), "");
assertEq(({}) + "!", "[object Object]!");
assertEq(1 + 2, 3); // pure numeric unaffected

// --- array iterator methods + substring ---
assertEq([...[5, 6, 7].keys()].join(","), "0,1,2");
let pairs = "";
for (const [i, v] of ["x", "y"].entries()) pairs += i + v;
assertEq(pairs, "0x1y");
assertEq("hello".substring(1, 3), "el");
assertEq("hello".substr(-2), "lo");

// --- async functions return promises ---
let asyncResult = "pending";
async function loadValue() {
  return 42;
}
loadValue().then((v) => {
  asyncResult = "got:" + v;
});

// --- a realistic pipeline ---
const records = [
  { name: "Ada", score: 95, team: "a" },
  { name: "Bob", score: 82, team: "b" },
  { name: "Cay", score: 88, team: "a" },
];
const byTeam = {};
for (const r of records) {
  (byTeam[r.team] ??= []).push(r.score);
}
const averages = Object.entries(byTeam)
  .map(([team, scores]) => ({
    team,
    avg: Math.round(scores.reduce((s, n) => s + n, 0) / scores.length),
  }))
  .sort((x, y) => y.avg - x.avg)
  .map(({ team, avg }) => `${team}=${avg}`)
  .join(",");
assertEq(averages, "a=92,b=82");
