// Conformance fixture: the advanced object/collection surface.

// --- getters / setters ---
const account = {
  _balance: 0,
  get balance() {
    return this._balance;
  },
  set balance(v) {
    this._balance = v < 0 ? 0 : v;
  },
};
account.balance = 100;
assertEq(account.balance, 100);
account.balance = -50; // clamped by the setter
assertEq(account.balance, 0);

// --- delete ---
const dict = { a: 1, b: 2, c: 3 };
assert(delete dict.b, "delete returns true");
assert(!("b" in dict), "property removed");
assertEq(Object.keys(dict).join(","), "a,c");

// --- Map / Set as first-class iterables ---
const seen = new Set();
for (const word of ["a", "b", "a", "c", "b"]) seen.add(word);
assertEq([...seen].join(","), "a,b,c");

const counts = new Map();
for (const ch of "banana") {
  counts.set(ch, (counts.get(ch) || 0) + 1);
}
assertEq(counts.get("a"), 3);
assertEq(counts.get("n"), 2);
assertEq(counts.get("b"), 1);

// --- tagged templates ---
function html(strings, ...values) {
  return strings.reduce(
    (acc, s, i) => acc + s + (i < values.length ? "<" + values[i] + ">" : ""),
    "",
  );
}
assertEq(html`a${1}b${2}c`, "a<1>b<2>c");

// --- constructor statics ---
assert(Number.isInteger(42), "isInteger");
assert(!Number.isInteger(4.2), "non-integer");
assertEq(String.fromCharCode(75, 97), "Ka");
assertEq(Array.from(new Set([1, 1, 2])).length, 2);
assertEq(Object.fromEntries([["k", "v"]]).k, "v");

// --- class with inherited accessor + super ---
class Vec2 {
  constructor(x, y) {
    this.x = x;
    this.y = y;
  }
  get length() {
    return Math.sqrt(this.x * this.x + this.y * this.y);
  }
  toString() {
    return "(" + this.x + "," + this.y + ")";
  }
}
class Vec3 extends Vec2 {
  constructor(x, y, z) {
    super(x, y);
    this.z = z;
  }
  get length() {
    return Math.sqrt(this.x * this.x + this.y * this.y + this.z * this.z);
  }
  toString() {
    return super.toString() + "+" + this.z;
  }
}
const v = new Vec3(2, 3, 6);
assertEq(v.length, 7); // sqrt(4+9+36) = 7
assertEq(v.toString(), "(2,3)+6");
assert(v instanceof Vec3, "instanceof self");
assert(v instanceof Vec2, "instanceof parent");

// --- destructuring with defaults / rest in real use ---
function configure({ host = "localhost", port = 8080, ...extra } = {}) {
  return host + ":" + port + "/" + Object.keys(extra).length;
}
assertEq(configure(), "localhost:8080/0");
assertEq(configure({ port: 3000, debug: true }), "localhost:3000/1");
