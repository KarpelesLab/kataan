// Snippets for the playground. Each is chosen to show something the engine does
// that is hard to fake — not to show that arithmetic works.
export const EXAMPLES = [
  {
    id: 'calendars',
    label: 'Lunisolar calendars',
    note: 'Chinese New Year is computed from new moons and solar terms, not a table.',
    source: `// Temporal's chinese calendar is astronomical: months begin at a
// new moon, and month 11 holds the December solstice.
for (const year of [2023, 2024, 2025, 2026]) {
  const newYear = Temporal.PlainDate.from(
    { year, month: 1, day: 1, calendar: "chinese" },
  );
  console.log(year, "->", newYear.withCalendar("iso8601").toString());
}

// A leap month takes the number of the month it follows.
const leap = Temporal.PlainDate.from(
  { year: 2025, monthCode: "M06L", day: 1, calendar: "chinese" },
);
leap.withCalendar("iso8601").toString();`,
  },
  {
    id: 'intl',
    label: 'Intl',
    note: 'Locale data, plural rules and currency formatting, all in Rust.',
    source: `const price = 1234567.891;
for (const locale of ["en-US", "de-DE", "ja-JP", "hi-IN"]) {
  console.log(
    locale.padEnd(6),
    new Intl.NumberFormat(locale, { style: "currency", currency: "EUR" })
      .format(price),
  );
}

console.log(new Intl.ListFormat("en", { type: "disjunction" })
  .format(["tokens", "a tree", "a value"]));

new Intl.RelativeTimeFormat("es").format(-3, "day");`,
  },
  {
    id: 'regexp',
    label: 'RegExp v flag',
    note: 'Unicode property escapes and set notation, from generated UCD tables.',
    source: `// Properties of strings match multi-code-point sequences.
console.log(/^\\p{RGI_Emoji}$/v.test("👨‍👩‍👧"));

// Set subtraction: Greek letters that are not lowercase.
const upperGreek = /[\\p{Script=Greek}--\\p{Lowercase}]/v;
console.log([..."αβΓΔε"].filter((c) => upperGreek.test(c)).join(""));

// Named groups, lookbehind, and the d flag for indices.
const m = /(?<=\\$)(?<amount>\\d+\\.\\d{2})/d.exec("total: $42.50");
JSON.stringify({ amount: m.groups.amount, at: m.indices.groups.amount });`,
  },
  {
    id: 'classes',
    label: 'Classes',
    note: 'Private fields and methods, static blocks, and brand checks.',
    source: `class Counter {
  #n = 0;
  static #instances = 0;
  static registry = new Set();

  static { this.created = 0; }          // static initialisation block

  constructor(name) {
    this.name = name;
    Counter.#instances++;
    Counter.created++;
    Counter.registry.add(this);
  }

  #bump(by) { this.#n += by; return this; }   // private method
  add(by = 1) { return this.#bump(by); }
  get value() { return this.#n; }

  static has(o) { return #n in o; }     // ergonomic brand check
}

const c = new Counter("a").add().add(41);
console.log(c.value, Counter.created, Counter.has(c), Counter.has({}));
c.value;`,
  },
  {
    id: 'async',
    label: 'Generators & async',
    note: 'Real suspension: yield suspends the frame, await resumes as a microtask.',
    source: `function* fib() {
  let [a, b] = [0n, 1n];
  for (;;) { yield a; [a, b] = [b, a + b]; }
}

// Iterator helpers, lazily.
console.log([...fib().drop(10).take(5)].join(" "));

async function* ticks(n) {
  for (let i = 1; i <= n; i++) yield await Promise.resolve(i * i);
}

const seen = [];
for await (const t of ticks(4)) seen.push(t);
console.log("ordering:", seen.join(","));

// A 90-digit Fibonacci number, exactly.
[...fib().drop(430).take(1)][0].toString().length;`,
  },
  {
    id: 'proxy',
    label: 'Proxy & Reflect',
    note: 'Traps fire with the right invariants, including on the prototype chain.',
    source: `const log = [];
const observed = new Proxy(
  { a: 1 },
  {
    get(t, k, r) { log.push(\`get \${String(k)}\`); return Reflect.get(t, k, r); },
    has(t, k) { log.push(\`has \${String(k)}\`); return Reflect.has(t, k); },
    ownKeys(t) { log.push("ownKeys"); return Reflect.ownKeys(t); },
  },
);

observed.a;
"b" in observed;
Object.keys(observed);

console.log(log.join(" | "));

// A revoked proxy throws on every operation.
const { proxy, revoke } = Proxy.revocable({}, {});
revoke();
try { proxy.x; } catch (e) { console.log(e.constructor.name); }
log.length;`,
  },
  {
    id: 'unicode',
    label: 'Unicode strings',
    note: 'Case mapping, segmentation, normalisation and locale collation — no ICU, no C.',
    source: `console.log("ß".toUpperCase(), "İ".toLowerCase().length);
console.log("e\\u0301".normalize("NFC") === "é");

// Grapheme segmentation keeps a family together.
const seg = new Intl.Segmenter("en", { granularity: "grapheme" });
const family = "👩‍👩‍👦";
console.log("code units:", family.length,
            "graphemes:", [...seg.segment(family)].length);

// Collation is locale-aware: Swedish orders å ä ö after z.
const sv = new Intl.Collator("sv").compare;
console.log(["zebra", "ängel", "åka", "apa"].sort(sv).join(" "));

// It understands numbers inside strings…
const natural = new Intl.Collator("en", { numeric: true });
console.log(["item10", "item2", "item1"].sort(natural.compare).join(" "));

// …and accents, when you ask it to ignore them.
new Intl.Collator("en", { sensitivity: "base" }).compare("résumé", "resume");`,
  },
  {
    id: 'errors',
    label: 'Errors',
    note: 'Every stage reports where it went wrong.',
    source: `// Delete a bracket and switch the pane to Tokens or Tree: the
// error moves earlier in the pipeline, and says where.
function risky(depth) {
  if (depth === 0) throw new RangeError("bottom reached");
  return risky(depth - 1);
}

try {
  risky(3);
} catch (e) {
  console.log(e instanceof RangeError, e.message);
}

// Errors carry a cause, and Error.isError sees through a realm.
const wrapped = new Error("upper", { cause: new RangeError("lower") });
console.log(wrapped.cause.name, Error.isError(wrapped.cause));

// Anything printed before the fault is kept — and the script's value
// becomes the error itself.
null.oops;`,
  },
];
