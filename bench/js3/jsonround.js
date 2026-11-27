// JSON round trips of a large nested document (original): build an order
// catalog with nested arrays/objects, escapes and unicode, stringify it with
// and without indentation, parse it back, transform with a reviver and a
// replacer array/function, and diff the structures.
const ORDERS = typeof JR_ORDERS === 'number' ? JR_ORDERS : 4000;

let seed = 31337;
function rnd(n) { seed = (Math.imul(seed, 22695477) + 1) >>> 0; return seed % n; }
const WORDS = ['widget', 'gadget', 'sprocket', 'naïve "quote"', 'tab\there', 'line\nbreak', 'ünïcödé', '日本', 'emoji 🎉', 'back\\slash'];
function order(i) {
  const items = [];
  for (let j = 0, n = 1 + rnd(6); j < n; j++) items.push({ sku: 'S' + rnd(99999), qty: 1 + rnd(9), price: rnd(100000) / 100, note: rnd(4) ? null : WORDS[rnd(WORDS.length)] });
  return { id: i, customer: { name: WORDS[rnd(WORDS.length)], vip: rnd(5) === 0, addr: { zip: String(10000 + rnd(89999)), lines: [WORDS[rnd(10)], WORDS[rnd(10)]] } },
    items, placed: '2024-0' + (1 + rnd(9)) + '-1' + rnd(10), total: 0, flags: { gift: !!rnd(2), rush: false }, score: rnd(3) ? rnd(1000) / 7 : -0 };
}
const doc = { version: 3, generated: 'bench', orders: [] };
for (let i = 0; i < ORDERS; i++) doc.orders.push(order(i));

let h = 0;
function mix(s) { for (let i = 0; i < s.length; i += 2) h = (Math.imul(h, 31) + s.charCodeAt(i)) | 0; }

const compact = JSON.stringify(doc);
const pretty = JSON.stringify(doc, null, 2);
mix(compact);
const back = JSON.parse(compact);
const back2 = JSON.parse(pretty);
let same = 0;
for (let i = 0; i < back.orders.length; i++) if (JSON.stringify(back.orders[i]) === JSON.stringify(back2.orders[i])) same++;
const revived = JSON.parse(compact, function (k, v) {
  if (k === 'price') return Math.round(v * 100);
  if (k === 'note' && v === null) return undefined;
  if (k === 'items') { let t = 0; for (const it of v) t += it.price * it.qty; this.total = t; }
  return v;
});
let grand = 0;
for (const o of revived.orders) grand += o.total;
const slim = JSON.stringify(doc.orders.slice(0, 1000), ['id', 'customer', 'name', 'items', 'sku', 'qty']);
mix(slim);
const replaced = JSON.stringify(doc, (k, v) => (typeof v === 'number' && !Number.isInteger(v) ? Math.floor(v) : k === 'flags' ? undefined : v));
mix(replaced);
let rounds = 0, cur = doc.orders.slice(0, 500);
for (let r = 0; r < 6; r++) { cur = JSON.parse(JSON.stringify(cur)); rounds += cur.length; }
const special = JSON.stringify({ a: [undefined, () => 1, NaN, Infinity, -0], d: new Date(0), s: ' \ud800', n: { toJSON() { return 'J'; } } });
console.log(compact.length, pretty.length, same, grand, slim.length, replaced.length, rounds, special, h);
