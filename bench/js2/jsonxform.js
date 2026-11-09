// JSON round-trips of nested records with transformation (original): build an
// "orders" document with customers, line items and tags, serialize it,
// parse it back with a reviver, reshape (group, aggregate, rename keys),
// serialize the result with a replacer and indentation, and repeat.
const ORDERS = typeof JX_ORDERS === 'number' ? JX_ORDERS : 1500;
const ROUNDS = typeof JX_ROUNDS === 'number' ? JX_ROUNDS : 6;

let seed = 4242;
function r(n) { seed = (Math.imul(seed, 69069) + 1) >>> 0; return (seed >>> 8) % n; }
const SKUS = ['bolt', 'nut', 'washer', 'gear', 'spring', 'cam', 'shaft', 'pulley', 'belt', 'chain'];
const CITIES = ['Lyon', 'Osaka', 'Quito', 'Perth', 'Oslo', 'Accra', 'Lima', 'Pune'];
const TAGS = ['rush', 'gift', 'fragile', 'bulk', 'repeat', 'promo'];

function makeDoc(n) {
  const customers = [];
  for (let i = 0; i < 120; i++) {
    customers.push({ id: 'c' + i, name: 'Customer ' + i, address: { city: CITIES[r(CITIES.length)], zip: String(10000 + r(89999)) }, vip: r(7) === 0 });
  }
  const orders = [];
  for (let i = 0; i < n; i++) {
    const items = [];
    const k = 1 + r(6);
    for (let j = 0; j < k; j++) items.push({ sku: SKUS[r(SKUS.length)], qty: 1 + r(9), price: (100 + r(9900)) / 100 });
    const tags = [];
    for (const t of TAGS) if (r(5) === 0) tags.push(t);
    orders.push({ id: i, customer: 'c' + r(customers.length), placed: '2026-' + String(1 + r(12)).padStart(2, '0') + '-' + String(1 + r(28)).padStart(2, '0'), items, tags, note: r(4) === 0 ? 'leave at "door" \\ side' : null });
  }
  return { version: 3, generated: 'synthetic', customers, orders };
}

function transform(doc) {
  const byId = new Map(doc.customers.map((c) => [c.id, c]));
  const cities = {};
  const skuTotals = {};
  for (const o of doc.orders) {
    const c = byId.get(o.customer);
    const city = c.address.city;
    const total = o.items.reduce((s, it) => s + it.qty * it.price, 0);
    const bucket = cities[city] || (cities[city] = { city, orders: 0, revenue: 0, vipOrders: 0, months: {} });
    bucket.orders++;
    bucket.revenue += total;
    if (c.vip) bucket.vipOrders++;
    const month = o.placed.slice(0, 7);
    bucket.months[month] = (bucket.months[month] || 0) + 1;
    for (const it of o.items) skuTotals[it.sku] = (skuTotals[it.sku] || 0) + it.qty;
  }
  const summary = Object.values(cities)
    .map((b) => ({ ...b, revenue: Math.round(b.revenue * 100) / 100, topMonth: Object.entries(b.months).sort((a, b2) => b2[1] - a[1] || (a[0] < b2[0] ? -1 : 1))[0][0] }))
    .sort((a, b) => b.revenue - a.revenue);
  const rush = doc.orders.filter((o) => o.tags.includes('rush')).map((o) => ({ orderId: o.id, lines: o.items.length, who: byId.get(o.customer).name }));
  return { schema: 'summary/1', summary, skuTotals, rush, counts: { orders: doc.orders.length, customers: doc.customers.length } };
}

const reviver = (key, value) => {
  if (key === 'placed') return value.replace(/-/g, '/').replace(/\//g, '-');
  if (key === 'price' && typeof value === 'number') return Math.round(value * 100) / 100;
  return value;
};
const replacer = (key, value) => (key === 'months' ? undefined : value);

let doc = makeDoc(ORDERS);
let checksum = 0;
let lastOut = '';
let bytes = 0;
for (let round = 0; round < ROUNDS; round++) {
  const text = JSON.stringify(doc);
  bytes += text.length;
  const back = JSON.parse(text, reviver);
  const out = transform(back);
  lastOut = JSON.stringify(out, replacer, 2);
  bytes += lastOut.length;
  const again = JSON.parse(lastOut);
  checksum = (checksum * 17 + again.summary.length + again.rush.length + lastOut.length) % 1000000007;
  // mutate the document slightly for the next round
  for (let i = round; i < back.orders.length; i += 7) back.orders[i].items.push({ sku: SKUS[i % SKUS.length], qty: 1, price: 0.5 });
  back.customers[round].vip = !back.customers[round].vip;
  doc = back;
}
const final = JSON.parse(lastOut);
console.log('bytes=' + bytes, 'checksum=' + checksum, 'cities=' + final.summary.map((s) => s.city + ':' + s.orders).join(','));
console.log('top=' + JSON.stringify(final.summary[0]), 'sku=' + JSON.stringify(final.skuTotals));
