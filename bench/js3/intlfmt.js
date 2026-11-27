// Locale-aware report rendering (original): format an invoice ledger with
// Intl.NumberFormat (currency, percent, compact, grouping), Intl.DateTimeFormat
// in UTC, sort customer names with Intl.Collator (accent- and case-aware), and
// hash the rendered strings.
const ROWS = typeof IF_ROWS === 'number' ? IF_ROWS : 6000;

let seed = 7;
function rnd(n) { seed = (seed * 48271) % 2147483647; return seed % n; }
const FIRST = ['Émile', 'zoë', 'Ana', 'Ångel', 'björn', 'Çelik', 'david', 'Élodie', 'Óscar', 'anna', 'Zack', 'ñandu', 'Ölaf', 'eva'];
const LAST = ['Müller', 'muller', 'Dupont', 'du Pont', 'Øster', 'Ortiz', 'Łukasz', 'Lopez', 'López', 'Smith'];

const usd = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' });
const eur = new Intl.NumberFormat('de-DE', { style: 'currency', currency: 'EUR' });
const pct = new Intl.NumberFormat('en-US', { style: 'percent', maximumFractionDigits: 1 });
const grp = new Intl.NumberFormat('en-US', { maximumFractionDigits: 2, minimumFractionDigits: 2 });
const plain = new Intl.NumberFormat('en-US', { useGrouping: false, maximumFractionDigits: 0 });
const dtf = new Intl.DateTimeFormat('en-US', { timeZone: 'UTC', year: 'numeric', month: '2-digit', day: '2-digit' });
const dtl = new Intl.DateTimeFormat('en-GB', { timeZone: 'UTC', dateStyle: 'medium' });
const coll = new Intl.Collator('en', { sensitivity: 'base' });
const collVariant = new Intl.Collator('en', { sensitivity: 'variant', caseFirst: 'upper' });

let h = 0;
function mix(s) { for (let i = 0; i < s.length; i++) h = (Math.imul(h, 31) + s.charCodeAt(i)) | 0; }

const rows = [];
for (let i = 0; i < ROWS; i++) {
  rows.push({
    name: FIRST[rnd(FIRST.length)] + ' ' + LAST[rnd(LAST.length)],
    amount: rnd(10000000) / 100,
    rate: rnd(1000) / 1000,
    at: Date.UTC(2020, 0, 1) + rnd(1500) * 86400000,
  });
}
for (const r of rows) {
  mix(usd.format(r.amount));
  mix(eur.format(r.amount));
  mix(pct.format(r.rate));
  mix(grp.format(r.amount * 13));
  mix(dtf.format(r.at));
}
for (let i = 0; i < rows.length; i += 4) mix(dtl.format(new Date(rows[i].at)));
const names = rows.map((r) => r.name);
const sorted = names.slice().sort(coll.compare);
const sorted2 = names.slice().sort(collVariant.compare);
let uniq = 0;
for (let i = 1; i < sorted.length; i++) if (coll.compare(sorted[i - 1], sorted[i]) !== 0) uniq++;
const parts = usd.formatToParts(-1234567.891).map((p) => p.type).join(',');
let total = 0;
for (const r of rows) total += r.amount;
console.log(h, sorted[0], sorted[sorted.length - 1], sorted2[0], sorted2[sorted2.length - 1], uniq, parts,
  usd.format(total), plain.format(total), eur.format(-total), pct.format(0.125));
