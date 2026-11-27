// Calendar arithmetic with Date (original): a recurring-event scheduler that
// expands rules (every N days, nth weekday of month, month-end) over decades in
// UTC, bins events by ISO week, formats ISO strings and re-parses them, and
// computes business-day differences.
const YEARS = typeof DT_YEARS === 'number' ? DT_YEARS : 150;
const DAY = 86400000;
const start = Date.UTC(2000, 0, 1);
const end = Date.UTC(2000 + YEARS, 0, 1);

function isoWeek(t) {
  const d = new Date(t);
  const dow = (d.getUTCDay() + 6) % 7;
  d.setUTCDate(d.getUTCDate() - dow + 3);
  const firstThu = new Date(Date.UTC(d.getUTCFullYear(), 0, 4));
  const w = 1 + Math.round(((d - firstThu) / DAY - 3 + ((firstThu.getUTCDay() + 6) % 7)) / 7);
  return d.getUTCFullYear() * 100 + w;
}
const events = [];
for (let t = start + 9.5 * 3600000; t < end; t += 3 * DAY) events.push(t);
for (let y = 2000; y < 2000 + YEARS; y++) for (let m = 0; m < 12; m++) {
  const d = new Date(Date.UTC(y, m, 1, 14));
  const shift = (2 - d.getUTCDay() + 7) % 7 + 7;
  d.setUTCDate(1 + shift);
  events.push(d.getTime());
}
for (let y = 2000; y < 2000 + YEARS; y++) for (let m = 0; m < 12; m++) events.push(Date.UTC(y, m + 1, 0, 23, 59, 59));
events.sort((a, b) => a - b);

const perWeek = new Map();
let isoHash = 0, roundTrip = 0;
for (let i = 0; i < events.length; i++) {
  const t = events[i];
  const k = isoWeek(t);
  perWeek.set(k, (perWeek.get(k) || 0) + 1);
  const s = new Date(t).toISOString();
  for (let j = 0; j < s.length; j += 3) isoHash = (isoHash * 33 + s.charCodeAt(j)) >>> 0;
  if (Date.parse(s) === t) roundTrip++;
}
function businessDays(a, b) {
  let n = 0;
  const d = new Date(a);
  while (d.getTime() < b) { const w = d.getUTCDay(); if (w !== 0 && w !== 6) n++; d.setUTCDate(d.getUTCDate() + 1); }
  return n;
}
let bd = 0;
for (let i = 0; i + 40 < events.length; i += 97) bd += businessDays(events[i], events[i + 40]);
let clamp = 0;
for (let i = 0; i < 6000; i++) {
  const d = new Date(Date.UTC(2001 + (i % 30), i % 12, 1 + (i % 31)));
  const want = (d.getUTCMonth() + 1) % 12;
  d.setUTCMonth(d.getUTCMonth() + 1);
  if (d.getUTCMonth() !== want) { d.setUTCDate(0); clamp++; }
  clamp += d.getUTCDate();
}
const parsed = ['2024-02-29T12:00:00Z', '2023-12-31', '1999-07-04T00:00:00.250Z', '2010-06-15T08:30:00+05:30']
  .map((s) => Date.parse(s));
const utcStr = new Date(events[events.length >> 1]).toUTCString();
let jsonDates = 0;
for (let i = 0; i < events.length; i += 11) jsonDates += JSON.stringify({ at: new Date(events[i]) }).length;
console.log(events.length, perWeek.size, Math.max(...perWeek.values()), isoHash, roundTrip, bd, clamp, parsed.join(','), utcStr,
  jsonDates, new Date(8.64e15).toISOString(), new Date(NaN).getTime());
