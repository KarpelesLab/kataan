// Immutable-update state churn (original): a reducer-free settings store
// updated with object spread and Object.assign, records reshaped with
// Object.entries / fromEntries / keys / values, grouping with Object.groupBy,
// structuredClone-free deep merges, and key-order-sensitive serialization.
const N = typeof OC_N === 'number' ? OC_N : 60000;

let seed = 2024;
function rnd(n) { seed = (Math.imul(seed, 1103515245) + 12345) >>> 0; return seed % n; }
const FIELDS = ['theme', 'lang', 'size', 'zoom', 'grid', 'snap', 'font', 'tabs'];
let state = { user: { name: 'n', prefs: { theme: 'dark', lang: 'en' } }, counters: {}, flags: { beta: false } };

function deepMerge(a, b) {
  const out = { ...a };
  for (const [k, v] of Object.entries(b)) out[k] = v && typeof v === 'object' && !Array.isArray(v) && a[k] ? deepMerge(a[k], v) : v;
  return out;
}
let h = 0;
for (let i = 0; i < N; i++) {
  const f = FIELDS[rnd(FIELDS.length)];
  switch (rnd(5)) {
    case 0: state = { ...state, user: { ...state.user, prefs: { ...state.user.prefs, [f]: i } } }; break;
    case 1: state = { ...state, counters: { ...state.counters, [f]: (state.counters[f] || 0) + 1 } }; break;
    case 2: state = Object.assign({}, state, { flags: Object.assign({}, state.flags, { [f]: !state.flags[f] }) }); break;
    case 3: state = deepMerge(state, { user: { prefs: { [f]: 'v' + (i & 15) } }, counters: { total: i } }); break;
    default: {
      const prefs = Object.fromEntries(Object.entries(state.user.prefs).filter(([k]) => k !== f).map(([k, v]) => [k, typeof v === 'number' ? v + 1 : v]));
      state = { ...state, user: { ...state.user, prefs } };
    }
  }
  if ((i & 255) === 0) { const s = JSON.stringify(state); for (let j = 0; j < s.length; j += 5) h = (h * 31 + s.charCodeAt(j)) | 0; }
}
const rows = [];
for (let i = 0; i < 20000; i++) rows.push({ id: i, dept: FIELDS[rnd(8)], salary: 1000 + rnd(9000), active: rnd(4) !== 0 });
const byDept = Object.groupBy(rows, (r) => r.dept);
const summary = Object.fromEntries(Object.entries(byDept).map(([d, rs]) => [d, rs.reduce((s, r) => s + (r.active ? r.salary : 0), 0)]));
const renamed = rows.slice(0, 5000).map((r) => Object.fromEntries(Object.entries(r).map(([k, v]) => ['x_' + k, v])));
let keyLen = 0, valSum = 0;
for (const r of renamed) { for (const k of Object.keys(r)) keyLen += k.length; for (const v of Object.values(r)) if (typeof v === 'number') valSum += v; }
const order = Object.keys({ b: 1, 2: 1, a: 1, 1: 1, [Symbol('s')]: 1, '-1': 1 }).join(',');
const frozen = Object.freeze({ ...state.flags });
const sealedOk = Object.isFrozen(frozen) && !Object.isExtensible(frozen);
console.log(h, Object.keys(state.user.prefs).sort().join(','), JSON.stringify(state.counters).length, JSON.stringify(summary), keyLen, valSum, order, sealedOk);
