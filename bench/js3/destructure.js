// Destructuring-heavy record processing (original): nested object/array
// patterns with defaults and renames, rest elements and rest properties,
// spread in calls and literals, parameter destructuring, swaps and
// computed-key patterns over a stream of synthetic events.
const N = typeof DS_N === 'number' ? DS_N : 400000;

let seed = 42;
function rnd(n) { seed = (seed * 16807) % 2147483647; return seed % n; }
const KINDS = ['click', 'view', 'buy', 'scroll'];
const events = [];
for (let i = 0; i < 4000; i++) {
  events.push({ id: i, kind: KINDS[rnd(4)], pos: [rnd(1920), rnd(1080)], meta: rnd(5) ? { user: { id: rnd(500), tier: rnd(3) }, tags: ['a', 'b', 'c'].slice(0, rnd(4)) } : {}, extra: rnd(2) ? i : undefined });
}

function score({ kind, pos: [x, y], meta: { user: { id: uid = -1, tier = 0 } = {}, tags = [] } = {}, extra = 7 }) {
  return (kind.length * 3 + x - y + uid * tier + tags.length + extra) | 0;
}
function head([first, second = 0, ...rest]) { return first + second * 2 + rest.length; }
function sum(...xs) { let s = 0; for (const x of xs) s += x; return s; }
function strip({ id, extra, ...rest }) { return rest; }

let acc = 0, swaps = 0, restKeys = 0;
for (let i = 0; i < N; i++) {
  const ev = events[i % events.length];
  acc = (acc + score(ev)) | 0;
  const [x, y] = ev.pos;
  let a = x, b = y;
  if (a > b) { [a, b] = [b, a]; swaps++; }
  acc = (acc + head([a, b, x, y]) + sum(...ev.pos, a, b)) | 0;
  if ((i & 7) === 0) {
    const r = strip(ev);
    for (const k in r) restKeys += k.length;
    const merged = { ...r, ...{ seen: i }, pos: [...ev.pos, i] };
    acc = (acc + merged.pos.length + merged.seen) | 0;
  }
  const key = KINDS[i & 3];
  const { [key]: hit = 0 } = { click: 1, view: 2, buy: 3 };
  acc = (acc + hit) | 0;
}
const pairs = events.map(({ id, pos: [px] }) => [id, px]);
let pairSum = 0;
for (const [id, px] of pairs) pairSum += id * px % 97;
const matrix = [];
for (let i = 0; i < 300; i++) matrix.push([i, i + 1, i + 2, i + 3]);
let diag = 0;
for (const [p, , q, ...more] of matrix) diag += p + q + more.length;
const { length, 0: first, [matrix.length - 1]: last } = matrix;
console.log(acc, swaps, restKeys, pairSum, diag, length, first.join(''), last.join(''));
