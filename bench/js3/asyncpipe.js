// An async job pipeline (original): simulated I/O stages built from
// async functions and resolved promises, fan-out with Promise.all, retry with
// rejection handling, a bounded-concurrency worker pool, Promise.allSettled /
// race / any, and async iteration over an async generator.
const JOBS = typeof AP_JOBS === 'number' ? AP_JOBS : 40000;

let seed = 1234;
function rnd(n) { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed % n; }

const fetchBlock = (id) => Promise.resolve({ id, size: 100 + (id * 37) % 900 });
async function decode(block) { await null; return { id: block.id, words: (block.size >> 3) + 1 }; }
async function maybeFail(id) {
  if (id % 13 === 0) throw new Error('transient ' + id);
  return id * 2;
}
async function withRetry(id, tries) {
  for (let t = 0; t < tries; t++) {
    try { return await maybeFail(id + t); } catch (e) { if (t === tries - 1) return -1; }
  }
}
async function stage(id) {
  const b = await fetchBlock(id);
  const d = await decode(b);
  const r = await withRetry(id, 2);
  return d.words + r;
}
async function pool(ids, width) {
  let next = 0, sum = 0;
  async function worker() {
    while (next < ids.length) { const id = ids[next++]; sum += await stage(id); }
  }
  const ws = [];
  for (let i = 0; i < width; i++) ws.push(worker());
  await Promise.all(ws);
  return sum;
}
async function* ticker(n) { for (let i = 0; i < n; i++) { await undefined; yield i * 3; } }

async function main() {
  const ids = [];
  for (let i = 0; i < JOBS; i++) ids.push(rnd(100000));
  const pooled = await pool(ids, 16);
  let batched = 0;
  for (let i = 0; i < ids.length; i += 50) {
    const vals = await Promise.all(ids.slice(i, i + 50).map(stage));
    for (const v of vals) batched += v;
  }
  const settled = await Promise.allSettled(ids.slice(0, 3000).map((id) => maybeFail(id)));
  let ok = 0, bad = 0;
  for (const s of settled) if (s.status === 'fulfilled') ok += s.value; else bad++;
  let raced = 0;
  for (let i = 0; i < 2000; i++) raced += await Promise.race([decode({ id: i, size: i }), new Promise((r) => r({ words: -1 }))]).then((v) => v.words);
  const anyv = await Promise.any([maybeFail(13), maybeFail(26), maybeFail(7)]);
  let ticks = 0;
  for await (const t of ticker(JOBS)) ticks += t;
  let chain = Promise.resolve(0);
  for (let i = 0; i < JOBS; i++) chain = chain.then((v) => v + (i & 7));
  const chained = await chain;
  console.log(pooled, batched, ok, bad, raced, anyv, ticks, chained);
}
void main().catch((e) => console.log('error', e && e.message));
