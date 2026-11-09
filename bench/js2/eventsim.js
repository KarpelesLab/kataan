// A discrete-event simulation of a small warehouse (original): customers
// arrive, queue for pickers, pickers walk aisles, packers batch orders.
// Actors are generator coroutines that yield "wait" requests to a scheduler
// built on a binary-heap event queue; statistics are gathered through closure
// callbacks. A second phase drives an async/await pipeline over Promises.
const HORIZON = typeof ES_HORIZON === 'number' ? ES_HORIZON : 600000;
const ASYNC_JOBS = typeof ES_JOBS === 'number' ? ES_JOBS : 60000;

let seed = 2024;
function rnd(n) { seed = (Math.imul(seed, 1103515245) + 12345) >>> 0; return (seed >>> 10) % n; }

class EventQueue {
  constructor() { this.h = []; this.seq = 0; }
  get size() { return this.h.length; }
  push(time, fn) {
    const h = this.h;
    const ev = { time, seq: this.seq++, fn };
    h.push(ev);
    let i = h.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (this.less(h[p], ev)) break;
      h[i] = h[p]; i = p;
    }
    h[i] = ev;
  }
  less(a, b) { return a.time < b.time || (a.time === b.time && a.seq < b.seq); }
  pop() {
    const h = this.h;
    const top = h[0];
    const last = h.pop();
    if (h.length) {
      let i = 0;
      for (;;) {
        let c = 2 * i + 1;
        if (c >= h.length) break;
        if (c + 1 < h.length && this.less(h[c + 1], h[c])) c++;
        if (this.less(last, h[c])) break;
        h[i] = h[c]; i = c;
      }
      h[i] = last;
    }
    return top;
  }
}

class Sim {
  constructor() { this.q = new EventQueue(); this.now = 0; this.listeners = new Map(); this.steps = 0; }
  on(kind, fn) { if (!this.listeners.has(kind)) this.listeners.set(kind, []); this.listeners.get(kind).push(fn); }
  emit(kind, data) { const ls = this.listeners.get(kind); if (ls) for (const fn of ls) fn(data, this.now); }
  at(delay, fn) { this.q.push(this.now + delay, fn); }
  spawn(gen) {
    const step = (value) => {
      this.steps++;
      const r = gen.next(value);
      if (r.done) return;
      const req = r.value;
      if (req.wait !== undefined) this.at(req.wait, () => step(this.now));
      else if (req.take !== undefined) req.take.acquire(() => step(this.now));
    };
    step(undefined);
  }
  run(until) {
    while (this.q.size && this.q.h[0].time <= until) {
      const ev = this.q.pop();
      this.now = ev.time;
      ev.fn();
    }
  }
}

class Resource {
  constructor(sim, name, capacity) { this.sim = sim; this.name = name; this.free = capacity; this.waiting = []; this.maxQueue = 0; }
  acquire(cb) {
    if (this.free > 0) { this.free--; cb(); return; }
    this.waiting.push(cb);
    if (this.waiting.length > this.maxQueue) this.maxQueue = this.waiting.length;
  }
  release() {
    const next = this.waiting.shift();
    if (next) this.sim.at(0, next); else this.free++;
  }
}

const sim = new Sim();
const pickers = new Resource(sim, 'pickers', 6);
const packers = new Resource(sim, 'packers', 3);
const stats = { arrived: 0, picked: 0, packed: 0, totalWait: 0, totalLines: 0, late: 0 };
sim.on('arrive', () => { stats.arrived++; });
sim.on('picked', (o) => { stats.picked++; stats.totalLines += o.lines; });
sim.on('packed', (o, now) => { stats.packed++; const t = now - o.created; stats.totalWait += t; if (t > 400) stats.late++; });

function* order(id) {
  const o = { id, lines: 1 + rnd(8), created: sim.now };
  sim.emit('arrive', o);
  yield { take: pickers };
  for (let l = 0; l < o.lines; l++) yield { wait: 3 + rnd(12) };
  pickers.release();
  sim.emit('picked', o);
  yield { take: packers };
  yield { wait: 10 + o.lines * 2 };
  packers.release();
  sim.emit('packed', o);
}

function* arrivals() {
  let id = 0;
  for (;;) {
    yield { wait: 1 + rnd(18) };
    sim.spawn(order(id++));
  }
}
sim.spawn(arrivals());
sim.run(HORIZON);

// Phase 2: async pipeline with Promise.all batches and a microtask chain.
function delayValue(v) { return Promise.resolve(v); }
async function fetchItem(i) {
  const a = await delayValue(i * 3);
  const b = await delayValue(a % 17);
  return { i, score: a + b };
}
async function stage(batch) {
  const results = await Promise.all(batch.map(fetchItem));
  let s = 0;
  for (const r of results) s += r.score;
  return s;
}
async function pipeline() {
  let total = 0;
  const batch = [];
  for (let i = 0; i < ASYNC_JOBS; i++) {
    batch.push(i);
    if (batch.length === 50) { total += await stage(batch.splice(0)); }
  }
  if (batch.length) total += await stage(batch);
  const chain = await [1, 2, 3, 4, 5].reduce((p, n) => p.then((acc) => acc * 2 + n), Promise.resolve(0));
  return total + chain;
}

const finished = pipeline().then((asyncTotal) => {
  console.log('now=' + sim.now, 'steps=' + sim.steps, 'arrived=' + stats.arrived, 'picked=' + stats.picked, 'packed=' + stats.packed);
  console.log('avgLines=' + (stats.totalLines / stats.picked).toFixed(3), 'avgWait=' + (stats.totalWait / stats.packed).toFixed(3), 'late=' + stats.late, 'maxQ=' + pickers.maxQueue + '/' + packers.maxQueue, 'async=' + asyncTotal);
});
