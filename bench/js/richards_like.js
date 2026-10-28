// A small task scheduler (original): polymorphic dispatch over task kinds,
// linked queues, and state machines.
class Packet { constructor(link, id, kind) { this.link = link; this.id = id; this.kind = kind; this.data = 0; } }
class Task {
  constructor(id, pri, queue) { this.id = id; this.pri = pri; this.queue = queue; this.count = 0; }
  take() { const p = this.queue; if (p) this.queue = p.link; return p; }
  give(p) { p.link = null; if (!this.queue) { this.queue = p; return; } let q = this.queue; while (q.link) q = q.link; q.link = p; }
}
class Worker extends Task { run(sched) { const p = this.take(); if (!p) return; p.data = (p.data + this.id * 7) % 1013; this.count++; sched.route(p, (this.id + 1) % sched.tasks.length); } }
class Handler extends Task { run(sched) { const p = this.take(); if (!p) return; p.kind ^= 1; this.count += p.kind; sched.route(p, (this.id + 2) % sched.tasks.length); } }
class Idle extends Task { run(sched) { this.count++; if (this.count % 3 === 0) sched.route(new Packet(null, this.count, 0), 0); } }
class Sched {
  constructor() { this.tasks = []; this.routed = 0; }
  route(p, to) { this.routed++; this.tasks[to].give(p); }
  step() { for (let i = 0; i < this.tasks.length; i++) this.tasks[i].run(this); }
}
const s = new Sched();
for (let i = 0; i < 12; i++) s.tasks.push(i % 3 === 0 ? new Worker(i, i, null) : i % 3 === 1 ? new Handler(i, i, null) : new Idle(i, i, null));
for (let i = 0; i < 40; i++) s.tasks[i % 12].give(new Packet(null, i, i & 1));
for (let i = 0; i < 60000; i++) s.step();
let sum = 0; for (const t of s.tasks) sum += t.count;
console.log(s.routed, sum);
