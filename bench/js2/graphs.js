// Graph algorithms over a generated road-like network (original): adjacency
// stored as Map<number, Array<{to, w}>>, Dijkstra with a hand-written binary
// heap of {node, dist} records, BFS layering with a Set of visited nodes,
// connected components by iterative DFS, and Prim's minimum spanning tree.
const N = typeof GR_N === 'number' ? GR_N : 20000;
const SOURCES = typeof GR_SOURCES === 'number' ? GR_SOURCES : 16;

let seed = 99;
function rnd(n) { seed = (Math.imul(seed, 48271) + 11) & 0x7fffffff; return seed % n; }

class Graph {
  constructor() { this.adj = new Map(); this.edges = 0; }
  addNode(id) { if (!this.adj.has(id)) this.adj.set(id, []); }
  addEdge(a, b, w) {
    this.addNode(a); this.addNode(b);
    this.adj.get(a).push({ to: b, w });
    this.adj.get(b).push({ to: a, w });
    this.edges++;
  }
  neighbors(id) { return this.adj.get(id) || []; }
  get size() { return this.adj.size; }
}

class MinHeap {
  constructor() { this.items = []; }
  get length() { return this.items.length; }
  push(node, dist) {
    const a = this.items;
    a.push({ node, dist });
    let i = a.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (a[p].dist <= a[i].dist) break;
      const t = a[p]; a[p] = a[i]; a[i] = t;
      i = p;
    }
  }
  pop() {
    const a = this.items;
    const top = a[0];
    const last = a.pop();
    if (a.length > 0) {
      a[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1, r = l + 1;
        let m = i;
        if (l < a.length && a[l].dist < a[m].dist) m = l;
        if (r < a.length && a[r].dist < a[m].dist) m = r;
        if (m === i) break;
        const t = a[m]; a[m] = a[i]; a[i] = t;
        i = m;
      }
    }
    return top;
  }
}

function build(n) {
  const g = new Graph();
  const side = Math.ceil(Math.sqrt(n));
  for (let i = 0; i < n; i++) {
    g.addNode(i);
    const x = i % side, y = (i / side) | 0;
    if (x + 1 < side && i + 1 < n && rnd(10) < 8) g.addEdge(i, i + 1, 1 + rnd(20));
    if (i + side < n && rnd(10) < 8) g.addEdge(i, i + side, 1 + rnd(20));
    if (rnd(50) === 0) g.addEdge(i, rnd(n), 5 + rnd(100));
    void y;
  }
  return g;
}

function dijkstra(g, src) {
  const dist = new Map();
  dist.set(src, 0);
  const heap = new MinHeap();
  heap.push(src, 0);
  let relax = 0;
  while (heap.length > 0) {
    const { node, dist: d } = heap.pop();
    if (d > dist.get(node)) continue;
    for (const e of g.neighbors(node)) {
      const nd = d + e.w;
      const old = dist.get(e.to);
      if (old === undefined || nd < old) {
        dist.set(e.to, nd);
        heap.push(e.to, nd);
        relax++;
      }
    }
  }
  return { dist, relax };
}

function bfsLayers(g, src) {
  const seen = new Set([src]);
  let frontier = [src];
  const widths = [];
  while (frontier.length) {
    widths.push(frontier.length);
    const next = [];
    for (const v of frontier) {
      for (const { to } of g.neighbors(v)) {
        if (!seen.has(to)) { seen.add(to); next.push(to); }
      }
    }
    frontier = next;
  }
  return widths;
}

function components(g) {
  const comp = new Map();
  let count = 0;
  const sizes = [];
  for (const start of g.adj.keys()) {
    if (comp.has(start)) continue;
    const stack = [start];
    comp.set(start, count);
    let size = 0;
    while (stack.length) {
      const v = stack.pop();
      size++;
      for (const e of g.neighbors(v)) if (!comp.has(e.to)) { comp.set(e.to, count); stack.push(e.to); }
    }
    sizes.push(size);
    count++;
  }
  sizes.sort((a, b) => b - a);
  return { count, largest: sizes[0], singles: sizes.filter((s) => s === 1).length };
}

function prim(g, start) {
  const inTree = new Set();
  const heap = new MinHeap();
  heap.push(start, 0);
  let total = 0;
  while (heap.length) {
    const { node, dist } = heap.pop();
    if (inTree.has(node)) continue;
    inTree.add(node);
    total += dist;
    for (const e of g.neighbors(node)) if (!inTree.has(e.to)) heap.push(e.to, e.w);
  }
  return { total, spanned: inTree.size };
}

const g = build(N);
let check = 0, relaxTotal = 0, far = 0;
for (let s = 0; s < SOURCES; s++) {
  const src = (s * 7919) % N;
  const { dist, relax } = dijkstra(g, src);
  relaxTotal += relax;
  let maxd = 0, sum = 0;
  for (const d of dist.values()) { sum += d; if (d > maxd) maxd = d; }
  far = Math.max(far, maxd);
  check = (check * 131 + sum + dist.size) % 2147483647;
}
const layers = bfsLayers(g, 0);
const cc = components(g);
const mst = prim(g, 0);
console.log('nodes=' + g.size, 'edges=' + g.edges, 'relax=' + relaxTotal, 'far=' + far, 'check=' + check);
console.log('bfs depth=' + layers.length, 'maxwidth=' + Math.max(...layers), 'cc=' + cc.count + '/' + cc.largest + '/' + cc.singles, 'mst=' + mst.total + '/' + mst.spanned);
