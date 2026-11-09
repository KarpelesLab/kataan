// A finite-domain constraint solver (original): variables with bitmask
// domains, a polymorphic hierarchy of constraints (AllDifferent, LinearSum,
// NotEqualOffset, Implies) that propagate to a fixpoint through a work queue,
// and depth-first search with smallest-domain-first branching and trail-based
// undo. It solves a batch of Sudoku puzzles, N-queens and a magic-square-ish
// arithmetic puzzle.
const QUEENS = typeof CS_QUEENS === 'number' ? CS_QUEENS : 14;
const PUZZLES = typeof CS_PUZZLES === 'number' ? CS_PUZZLES : 30;

function bitCount(x) { let c = 0; while (x) { x &= x - 1; c++; } return c; }
function lowBit(x) { return 31 - Math.clz32(x & -x); }

class Var {
  constructor(id, name, dom) { this.id = id; this.name = name; this.dom = dom; this.watchers = []; }
  get fixed() { return (this.dom & (this.dom - 1)) === 0; }
  get value() { return lowBit(this.dom); }
  get size() { return bitCount(this.dom); }
}

class Constraint {
  constructor(vars) { this.vars = vars; this.queued = false; }
  propagate(_s) { return true; }
}
class AllDifferent extends Constraint {
  propagate(s) {
    let changed = true;
    while (changed) {
      changed = false;
      let union = 0, count = 0;
      for (const v of this.vars) {
        if (v.fixed) {
          for (const w of this.vars) {
            if (w !== v && (w.dom & v.dom)) {
              if (!s.set(w, w.dom & ~v.dom)) return false;
              changed = true;
            }
          }
        }
        union |= v.dom; count++;
      }
      if (bitCount(union) < count) return false;
    }
    // hidden singles: a value only one variable can take
    for (let bit = 1; bit < (1 << 30) && bit > 0; bit <<= 1) {
      let owner = null, n = 0;
      for (const v of this.vars) if (v.dom & bit) { owner = v; n++; if (n > 1) break; }
      if (n === 1 && owner.dom !== bit && this.vars.length === s.hiddenSingleWidth) { if (!s.set(owner, bit)) return false; }
      if (bit > s.maxBit) break;
    }
    return true;
  }
}
class NotEqualOffset extends Constraint {
  constructor(a, b, off) { super([a, b]); this.a = a; this.b = b; this.off = off; }
  propagate(s) {
    const { a, b, off } = this;
    if (a.fixed) {
      const v = a.value + off;
      if (v >= 0 && v < 31 && (b.dom >> v) & 1) { if (!s.set(b, b.dom & ~(1 << v))) return false; }
    }
    if (b.fixed) {
      const v = b.value - off;
      if (v >= 0 && v < 31 && (a.dom >> v) & 1) { if (!s.set(a, a.dom & ~(1 << v))) return false; }
    }
    return true;
  }
}
class LinearSum extends Constraint {
  constructor(vars, coeffs, total) { super(vars); this.coeffs = coeffs; this.total = total; }
  bounds(v) { let lo = lowBit(v.dom), hi = 31 - Math.clz32(v.dom); return [lo, hi]; }
  propagate(s) {
    let minSum = 0, maxSum = 0;
    const bs = this.vars.map((v) => this.bounds(v));
    for (let i = 0; i < bs.length; i++) {
      const c = this.coeffs[i];
      if (c > 0) { minSum += c * bs[i][0]; maxSum += c * bs[i][1]; } else { minSum += c * bs[i][1]; maxSum += c * bs[i][0]; }
    }
    if (minSum > this.total || maxSum < this.total) return false;
    for (let i = 0; i < bs.length; i++) {
      const v = this.vars[i], c = this.coeffs[i];
      const restMin = minSum - (c > 0 ? c * bs[i][0] : c * bs[i][1]);
      const restMax = maxSum - (c > 0 ? c * bs[i][1] : c * bs[i][0]);
      let dom = v.dom;
      for (let x = bs[i][0]; x <= bs[i][1]; x++) {
        if (!((dom >> x) & 1)) continue;
        const t = c * x;
        if (t + restMin > this.total || t + restMax < this.total) dom &= ~(1 << x);
      }
      if (dom !== v.dom && !s.set(v, dom)) return false;
    }
    return true;
  }
}
class Implies extends Constraint {
  constructor(a, va, b, vb) { super([a, b]); this.a = a; this.va = va; this.b = b; this.vb = vb; }
  propagate(s) {
    if (this.a.fixed && this.a.value === this.va) return s.set(this.b, this.b.dom & (1 << this.vb));
    return true;
  }
}

class Solver {
  constructor() { this.vars = []; this.cons = []; this.trail = []; this.queue = []; this.nodes = 0; this.props = 0; this.hiddenSingleWidth = -1; this.maxBit = 1 << 30; }
  newVar(name, lo, hi) {
    let dom = 0;
    for (let x = lo; x <= hi; x++) dom |= 1 << x;
    const v = new Var(this.vars.length, name, dom);
    this.vars.push(v);
    return v;
  }
  post(c) { this.cons.push(c); for (const v of c.vars) v.watchers.push(c); return c; }
  set(v, dom) {
    if (dom === v.dom) return true;
    if (dom === 0) return false;
    this.trail.push(v, v.dom);
    v.dom = dom;
    for (const c of v.watchers) if (!c.queued) { c.queued = true; this.queue.push(c); }
    return true;
  }
  fixpoint() {
    let ok = true;
    while (this.queue.length) {
      const c = this.queue.pop();
      c.queued = false;
      if (!ok) continue;
      this.props++;
      if (!c.propagate(this)) ok = false;
    }
    return ok;
  }
  undo(mark) {
    while (this.trail.length > mark) { const dom = this.trail.pop(); const v = this.trail.pop(); v.dom = dom; }
  }
  search(limit) {
    const solutions = [];
    const go = () => {
      this.nodes++;
      let best = null;
      for (const v of this.vars) if (!v.fixed && (best === null || v.size < best.size)) best = v;
      if (best === null) { solutions.push(this.vars.map((v) => v.value)); return solutions.length >= limit; }
      let dom = best.dom;
      while (dom) {
        const bit = dom & -dom;
        dom &= dom - 1;
        const mark = this.trail.length;
        if (this.set(best, bit) && this.fixpoint() && go()) return true;
        this.queue.length = 0;
        for (const c of this.cons) c.queued = false;
        this.undo(mark);
      }
      return false;
    };
    for (const c of this.cons) { c.queued = true; this.queue.push(c); }
    if (this.fixpoint()) go();
    return solutions;
  }
}

function sudoku(givens) {
  const s = new Solver();
  s.hiddenSingleWidth = 9; s.maxBit = 1 << 9;
  const cells = [];
  for (let i = 0; i < 81; i++) {
    const g = givens.charCodeAt(i) - 48;
    cells.push(g >= 1 && g <= 9 ? s.newVar('c' + i, g, g) : s.newVar('c' + i, 1, 9));
  }
  for (let r = 0; r < 9; r++) s.post(new AllDifferent(cells.slice(r * 9, r * 9 + 9)));
  for (let c = 0; c < 9; c++) s.post(new AllDifferent([0, 1, 2, 3, 4, 5, 6, 7, 8].map((r) => cells[r * 9 + c])));
  for (let b = 0; b < 9; b++) {
    const br = Math.floor(b / 3) * 3, bc = (b % 3) * 3, grp = [];
    for (let i = 0; i < 9; i++) grp.push(cells[(br + Math.floor(i / 3)) * 9 + bc + (i % 3)]);
    s.post(new AllDifferent(grp));
  }
  const sol = s.search(1);
  return { s, sol: sol[0] };
}

function queens(n, limit) {
  const s = new Solver();
  const q = [];
  for (let i = 0; i < n; i++) q.push(s.newVar('q' + i, 0, n - 1));
  s.post(new AllDifferent(q));
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) { s.post(new NotEqualOffset(q[i], q[j], j - i)); s.post(new NotEqualOffset(q[i], q[j], i - j)); }
  return { s, sols: s.search(limit) };
}

function arith() {
  // A 3x3 grid of distinct digits 1..9 with every row and column summing to 15,
  // plus an implication tying two corners.
  const s = new Solver();
  const g = [];
  for (let i = 0; i < 9; i++) g.push(s.newVar('g' + i, 1, 9));
  s.post(new AllDifferent(g));
  for (let r = 0; r < 3; r++) s.post(new LinearSum([g[r * 3], g[r * 3 + 1], g[r * 3 + 2]], [1, 1, 1], 15));
  for (let c = 0; c < 3; c++) s.post(new LinearSum([g[c], g[c + 3], g[c + 6]], [1, 1, 1], 15));
  s.post(new LinearSum([g[0], g[8]], [1, -1], -3));
  s.post(new Implies(g[0], 2, g[2], 4));
  return { s, sols: s.search(100) };
}

// A rotation of one seed grid, digits relabelled, gives distinct puzzles.
const SEED = '530070000600195000098000060800060003400803001700020006060000280000419005000080079';
const results = [];
let nodes = 0, props = 0;
for (let p = 0; p < PUZZLES; p++) {
  let grid = '';
  for (let i = 0; i < 81; i++) {
    const r = Math.floor(i / 9), c = i % 9;
    const src = p % 2 === 0 ? SEED[i] : SEED[c * 9 + r];
    grid += src === '0' ? '0' : String(((src.charCodeAt(0) - 49 + p) % 9) + 1);
  }
  // Remove a few givens to make search do real work.
  const holes = grid.split('');
  for (let k = 0; k < 6 + p; k++) holes[(k * 31 + p * 7) % 81] = '0';
  const { s, sol } = sudoku(holes.join(''));
  nodes += s.nodes; props += s.props;
  results.push(sol ? sol.slice(0, 9).join('') : 'none');
}
const qr = queens(QUEENS, 400);
const ar = arith();
nodes += qr.s.nodes + ar.s.nodes; props += qr.s.props + ar.s.props;
let qsum = 0;
for (const sol of qr.sols) for (let i = 0; i < sol.length; i++) qsum = (qsum * 7 + sol[i] * (i + 1)) % 1000003;
console.log('sudoku ' + results.join(' '));
console.log('queens=' + qr.sols.length + ' qsum=' + qsum + ' arith=' + ar.sols.length + ' first=' + (ar.sols[0] || []).join('') + ' nodes=' + nodes + ' props=' + props);
