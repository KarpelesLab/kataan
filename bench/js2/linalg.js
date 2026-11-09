// Numerical linear algebra on Float64Array (original): dense LU decomposition
// with partial pivoting and forward/back substitution, a sparse matrix in CSR
// form assembled from a 2-D Poisson stencil, conjugate-gradient solve, and a
// power iteration for the dominant eigenvalue. Results are rounded for output.
const DN = typeof LA_DENSE === 'number' ? LA_DENSE : 320;
const GRID = typeof LA_GRID === 'number' ? LA_GRID : 120;

let seed = 31337;
function rnd() { seed = (Math.imul(seed, 1103515245) + 12345) >>> 0; return (seed >>> 8) / 16777216; }

class Dense {
  constructor(n) { this.n = n; this.a = new Float64Array(n * n); }
  get(i, j) { return this.a[i * this.n + j]; }
  set(i, j, v) { this.a[i * this.n + j] = v; }
}

function luDecompose(m) {
  const n = m.n, a = m.a;
  const piv = new Int32Array(n);
  for (let i = 0; i < n; i++) piv[i] = i;
  for (let k = 0; k < n; k++) {
    let p = k, best = Math.abs(a[k * n + k]);
    for (let i = k + 1; i < n; i++) {
      const v = Math.abs(a[i * n + k]);
      if (v > best) { best = v; p = i; }
    }
    if (p !== k) {
      for (let j = 0; j < n; j++) { const t = a[k * n + j]; a[k * n + j] = a[p * n + j]; a[p * n + j] = t; }
      const t = piv[k]; piv[k] = piv[p]; piv[p] = t;
    }
    const d = a[k * n + k];
    for (let i = k + 1; i < n; i++) {
      const f = (a[i * n + k] /= d);
      if (f === 0) continue;
      const ri = i * n, rk = k * n;
      for (let j = k + 1; j < n; j++) a[ri + j] -= f * a[rk + j];
    }
  }
  return piv;
}
function luSolve(m, piv, b) {
  const n = m.n, a = m.a;
  const x = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = b[piv[i]];
    for (let j = 0; j < i; j++) s -= a[i * n + j] * x[j];
    x[i] = s;
  }
  for (let i = n - 1; i >= 0; i--) {
    let s = x[i];
    for (let j = i + 1; j < n; j++) s -= a[i * n + j] * x[j];
    x[i] = s / a[i * n + i];
  }
  return x;
}

class CSR {
  constructor(n, rowPtr, cols, vals) { this.n = n; this.rowPtr = rowPtr; this.cols = cols; this.vals = vals; }
  static fromTriplets(n, trip) {
    trip.sort((p, q) => p.r - q.r || p.c - q.c);
    const rowPtr = new Int32Array(n + 1);
    const cols = [], vals = [];
    let last = null;
    for (const t of trip) {
      if (last && last.r === t.r && last.c === t.c) { vals[vals.length - 1] += t.v; continue; }
      cols.push(t.c); vals.push(t.v); rowPtr[t.r + 1]++;
      last = t;
    }
    for (let i = 0; i < n; i++) rowPtr[i + 1] += rowPtr[i];
    return new CSR(n, rowPtr, Int32Array.from(cols), Float64Array.from(vals));
  }
  mul(x, y) {
    const { rowPtr, cols, vals } = this;
    for (let i = 0; i < this.n; i++) {
      let s = 0;
      for (let k = rowPtr[i]; k < rowPtr[i + 1]; k++) s += vals[k] * x[cols[k]];
      y[i] = s;
    }
  }
}

function dot(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
function axpy(alpha, x, y) { for (let i = 0; i < y.length; i++) y[i] += alpha * x[i]; }

function conjugateGradient(A, b, tol, maxIter) {
  const n = A.n;
  const x = new Float64Array(n), r = Float64Array.from(b), p = Float64Array.from(b), Ap = new Float64Array(n);
  let rr = dot(r, r);
  let it = 0;
  for (; it < maxIter && rr > tol * tol; it++) {
    A.mul(p, Ap);
    const alpha = rr / dot(p, Ap);
    axpy(alpha, p, x);
    axpy(-alpha, Ap, r);
    const rr2 = dot(r, r);
    const beta = rr2 / rr;
    for (let i = 0; i < n; i++) p[i] = r[i] + beta * p[i];
    rr = rr2;
  }
  return { x, it, res: Math.sqrt(rr) };
}

function powerIteration(A, iters) {
  let v = new Float64Array(A.n).fill(1);
  let w = new Float64Array(A.n);
  let lambda = 0;
  for (let k = 0; k < iters; k++) {
    A.mul(v, w);
    lambda = Math.sqrt(dot(w, w));
    for (let i = 0; i < w.length; i++) w[i] /= lambda;
    const t = v; v = w; w = t;
  }
  return lambda;
}

const r6 = (x) => Math.round(x * 1e6) / 1e6;

// Dense part.
const M = new Dense(DN);
for (let i = 0; i < DN; i++) {
  let rowSum = 0;
  for (let j = 0; j < DN; j++) { const v = rnd() - 0.5; M.set(i, j, v); rowSum += Math.abs(v); }
  M.set(i, i, rowSum + 1);
}
const orig = Float64Array.from(M.a);
const bDense = new Float64Array(DN);
for (let i = 0; i < DN; i++) bDense[i] = i % 7 - 3;
const piv = luDecompose(M);
const xd = luSolve(M, piv, bDense);
let maxRes = 0;
for (let i = 0; i < DN; i++) {
  let s = 0;
  for (let j = 0; j < DN; j++) s += orig[i * DN + j] * xd[j];
  maxRes = Math.max(maxRes, Math.abs(s - bDense[i]));
}

// Sparse part: 5-point Laplacian on GRID x GRID.
const n = GRID * GRID;
const trip = [];
for (let y = 0; y < GRID; y++) {
  for (let x = 0; x < GRID; x++) {
    const i = y * GRID + x;
    trip.push({ r: i, c: i, v: 4 });
    if (x > 0) trip.push({ r: i, c: i - 1, v: -1 });
    if (x < GRID - 1) trip.push({ r: i, c: i + 1, v: -1 });
    if (y > 0) trip.push({ r: i, c: i - GRID, v: -1 });
    if (y < GRID - 1) trip.push({ r: i, c: i + GRID, v: -1 });
    if ((x * 13 + y * 7) % 29 === 0) trip.push({ r: i, c: i, v: 0.5 });
  }
}
const A = CSR.fromTriplets(n, trip);
const b = new Float64Array(n);
for (let i = 0; i < n; i++) b[i] = ((i * 2654435761) >>> 0) % 100 / 100;
const cg = conjugateGradient(A, b, 1e-8, 2000);
const lam = powerIteration(A, 150);
let xsum = 0;
for (let i = 0; i < n; i++) xsum += cg.x[i];
console.log('dense n=' + DN, 'x0=' + r6(xd[0]), 'xsum=' + r6(xd.reduce((s, v) => s + v, 0)), 'resOk=' + (maxRes < 1e-9));
console.log('sparse n=' + n, 'nnz=' + A.vals.length, 'cgIters=' + cg.it, 'cgOk=' + (cg.res < 1e-8), 'xsum=' + xsum.toFixed(3), 'lambda=' + lam.toFixed(4));
