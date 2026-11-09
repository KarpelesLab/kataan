// Cellular automata (original): Conway's Life on a toroidal Uint8Array grid
// with double buffering, a "generations" variant with decaying cells stored in
// a plain Array of row Arrays, an elementary 1-D rule evolved over a bit-packed
// Uint32Array, and pattern bookkeeping in a Map keyed by row strings.
const SIZE = typeof CA_SIZE === 'number' ? CA_SIZE : 256;
const GENS = typeof CA_GENS === 'number' ? CA_GENS : 240;

let seed = 8;
function rnd() { seed ^= seed << 13; seed ^= seed >>> 17; seed ^= seed << 5; return seed >>> 0; }

function lifeStep(cur, next, w, h) {
  let alive = 0;
  for (let y = 0; y < h; y++) {
    const up = ((y + h - 1) % h) * w, row = y * w, dn = ((y + 1) % h) * w;
    for (let x = 0; x < w; x++) {
      const l = (x + w - 1) % w, r = (x + 1) % w;
      const n = cur[up + l] + cur[up + x] + cur[up + r] + cur[row + l] + cur[row + r] + cur[dn + l] + cur[dn + x] + cur[dn + r];
      const v = n === 3 || (n === 2 && cur[row + x] === 1) ? 1 : 0;
      next[row + x] = v;
      alive += v;
    }
  }
  return alive;
}

// "Generations" rule: survive on 3-4-5, born on 2, cells decay over 4 states.
function genStep(grid) {
  const h = grid.length, w = grid[0].length;
  const out = [];
  for (let y = 0; y < h; y++) {
    const row = [];
    for (let x = 0; x < w; x++) {
      const s = grid[y][x];
      if (s > 1) { row.push(s === 3 ? 0 : s + 1); continue; }
      let n = 0;
      for (let dy = -1; dy <= 1; dy++) {
        const rr = grid[(y + dy + h) % h];
        for (let dx = -1; dx <= 1; dx++) if ((dx || dy) && rr[(x + dx + w) % w] === 1) n++;
      }
      if (s === 1) row.push(n >= 3 && n <= 5 ? 1 : 2);
      else row.push(n === 2 ? 1 : 0);
    }
    out.push(row);
  }
  return out;
}

function rule1d(bits, words, rule) {
  const out = new Uint32Array(words);
  const nbits = words * 32;
  for (let i = 0; i < nbits; i++) {
    const l = (i + nbits - 1) % nbits, r = (i + 1) % nbits;
    const lb = (bits[l >>> 5] >>> (l & 31)) & 1;
    const cb = (bits[i >>> 5] >>> (i & 31)) & 1;
    const rb = (bits[r >>> 5] >>> (r & 31)) & 1;
    if ((rule >>> ((lb << 2) | (cb << 1) | rb)) & 1) out[i >>> 5] |= 1 << (i & 31);
  }
  return out;
}
function popcount(x) { x -= (x >>> 1) & 0x55555555; x = (x & 0x33333333) + ((x >>> 2) & 0x33333333); return (Math.imul((x + (x >>> 4)) & 0x0f0f0f0f, 0x01010101) >>> 24); }

// Life.
let a = new Uint8Array(SIZE * SIZE), b = new Uint8Array(SIZE * SIZE);
for (let i = 0; i < a.length; i++) a[i] = rnd() % 4 === 0 ? 1 : 0;
// a glider gun-ish seed in a corner
const glider = [[1, 0], [2, 1], [0, 2], [1, 2], [2, 2]];
for (const [x, y] of glider) a[(y + 5) * SIZE + x + 5] = 1;
const population = [];
const seenRows = new Map();
for (let g = 0; g < GENS; g++) {
  population.push(lifeStep(a, b, SIZE, SIZE));
  const t = a; a = b; b = t;
  if (g % 10 === 0) {
    for (let y = 0; y < SIZE; y += 8) {
      const key = Array.prototype.join.call(a.subarray(y * SIZE, y * SIZE + 32), '');
      seenRows.set(key, (seenRows.get(key) || 0) + 1);
    }
  }
}
let lifeHash = 0;
for (let i = 0; i < a.length; i++) lifeHash = (Math.imul(lifeHash, 3) + a[i]) | 0;

// Generations.
const GS = SIZE >> 1;
let grid = [];
for (let y = 0; y < GS; y++) { const row = []; for (let x = 0; x < GS; x++) row.push(rnd() % 5 === 0 ? 1 : 0); grid.push(row); }
for (let g = 0; g < GENS >> 1; g++) grid = genStep(grid);
const states = [0, 0, 0, 0];
for (const row of grid) for (const s of row) states[s]++;

// 1-D rule 110 and rule 30.
const WORDS = SIZE >> 2;
let bits = new Uint32Array(WORDS);
bits[WORDS >> 1] = 1;
let ones110 = 0;
for (let g = 0; g < GENS * 2; g++) { bits = rule1d(bits, WORDS, 110); for (const w of bits) ones110 += popcount(w); }
let bits30 = new Uint32Array(WORDS);
bits30[WORDS >> 1] = 1;
let center = '';
for (let g = 0; g < GENS * 2; g++) { bits30 = rule1d(bits30, WORDS, 30); center += (bits30[WORDS >> 1] & 1); }

console.log('life final=' + population[population.length - 1], 'min=' + Math.min(...population), 'max=' + Math.max(...population), 'hash=' + (lifeHash >>> 0), 'rows=' + seenRows.size);
console.log('gen states=' + states.join('/'), 'r110=' + ones110, 'r30=' + parseInt(center.slice(0, 30), 2));
