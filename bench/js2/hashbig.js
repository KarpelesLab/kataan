// Hashing and big-integer arithmetic (original): a 32-bit ARX block mixer over
// byte buffers (Math.imul, rotates, DataView reads), a Merkle tree of those
// digests, then BigInt modular exponentiation, a Miller-Rabin primality scan
// and a BigInt Fibonacci by fast doubling.
const MB = typeof HB_BLOCKS === 'number' ? HB_BLOCKS : 16384;
const PRIME_SCAN = typeof HB_PRIMES === 'number' ? HB_PRIMES : 1500;

function rotl(x, r) { return (x << r) | (x >>> (32 - r)); }

// Digest 64-byte blocks into an 8-word state.
function digest(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const st = new Uint32Array([0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]);
  const w = new Uint32Array(16);
  for (let off = 0; off + 64 <= bytes.length; off += 64) {
    for (let i = 0; i < 16; i++) w[i] = view.getUint32(off + i * 4, true);
    let a = st[0], b = st[1], c = st[2], d = st[3], e = st[4], f = st[5], g = st[6], h = st[7];
    for (let round = 0; round < 24; round++) {
      const m = w[round & 15] ^ w[(round * 7 + 3) & 15];
      a = (a + b + m) | 0; d = rotl(d ^ a, 16);
      c = (c + d) | 0; b = rotl(b ^ c, 12);
      e = (e + f + Math.imul(m, 0x9e3779b1)) | 0; h = rotl(h ^ e, 8);
      g = (g + h) | 0; f = rotl(f ^ g, 7);
      const t = a; a = b; b = c; c = d; d = e; e = f; f = g; g = h; h = t;
      w[round & 15] = (w[round & 15] + rotl(m, 3)) | 0;
    }
    st[0] += a; st[1] += b; st[2] += c; st[3] += d; st[4] += e; st[5] += f; st[6] += g; st[7] += h;
  }
  return st;
}

function hex(words) {
  let s = '';
  for (let i = 0; i < words.length; i++) s += (words[i] >>> 0).toString(16).padStart(8, '0');
  return s;
}

// Build input data and a Merkle tree over 1 KiB leaves.
const data = new Uint8Array(MB * 64);
let x = 0x1234567;
for (let i = 0; i < data.length; i++) { x ^= x << 13; x ^= x >>> 17; x ^= x << 5; data[i] = x & 255; }
let level = [];
for (let off = 0; off < data.length; off += 1024) level.push(digest(data.subarray(off, off + 1024)));
let depth = 0;
while (level.length > 1) {
  const next = [];
  for (let i = 0; i < level.length; i += 2) {
    const pair = new Uint8Array(64);
    pair.set(new Uint8Array(level[i].buffer), 0);
    pair.set(new Uint8Array((level[i + 1] || level[i]).buffer), 32);
    next.push(digest(pair));
  }
  level = next;
  depth++;
}
const root = hex(level[0]);
const whole = hex(digest(data));

// BigInt part.
function modpow(b, e, m) {
  let r = 1n;
  b %= m;
  while (e > 0n) {
    if (e & 1n) r = (r * b) % m;
    b = (b * b) % m;
    e >>= 1n;
  }
  return r;
}
function isProbablePrime(n) {
  if (n < 2n) return false;
  for (const p of [2n, 3n, 5n, 7n, 11n, 13n, 17n, 19n, 23n, 29n, 31n, 37n]) {
    if (n === p) return true;
    if (n % p === 0n) return false;
  }
  let d = n - 1n, s = 0;
  while ((d & 1n) === 0n) { d >>= 1n; s++; }
  outer: for (const a of [2n, 3n, 5n, 7n, 11n, 13n, 17n, 19n, 23n, 29n, 31n, 37n]) {
    let y = modpow(a, d, n);
    if (y === 1n || y === n - 1n) continue;
    for (let i = 1; i < s; i++) {
      y = (y * y) % n;
      if (y === n - 1n) continue outer;
    }
    return false;
  }
  return true;
}
function fibPair(n) { // returns [F(n), F(n+1)]
  if (n === 0) return [0n, 1n];
  const [a, b] = fibPair(n >> 1);
  const c = a * (2n * b - a);
  const d = a * a + b * b;
  return (n & 1) ? [d, c + d] : [c, d];
}

const base = (1n << 255n) + BigInt('0x' + root.slice(0, 16));
let found = [];
for (let k = 0n; k < BigInt(PRIME_SCAN); k++) {
  const c = (base | 1n) + 2n * k;
  if (isProbablePrime(c)) found.push(c);
}
const f = fibPair(30000)[0];
const fdigits = f.toString().length;
let mix = 0n;
for (const p of found) mix = (mix * 1000003n + (p % 1000000007n)) % 18446744073709551557n;
console.log('blocks=' + MB, 'depth=' + depth, 'root=' + root.slice(0, 24), 'whole=' + whole.slice(0, 24));
console.log('primes=' + found.length, 'mix=' + mix.toString(36), 'fibdigits=' + fdigits, 'fibmod=' + (f % 999999937n));
