// BigInt number theory (original): modular exponentiation, Miller-Rabin
// primality on 128-bit candidates, an iterative factorial, Fibonacci by fast
// doubling, a Pollard rho factorization and radix conversion of big results.
const SCALE = typeof BM_SCALE === 'number' ? BM_SCALE : 5;

function modpow(b, e, m) {
  let r = 1n; b %= m;
  while (e > 0n) { if (e & 1n) r = (r * b) % m; b = (b * b) % m; e >>= 1n; }
  return r;
}
const WITNESSES = [2n, 3n, 5n, 7n, 11n, 13n, 17n, 19n, 23n, 29n, 31n, 37n];
function isPrime(n) {
  if (n < 2n) return false;
  for (const p of WITNESSES) { if (n === p) return true; if (n % p === 0n) return false; }
  let d = n - 1n, s = 0;
  while ((d & 1n) === 0n) { d >>= 1n; s++; }
  outer: for (const a of WITNESSES) {
    let x = modpow(a, d, n);
    if (x === 1n || x === n - 1n) continue;
    for (let i = 1; i < s; i++) { x = (x * x) % n; if (x === n - 1n) continue outer; }
    return false;
  }
  return true;
}
function gcd(a, b) { while (b) { [a, b] = [b, a % b]; } return a; }
function rho(n, c) {
  let x = 2n, y = 2n, d = 1n;
  while (d === 1n) {
    x = (x * x + c) % n; y = (y * y + c) % n; y = (y * y + c) % n;
    d = gcd(x > y ? x - y : y - x, n);
  }
  return d;
}
function fibPair(n) { // [F(n), F(n+1)]
  if (n === 0) return [0n, 1n];
  const [a, b] = fibPair(n >> 1);
  const c = a * (2n * b - a), d = a * a + b * b;
  return n & 1 ? [d, c + d] : [c, d];
}

let acc = 0n;
const base = (1n << 127n) + 12345n;
let primes = 0;
for (let i = 0n; i < BigInt(120 * SCALE); i++) {
  const n = base + 2n * i;
  if (isPrime(n)) { primes++; acc ^= n; }
}
let fact = 1n;
for (let i = 1n; i <= 1500n; i++) fact *= i;
const factStr = fact.toString();
let digitSum = 0;
for (let i = 0; i < factStr.length; i++) digitSum += factStr.charCodeAt(i) - 48;
const fib = fibPair(20000 * SCALE)[0];
const fibHex = fib.toString(16);
const semis = [1000003n * 998244353n, 1000000007n * 1000000009n, 2147483647n * 4294967291n];
const factors = semis.map((n) => { const f = rho(n, 1n); return String(f < n / f ? f : n / f); });
let mix = 0n;
for (let i = 0; i < 3000 * SCALE; i++) {
  const v = BigInt.asUintN(64, BigInt(i) * 0x9e3779b97f4a7c15n);
  mix = BigInt.asIntN(96, mix * 31n + (v >> 7n) - (v & 0xffffn));
}
console.log(primes, (acc % 1000000007n).toString(), factStr.length, digitSum, fibHex.length, fibHex.slice(0, 12),
  factors.join(','), mix.toString(36), (fact % (2n ** 61n - 1n)).toString(), String(BigInt(Number.MAX_SAFE_INTEGER) * -3n));
