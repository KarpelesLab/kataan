// Binary record parsing (original): serialize a stream of variable-length
// records into an ArrayBuffer with DataView (mixed endianness, varints, f32/f64,
// length-prefixed names), then parse it back, checksum it with a table-driven
// CRC-32 over a Uint8Array, and reinterpret the bytes through other views.
const RECORDS = typeof BP_RECORDS === 'number' ? BP_RECORDS : 150000;

let seed = 2463534242;
function rnd() { seed ^= seed << 13; seed ^= seed >>> 17; seed ^= seed << 5; return seed >>> 0; }

const CRC = new Uint32Array(256);
for (let n = 0; n < 256; n++) {
  let c = n;
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  CRC[n] = c >>> 0;
}
function crc32(bytes, from, to) {
  let c = 0xffffffff;
  for (let i = from; i < to; i++) c = CRC[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

const NAMES = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta', 'iota', 'kappa'];
const enc = [];
for (const s of NAMES) { const b = []; for (let i = 0; i < s.length; i++) b.push(s.charCodeAt(i)); enc.push(b); }

const buf = new ArrayBuffer(RECORDS * 48 + 16);
const dv = new DataView(buf);
const u8 = new Uint8Array(buf);
let off = 0;
function putVarint(v) {
  while (v >= 0x80) { dv.setUint8(off++, (v & 0x7f) | 0x80); v >>>= 7; }
  dv.setUint8(off++, v);
}
dv.setUint32(off, 0x4b415441); off += 4;           // magic, big-endian
dv.setUint32(off, RECORDS, true); off += 4;        // count, little-endian
for (let r = 0; r < RECORDS; r++) {
  const kind = rnd() % 4;
  dv.setUint8(off++, kind);
  putVarint(rnd() % 300000);
  const nm = enc[rnd() % enc.length];
  dv.setUint8(off++, nm.length);
  for (let i = 0; i < nm.length; i++) u8[off++] = nm[i];
  if (kind === 0) { dv.setFloat64(off, (rnd() % 100000) / 7, true); off += 8; }
  else if (kind === 1) { dv.setFloat32(off, (rnd() % 1000) / 3); off += 4; }
  else if (kind === 2) { dv.setInt16(off, (rnd() % 65536) - 32768, true); off += 2; dv.setInt32(off, rnd() | 0); off += 4; }
  else { dv.setBigUint64(off, BigInt(rnd()) * 4096n + BigInt(rnd() & 4095), true); off += 8; }
}
const total = off;

function parse() {
  let p = 0;
  const magic = dv.getUint32(p); p += 4;
  const count = dv.getUint32(p, true); p += 4;
  const sums = new Float64Array(4);
  const counts = new Int32Array(4);
  const nameHits = new Map();
  let ids = 0, big = 0n;
  for (let r = 0; r < count; r++) {
    const kind = dv.getUint8(p++);
    let v = 0, shift = 0, b;
    do { b = dv.getUint8(p++); v |= (b & 0x7f) << shift; shift += 7; } while (b & 0x80);
    ids = (ids + v) % 1000000007;
    const len = dv.getUint8(p++);
    let name = '';
    for (let i = 0; i < len; i++) name += String.fromCharCode(u8[p + i]);
    p += len;
    nameHits.set(name, (nameHits.get(name) || 0) + 1);
    counts[kind]++;
    if (kind === 0) { sums[0] += dv.getFloat64(p, true); p += 8; }
    else if (kind === 1) { sums[1] += dv.getFloat32(p); p += 4; }
    else if (kind === 2) { sums[2] += dv.getInt16(p, true) + dv.getInt32(p + 2) / 65536; p += 6; }
    else { big += dv.getBigUint64(p, true) & 0xffffn; p += 8; }
  }
  return { magic, ids, sums, counts, nameHits, big, end: p };
}

let out;
for (let pass = 0; pass < 3; pass++) out = parse();
const crc = crc32(u8, 0, total);
const f32 = new Float32Array(buf, 0, (total >> 2) & ~1);
let fsum = 0;
for (let i = 0; i < f32.length; i++) { const x = f32[i]; if (x === x && Math.abs(x) < 1e6) fsum += x; }
const u16 = new Uint16Array(buf.slice(0, total & ~1));
let x16 = 0;
for (let i = 0; i < u16.length; i++) x16 = (x16 * 31 + u16[i]) >>> 0;
const sorted = Int32Array.from(out.counts).sort();
const sub = u8.subarray(8, 8 + 4096);
const copy = new Uint8Array(sub.length);
copy.set(sub);
copy.reverse();
let sx = 0;
for (let i = 0; i < copy.length; i += 7) sx ^= copy[i] << (i & 15);
console.log(out.magic.toString(16), out.end === total, out.ids, out.counts.join(','),
  out.sums[0].toFixed(3), out.sums[1].toFixed(3), out.sums[2].toFixed(3), String(out.big),
  [...out.nameHits].map(([k, v]) => k[0] + v).join(''), crc.toString(16), fsum.toFixed(2), x16, sorted.join('|'), sx,
  copy.indexOf(0x61), copy.includes(255));
