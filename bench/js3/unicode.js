// Unicode text handling (original): build a multilingual corpus with accented
// Latin, Greek, Cyrillic, CJK and astral emoji, then normalize (NFC/NFD/NFKC),
// case-map, walk code points across surrogate pairs, strip combining marks,
// group by folded keys and rebuild strings with String.fromCodePoint.
const N = typeof UC_N === 'number' ? UC_N : 30000;

let seed = 99;
function rnd(n) { seed = (seed * 1103515245 + 12345) % 2147483648; return seed % n; }
const PIECES = ['café', 'naïve', 'Straße', 'Ångström', 'ﬁnancial', 'Ελληνικά', 'Привет', 'İstanbul',
  '東京', '한국어', '😀', '👍🏽', '𝔘𝔫𝔦𝔠𝔬𝔡𝔢', 'é', 'Å', 'ǅemal', 'ﬀ', '½', 'ß', 'ΣΟΦΟΣ', 'ǰ'];
const words = [];
for (let i = 0; i < N; i++) {
  let w = PIECES[rnd(PIECES.length)];
  if (rnd(3) === 0) w += PIECES[rnd(PIECES.length)];
  words.push(w);
}
const text = words.join(' ');

let h = 0;
function mixStr(s) { for (let i = 0; i < s.length; i++) h = (Math.imul(h, 31) + s.charCodeAt(i)) | 0; }

const nfd = text.normalize('NFD');
const nfc = nfd.normalize('NFC');
const nfkc = text.normalize('NFKC');
mixStr(nfc); mixStr(nfkc);
const stripped = nfd.replace(/\p{M}/gu, '');
mixStr(stripped);
const upper = text.toUpperCase();
const lower = upper.toLowerCase();
mixStr(upper); mixStr(lower);

let astral = 0, bmp = 0, cpSum = 0;
for (let i = 0; i < text.length; i++) {
  const cp = text.codePointAt(i);
  if (cp > 0xffff) { astral++; i++; } else bmp++;
  cpSum = (cpSum + cp) % 1000003;
}
const scripts = { latin: 0, greek: 0, cyrillic: 0, cjk: 0, hangul: 0, emoji: 0, other: 0 };
for (const ch of text) {
  const cp = ch.codePointAt(0);
  if (cp < 0x250) scripts.latin++;
  else if (cp >= 0x370 && cp < 0x400) scripts.greek++;
  else if (cp >= 0x400 && cp < 0x500) scripts.cyrillic++;
  else if (cp >= 0x4e00 && cp < 0xa000) scripts.cjk++;
  else if (cp >= 0xac00 && cp < 0xd7b0) scripts.hangul++;
  else if (cp >= 0x1f000) scripts.emoji++;
  else scripts.other++;
}
let rev = '';
for (let i = 0; i < words.length; i += 3) {
  const cps = Array.from(words[i], (c) => c.codePointAt(0)).reverse();
  rev += String.fromCodePoint(...cps);
}
mixStr(rev);
const groups = new Map();
for (const w of words) {
  const k = w.normalize('NFKD').toLowerCase().replace(/[̀-ͯ]/g, '');
  groups.set(k, (groups.get(k) || 0) + 1);
}
let wellFormed = 0;
for (let i = 0; i < words.length; i++) {
  const s = words[i] + (i % 97 === 0 ? '\ud800' : '');
  if (s.isWellFormed()) wellFormed++; else mixStr(s.toWellFormed());
}
let padded = 0;
for (let i = 0; i < words.length; i += 5) padded += words[i].padEnd(12, '·').at(-1) === '·' ? 1 : 0;
console.log(text.length, nfd.length, nfc.length, nfkc.length, stripped.length, upper.length, astral, bmp, cpSum,
  JSON.stringify(scripts), groups.size, wellFormed, padded, h, nfc === text.normalize());
