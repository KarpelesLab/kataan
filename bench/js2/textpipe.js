// A text-processing pipeline (original): synthesize a corpus of sentences from
// a seeded generator, then split into lines and words, normalize with regular
// expressions, count with Maps, find bigrams, rank by frequency with a
// comparator sort, and render a report with join/padStart.
const LINES = typeof TP_LINES === 'number' ? TP_LINES : 16000;

let seed = 12345;
function rand() { seed = (Math.imul(seed, 1103515245) + 12345) & 0x7fffffff; return seed; }

const SYLLABLES = ['ka', 'ta', 'an', 'ro', 'mi', 'su', 'le', 'vo', 'ne', 'qui', 'zor', 'pha', 'el', 'th', 'ar'];
const vocab = [];
for (let i = 0; i < 400; i++) {
  let w = '';
  const n = 1 + (rand() % 3);
  for (let j = 0; j < n; j++) w += SYLLABLES[rand() % SYLLABLES.length];
  vocab.push(w);
}
const PUNCT = ['.', ',', ';', '!', '?', ' -', ':'];

function sentence() {
  const len = 4 + (rand() % 12);
  const parts = [];
  for (let i = 0; i < len; i++) {
    // Zipf-ish: smaller indices more likely.
    const r = rand() % 1000;
    const idx = Math.floor((r * r) / 2500) % vocab.length;
    let w = vocab[idx];
    if (i === 0 || rand() % 17 === 0) w = w[0].toUpperCase() + w.slice(1);
    if (rand() % 23 === 0) w = w + (rand() % 100);
    parts.push(w);
    if (rand() % 9 === 0) parts.push(PUNCT[rand() % PUNCT.length]);
  }
  return parts.join(' ') + '.';
}

const lines = [];
for (let i = 0; i < LINES; i++) lines.push((i % 50 === 0 ? '## Section ' + (i / 50) + '\n' : '') + sentence());
const corpus = lines.join('\n');

// Stage 1: tokenize.
const wordRe = /[A-Za-z]+(?:\d+)?/g;
const freq = new Map();
const bigrams = new Map();
const lengths = new Map();
let totalWords = 0;
let headings = 0;
for (const line of corpus.split('\n')) {
  if (line.startsWith('## ')) { headings++; continue; }
  const words = line.toLowerCase().match(wordRe) || [];
  let prev = null;
  for (const raw of words) {
    const w = raw.replace(/\d+$/, '');
    totalWords++;
    freq.set(w, (freq.get(w) || 0) + 1);
    lengths.set(w.length, (lengths.get(w.length) || 0) + 1);
    if (prev !== null) {
      const key = prev + ' ' + w;
      bigrams.set(key, (bigrams.get(key) || 0) + 1);
    }
    prev = w;
  }
}

// Stage 2: rank.
const byCount = (a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0);
const topWords = [...freq.entries()].sort(byCount).slice(0, 25);
const topBigrams = Array.from(bigrams).sort(byCount).slice(0, 15);

// Stage 3: per-sentence statistics with string methods.
let longest = '';
let vowelHeavy = 0;
let capitals = 0;
const sentenceRe = /[^.!?]+[.!?]/g;
let m;
let sentences = 0;
while ((m = sentenceRe.exec(corpus)) !== null) {
  sentences++;
  const s = m[0].trim();
  if (s.length > longest.length) longest = s;
  const vowels = s.replace(/[^aeiou]/gi, '').length;
  if (vowels * 2 > s.length) vowelHeavy++;
  capitals += (s.match(/\b[A-Z]/g) || []).length;
}

// Stage 4: a substitution pass and a word-wrap of a slice.
const redacted = corpus.slice(0, 20000).replace(/\b(ka|ta)(\w*)/g, (all, p, rest) => p.toUpperCase() + rest.length);
const wrapped = [];
let cur = '';
for (const w of redacted.split(/\s+/)) {
  if (cur.length + w.length + 1 > 72) { wrapped.push(cur); cur = w; } else cur = cur ? cur + ' ' + w : w;
}
if (cur) wrapped.push(cur);

// Report.
const report = [];
report.push('words ' + totalWords + ' distinct ' + freq.size + ' bigrams ' + bigrams.size + ' headings ' + headings);
for (const [w, c] of topWords) report.push(w.padEnd(12, '.') + String(c).padStart(7));
for (const [b, c] of topBigrams) report.push('[' + b + '] ' + c);
const lens = [...lengths.keys()].sort((a, b) => a - b).map((k) => k + ':' + lengths.get(k));
report.push(lens.join(' '));
report.push('sentences ' + sentences + ' vowelHeavy ' + vowelHeavy + ' capitals ' + capitals + ' longest ' + longest.length);
report.push('wrapped ' + wrapped.length + ' ' + wrapped[wrapped.length >> 1]);
const text = report.join('\n');
let h = 0;
for (let i = 0; i < text.length; i++) h = (Math.imul(h, 33) ^ text.charCodeAt(i)) >>> 0;
console.log(report[0]);
console.log(report[1], '|', report[26]);
console.log('report', text.length, 'hash', h);
