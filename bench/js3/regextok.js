// A regex-driven tokenizer and log scanner (original): sticky-flag lexing of
// a small config language with named groups, lookbehind/lookahead field
// extraction from log lines, Unicode property and v-flag set classes,
// matchAll with indices, and replace with named-group callbacks.
const LINES = typeof RT_LINES === 'number' ? RT_LINES : 12000;

let seed = 5;
function rnd(n) { seed = (seed * 69069 + 1) >>> 0; return (seed >>> 16) % n; }
const IDENTS = ['alpha', 'βeta', 'gamma_2', 'délta', 'x', 'ω', 'naïve', 'path'];
const src = [];
for (let i = 0; i < LINES; i++) {
  const id = IDENTS[rnd(IDENTS.length)];
  switch (rnd(4)) {
    case 0: src.push(`${id} = ${rnd(100000) / 100};`); break;
    case 1: src.push(`${id} = "str ${i} \\"q\\"";  # comment ${i}`); break;
    case 2: src.push(`if (${id} >= 0x${rnd(65535).toString(16)}) { ${id} += ${rnd(9)}; }`); break;
    default: src.push(`[${IDENTS[rnd(8)]}, ${rnd(50)}, true] -> ${id}`);
  }
}
const program = src.join('\n');
const TOKEN = /(?<ws>\s+)|(?<comment>#[^\n]*)|(?<num>0x[0-9a-f]+|\d+(?:\.\d+)?)|(?<str>"(?:[^"\\]|\\.)*")|(?<id>[\p{L}_][\p{L}\p{N}_]*)|(?<op>>=|->|\+=|[=;(){}\[\],<>+])/uy;
const counts = Object.create(null);
let pos = 0, numSum = 0, idLen = 0;
while (pos < program.length) {
  TOKEN.lastIndex = pos;
  const m = TOKEN.exec(program);
  if (!m) throw new Error('lex error at ' + pos);
  const g = m.groups;
  let kind;
  for (const k in g) if (g[k] !== undefined) { kind = k; break; }
  counts[kind] = (counts[kind] || 0) + 1;
  if (kind === 'num') numSum += Number(m[0]);
  else if (kind === 'id') idLen += m[0].length;
  pos = TOKEN.lastIndex;
}
const logs = [];
for (let i = 0; i < LINES; i++) logs.push(`2024-05-${10 + (i % 20)}T${String(i % 24).padStart(2, '0')}:00:00Z level=${['info', 'warn', 'error'][i % 3]} user=u${rnd(900)} latency=${rnd(5000)}ms path=/api/v${1 + (i % 3)}/items/${rnd(10000)} msg="ok ${i}"`);
const logText = logs.join('\n');
let lat = 0, errs = 0, paths = 0;
for (const m of logText.matchAll(/(?<=latency=)(?<ms>\d+)(?=ms)/g)) lat += +m.groups.ms;
for (const m of logText.matchAll(/(?<=level=)error(?= user=u(?<uid>\d+))/g)) errs += +m.groups.uid;
for (const m of logText.matchAll(/path=(?<p>(?:\/[a-z0-9]+)+)/dg)) paths += m.indices.groups.p[1] - m.indices.groups.p[0];
const dated = logText.replace(/(?<y>\d{4})-(?<m>\d{2})-(?<d>\d{2})T/g, (...args) => { const g = args.at(-1); return `${g.d}/${g.m}/${g.y} `; });
const greekOrCyr = (program.match(/[\p{Script=Greek}--[α-γ]]/gv) || []).length;
const letters = (program.match(/[\p{L}&&\p{Ll}]/gv) || []).length;
const notAscii = program.replace(/[^\x00-\x7f]/gu, '?').split('?').length;
const words = logText.split(/\s+/).filter((w) => /^[a-z]+=/.test(w)).length;
console.log(JSON.stringify(counts), numSum.toFixed(2), idLen, lat, errs, paths, dated.length, greekOrCyr, letters, notAscii, words);
