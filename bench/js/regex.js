// Regex test/exec/replace over generated text.
const lines = [];
for (let i = 0; i < 50000; i++) lines.push(`user${i}@host${i % 97}.example.com visited /page/${i % 1000}?q=${i * 7}`);
const text = lines.join('\n');
const email = /([a-z0-9]+)@([a-z0-9.]+)/g;
let c = 0, mm;
while ((mm = email.exec(text)) !== null) c += mm[1].length;
const replaced = text.replace(/\d+/g, (d) => String(d.length));
let hits = 0; for (const l of lines) if (/page\/(\d)9\d/.test(l)) hits++;
console.log(c, replaced.length, hits);
