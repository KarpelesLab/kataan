// Map/Set with integer and string keys.
const m = new Map(); const st = new Set();
for (let i = 0; i < 300000; i++) { m.set(i, i * 2); st.add('k' + (i % 50000)); }
let s = 0;
for (let i = 0; i < 300000; i++) s += m.get(i) + (st.has('k' + i) ? 1 : 0);
for (const [k, v] of m) s += k & v & 7;
console.log(s, m.size, st.size);
