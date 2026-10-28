// JSON.stringify / JSON.parse round trips on nested data.
const data = [];
for (let i = 0; i < 20000; i++) data.push({ id: i, name: 'user' + i, tags: ['a', 'b', i % 5], score: i / 7, ok: i % 2 === 0 });
let n = 0;
for (let r = 0; r < 10; r++) { const txt = JSON.stringify(data); const back = JSON.parse(txt); n += back.length + txt.length; }
console.log(n);
