// Closure creation and invocation, captured variable updates.
function counter() { let c = 0; return () => ++c; }
let total = 0;
for (let i = 0; i < 300000; i++) { const f = counter(); total += f() + f() + f(); }
const fs = []; for (let i = 0; i < 1000; i++) fs.push((x) => x + i);
for (let r = 0; r < 2000; r++) for (let i = 0; i < fs.length; i++) total = (total + fs[i](r)) | 0;
console.log(total);
