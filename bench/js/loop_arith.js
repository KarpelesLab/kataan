// Tight numeric loop: integer and float arithmetic, local variables.
let a = 0, b = 1.5;
for (let i = 0; i < 30000000; i++) { a = (a + i * 3) | 0; b = b * 0.999999 + 0.5; }
console.log(a, Math.round(b));
