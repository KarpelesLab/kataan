// Monomorphic property reads/writes on small objects.
function Point(x, y) { this.x = x; this.y = y; }
const pts = [];
for (let i = 0; i < 1000; i++) pts.push(new Point(i, i * 2));
let s = 0;
for (let r = 0; r < 5000; r++) {
  for (let i = 0; i < pts.length; i++) { const p = pts[i]; s += p.x + p.y; p.x = p.y - p.x; }
}
console.log(s);
