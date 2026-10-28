// Class methods, `this`, getters, prototype dispatch.
class Vec { constructor(x, y) { this.x = x; this.y = y; }
  add(o) { return new Vec(this.x + o.x, this.y + o.y); }
  get len2() { return this.x * this.x + this.y * this.y; } }
let v = new Vec(0, 0); const d = new Vec(1, 2); let t = 0;
for (let i = 0; i < 2000000; i++) { v = v.add(d); t += v.len2 % 7; }
console.log(v.x, v.y, t);
