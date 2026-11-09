// A small recursive ray tracer (original): immutable vector objects, a class
// hierarchy of scene objects with polymorphic intersect/normal methods,
// materials with reflection, point lights with shadow rays, and supersampling.
// Uses only + - * / and Math.sqrt so the result is bit-identical everywhere.
const SCALE = typeof RT_SCALE === 'number' ? RT_SCALE : 1;

class Vec {
  constructor(x, y, z) { this.x = x; this.y = y; this.z = z; }
  add(o) { return new Vec(this.x + o.x, this.y + o.y, this.z + o.z); }
  sub(o) { return new Vec(this.x - o.x, this.y - o.y, this.z - o.z); }
  mul(s) { return new Vec(this.x * s, this.y * s, this.z * s); }
  hadamard(o) { return new Vec(this.x * o.x, this.y * o.y, this.z * o.z); }
  dot(o) { return this.x * o.x + this.y * o.y + this.z * o.z; }
  cross(o) {
    return new Vec(this.y * o.z - this.z * o.y, this.z * o.x - this.x * o.z, this.x * o.y - this.y * o.x);
  }
  length() { return Math.sqrt(this.dot(this)); }
  norm() { const l = this.length(); return l === 0 ? this : this.mul(1 / l); }
  reflect(n) { return this.sub(n.mul(2 * this.dot(n))); }
}
const ZERO = new Vec(0, 0, 0);

class Ray {
  constructor(origin, dir) { this.origin = origin; this.dir = dir; }
  at(t) { return this.origin.add(this.dir.mul(t)); }
}

class Material {
  constructor(color, diffuse, specular, shininess, reflect) {
    this.color = color; this.diffuse = diffuse; this.specular = specular;
    this.shininess = shininess; this.reflect = reflect;
  }
  colorAt(_p) { return this.color; }
}
class Checker extends Material {
  constructor(a, b, size) { super(a, 0.8, 0.1, 8, 0.25); this.other = b; this.size = size; }
  colorAt(p) {
    const ix = Math.floor(p.x / this.size), iz = Math.floor(p.z / this.size);
    return ((ix + iz) & 1) === 0 ? this.color : this.other;
  }
}

class Shape {
  constructor(material) { this.material = material; }
  intersect(_ray) { return Infinity; }
}
class Sphere extends Shape {
  constructor(center, radius, material) { super(material); this.center = center; this.radius = radius; this.r2 = radius * radius; }
  intersect(ray) {
    const oc = ray.origin.sub(this.center);
    const b = oc.dot(ray.dir);
    const c = oc.dot(oc) - this.r2;
    const disc = b * b - c;
    if (disc < 0) return Infinity;
    const s = Math.sqrt(disc);
    let t = -b - s;
    if (t > 1e-6) return t;
    t = -b + s;
    return t > 1e-6 ? t : Infinity;
  }
  normalAt(p) { return p.sub(this.center).mul(1 / this.radius); }
}
class Plane extends Shape {
  constructor(point, normal, material) { super(material); this.point = point; this.normal = normal.norm(); }
  intersect(ray) {
    const denom = this.normal.dot(ray.dir);
    if (Math.abs(denom) < 1e-9) return Infinity;
    const t = this.point.sub(ray.origin).dot(this.normal) / denom;
    return t > 1e-6 ? t : Infinity;
  }
  normalAt(_p) { return this.normal; }
}
class Box extends Shape {
  constructor(lo, hi, material) { super(material); this.lo = lo; this.hi = hi; }
  intersect(ray) {
    let tmin = -Infinity, tmax = Infinity;
    const axes = ['x', 'y', 'z'];
    for (let i = 0; i < 3; i++) {
      const a = axes[i];
      const d = ray.dir[a], o = ray.origin[a];
      if (Math.abs(d) < 1e-12) {
        if (o < this.lo[a] || o > this.hi[a]) return Infinity;
        continue;
      }
      let t1 = (this.lo[a] - o) / d, t2 = (this.hi[a] - o) / d;
      if (t1 > t2) { const tmp = t1; t1 = t2; t2 = tmp; }
      if (t1 > tmin) tmin = t1;
      if (t2 < tmax) tmax = t2;
      if (tmin > tmax) return Infinity;
    }
    if (tmin > 1e-6) return tmin;
    return tmax > 1e-6 ? tmax : Infinity;
  }
  normalAt(p) {
    const eps = 1e-4;
    if (Math.abs(p.x - this.lo.x) < eps) return new Vec(-1, 0, 0);
    if (Math.abs(p.x - this.hi.x) < eps) return new Vec(1, 0, 0);
    if (Math.abs(p.y - this.lo.y) < eps) return new Vec(0, -1, 0);
    if (Math.abs(p.y - this.hi.y) < eps) return new Vec(0, 1, 0);
    if (Math.abs(p.z - this.lo.z) < eps) return new Vec(0, 0, -1);
    return new Vec(0, 0, 1);
  }
}

class Light { constructor(pos, color) { this.pos = pos; this.color = color; } }

class Scene {
  constructor() { this.shapes = []; this.lights = []; this.ambient = new Vec(0.08, 0.08, 0.1); this.rays = 0; }
  add(s) { this.shapes.push(s); return this; }
  closest(ray) {
    this.rays++;
    let best = Infinity, hit = null;
    for (const s of this.shapes) {
      const t = s.intersect(ray);
      if (t < best) { best = t; hit = s; }
    }
    return hit === null ? null : { shape: hit, t: best };
  }
  occluded(p, toLight, dist) {
    const ray = new Ray(p, toLight);
    for (const s of this.shapes) {
      if (s.intersect(ray) < dist) return true;
    }
    return false;
  }
  trace(ray, depth) {
    const h = this.closest(ray);
    if (h === null) {
      const k = 0.5 * (ray.dir.y + 1);
      return new Vec(0.6, 0.7, 0.9).mul(k).add(new Vec(0.2, 0.2, 0.25).mul(1 - k));
    }
    const p = ray.at(h.t);
    const n = h.shape.normalAt(p);
    const m = h.shape.material;
    const base = m.colorAt(p);
    let col = base.hadamard(this.ambient);
    for (const light of this.lights) {
      const toL = light.pos.sub(p);
      const dist = toL.length();
      const l = toL.mul(1 / dist);
      const ndl = n.dot(l);
      if (ndl <= 0) continue;
      if (this.occluded(p.add(n.mul(1e-4)), l, dist)) continue;
      col = col.add(base.hadamard(light.color).mul(m.diffuse * ndl));
      const r = l.mul(-1).reflect(n);
      let spec = -r.dot(ray.dir);
      if (spec > 0) {
        let s = 1;
        for (let i = 0; i < m.shininess; i++) s *= spec;
        col = col.add(light.color.mul(m.specular * s));
      }
    }
    if (m.reflect > 0 && depth < 4) {
      const rd = ray.dir.reflect(n).norm();
      const rc = this.trace(new Ray(p.add(n.mul(1e-4)), rd), depth + 1);
      col = col.mul(1 - m.reflect).add(rc.mul(m.reflect));
    }
    return col;
  }
}

function buildScene() {
  const sc = new Scene();
  sc.add(new Plane(new Vec(0, -1, 0), new Vec(0, 1, 0), new Checker(new Vec(0.9, 0.9, 0.9), new Vec(0.2, 0.2, 0.2), 1)));
  const palette = [new Vec(0.9, 0.2, 0.2), new Vec(0.2, 0.8, 0.3), new Vec(0.2, 0.3, 0.9), new Vec(0.9, 0.8, 0.2)];
  for (let i = 0; i < 6; i++) {
    const ang = i / 6;
    const x = (ang - 0.5) * 6, z = 5 + (i % 3);
    const r = 0.5 + (i % 2) * 0.3;
    sc.add(new Sphere(new Vec(x, r - 1, z), r, new Material(palette[i % 4], 0.7, 0.4, 16, i % 2 === 0 ? 0.3 : 0)));
  }
  sc.add(new Box(new Vec(-0.6, -1, 3), new Vec(0.6, -0.2, 4), new Material(new Vec(0.7, 0.5, 0.9), 0.6, 0.5, 32, 0.15)));
  sc.lights.push(new Light(new Vec(-4, 6, 0), new Vec(0.8, 0.8, 0.8)));
  sc.lights.push(new Light(new Vec(5, 4, 2), new Vec(0.4, 0.35, 0.3)));
  return sc;
}

function render(sc, width, height, samples) {
  const eye = new Vec(0, 1, -2);
  const target = new Vec(0, 0, 5);
  const fwd = target.sub(eye).norm();
  const right = fwd.cross(new Vec(0, 1, 0)).norm().mul(-1);
  const up = right.cross(fwd).mul(-1);
  const pixels = new Uint8Array(width * height * 3);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let acc = ZERO;
      for (let s = 0; s < samples; s++) {
        const jx = (s % 2) * 0.5 + 0.25, jy = ((s >> 1) % 2) * 0.5 + 0.25;
        const u = ((x + jx) / width - 0.5) * 1.6;
        const v = (0.5 - (y + jy) / height) * 1.2;
        const dir = fwd.add(right.mul(u)).add(up.mul(v)).norm();
        acc = acc.add(sc.trace(new Ray(eye, dir), 0));
      }
      acc = acc.mul(1 / samples);
      const o = (y * width + x) * 3;
      pixels[o] = Math.min(255, Math.floor(acc.x * 255));
      pixels[o + 1] = Math.min(255, Math.floor(acc.y * 255));
      pixels[o + 2] = Math.min(255, Math.floor(acc.z * 255));
    }
  }
  return pixels;
}

const scene = buildScene();
const W = Math.round(200 * SCALE), H = Math.round(150 * SCALE);
const img = render(scene, W, H, 4);
let h = 0;
for (let i = 0; i < img.length; i++) h = (Math.imul(h, 31) + img[i]) | 0;
let bright = 0;
for (let i = 0; i < img.length; i += 3) if (img[i] + img[i + 1] + img[i + 2] > 450) bright++;
console.log(W + 'x' + H, 'rays=' + scene.rays, 'bright=' + bright, 'hash=' + (h >>> 0));
