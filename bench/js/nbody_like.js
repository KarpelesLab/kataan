// Float-heavy physics loop (original): arrays of objects, Math.sqrt.
const bodies = [];
for (let i = 0; i < 6; i++) bodies.push({ x: i, y: i * 0.5, z: -i, vx: 0, vy: 0, vz: 0, m: 1 + i * 0.1 });
function step(dt) {
  for (let i = 0; i < bodies.length; i++) { const a = bodies[i];
    for (let j = i + 1; j < bodies.length; j++) { const b = bodies[j];
      const dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
      const d2 = dx * dx + dy * dy + dz * dz + 0.01; const mag = dt / (d2 * Math.sqrt(d2));
      a.vx -= dx * b.m * mag; a.vy -= dy * b.m * mag; a.vz -= dz * b.m * mag;
      b.vx += dx * a.m * mag; b.vy += dy * a.m * mag; b.vz += dz * a.m * mag; } }
  for (const b of bodies) { b.x += dt * b.vx; b.y += dt * b.vy; b.z += dt * b.vz; }
}
for (let i = 0; i < 300000; i++) step(0.001);
let e = 0; for (const b of bodies) e += b.m * (b.vx * b.vx + b.vy * b.vy + b.vz * b.vz);
console.log(e.toFixed(6));
