// Meta-programming with Proxy and Reflect (original): a validating schema
// proxy, an observable store that records get/set/delete traps, a
// default-value proxy, method-call tracing through apply traps, and
// Reflect-driven object copying and construction.
const N = typeof MP_N === 'number' ? MP_N : 60000;

const log = { get: 0, set: 0, del: 0, has: 0, keys: 0, apply: 0, construct: 0 };
function observable(target) {
  return new Proxy(target, {
    get(t, k, r) { log.get++; return Reflect.get(t, k, r); },
    set(t, k, v, r) { log.set++; return Reflect.set(t, k, v, r); },
    deleteProperty(t, k) { log.del++; return Reflect.deleteProperty(t, k); },
    has(t, k) { log.has++; return Reflect.has(t, k); },
    ownKeys(t) { log.keys++; return Reflect.ownKeys(t); },
  });
}
function schema(types) {
  return new Proxy({}, {
    set(t, k, v) {
      if (!(k in types)) throw new TypeError('unknown field ' + String(k));
      if (typeof v !== types[k]) throw new TypeError('bad type for ' + k);
      t[k] = v; return true;
    },
  });
}
const withDefault = (d) => new Proxy({}, { get: (t, k) => (k in t ? t[k] : d) });
function traced(fn) { return new Proxy(fn, { apply(f, self, args) { log.apply++; return Reflect.apply(f, self, args); } }); }
class Point { constructor(x, y) { this.x = x; this.y = y; } }
const PointP = new Proxy(Point, { construct(T, args, nt) { log.construct++; return Reflect.construct(T, args, nt); } });

const store = observable({ count: 0, items: [] });
let errors = 0, acc = 0;
const person = schema({ name: 'string', age: 'number' });
const hist = withDefault(0);
const add = traced((a, b) => a + b);
for (let i = 0; i < N; i++) {
  store.count = store.count + 1;
  if ('count' in store) acc += store.count & 1;
  store['k' + (i & 15)] = i;
  if ((i & 3) === 0) delete store['k' + ((i >> 2) & 15)];
  try { person.age = i & 1 ? i : 'x'; } catch (e) { errors++; }
  try { if (i % 101 === 0) person.email = 'a'; } catch (e) { errors++; }
  hist[i % 37] = hist[i % 37] + 1;
  acc = add(acc, i & 255) % 1000003;
  if ((i & 63) === 0) { const p = new PointP(i, -i); acc += p.x + p.y + (p instanceof Point ? 1 : 0); }
  if ((i & 1023) === 0) acc += Object.keys(store).length;
}
const copy = {};
for (const k of Reflect.ownKeys(store)) Reflect.defineProperty(copy, k, Reflect.getOwnPropertyDescriptor(store, k));
const frozen = Object.freeze({ a: 1 });
const okSet = Reflect.set(frozen, 'a', 2);
const proto = Reflect.getPrototypeOf(new PointP(1, 2)) === Point.prototype;
const revocable = Proxy.revocable({ v: 1 }, {});
revocable.revoke();
let revoked = false;
try { revocable.proxy.v; } catch (e) { revoked = e instanceof TypeError; }
console.log(acc, errors, JSON.stringify(log), Object.keys(copy).length, okSet, proto, revoked, hist[5], person.age);
