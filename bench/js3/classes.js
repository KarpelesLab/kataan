// A class-based simulation of bank accounts (original): an inheritance
// chain with super calls, private fields and private methods, getters and
// setters with validation, static blocks and static private state, `#x in o`
// brand checks, accessor overrides and instanceof dispatch.
const N = typeof CL_N === 'number' ? CL_N : 600000;

class Account {
  static #nextId = 1;
  static registry;
  static { Account.registry = new Map(); }
  #balance = 0;
  #history = 0;
  constructor(owner) { this.id = Account.#nextId++; this.owner = owner; Account.registry.set(this.id, this); }
  get balance() { return this.#balance; }
  set balance(v) { if (!Number.isFinite(v)) throw new RangeError('bad'); this.#balance = v; }
  #record(delta) { this.#history = (this.#history * 31 + delta) | 0; }
  deposit(x) { this.balance = this.#balance + x; this.#record(x); return this; }
  withdraw(x) { if (x > this.#balance) return false; this.balance = this.#balance - x; this.#record(-x); return true; }
  get digest() { return this.#history; }
  fee() { return 1; }
  static isAccount(o) { return #balance in o; }
}
class Savings extends Account {
  #rate;
  constructor(owner, rate) { super(owner); this.#rate = rate; }
  fee() { return 0; }
  accrue() { this.deposit(Math.floor(this.balance * this.#rate)); }
  get rate() { return this.#rate; }
}
class Premium extends Savings {
  static tier = 'gold';
  static #count = 0;
  static { Premium.#count = 100; }
  constructor(owner) { super(owner, 0.002); Premium.#count++; }
  fee() { return super.fee() + 2; }
  withdraw(x) { return super.withdraw(x + this.fee()); }
  get balance() { return super.balance; }
  set balance(v) { super.balance = Math.max(v, 0); }
  static get count() { return Premium.#count; }
}

let seed = 77;
function rnd(n) { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed % n; }
const accounts = [];
for (let i = 0; i < 900; i++) accounts.push(i % 3 === 0 ? new Account('a' + i) : i % 3 === 1 ? new Savings('s' + i, 0.001) : new Premium('p' + i));
let failed = 0, fees = 0, errors = 0;
for (let i = 0; i < N; i++) {
  const a = accounts[rnd(accounts.length)];
  const op = rnd(10);
  if (op < 5) a.deposit(rnd(500));
  else if (op < 9) { if (!a.withdraw(rnd(400))) failed++; fees += a.fee(); }
  else if (a instanceof Savings) a.accrue();
  if (i % 5000 === 0) { try { a.balance = NaN; } catch (e) { if (e instanceof RangeError) errors++; } }
}
let total = 0, digest = 0, brands = 0, rates = 0;
for (const a of accounts) {
  total += a.balance; digest = (digest ^ a.digest) | 0;
  if (Account.isAccount(a)) brands++;
  if (a instanceof Premium) rates += a.rate * 1000;
}
const notAcct = Account.isAccount({ balance: 1 });
console.log(total, digest, failed, fees, errors, brands, rates, Premium.count, Premium.tier, Account.registry.size, notAcct,
  Object.getOwnPropertyNames(Premium.prototype).join(','));
