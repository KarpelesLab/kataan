/*---
description: private accessors (get #x / set #x) dispatch on read and write
features: [class-fields-private, class-methods-private]
---*/
// A private get/set pair: writing invokes the setter, reading the getter.
class G {
  #v = 10;
  get #priv() { return this.#v; }
  set #priv(x) { this.#v = x * 2; }
  run() { var before = this.#priv; this.#priv = 50; return before + "," + this.#priv; }
}
assert.sameValue(new G().run(), "10,100", "setter runs (50*2) then getter reads 100");

// A setter-only private accessor.
class H {
  #log = [];
  set #add(x) { this.#log.push(x); }
  run() { this.#add = 1; this.#add = 2; return this.#log.join(","); }
}
assert.sameValue(new H().run(), "1,2", "setter-only accessor collects writes");

// Public accessor delegating through a private accessor.
class P {
  #v = 0;
  get #pv() { return this.#v; }
  set #pv(x) { this.#v = x; }
  get pub() { return this.#pv; }
  set pub(x) { this.#pv = x + 1; }
}
var p = new P();
p.pub = 10;
assert.sameValue(p.pub, 11, "public->private accessor delegation");

// Private fields and methods are unaffected.
class F { #x = 1; bump() { this.#x++; return this.#x; } }
assert.sameValue(new F().bump(), 2, "private field");
class M { #m() { return 7; } call() { return this.#m(); } }
assert.sameValue(new M().call(), 7, "private method");
