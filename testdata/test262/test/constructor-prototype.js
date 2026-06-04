/*---
description: Constructor function .prototype and prototype-chain inheritance
esid: sec-ordinary-object-internal-methods
---*/
function Animal(name) { this.name = name; }
Animal.prototype.speak = function () { return this.name + " makes a sound"; };
Object.defineProperty(Animal.prototype, "loud", {
  get: function () { return this.name.toUpperCase(); }
});
var a = new Animal("cat");
assert.sameValue(a.speak(), "cat makes a sound", "method on the prototype");
assert.sameValue(a.loud, "CAT", "getter on the prototype");
assert.sameValue(a instanceof Animal, true);

function Dog(name) { Animal.call(this, name); this.legs = 4; }
Dog.prototype = Object.create(Animal.prototype);
var d = new Dog("rex");
assert.sameValue(d.speak(), "rex makes a sound", "inherited through two prototype levels");
assert.sameValue(d.legs, 4);
assert.sameValue(d.name, "rex");
