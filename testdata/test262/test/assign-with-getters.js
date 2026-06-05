/*---
description: Object.assign invokes getters and copies enumerable own props
esid: sec-object.assign
---*/
var src = { a: 1, get b() { return 2; } };
var target = Object.assign({}, src);
assert.sameValue(target.a, 1);
assert.sameValue(target.b, 2, "getter invoked during assign");
var multi = Object.assign({}, { x: 1 }, { y: 2 }, { x: 9 });
assert.sameValue(multi.x, 9, "later source wins");
assert.sameValue(multi.y, 2);
var withNonEnum = {};
Object.defineProperty(withNonEnum, "hidden", { value: 5, enumerable: false });
withNonEnum.visible = 6;
var copy = Object.assign({}, withNonEnum);
assert.sameValue(copy.visible, 6);
assert.sameValue(copy.hidden, undefined, "non-enumerable not copied");
