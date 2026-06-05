/*---
description: Property enumeration order with mixed integer and string keys
esid: sec-ordinaryownpropertykeys
---*/
var obj = {};
obj["2"] = "a";
obj["1"] = "b";
obj["banana"] = "c";
obj["10"] = "d";
obj["apple"] = "e";
assert.sameValue(Object.keys(obj).join(","), "1,2,10,banana,apple", "integers ascending then insertion");
var forIn = [];
for (var key in obj) forIn.push(key);
assert.sameValue(forIn.join(","), "1,2,10,banana,apple", "for-in same order");
var values = Object.values(obj);
assert.sameValue(values.join(","), "b,a,d,c,e", "values follow key order");
var entries = Object.entries(obj);
assert.sameValue(entries[0].join("="), "1=b");
assert.sameValue(entries.length, 5);
var mixed = { z: 1, "0": 2, y: 3, "5": 4 };
assert.sameValue(Object.keys(mixed).join(","), "0,5,z,y");
var spread = { ...obj };
assert.sameValue(Object.keys(spread).join(","), "1,2,10,banana,apple", "spread preserves order");
