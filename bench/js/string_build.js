// String concatenation, templates, slicing, charCodeAt, indexOf.
let str = '';
for (let i = 0; i < 200000; i++) str += `${i % 10}`;
let h = 0;
for (let i = 0; i < str.length; i++) h = (h * 31 + str.charCodeAt(i)) | 0;
const parts = [];
for (let i = 0; i < 100000; i++) parts.push('item' + i);
const joined = parts.join(',');
console.log(h, joined.length, joined.indexOf('item99999'), str.slice(1000, 1010));
