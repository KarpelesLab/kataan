// Array push/index/length, higher-order methods with callbacks.
let arr = [];
for (let i = 0; i < 1000000; i++) arr.push(i * 2);
let s = 0;
for (let r = 0; r < 5; r++) {
  s += arr.map((x) => x + 1).filter((x) => x % 3 === 0).reduce((a, b) => a + b, 0);
}
for (let i = 0; i < arr.length; i++) s += arr[i] & 15;
console.log(s);
