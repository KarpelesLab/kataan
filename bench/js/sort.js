// Array.prototype.sort: numeric comparator and default string sort.
let seed = 12345; function rnd() { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed; }
const nums = []; for (let i = 0; i < 300000; i++) nums.push(rnd());
nums.sort((a, b) => a - b);
const strs = []; for (let i = 0; i < 100000; i++) strs.push('s' + rnd());
strs.sort();
console.log(nums[0], nums[nums.length - 1], strs[0], strs[strs.length - 1]);
