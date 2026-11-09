// A typed-array image pipeline (original): procedurally generate an RGBA
// image, convert to grayscale, separable Gaussian blur with integer kernels,
// Sobel edge magnitude, a 3x3 median filter, histogram equalization and a
// box downscale. Everything is integer arithmetic on Uint8/Int32/Float32 arrays.
const W = typeof IF_W === 'number' ? IF_W : 1024;
const H = typeof IF_H === 'number' ? IF_H : 768;
const PASSES = typeof IF_PASSES === 'number' ? IF_PASSES : 3;

function generate(w, h) {
  const img = new Uint8ClampedArray(w * h * 4);
  let s = 7;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      s = (Math.imul(s, 1664525) + 1013904223) | 0;
      const noise = (s >>> 24) & 31;
      const o = (y * w + x) * 4;
      const cx = x - w / 2, cy = y - h / 2;
      const ring = ((cx * cx + cy * cy) >> 9) & 1;
      img[o] = ((x * 255) / w) + noise;
      img[o + 1] = ring ? 200 - noise : 40 + noise;
      img[o + 2] = (((x ^ y) & 32) ? 220 : 30) + noise;
      img[o + 3] = 255;
    }
  }
  return img;
}

function grayscale(rgba, w, h) {
  const g = new Uint8Array(w * h);
  for (let i = 0, o = 0; i < g.length; i++, o += 4) {
    g[i] = (rgba[o] * 77 + rgba[o + 1] * 150 + rgba[o + 2] * 29) >> 8;
  }
  return g;
}

const KERNEL = new Int32Array([1, 6, 15, 20, 15, 6, 1]);
const KSUM = 64;
function blur(src, w, h) {
  const tmp = new Uint8Array(w * h);
  const out = new Uint8Array(w * h);
  const r = 3;
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      let acc = 0;
      for (let k = -r; k <= r; k++) {
        let xx = x + k;
        if (xx < 0) xx = 0; else if (xx >= w) xx = w - 1;
        acc += src[row + xx] * KERNEL[k + r];
      }
      tmp[row + x] = (acc + 32) >> 6;
    }
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let acc = 0;
      for (let k = -r; k <= r; k++) {
        let yy = y + k;
        if (yy < 0) yy = 0; else if (yy >= h) yy = h - 1;
        acc += tmp[yy * w + x] * KERNEL[k + r];
      }
      out[y * w + x] = (acc + KSUM / 2) / KSUM | 0;
    }
  }
  return out;
}

function sobel(src, w, h) {
  const mag = new Float32Array(w * h);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const a = src[i - w - 1], b = src[i - w], c = src[i - w + 1];
      const d = src[i - 1], f = src[i + 1];
      const g = src[i + w - 1], hh = src[i + w], k = src[i + w + 1];
      const gx = (c + 2 * f + k) - (a + 2 * d + g);
      const gy = (g + 2 * hh + k) - (a + 2 * b + c);
      mag[i] = Math.sqrt(gx * gx + gy * gy);
    }
  }
  return mag;
}

function median3(src, w, h) {
  const out = new Uint8Array(w * h);
  const win = new Int32Array(9);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      let n = 0;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) win[n++] = src[(y + dy) * w + x + dx];
      // insertion sort of 9
      for (let i = 1; i < 9; i++) {
        const v = win[i];
        let j = i - 1;
        while (j >= 0 && win[j] > v) { win[j + 1] = win[j]; j--; }
        win[j + 1] = v;
      }
      out[y * w + x] = win[4];
    }
  }
  return out;
}

function equalize(src) {
  const hist = new Int32Array(256);
  for (let i = 0; i < src.length; i++) hist[src[i]]++;
  const cdf = new Int32Array(256);
  let run = 0;
  for (let i = 0; i < 256; i++) { run += hist[i]; cdf[i] = run; }
  let cmin = 0;
  for (let i = 0; i < 256; i++) if (cdf[i] > 0) { cmin = cdf[i]; break; }
  const lut = new Uint8Array(256);
  const denom = src.length - cmin || 1;
  for (let i = 0; i < 256; i++) lut[i] = Math.round(((cdf[i] - cmin) * 255) / denom);
  const out = new Uint8Array(src.length);
  for (let i = 0; i < src.length; i++) out[i] = lut[src[i]];
  return out;
}

function downscale(src, w, h, f) {
  const ow = (w / f) | 0, oh = (h / f) | 0;
  const out = new Uint8Array(ow * oh);
  for (let y = 0; y < oh; y++) {
    for (let x = 0; x < ow; x++) {
      let s = 0;
      for (let yy = 0; yy < f; yy++) for (let xx = 0; xx < f; xx++) s += src[(y * f + yy) * w + x * f + xx];
      out[y * ow + x] = (s / (f * f)) | 0;
    }
  }
  return out;
}

function hashBytes(a) {
  let h = 2166136261;
  for (let i = 0; i < a.length; i++) h = Math.imul(h ^ a[i], 16777619);
  return h >>> 0;
}

const rgba = generate(W, H);
let gray = grayscale(rgba, W, H);
for (let p = 0; p < PASSES; p++) gray = blur(gray, W, H);
const edges = sobel(gray, W, H);
let edgeSum = 0, strong = 0;
for (let i = 0; i < edges.length; i++) { edgeSum += Math.floor(edges[i]); if (edges[i] > 40) strong++; }
const med = median3(gray, W, H);
const eq = equalize(med);
const small = downscale(eq, W, H, 4);
console.log(W + 'x' + H, 'gray=' + hashBytes(gray), 'edges=' + edgeSum + '/' + strong, 'median=' + hashBytes(med), 'eq=' + hashBytes(eq), 'small=' + hashBytes(small));
