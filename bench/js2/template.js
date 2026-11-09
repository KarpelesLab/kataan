// HTML rendering with a tiny component system (original): template literals,
// an escape function using replace with a callback, nested components that
// return strings, array map/join, a mustache-style template compiler built on
// a regex scanner, and attribute building from objects.
const PAGES = typeof TM_PAGES === 'number' ? TM_PAGES : 500;

const ESC = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
const escapeHtml = (s) => String(s).replace(/[&<>"']/g, (c) => ESC[c]);

function attrs(obj) {
  let out = '';
  for (const k of Object.keys(obj)) {
    const v = obj[k];
    if (v === false || v == null) continue;
    out += v === true ? ` ${k}` : ` ${k}="${escapeHtml(v)}"`;
  }
  return out;
}
const h = (tag, props, ...children) => `<${tag}${attrs(props || {})}>${children.flat().join('')}</${tag}>`;

// A mustache-ish compiler: {{name}}, {{{raw}}}, {{#list}}...{{/list}}, {{^cond}}...{{/cond}}.
function compile(src) {
  const re = /\{\{(\{?)([#^/]?)\s*([\w.]+)\s*\}?\}\}/g;
  const root = { kind: 'root', children: [] };
  const stack = [root];
  let last = 0, m;
  while ((m = re.exec(src)) !== null) {
    const top = stack[stack.length - 1];
    if (m.index > last) top.children.push({ kind: 'text', text: src.slice(last, m.index) });
    last = re.lastIndex;
    const [, raw, sigil, name] = m;
    if (sigil === '#' || sigil === '^') {
      const node = { kind: sigil === '#' ? 'section' : 'inverted', name, children: [] };
      top.children.push(node);
      stack.push(node);
    } else if (sigil === '/') {
      stack.pop();
    } else top.children.push({ kind: raw ? 'raw' : 'var', name });
  }
  if (last < src.length) root.children.push({ kind: 'text', text: src.slice(last) });
  return (ctx) => renderNodes(root.children, [ctx]);
}
function lookup(stack, name) {
  if (name === '.') return stack[stack.length - 1];
  const parts = name.split('.');
  for (let i = stack.length - 1; i >= 0; i--) {
    let v = stack[i];
    if (v != null && typeof v === 'object' && parts[0] in v) {
      for (const p of parts) v = v == null ? undefined : v[p];
      return v;
    }
  }
  return undefined;
}
function renderNodes(nodes, stack) {
  let out = '';
  for (const n of nodes) {
    switch (n.kind) {
      case 'text': out += n.text; break;
      case 'var': { const v = lookup(stack, n.name); out += v == null ? '' : escapeHtml(v); break; }
      case 'raw': { const v = lookup(stack, n.name); out += v == null ? '' : String(v); break; }
      case 'section': {
        const v = lookup(stack, n.name);
        if (Array.isArray(v)) for (const item of v) { stack.push(item); out += renderNodes(n.children, stack); stack.pop(); }
        else if (v) { stack.push(v); out += renderNodes(n.children, stack); stack.pop(); }
        break;
      }
      case 'inverted': {
        const v = lookup(stack, n.name);
        if (!v || (Array.isArray(v) && v.length === 0)) out += renderNodes(n.children, stack);
        break;
      }
    }
  }
  return out;
}

const rowTemplate = compile(`<li class="item{{#featured}} featured{{/featured}}">
  <a href="/p/{{slug}}">{{title}}</a> <span class="price">{{price}}</span>
  {{#tags}}<em>{{.}}</em>{{/tags}}{{^tags}}<em>untagged</em>{{/tags}}
  <p>{{{blurb}}}</p></li>`);

function Product(p) {
  return h('article', { class: 'product', 'data-id': p.id, hidden: p.stock === 0 },
    h('h2', null, escapeHtml(p.title)),
    h('ul', { class: 'specs' }, Object.entries(p.specs).map(([k, v]) => h('li', null, `${escapeHtml(k)}: <b>${escapeHtml(v)}</b>`))),
    p.stock < 3 ? h('p', { class: 'warn' }, `Only ${p.stock} left & "selling" fast`) : '');
}

function Page(n, products) {
  const nav = h('nav', null, ['Home', 'Shop', 'About <us>', 'Contact'].map((t, i) => h('a', { href: '#' + i, class: i === n % 4 ? 'active' : false }, escapeHtml(t))));
  const list = `<ul>${products.map((p) => rowTemplate(p)).join('\n')}</ul>`;
  const cards = products.slice(0, 8).map(Product).join('');
  const total = products.reduce((s, p) => s + p.priceCents, 0);
  return `<!doctype html><html><head><title>Page ${n} &mdash; ${products.length} items</title></head><body>${nav}<main>${list}${cards}</main><footer>Total: ${(total / 100).toFixed(2)}</footer></body></html>`;
}

const WORDS = ['brass', 'widget', 'deluxe', 'compact', '<pro>', 'eco', 'smart', 'retro', 'mini', 'quad', "o'neil", 'ultra'];
function product(i) {
  const w1 = WORDS[i % WORDS.length], w2 = WORDS[(i * 7 + 3) % WORDS.length];
  const tags = [];
  for (let t = 0; t < (i % 4); t++) tags.push(WORDS[(i + t * 5) % WORDS.length]);
  return {
    id: i, slug: `${w1}-${w2}-${i}`.replace(/[^a-z0-9-]/g, ''), title: `${w1} ${w2} #${i}`,
    priceCents: 199 + ((i * 7919) % 50000), get price() { return '$' + (this.priceCents / 100).toFixed(2); },
    featured: i % 9 === 0, tags, blurb: `<i>${w2}</i> made for ${w1} & friends`, stock: i % 11,
    specs: { weight: (i % 50) / 10 + 'kg', color: ['red', 'blue', 'green'][i % 3], size: 'L' + (i % 5) },
  };
}

let bytes = 0, hash = 0, longest = 0;
for (let n = 0; n < PAGES; n++) {
  const products = [];
  for (let i = 0; i < 40; i++) products.push(product(n * 40 + i));
  const html = Page(n, products);
  bytes += html.length;
  if (html.length > longest) longest = html.length;
  for (let i = 0; i < html.length; i += 7) hash = (Math.imul(hash, 31) + html.charCodeAt(i)) | 0;
}
const sample = Page(1, [product(7), product(9)]);
console.log('pages=' + PAGES, 'bytes=' + bytes, 'longest=' + longest, 'hash=' + (hash >>> 0));
console.log(sample.slice(0, 160).replace(/\n/g, ' '));
