// A tokenizer, recursive-descent parser and tree-walking evaluator for a
// small expression language (original). The language has numbers, strings,
// let-bindings, closures, if/while, arrays and calls; programs are parsed into
// AST objects and evaluated with lexical environments built from Maps.
const REPS = typeof ML_REPS === 'number' ? ML_REPS : 1;

const KEYWORDS = new Set(['let', 'fn', 'if', 'else', 'while', 'return', 'true', 'false', 'nil']);

function tokenize(src) {
  const toks = [];
  let i = 0;
  const n = src.length;
  while (i < n) {
    const c = src[i];
    if (c === ' ' || c === '\n' || c === '\t' || c === '\r') { i++; continue; }
    if (c === '#') { while (i < n && src[i] !== '\n') i++; continue; }
    if (c >= '0' && c <= '9') {
      let j = i;
      while (j < n && ((src[j] >= '0' && src[j] <= '9') || src[j] === '.')) j++;
      toks.push({ t: 'num', v: parseFloat(src.slice(i, j)) });
      i = j; continue;
    }
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c === '_') {
      let j = i;
      while (j < n && /[A-Za-z0-9_]/.test(src[j])) j++;
      const w = src.slice(i, j);
      toks.push(KEYWORDS.has(w) ? { t: w, v: w } : { t: 'id', v: w });
      i = j; continue;
    }
    if (c === '"') {
      let j = i + 1, s = '';
      while (src[j] !== '"') {
        if (src[j] === '\\') { j++; s += src[j] === 'n' ? '\n' : src[j]; } else s += src[j];
        j++;
      }
      toks.push({ t: 'str', v: s });
      i = j + 1; continue;
    }
    const two = src.substr(i, 2);
    if (two === '==' || two === '!=' || two === '<=' || two === '>=' || two === '&&' || two === '||') {
      toks.push({ t: two, v: two }); i += 2; continue;
    }
    toks.push({ t: c, v: c }); i++;
  }
  toks.push({ t: 'eof', v: null });
  return toks;
}

const PREC = { '||': 1, '&&': 2, '==': 3, '!=': 3, '<': 4, '>': 4, '<=': 4, '>=': 4, '+': 5, '-': 5, '*': 6, '/': 6, '%': 6 };

class Parser {
  constructor(toks) { this.toks = toks; this.pos = 0; }
  peek() { return this.toks[this.pos]; }
  next() { return this.toks[this.pos++]; }
  expect(t) {
    const tok = this.next();
    if (tok.t !== t) throw new SyntaxError('expected ' + t + ' got ' + tok.t + ' at ' + this.pos);
    return tok;
  }
  accept(t) { if (this.peek().t === t) { this.pos++; return true; } return false; }
  program() {
    const body = [];
    while (this.peek().t !== 'eof') body.push(this.statement());
    return { k: 'block', body };
  }
  block() {
    this.expect('{');
    const body = [];
    while (!this.accept('}')) body.push(this.statement());
    return { k: 'block', body };
  }
  statement() {
    const t = this.peek().t;
    if (t === 'let') {
      this.next();
      const name = this.expect('id').v;
      this.expect('=');
      const init = this.expr(0);
      this.accept(';');
      return { k: 'let', name, init };
    }
    if (t === 'while') {
      this.next();
      const cond = this.expr(0);
      return { k: 'while', cond, body: this.block() };
    }
    if (t === 'return') {
      this.next();
      const e = this.expr(0);
      this.accept(';');
      return { k: 'return', e };
    }
    if (t === '{') return this.block();
    const e = this.expr(0);
    if (this.accept('=')) {
      const rhs = this.expr(0);
      this.accept(';');
      return { k: 'assign', target: e, rhs };
    }
    this.accept(';');
    return { k: 'expr', e };
  }
  expr(minPrec) {
    let left = this.unary();
    for (;;) {
      const op = this.peek().t;
      const p = PREC[op];
      if (p === undefined || p <= minPrec) return left;
      this.next();
      const right = this.expr(p);
      left = { k: 'bin', op, left, right };
    }
  }
  unary() {
    if (this.accept('-')) return { k: 'neg', e: this.unary() };
    if (this.accept('!')) return { k: 'not', e: this.unary() };
    return this.postfix(this.primary());
  }
  postfix(e) {
    for (;;) {
      if (this.accept('(')) {
        const args = [];
        if (!this.accept(')')) {
          do args.push(this.expr(0)); while (this.accept(','));
          this.expect(')');
        }
        e = { k: 'call', callee: e, args };
      } else if (this.accept('[')) {
        const idx = this.expr(0);
        this.expect(']');
        e = { k: 'index', obj: e, idx };
      } else return e;
    }
  }
  primary() {
    const tok = this.next();
    switch (tok.t) {
      case 'num': return { k: 'lit', v: tok.v };
      case 'str': return { k: 'lit', v: tok.v };
      case 'true': return { k: 'lit', v: true };
      case 'false': return { k: 'lit', v: false };
      case 'nil': return { k: 'lit', v: null };
      case 'id': return { k: 'var', name: tok.v };
      case '(': { const e = this.expr(0); this.expect(')'); return e; }
      case '[': {
        const items = [];
        if (!this.accept(']')) { do items.push(this.expr(0)); while (this.accept(',')); this.expect(']'); }
        return { k: 'array', items };
      }
      case 'fn': {
        this.expect('(');
        const params = [];
        if (!this.accept(')')) { do params.push(this.expect('id').v); while (this.accept(',')); this.expect(')'); }
        return { k: 'fn', params, body: this.block() };
      }
      case 'if': {
        const cond = this.expr(0);
        const then = this.block();
        let otherwise = null;
        if (this.accept('else')) otherwise = this.peek().t === 'if' ? { k: 'block', body: [{ k: 'expr', e: this.primary() }] } : this.block();
        return { k: 'if', cond, then, otherwise };
      }
      default: throw new SyntaxError('unexpected ' + tok.t);
    }
  }
}

class Env {
  constructor(parent) { this.vars = new Map(); this.parent = parent; }
  lookup(name) {
    let e = this;
    while (e !== null) { if (e.vars.has(name)) return e.vars.get(name); e = e.parent; }
    throw new ReferenceError('undefined variable ' + name);
  }
  set(name, v) {
    let e = this;
    while (e !== null) { if (e.vars.has(name)) { e.vars.set(name, v); return; } e = e.parent; }
    throw new ReferenceError('assignment to undeclared ' + name);
  }
  define(name, v) { this.vars.set(name, v); }
}

class Closure { constructor(params, body, env) { this.params = params; this.body = body; this.env = env; } }
class ReturnSignal { constructor(v) { this.v = v; } }

let steps = 0;
function evalBlock(node, env) {
  let last = null;
  for (const s of node.body) {
    last = exec(s, env);
    if (last instanceof ReturnSignal) return last;
  }
  return last;
}
function exec(s, env) {
  steps++;
  switch (s.k) {
    case 'let': env.define(s.name, evaluate(s.init, env)); return null;
    case 'assign': {
      const v = evaluate(s.rhs, env);
      if (s.target.k === 'var') env.set(s.target.name, v);
      else evaluate(s.target.obj, env)[evaluate(s.target.idx, env)] = v;
      return null;
    }
    case 'while': {
      while (truthy(evaluate(s.cond, env))) {
        const r = evalBlock(s.body, new Env(env));
        if (r instanceof ReturnSignal) return r;
      }
      return null;
    }
    case 'return': return new ReturnSignal(evaluate(s.e, env));
    case 'block': return evalBlock(s, new Env(env));
    case 'expr': return evaluate(s.e, env);
  }
  throw new Error('bad stmt ' + s.k);
}
function truthy(v) { return v !== null && v !== false && v !== 0; }
function evaluate(e, env) {
  steps++;
  switch (e.k) {
    case 'lit': return e.v;
    case 'var': return env.lookup(e.name);
    case 'neg': return -evaluate(e.e, env);
    case 'not': return !truthy(evaluate(e.e, env));
    case 'array': return e.items.map((x) => evaluate(x, env));
    case 'index': return evaluate(e.obj, env)[evaluate(e.idx, env)];
    case 'fn': return new Closure(e.params, e.body, env);
    case 'if': {
      if (truthy(evaluate(e.cond, env))) return evalBlock(e.then, new Env(env));
      return e.otherwise ? evalBlock(e.otherwise, new Env(env)) : null;
    }
    case 'bin': {
      if (e.op === '&&') { const l = evaluate(e.left, env); return truthy(l) ? evaluate(e.right, env) : l; }
      if (e.op === '||') { const l = evaluate(e.left, env); return truthy(l) ? l : evaluate(e.right, env); }
      const a = evaluate(e.left, env), b = evaluate(e.right, env);
      switch (e.op) {
        case '+': return a + b;
        case '-': return a - b;
        case '*': return a * b;
        case '/': return a / b;
        case '%': return a % b;
        case '<': return a < b;
        case '>': return a > b;
        case '<=': return a <= b;
        case '>=': return a >= b;
        case '==': return a === b;
        case '!=': return a !== b;
      }
      throw new Error('bad op');
    }
    case 'call': {
      const f = evaluate(e.callee, env);
      const args = e.args.map((a) => evaluate(a, env));
      if (typeof f === 'function') return f(...args);
      const fenv = new Env(f.env);
      for (let i = 0; i < f.params.length; i++) fenv.define(f.params[i], args[i] === undefined ? null : args[i]);
      const r = evalBlock(f.body, fenv);
      if (r instanceof ReturnSignal) return r.v;
      return r === undefined ? null : r;
    }
  }
  throw new Error('bad expr ' + e.k);
}

const SOURCE = `
# recursive and iterative helpers
let fib = fn(n) { if n < 2 { n } else { fib(n - 1) + fib(n - 2) } };
let range = fn(n) { let a = []; let i = 0; while i < n { push(a, i); i = i + 1; } return a; };
let map = fn(a, f) { let out = []; let i = 0; while i < len(a) { push(out, f(a[i])); i = i + 1; } return out; };
let fold = fn(a, z, f) { let acc = z; let i = 0; while i < len(a) { acc = f(acc, a[i]); i = i + 1; } return acc; };
let make_counter = fn() { let c = 0; return fn() { c = c + 1; return c; }; };
let primes = fn(limit) {
  let sieve = []; let i = 0;
  while i <= limit { push(sieve, true); i = i + 1; }
  let p = 2; let count = 0;
  while p <= limit {
    if sieve[p] { count = count + 1; let m = p * p; while m <= limit { sieve[m] = false; m = m + p; } }
    p = p + 1;
  }
  return count;
};
let words = ["alpha", "beta", "gamma", "delta"];
let label = fn(i) { words[i % 4] + "-" + str(i) };
let ctr = make_counter();
let squares = map(range(200), fn(x) { x * x % 97 });
let total = fold(squares, 0, fn(a, b) { a + b });
let s = "";
let k = 0;
while k < 30 { s = s + label(k); ctr(); k = k + 1; }
let result = [fib(16), total, primes(3000), len(s), ctr()];
result
`;

const builtins = {
  push: (a, v) => { a.push(v); return null; },
  len: (a) => a.length,
  str: (v) => String(v),
};

let checksum = 0;
let lastResult = null;
const tokens0 = tokenize(SOURCE);
for (let rep = 0; rep < REPS * 24; rep++) {
  const toks = tokenize(SOURCE);
  const ast = new Parser(toks).program();
  const global = new Env(null);
  for (const name of Object.keys(builtins)) global.define(name, builtins[name]);
  const r = evalBlock(ast, global);
  lastResult = r;
  checksum = (checksum * 31 + r[0] + r[1] + r[2] * 7 + r[3]) % 1000000007;
}
console.log('tokens=' + tokens0.length, 'result=' + JSON.stringify(lastResult), 'steps=' + steps, 'sum=' + checksum);
