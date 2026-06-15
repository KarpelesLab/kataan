#!/usr/bin/env python3
import os, sys, re, subprocess, tempfile

ROOT = "/home/magicaltux/projects/kataan/.claude/worktrees/agent-ad406db8f64de3f82"
T262 = "/home/magicaltux/projects/kataan/vendor/test262"
HARNESS = os.path.join(T262, "harness")
BIN = os.path.join(ROOT, "target/release/kataan")

def parse_meta(src):
    meta = {"negative": None, "includes": [], "flags": [], "features": []}
    m = re.search(r"/\*---(.*?)---\*/", src, re.S)
    if not m:
        return meta
    block = m.group(1)
    fm = re.search(r"flags:\s*\[(.*?)\]", block)
    if fm: meta["flags"] = [x.strip() for x in fm.group(1).split(",") if x.strip()]
    im = re.search(r"includes:\s*\[(.*?)\]", block)
    if im: meta["includes"] = [x.strip() for x in im.group(1).split(",") if x.strip()]
    im2 = re.search(r"includes:\s*\n((?:\s*-\s*.+\n)+)", block)
    if im2:
        meta["includes"] += [l.strip().lstrip("-").strip() for l in im2.group(1).splitlines() if l.strip()]
    fem = re.search(r"features:\s*\[(.*?)\]", block)
    if fem: meta["features"] = [x.strip() for x in fem.group(1).split(",") if x.strip()]
    fem2 = re.search(r"features:\s*\n((?:\s*-\s*.+\n)+)", block)
    if fem2:
        meta["features"] += [l.strip().lstrip("-").strip() for l in fem2.group(1).splitlines() if l.strip()]
    nm = re.search(r"negative:\s*\n\s*phase:\s*(\S+)\s*\n\s*type:\s*(\S+)", block)
    if nm: meta["negative"] = (nm.group(1), nm.group(2))
    return meta

_hcache = {}
def read_harness(name):
    if name not in _hcache:
        with open(os.path.join(HARNESS, name)) as f:
            _hcache[name] = f.read()
    return _hcache[name]

def assemble(meta, src, strict=False):
    out = []
    if strict:
        out.append('"use strict";')
    if "raw" not in meta["flags"]:
        out.append(read_harness("sta.js"))
        out.append(read_harness("assert.js"))
        for inc in meta["includes"]:
            out.append(read_harness(inc))
    if os.environ.get("WRAP"):
        out.append("try {\n" + src + "\n} catch (e) { throw new Error('T262FAIL: ' + (e && e.message ? e.message : e)); }")
    else:
        out.append(src)
    return "\n".join(out)

def run_src(combined):
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as tf:
        tf.write(combined)
        tfn = tf.name
    try:
        p = subprocess.run([BIN, "nbrun", tfn], capture_output=True, text=True, timeout=20)
        ok = (p.returncode == 0)
        out = (p.stdout + p.stderr).strip()
    except subprocess.TimeoutExpired:
        ok, out = False, "TIMEOUT"
    os.unlink(tfn)
    return ok, out

def run_one(path):
    with open(path, errors="replace") as f:
        src = f.read()
    meta = parse_meta(src)
    flags = meta["flags"]
    if "module" in flags or "async" in flags:
        return ("skip", "async/module")
    neg = meta["negative"]
    combined = assemble(meta, src, strict=("onlyStrict" in flags))
    ok, out = run_src(combined)
    if neg:
        if not ok:
            return ("pass", "")
        return ("fail", "expected-negative")
    if ok:
        return ("pass", "")
    m = re.search(r"T262FAIL: (.*)", out)
    if m:
        return ("fail", m.group(1))
    return ("fail", out[-500:])

def walk(subdir):
    base = os.path.join(T262, "test/built-ins", subdir)
    results = {"pass":0, "fail":0, "skip":0}
    fails = []
    for dp, dn, fn in os.walk(base):
        for f in sorted(fn):
            if not f.endswith(".js"): continue
            if f.endswith("_FIXTURE.js"): continue
            path = os.path.join(dp, f)
            st, msg = run_one(path)
            results[st]+=1
            if st=="fail":
                fails.append((os.path.relpath(path, base), msg))
    return results, fails

if __name__ == "__main__":
    subdir = sys.argv[1]
    showfails = "--fails" in sys.argv
    res, fails = walk(subdir)
    print(f"{subdir}: pass={res['pass']} fail={res['fail']} skip={res['skip']}")
    if showfails:
        for p, m in fails:
            print("FAIL", p)
            print("   ", m.replace("\n"," | ")[:300])
