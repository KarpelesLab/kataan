#!/usr/bin/env python3
"""Fills in the expected words of src/jit/a64_enc_tests.rs.

Each `case!("<asm text>", <closure>, [<words>])` line between the
A64ENC-BEGIN/END markers is assembled with the GNU cross assembler
(`aarch64-unknown-linux-gnu-as`, or $AS) and its `[<words>]` is rewritten
with the instruction words the assembler produced. Run it after adding or
changing a case; the Rust test then compares the encoder against them.

    python3 tools/gen_a64_table.py [path/to/a64_enc_tests.rs]
"""
import os
import re
import subprocess
import sys
import tempfile

AS = os.environ.get("AS", "aarch64-unknown-linux-gnu-as")
OBJCOPY = os.environ.get("OBJCOPY", "aarch64-unknown-linux-gnu-objcopy")
CASE = re.compile(r'^(\s*case!\(")((?:[^"\\]|\\.)*)(",.*, )\[[^\]]*\](\),\s*)$')


def assemble(text):
    """The little-endian words `text` (';'-separated statements) assembles to."""
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "a.s")
        obj = os.path.join(d, "a.o")
        raw = os.path.join(d, "a.bin")
        with open(src, "w") as f:
            f.write(".text\n" + text.replace(";", "\n") + "\n")
        subprocess.run([AS, "-o", obj, src], check=True)
        subprocess.run([OBJCOPY, "-O", "binary", "-j", ".text", obj, raw], check=True)
        data = open(raw, "rb").read()
    return [int.from_bytes(data[i:i + 4], "little") for i in range(0, len(data), 4)]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "src/jit/a64_enc_tests.rs"
    out, inside = [], False
    for line in open(path):
        if "A64ENC-BEGIN" in line:
            inside = True
        elif "A64ENC-END" in line:
            inside = False
        elif inside:
            m = CASE.match(line.rstrip("\n"))
            if not m:
                sys.exit("unparsed case line: " + line)
            words = ", ".join("0x%08x" % w for w in assemble(m.group(2)))
            line = "%s%s%s[%s]%s\n" % (m.group(1), m.group(2), m.group(3), words, m.group(4))
        out.append(line)
    open(path, "w").write("".join(out))


if __name__ == "__main__":
    main()
