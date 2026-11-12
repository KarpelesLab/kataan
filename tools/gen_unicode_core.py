#!/usr/bin/env python3
"""Generate `src/unicode_data.rs` -- the Unicode tables behind the engine's own
JS-observable string semantics, pinned to one UCD version so that behaviour does
not drift with the Rust toolchain's `char`/`str` Unicode tables.

Usage:

    tools/gen_unicode_core.py <ucd-dir> > src/unicode_data.rs

`<ucd-dir>` must contain these files, from
https://www.unicode.org/Public/<version>/ucd/ :

    UnicodeData.txt  SpecialCasing.txt  PropList.txt  DerivedCoreProperties.txt

What is emitted:

* range tables (sorted, coalesced, inclusive `(lo, hi)`) for `White_Space`,
  general category `Zs`, `ID_Start`, `ID_Continue`, `Cased` and
  `Case_Ignorable`; plus `Alphabetic`, `Lowercase`, `Uppercase` and general
  category `N` (the `char::is_numeric` set), which only builds without the
  `intl` crate need (with it, `intl` answers those);
* the simple upper/lowercase mappings (`UnicodeData.txt` fields 12 and 13) as
  delta runs: `(lo, hi, delta, step)` maps every `step`-th code point from `lo`
  through `hi` to itself plus `delta`;
* the unconditional `SpecialCasing.txt` mappings that differ from the simple
  ones (`ß` -> `SS`, `İ` -> `i̇`, ...), at most three code points each. The
  conditional entries are either language-specific (`tr`, `az`, `lt`; served by
  the `intl` crate for `toLocale*Case`) or `Final_Sigma`, which the string
  mapper implements in code from the `Cased`/`Case_Ignorable` tables.
"""

import os
import re
import sys

UNICODE_VERSION = "17.0.0"

LINE = re.compile(
    r"^\s*([0-9A-Fa-f]{4,6})(?:\.\.([0-9A-Fa-f]{4,6}))?\s*;\s*([A-Za-z_0-9]+)"
)


def coalesce(ranges):
    out = []
    for lo, hi in sorted(ranges):
        if out and lo <= out[-1][1] + 1:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def parse_props(path):
    raw = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            m = LINE.match(line.split("#", 1)[0])
            if not m:
                continue
            lo = int(m.group(1), 16)
            hi = int(m.group(2), 16) if m.group(2) else lo
            raw.setdefault(m.group(3), []).append((lo, hi))
    return {name: coalesce(rs) for name, rs in raw.items()}


def parse_unicode_data(path):
    """-> (gc ranges by category, simple upper map, simple lower map)."""
    gc = {}
    upper = {}
    lower = {}
    first = None
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            f = line.rstrip("\n").split(";")
            cp = int(f[0], 16)
            name, cat = f[1], f[2]
            if name.endswith(", First>"):
                first = cp
                continue
            lo = cp
            if name.endswith(", Last>"):
                lo = first
                first = None
            gc.setdefault(cat, []).append((lo, cp))
            if f[12]:
                upper[cp] = int(f[12], 16)
            if f[13]:
                lower[cp] = int(f[13], 16)
    return {k: coalesce(v) for k, v in gc.items()}, upper, lower


def parse_special(path):
    """Unconditional SpecialCasing entries -> (upper, lower) {cp: [cps]}."""
    upper = {}
    lower = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            f = [x.strip() for x in line.split(";")]
            # code; lower; title; upper; (condition_list;)?
            if len(f) > 4 and f[4]:
                continue  # conditional (language-specific or Final_Sigma)
            cp = int(f[0], 16)
            upper[cp] = [int(x, 16) for x in f[3].split()]
            lower[cp] = [int(x, 16) for x in f[1].split()]
    return upper, lower


def delta_runs(mapping):
    runs = []  # [lo, hi, delta, step]
    for cp in sorted(mapping):
        d = mapping[cp] - cp
        if runs:
            r = runs[-1]
            gap = cp - r[1]
            if d == r[2] and (
                (r[0] == r[1] and gap in (1, 2)) or (r[0] != r[1] and gap == r[3])
            ):
                r[3] = gap
                r[1] = cp
                continue
        runs.append([cp, cp, d, 1])
    return runs


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    ucd = sys.argv[1]
    plist = parse_props(os.path.join(ucd, "PropList.txt"))
    dcp = parse_props(os.path.join(ucd, "DerivedCoreProperties.txt"))
    gc, su, sl = parse_unicode_data(os.path.join(ucd, "UnicodeData.txt"))
    spu, spl = parse_special(os.path.join(ucd, "SpecialCasing.txt"))

    # Keep only the special mappings that differ from the simple mapping.
    spu = {c: v for c, v in spu.items() if v != [su.get(c, c)]}
    spl = {c: v for c, v in spl.items() if v != [sl.get(c, c)]}
    for v in list(spu.values()) + list(spl.values()):
        assert 1 <= len(v) <= 3, v

    numeric = coalesce(gc["Nd"] + gc["Nl"] + gc["No"])

    major, minor, patch = UNICODE_VERSION.split(".")
    out = sys.stdout.write
    out(
        f"""//! Unicode tables for the engine's JS-observable string semantics.
//!
//! @generated by `tools/gen_unicode_core.py` from the Unicode Character
//! Database v{UNICODE_VERSION} (`UnicodeData.txt`, `SpecialCasing.txt`,
//! `PropList.txt`, `DerivedCoreProperties.txt`) — DO NOT EDIT. Regenerate with:
//!
//! ```text
//! tools/gen_unicode_core.py <ucd-dir> > src/unicode_data.rs
//! ```
//!
//! Pinned in-house so that case mapping, identifier and white-space
//! classification do not change with the Rust toolchain's own Unicode tables.
//! The lookups live in `crate::unicode`.

/// The UCD version these tables were generated from.
pub const UNICODE_VERSION: (u8, u8, u8) = ({major}, {minor}, {patch});

/// Whether `cp` falls inside one of `ranges` (sorted, non-overlapping, inclusive).
#[inline]
pub(crate) fn in_ranges(ranges: &[(u32, u32)], cp: u32) -> bool {{
    ranges
        .binary_search_by(|&(lo, hi)| {{
            if cp < lo {{
                core::cmp::Ordering::Greater
            }} else if cp > hi {{
                core::cmp::Ordering::Less
            }} else {{
                core::cmp::Ordering::Equal
            }}
        }})
        .is_ok()
}}
"""
    )

    def ranges(const, doc, rs, cfg=None):
        out(f"\n/// {doc} — {len(rs)} ranges.\n")
        if cfg:
            out(f"{cfg}\n")
        out(f"pub(crate) static {const}: &[(u32, u32)] = &[\n")
        for lo, hi in rs:
            out(f"    (0x{lo:04X}, 0x{hi:04X}),\n")
        out("];\n")

    ranges("WHITE_SPACE", "`White_Space`", plist["White_Space"])
    ranges("SPACE_SEPARATOR", "General category `Zs`", gc["Zs"])
    ranges("ID_START", "`ID_Start`", dcp["ID_Start"])
    ranges("ID_CONTINUE", "`ID_Continue`", dcp["ID_Continue"])
    ranges("CASED", "`Cased`", dcp["Cased"])
    ranges("CASE_IGNORABLE", "`Case_Ignorable`", dcp["Case_Ignorable"])
    no_intl = '#[cfg(not(feature = "intl"))]'
    ranges("ALPHABETIC", "`Alphabetic` (builds without `intl`)", dcp["Alphabetic"], no_intl)
    ranges("LOWERCASE", "`Lowercase` (builds without `intl`)", dcp["Lowercase"], no_intl)
    ranges("UPPERCASE", "`Uppercase` (builds without `intl`)", dcp["Uppercase"], no_intl)
    ranges("NUMERIC", "General category `N` (builds without `intl`)", numeric, no_intl)

    def runs(const, doc, mapping):
        rs = delta_runs(mapping)
        out(f"\n/// {doc}: `(lo, hi, delta, step)` runs ({len(mapping)} mappings).\n")
        out(f"pub(crate) static {const}: &[(u32, u32, i32, u32)] = &[\n")
        for lo, hi, d, step in rs:
            out(f"    (0x{lo:04X}, 0x{hi:04X}, {d}, {step}),\n")
        out("];\n")

    runs("UPPER_RUNS", "Simple uppercase mapping", su)
    runs("LOWER_RUNS", "Simple lowercase mapping", sl)

    def special(const, doc, mapping):
        out(f"\n/// {doc} — {len(mapping)} entries, `'\\0'`-padded.\n")
        out(f"pub(crate) static {const}: &[(u32, [char; 3])] = &[\n")
        for cp in sorted(mapping):
            v = mapping[cp] + [0] * (3 - len(mapping[cp]))
            chars = ", ".join("'\\0'" if c == 0 else f"'\\u{{{c:04X}}}'" for c in v)
            out(f"    (0x{cp:04X}, [{chars}]),\n")
        out("];\n")

    special("SPECIAL_UPPER", "Unconditional full uppercase mappings", spu)
    special("SPECIAL_LOWER", "Unconditional full lowercase mappings", spl)


if __name__ == "__main__":
    main()
