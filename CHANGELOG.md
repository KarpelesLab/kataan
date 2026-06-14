# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.3](https://github.com/KarpelesLab/kataan/compare/v0.0.2...v0.0.3) - 2026-06-14

### Added

- *(strings)* B5 — surrogate-aware case + normalize ([#12](https://github.com/KarpelesLab/kataan/pull/12))
- *(strings)* B4 — wire regex builtins to the UTF-16 code-unit engine ([#12](https://github.com/KarpelesLab/kataan/pull/12))
- *(strings)* B2/B3 — surrogate-correct string evaluation and UTF-16 ops
- *(strings)* B2 — carry string-literal cooked values as WTF-8 bytes
- *(ffi)* embedder buffer-creation API — owned + external (A6, #11)
- *(wasm)* WebAssembly.Memory shares one byte store with JS (A5, #11)
- *(wtf8)* WTF-8 string-storage foundation (B1)
- *(cell)* add Cell::Bytes byte store + TypedArray view variant (A1)

### Fixed

- *(typedarrays)* stable/shared .buffer object + restore wasm_bytes over byte stores

### Other

- *(regex)* fix broken intra-doc links in module docs
- *(regex)* match over UTF-16 code units with JS u-flag semantics [B4-engine]
- *(wtf8)* fix broken intra-doc links in rope/atom/wtf8 module docs
- *(typedarrays)* real byte-backed ArrayBuffer/TypedArray/DataView (A2/A3/A4)

## [0.0.2](https://github.com/KarpelesLab/kataan/compare/v0.0.1...v0.0.2) - 2026-06-14

### Added

- *(object)* dictionary-mode objects to bound shape-tree growth (MEM-3)
- *(limits)* add configurable Limits; migrate caps to read from Realm
- *(math)* drop fixed Math.random seed; entropy-mix the fallback
- *(math)* seed Math.random from purecrypto's OS CSPRNG
- *(math)* back Math.random with xorshift128+ (was xorshift64)

### Fixed

- *(wasm)* configurable limits, fuel metering, multi-byte blocktype (WASM-6/7/9)
- *(nbvm)* collect subject chars once in regex match/split loops (RE-7)
- *(regex)* collect subject chars once per match/replace/split loop (RE-7)
- *(rng)* don't let OsRng panic abort Interp::new; fall back to entropy mix (RNG-1)
- *(bigint)* cap asUintN/asIntN/** bit-size to prevent allocation bomb (MEM-6)
- *(regex)* tighten backtracking budget + share it across starts; add char-slice match API (RE-8, RE-7)
- *(regex)* avoid exponential compile-time blowup for nested unbounded quantifiers (RE-9)
- *(nbvm)* cap repeat + guard compiler register overflow (NBVM-1/NBVM-2)
- *(flatbc)* read untrusted program via fs::read, drop mmap (VM-9)
- *(bytecode)* verify untrusted load paths, bound allocs, drop mmap (VM-7/8/9)
- *(snapshot)* SNAP-3 read snapshot file instead of mmap (SIGBUS hazard)
- *(wasm)* use checked_add for element-segment slot index (WASM-8)
- *(heap)* clamp free-path generation below compaction range (MEM-4)
- *(parser)* guard recursion in parse_new/parse_exponent/parse_class (PARSE-2/3/4)
- *(date)* parse_iso_date panics on malformed/non-ASCII input (EXEC-7)
- *(lint)* satisfy clippy + rustdoc + rustfmt gates

### Other

- *(limits)* end-to-end config-override regression test
- *(snapshot)* validate restored func_id against a recorded bound (SNAP-2)
