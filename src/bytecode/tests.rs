//! Bytecode container + codec tests.

use super::{BytecodeError, Chunk, Const, MAGIC, Module, Op, VERSION, deserialize, serialize};

/// Builds a representative module: `(a + b) * 2`, returned.
fn sample_module() -> Module {
    let mut chunk = Chunk::new("<main>");
    chunk.register_count = 4;
    chunk.param_count = 2;
    let k2 = chunk.add_constant(Const::Number(2.0));
    let kmsg = chunk.add_constant(Const::Str("hello".into()));
    chunk.emit(Op::LoadConst { dst: 2, k: k2 });
    chunk.emit(Op::Add { dst: 3, a: 0, b: 1 });
    chunk.emit(Op::Mul { dst: 3, a: 3, b: 2 });
    chunk.emit(Op::GetGlobal { dst: 0, name: kmsg });
    chunk.emit(Op::Return { src: 3 });
    Module::new(chunk)
}

#[test]
fn round_trips() {
    let module = sample_module();
    let bytes = serialize(&module);
    let back = deserialize(&bytes).expect("deserialize ok");
    assert_eq!(module, back);
}

#[test]
fn constants_are_deduplicated() {
    let mut chunk = Chunk::new("c");
    let a = chunk.add_constant(Const::Number(1.0));
    let b = chunk.add_constant(Const::Number(1.0));
    let c = chunk.add_constant(Const::Str("x".into()));
    assert_eq!(a, b);
    assert_ne!(a, c);
    assert_eq!(chunk.constants.len(), 2);
}

#[test]
fn header_is_well_formed() {
    let bytes = serialize(&sample_module());
    assert_eq!(&bytes[0..4], &MAGIC);
    // Version is little/native-endian u16 right after the magic.
    let version = u16::from_ne_bytes([bytes[4], bytes[5]]);
    assert_eq!(version, VERSION);
}

#[test]
fn rejects_bad_magic() {
    let mut bytes = serialize(&sample_module());
    bytes[0] = b'X';
    assert_eq!(deserialize(&bytes), Err(BytecodeError::BadMagic));
}

#[test]
fn rejects_unsupported_version() {
    let mut bytes = serialize(&sample_module());
    // Bump the version field (native-endian u16 at offset 4).
    let bad = (VERSION + 99).to_ne_bytes();
    bytes[4] = bad[0];
    bytes[5] = bad[1];
    assert_eq!(
        deserialize(&bytes),
        Err(BytecodeError::UnsupportedVersion(VERSION + 99))
    );
}

#[test]
fn rejects_truncated() {
    let bytes = serialize(&sample_module());
    let truncated = &bytes[..bytes.len() - 3];
    assert_eq!(deserialize(truncated), Err(BytecodeError::Truncated));
}

#[test]
fn endian_marker_records_host_order() {
    let bytes = serialize(&sample_module());
    // The marker (offset 6) reflects the host that produced the buffer.
    let expected = if cfg!(target_endian = "little") { 0 } else { 1 };
    assert_eq!(bytes[6], expected);
}

#[test]
fn disassembly_mentions_opcodes() {
    let text = sample_module().disassemble();
    assert!(text.contains("Add"));
    assert!(text.contains("Mul"));
    assert!(text.contains("Return"));
    assert!(text.contains("hello"));
}
