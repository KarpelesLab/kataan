//! The string/RegExp protocol over a *pristine* RegExp, in one step.
//!
//! `s.replace(re, t)` / `s.match(re)` with a RegExp `re` is, by the letter of
//! ECMA-262, a chain of observable property reads: `String.prototype.replace`
//! looks up `re[@@replace]` and calls it; that reads `re.flags` (whose getter
//! reads eight flag getters), `re.lastIndex` and `re.exec`, and calls `exec`,
//! which builds a match array that `@@replace` then reads back property by
//! property. When `re` owns no property at all (its `lastIndex` is the cell's
//! own slot) and inherits straight from this realm's `%RegExp.prototype%`, and
//! that prototype still holds the intrinsic `exec`, `@@replace`, `@@match`,
//! `flags` and flag getters, every one of those reads lands on an intrinsic
//! that runs no user code — so the whole chain is equivalent to running the
//! matcher and splicing the result, with nothing observable skipped. That is
//! what this module does ([`Interp::regexp_pristine`] decides the condition).

use super::*;

/// What [`Interp::regexp_pristine`] last proved of a `%RegExp.prototype%`:
/// that, at this stamp, it held the intrinsic methods in [`PROOF_METHODS`]
/// (at the recorded slots, as the recorded values) and the intrinsic `flags`
/// getter and flag getters.
pub(crate) struct RegexpProtoProof {
    /// The realm (by address), the prototype, its key epoch (which every
    /// change to its set of keys or their kinds moves) and the realm's
    /// cell-swap count.
    stamp: (usize, u64, u32, u32),
    /// Per method: where the prototype holds it (see
    /// `Realm::own_data_probe`) and the intrinsic function found there.
    keys: [(u32, u64); 4],
    /// Each method's storage key (a symbol's is its registered key string).
    names: [alloc::rc::Rc<str>; 4],
    /// What [`Interp::regexp_species_default`] proved under this stamp: the
    /// slot of the prototype's `constructor` and the value there (`%RegExp%`),
    /// and the key epoch of the object holding `%RegExp%`'s own properties
    /// (with that object), which every accessor (re)definition moves.
    species: Option<(u32, u64, (u64, u32))>,
    /// The key `constructor`, for rechecking that slot.
    ctor_key: [alloc::rc::Rc<str>; 1],
}

/// The prototype's methods the protocol reads, by intrinsic name: `exec`,
/// `[@@replace]`, `[@@match]`, `[@@split]`.
const PROOF_METHODS: [&[u8]; 4] = [b"exec", b"replace", b"match", b"split"];

/// A one-step operation's subject code units: the subject memo's, or
/// converted into the interpreter's reusable scratch buffer — no memo entry
/// and no allocation, for the common case of an operation that finds
/// nothing to keep them for (see [`Interp::lean_subject_rc`]).
pub(crate) enum Subject {
    Memo(alloc::rc::Rc<Vec<u16>>),
    Scratch(Vec<u16>),
}

impl Subject {
    fn units(&self) -> &[u16] {
        match self {
            Subject::Memo(rc) => rc,
            Subject::Scratch(v) => v,
        }
    }
}

/// [`Interp::regexp_pristine`]'s `need` bit for `exec`.
pub(crate) const NEED_EXEC: u32 = 1;
/// … for `[@@replace]`.
pub(crate) const NEED_REPLACE: u32 = 2;
/// … for `[@@match]`.
pub(crate) const NEED_MATCH: u32 = 4;
/// … for `[@@split]`.
pub(crate) const NEED_SPLIT: u32 = 8;

impl Interp {
    /// Whether `h` is a RegExp whose every protocol read is an intrinsic's
    /// (see the module docs): a RegExp cell with no own property beside its
    /// `lastIndex` slot (no auxiliary object), whose `[[Prototype]]` is this
    /// realm's `%RegExp.prototype%`, which holds the intrinsic `exec`,
    /// `@@replace`, `@@match`, `flags` and flag getters, in a single-realm
    /// run. Of the methods, only those in `need` (bits [`NEED_EXEC`],
    /// [`NEED_REPLACE`], [`NEED_MATCH`]) are rechecked on a proven prototype:
    /// the caller's operation reads no other.
    pub(crate) fn regexp_pristine(&mut self, h: Handle, need: u32) -> bool {
        if !self.created_realms.is_empty() {
            return false;
        }
        let Some(proto) = self.current_regexp_proto() else {
            return false;
        };
        if self.realm.plain_regexp_default_proto(h) != Some(proto) {
            return false;
        }
        let Some(epoch) = self.realm.object_key_epoch(proto) else {
            return false;
        };
        let stamp = (
            core::ptr::from_ref::<Realm>(&self.realm) as usize,
            proto.to_raw(),
            epoch,
            self.realm.cell_swap_count(),
        );
        if let Some(p) = &self.regexp_proto_proof
            && p.stamp == stamp
            && self.realm.own_data_all_hold(proto, &p.keys, &p.names, need)
        {
            return true;
        }
        self.regexp_proto_prove(proto, stamp)
    }

    /// The full check behind [`regexp_pristine`](Self::regexp_pristine),
    /// recording its proof on success.
    fn regexp_proto_prove(&mut self, proto: Handle, stamp: (usize, u64, u32, u32)) -> bool {
        self.regexp_proto_proof = None;
        if self.regexp_flags_check(proto).is_none() {
            return false;
        }
        let names: [alloc::rc::Rc<str>; 4] = [
            alloc::rc::Rc::from("exec"),
            self.regexp_symbol_key(0),
            self.regexp_symbol_key(1),
            self.regexp_symbol_key(2),
        ];
        let mut keys = [(0u32, 0u64); 4];
        for ((key, name), method) in keys.iter_mut().zip(&names).zip(PROOF_METHODS) {
            let Some((slot, v)) = self.realm.own_data_probe(proto, name) else {
                return false;
            };
            let Some(fh) = v.as_handle().map(Handle::from_raw) else {
                return false;
            };
            let Some((id, target)) = self.realm.bound_native_at(fh) else {
                return false;
            };
            if id != N_REGEXP_PROTO_FN
                || self.realm.string_leaf_bytes(target) != Some(method)
                || self.get_function_realm(fh) != self.cur_realm
            {
                return false;
            }
            *key = (slot, v.to_bits());
        }
        self.regexp_proto_proof = Some(RegexpProtoProof {
            stamp,
            keys,
            names,
            species: None,
            ctor_key: [alloc::rc::Rc::from("constructor")],
        });
        true
    }

    /// Whether `SpeciesConstructor(R, %RegExp%)` of a pristine RegExp `R`
    /// (see [`regexp_pristine`](Self::regexp_pristine)) is this realm's
    /// `%RegExp%` with no user code run: `%RegExp.prototype%.constructor` is
    /// a data property holding `%RegExp%`, whose `@@species` is the
    /// intrinsic getter (which returns its receiver).
    fn regexp_species_default(&mut self) -> bool {
        let (Some(proto), Some(ctor)) = (self.current_regexp_proto(), self.regexp_ctor) else {
            return false;
        };
        let ctor_epoch = self.realm.own_props_epoch(ctor);
        // Proven under the current prototype proof (the caller's
        // `regexp_pristine`, whose stamp pins the prototype's layout): the
        // `constructor` slot still holds `%RegExp%` and `%RegExp%`'s keys
        // and accessors are as they were.
        if let Some(p) = &self.regexp_proto_proof
            && let Some((slot, bits, epoch)) = p.species
            && Some(epoch) == ctor_epoch
            && self
                .realm
                .own_data_all_hold(proto, &[(slot, bits)], &p.ctor_key, 1)
        {
            return true;
        }
        let Some((slot, v)) = self.realm.own_data_probe(proto, "constructor") else {
            return false;
        };
        if v.as_handle() != Some(ctor.to_raw()) {
            return false;
        }
        let key = self.regexp_symbol_key(3);
        let ok = self.realm.accessor(ctor, &key).is_some_and(|(g, _)| {
            g.as_handle().map(Handle::from_raw).is_some_and(|g| {
                self.realm
                    .bound_native_at(g)
                    .is_some_and(|(id, _)| id == N_REGEXP_SPECIES)
                    && self.get_function_realm(g) == self.cur_realm
            })
        });
        if ok
            && let Some(epoch) = ctor_epoch
            && let Some(p) = &mut self.regexp_proto_proof
        {
            p.species = Some((slot, v.to_bits(), epoch));
        }
        ok
    }

    /// `String.prototype.split` of the string `o` with the pristine RegExp
    /// `h` (see [`regexp_pristine`](Self::regexp_pristine), which the caller
    /// has checked for `exec`, `@@match` and `@@split`) and the limit
    /// `limit`: the result array, or `None` (nothing done) when a step could
    /// run user code (a limit that is not a number or `undefined`, a
    /// user species) or the subject is empty.
    ///
    /// `@@split` constructs its splitter through `SpeciesConstructor`; with
    /// the intrinsic `%RegExp%`, a pristine `R` (the constructor's
    /// `Get(R, @@match)`, the splitter's `exec`) and a limit converted
    /// without user code, nothing can observe that object, so its program
    /// (`R`'s source with the new flags) runs alone.
    pub(crate) fn regexp_split_pristine(
        &mut self,
        h: Handle,
        o: NanBox,
        limit: NanBox,
    ) -> Result<Option<NanBox>, ExecError> {
        let lim = match limit.unpack() {
            Unpacked::Undefined => u32::MAX as usize,
            _ => match limit.as_number() {
                Some(n) => to_uint32(n) as usize,
                None => return Ok(None),
            },
        };
        if !self.regexp_species_default() {
            return Ok(None);
        }
        let Some(unicode) = self
            .realm
            .regexp_flags(h)
            .map(|f| f.contains('u') || f.contains('v'))
        else {
            return Ok(None);
        };
        let Some(own) = self.realm.regex_compiled(h) else {
            return Ok(None);
        };
        let re = match &self.regexp_split_memo {
            Some((k, re)) if alloc::rc::Rc::ptr_eq(k, &own) => re.clone(),
            _ => {
                let Some(re) = self
                    .realm
                    .regexp_flags(h)
                    .map(|f| {
                        if f.contains('y') {
                            alloc::string::String::from(f)
                        } else {
                            alloc::format!("{f}y")
                        }
                    })
                    .and_then(|new_flags| {
                        let src = self.realm.regexp_source_bytes(h)?;
                        self.realm.regex_program_for(&src, &new_flags)
                    })
                else {
                    return Ok(None);
                };
                self.regexp_split_memo = Some((own, re.clone()));
                re
            }
        };
        let (_, units) = self.subject_units_cached(o)?;
        if units.is_empty() {
            return Ok(None);
        }
        let parts = if lim == 0 {
            Vec::new()
        } else {
            self.regexp_split_run(&re, &units, lim, unicode)
        };
        Ok(Some(NanBox::handle(self.realm.new_array(parts).to_raw())))
    }

    /// The code units of the string `o` for a one-step operation (see
    /// [`Subject`]): the memo's when it has them, else the scratch buffer
    /// refilled.
    fn lean_subject(&mut self, o: NanBox) -> Option<Subject> {
        let raw = o.as_handle()?;
        for entry in &self.subject_units_cache {
            if let Some((r, units)) = entry
                && *r == raw
            {
                return Some(Subject::Memo(units.clone()));
            }
        }
        let h = Handle::from_raw(raw);
        self.realm.flatten_string(h);
        let mut units = core::mem::take(&mut self.lean_units);
        units.clear();
        utf16_units_into(self.realm.string_leaf_bytes(h)?, &mut units);
        Some(Subject::Scratch(units))
    }

    /// `sub` (the units of the string `o`) as a shared buffer a match can
    /// keep (the legacy match record, a replacer plan), remembered in the
    /// subject memo as `subject_units_cached` would have.
    fn lean_subject_rc(&mut self, o: NanBox, sub: Subject) -> alloc::rc::Rc<Vec<u16>> {
        match sub {
            Subject::Memo(rc) => rc,
            Subject::Scratch(v) => {
                let rc = alloc::rc::Rc::new(v);
                if let Some(raw) = o.as_handle() {
                    self.subject_units_cache[1] = self.subject_units_cache[0].take();
                    self.subject_units_cache[0] = Some((raw, rc.clone()));
                }
                rc
            }
        }
    }

    /// Gives a scratch buffer back for the next operation (a long
    /// subject's is dropped rather than kept for short ones).
    fn lean_subject_done(&mut self, sub: Subject) {
        if let Subject::Scratch(v) = sub
            && v.capacity() <= 4096
        {
            self.lean_units = v;
        }
    }

    /// Records the last match of `spans` (`stride` per match) in the legacy
    /// match record, the subject `o`'s units kept for it.
    fn lean_legacy(
        &mut self,
        o: NanBox,
        sub: Subject,
        spans: &[Option<(usize, usize)>],
        stride: usize,
    ) -> alloc::rc::Rc<Vec<u16>> {
        let rc = self.lean_subject_rc(o, sub);
        if let Some(last) = spans.rchunks(stride).next() {
            let caps = crate::regex::Captures {
                groups: last.to_vec(),
            };
            self.update_legacy_regexp(&rc, &caps);
        }
        rc
    }

    /// Every match `RegExpBuiltinExec` would make on the pristine RegExp `h`
    /// over `units` for one `@@replace`/`@@match`: for a global RegExp the
    /// whole collection loop (from `lastIndex` 0, which it leaves at 0), else
    /// the one exec (from `lastIndex` when sticky, which it then updates).
    /// The spans (`group_count() + 1` per match) are appended to `spans`;
    /// the caller records the last one in the legacy match record.
    #[allow(clippy::too_many_arguments)]
    fn pristine_matches(
        &mut self,
        h: Handle,
        re: &crate::regex::Regex,
        units: &[u16],
        global: bool,
        sticky: bool,
        unicode: bool,
        spans: &mut Vec<Option<(usize, usize)>>,
    ) {
        let len = units.len();
        if global {
            re.global_scan_into(units, 0, unicode, spans);
            self.realm.set_regex_last_index(h, 0);
        } else {
            let start = if sticky {
                self.realm.regex_last_index(h)
            } else {
                0
            };
            let found = start <= len && re.captures_into_u16(units, start, spans);
            if sticky {
                let end = if found {
                    spans[0].map_or(0, |(_, b)| b)
                } else {
                    0
                };
                self.realm.set_regex_last_index(h, end);
            }
        }
    }

    /// `String.prototype.replace` (`all`: `replaceAll`) of the string `o`
    /// with the pristine RegExp `h` (see
    /// [`regexp_pristine`](Self::regexp_pristine)) and a replacement that is
    /// a string template or a VM function: the result, or for a function the
    /// matches its calls are to be made over
    /// ([`crate::nbvm::ReplaceStep::Plan`]). `None`, with nothing done, for
    /// anything else (a RegExp with named groups, another replacement).
    pub(crate) fn regexp_replace_pristine(
        &mut self,
        h: Handle,
        o: NanBox,
        replace: NanBox,
        all: bool,
    ) -> Result<Option<crate::nbvm::ReplaceStep>, ExecError> {
        use crate::nbvm::{ReplacePlan, ReplaceStep};
        let Some(rh) = replace.as_handle().map(Handle::from_raw) else {
            return Ok(None);
        };
        let func = self.realm.is_vm_function(rh);
        if !func && !(self.realm.is_string_handle(rh) && self.realm.typeof_is_string(rh)) {
            return Ok(None);
        }
        let Some(re) = self.realm.regex_compiled(h) else {
            return Ok(None);
        };
        if !re.group_names().is_empty() {
            return Ok(None);
        }
        let Some((global, sticky, unicode)) = self.realm.regexp_flags(h).map(|f| {
            (
                f.contains('g'),
                f.contains('y'),
                f.contains('u') || f.contains('v'),
            )
        }) else {
            return Ok(None);
        };
        if all && !global {
            return Err(self.type_error(
                "String.prototype.replaceAll called with a non-global RegExp argument",
            ));
        }
        // `ToString(S)` is the identity on a string; then, for a template,
        // `ToString(replaceValue)` likewise.
        let str_v = o;
        // A pattern whose matches all begin with a unit of a known set
        // (`/[&<>"']/g`, `/\d+$/`) over a flat ASCII subject holding none of
        // them: no match at all, so the subject itself, found by a scan of
        // its bytes (no code-unit copy, no replacer call). A failing exec
        // leaves a global or sticky RegExp's `lastIndex` 0 and the legacy
        // match record as it was.
        if str_v
            .as_handle()
            .and_then(|s| self.realm.string_leaf_bytes(Handle::from_raw(s)))
            .is_some_and(|b| re.ascii_never_matches(b))
        {
            if global || sticky {
                self.realm.set_regex_last_index(h, 0);
            }
            return Ok(Some(ReplaceStep::Done(str_v)));
        }
        let Some(sub) = self.lean_subject(o) else {
            return Ok(None);
        };
        let repl: Vec<u8> = if func {
            Vec::new()
        } else {
            self.realm.flatten_string(rh);
            self.realm
                .string_leaf_bytes(rh)
                .unwrap_or_default()
                .to_vec()
        };
        let plain = !repl.contains(&b'$');
        let length = sub.units().len();
        // An ASCII subject's unmatched runs are its code units narrowed.
        // (Only ASCII text has as many WTF-8 bytes as UTF-16 units.)
        let ascii = str_v
            .as_handle()
            .and_then(|s| self.realm.string_leaf_bytes(Handle::from_raw(s)))
            .is_some_and(|b| b.len() == length);
        let copy = |out: &mut Vec<u8>, run: &[u16]| {
            if ascii {
                out.extend(run.iter().map(|&u| u as u8));
            } else {
                crate::wtf8::append_utf16(out, run);
            }
        };
        // A global one-character atom (`/[&<>]/g`, `/[^aeiou]/gi`) with a
        // plain template: every match is one unit the atom accepts, so the
        // result is one pass over the subject, with no match list.
        if global
            && !sticky
            && !func
            && plain
            && let Some(accepts) = re.single_atom_matcher()
        {
            let units: &[u16] = sub.units();
            let max_len = self.realm.limits.max_string_len;
            let mut out: Vec<u8> = Vec::with_capacity(length);
            let (mut run, mut last) = (0usize, None);
            let mut too_long = false;
            for (i, &u) in units.iter().enumerate() {
                if accepts(u) {
                    last = Some(i);
                    if too_long {
                        continue;
                    }
                    copy(&mut out, &units[run..i]);
                    crate::wtf8::append(&mut out, &repl);
                    run = i + 1;
                    // Past the limit only the last match is still wanted
                    // (the collection completes before the splice throws).
                    too_long = out.len() > max_len;
                }
            }
            // The collection loop's final, failing exec leaves `lastIndex` 0.
            self.realm.set_regex_last_index(h, 0);
            let Some(l) = last else {
                self.lean_subject_done(sub);
                return Ok(Some(ReplaceStep::Done(str_v)));
            };
            copy(&mut out, &units[run..]);
            self.lean_legacy(o, sub, &[Some((l, l + 1))], 1);
            if too_long {
                let m = self.new_str("Invalid string length");
                return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
            }
            return Ok(Some(ReplaceStep::Done(self.new_str_bytes(out))));
        }
        let mut spans = Vec::new();
        self.pristine_matches(h, &re, sub.units(), global, sticky, unicode, &mut spans);
        let stride = re.group_count() + 1;
        // No match: the subject itself, with no replacer call.
        if spans.is_empty() {
            self.lean_subject_done(sub);
            return Ok(Some(ReplaceStep::Done(str_v)));
        }
        let units_rc = self.lean_legacy(o, sub, &spans, stride);
        if func {
            return Ok(Some(ReplaceStep::Plan(ReplacePlan {
                subject: str_v,
                units: units_rc,
                stride,
                spans,
            })));
        }
        let units: &[u16] = &units_rc;
        let mut out: Vec<u8> = Vec::with_capacity(length + repl.len());
        let mut next = 0usize;
        for m in spans.chunks(stride) {
            let Some((pos, end)) = m[0] else {
                continue;
            };
            let pos = pos.min(length);
            let end = end.min(length).max(pos);
            if pos < next {
                continue;
            }
            copy(&mut out, &units[next..pos]);
            if plain {
                crate::wtf8::append(&mut out, &repl);
            } else {
                let captures: Vec<Option<Vec<u8>>> = m[1..]
                    .iter()
                    .map(|g| g.map(|(a, b)| u16_slice(units, a, b)))
                    .collect();
                let r = self.get_substitution(
                    &units[pos..end],
                    units,
                    pos,
                    &captures,
                    NanBox::undefined(),
                    &repl,
                )?;
                crate::wtf8::append(&mut out, &r);
            }
            next = end;
            if out.len() > self.realm.limits.max_string_len {
                let m = self.new_str("Invalid string length");
                return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
            }
        }
        if next < length {
            copy(&mut out, &units[next..]);
        }
        Ok(Some(ReplaceStep::Done(self.new_str_bytes(out))))
    }

    /// `String.prototype.match` of the string `o` with the pristine RegExp
    /// `h`: for a global RegExp the array of every match's text (or null),
    /// else `RegExpBuiltinExec`'s result.
    pub(crate) fn regexp_match_pristine(
        &mut self,
        h: Handle,
        o: NanBox,
    ) -> Result<Option<NanBox>, ExecError> {
        let Some((global, unicode)) = self
            .realm
            .regexp_flags(h)
            .map(|f| (f.contains('g'), f.contains('u') || f.contains('v')))
        else {
            return Ok(None);
        };
        if !global {
            return self.regexp_builtin_exec(h, o).map(Some);
        }
        let Some(re) = self.realm.regex_compiled(h) else {
            return Ok(None);
        };
        let Some(sub) = self.lean_subject(o) else {
            return Ok(None);
        };
        let mut spans = Vec::new();
        self.pristine_matches(h, &re, sub.units(), true, false, unicode, &mut spans);
        if spans.is_empty() {
            self.lean_subject_done(sub);
            return Ok(Some(NanBox::null()));
        }
        let stride = re.group_count() + 1;
        let units_rc = self.lean_legacy(o, sub, &spans, stride);
        let units: &[u16] = &units_rc;
        let results: Vec<NanBox> = spans
            .chunks(stride)
            .map(|g| {
                let (a, b) = g[0].unwrap_or_default();
                NanBox::handle(self.realm.new_string_units(&units[a..b]).to_raw())
            })
            .collect();
        Ok(Some(NanBox::handle(self.realm.new_array(results).to_raw())))
    }
}
