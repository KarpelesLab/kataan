//! `Temporal.*` engine glue: native-id constants, constructor/prototype
//! registration, and the runtime dispatch that routes an instance method,
//! getter, static, or `new` into the per-type logic module
//! (`temporal_plaindate.rs`, `temporal_plaintime.rs`, …).
//!
//! One generic `Cell::Temporal` brand backs every type (see
//! [`crate::temporal_iso::TemporalData`]); each type's *logic* lives in its own
//! file as `impl Interp` methods so the implementations can be developed
//! independently without touching this shared file. This module owns only the
//! wiring: the method/getter name tables come from the per-type modules, and the
//! dispatch `match`es fan out to them.

use super::*;
use crate::temporal_iso::TemporalKind;

// --- Native-id block (1200+, well clear of the existing ~912 ceiling) --------
// One constructor id per type (needed for is_native_constructor, construct
// dispatch, static brand-check, and instanceof), plus one shared proto-method id
// and one shared getter id (both dispatch by the receiver's kind + bound name).
pub(crate) const N_TEMPORAL_PLAINDATE: u16 = 1200;
pub(crate) const N_TEMPORAL_PLAINTIME: u16 = 1201;
pub(crate) const N_TEMPORAL_PLAINDATETIME: u16 = 1202;
pub(crate) const N_TEMPORAL_DURATION: u16 = 1203;
pub(crate) const N_TEMPORAL_INSTANT: u16 = 1204;
pub(crate) const N_TEMPORAL_PLAINYEARMONTH: u16 = 1205;
pub(crate) const N_TEMPORAL_PLAINMONTHDAY: u16 = 1206;
pub(crate) const N_TEMPORAL_ZONEDDATETIME: u16 = 1207;
/// A bound Temporal prototype *method* (name carried on the bound-native cell).
pub(crate) const N_TEMPORAL_PROTO_FN: u16 = 1210;
/// A bound Temporal prototype *getter* accessor (name carried on the cell).
pub(crate) const N_TEMPORAL_GETTER: u16 = 1211;
/// A bound Temporal *static* method installed as an own property of a constructor
/// (name carried on the cell; the kind comes from the `this` constructor).
pub(crate) const N_TEMPORAL_STATIC_FN: u16 = 1212;
/// A bound method of the `Temporal.Now` namespace (name carried on the cell).
pub(crate) const N_TEMPORAL_NOW_FN: u16 = 1213;

/// `Temporal.Now.<method>` names + `.length`.
const NOW_METHODS: [(&str, u32); 6] = [
    ("timeZoneId", 0),
    ("instant", 0),
    ("plainDateTimeISO", 0),
    ("zonedDateTimeISO", 0),
    ("plainDateISO", 0),
    ("plainTimeISO", 0),
];

/// The static method names + `.length` for a kind (installed as own ctor props).
fn statics_for(kind: TemporalKind) -> &'static [(&'static str, u32)] {
    match kind {
        TemporalKind::Instant => &[
            ("from", 1),
            ("fromEpochMilliseconds", 1),
            ("fromEpochNanoseconds", 1),
            ("compare", 2),
        ],
        TemporalKind::PlainMonthDay => &[("from", 1)],
        _ => &[("from", 1), ("compare", 2)],
    }
}

/// The constructor `.length` (count of required leading parameters) per kind.
fn ctor_len(kind: TemporalKind) -> u32 {
    match kind {
        TemporalKind::PlainDate | TemporalKind::PlainDateTime => 3,
        TemporalKind::PlainTime | TemporalKind::Duration => 0,
        TemporalKind::Instant => 1,
        TemporalKind::PlainYearMonth
        | TemporalKind::PlainMonthDay
        | TemporalKind::ZonedDateTime => 2,
    }
}

/// The `.length` of a prototype method by name (spec required-arg count; default 0).
fn method_len(name: &str) -> u32 {
    match name {
        "with"
        | "add"
        | "subtract"
        | "until"
        | "since"
        | "round"
        | "total"
        | "equals"
        | "toZonedDateTime"
        | "withPlainTime"
        | "withCalendar"
        | "toPlainDate"
        | "withTimeZone"
        | "getTimeZoneTransition" => 1,
        _ => 0,
    }
}

/// The eight constructor ids, aligned with [`TemporalKind`].
pub(crate) const TEMPORAL_CTOR_IDS: [(TemporalKind, u16); 8] = [
    (TemporalKind::PlainDate, N_TEMPORAL_PLAINDATE),
    (TemporalKind::PlainTime, N_TEMPORAL_PLAINTIME),
    (TemporalKind::PlainDateTime, N_TEMPORAL_PLAINDATETIME),
    (TemporalKind::Duration, N_TEMPORAL_DURATION),
    (TemporalKind::Instant, N_TEMPORAL_INSTANT),
    (TemporalKind::PlainYearMonth, N_TEMPORAL_PLAINYEARMONTH),
    (TemporalKind::PlainMonthDay, N_TEMPORAL_PLAINMONTHDAY),
    (TemporalKind::ZonedDateTime, N_TEMPORAL_ZONEDDATETIME),
];

/// The `TemporalKind` for a constructor native id, if any.
pub(crate) fn kind_for_ctor_id(id: u16) -> Option<TemporalKind> {
    TEMPORAL_CTOR_IDS
        .iter()
        .find(|(_, cid)| *cid == id)
        .map(|(k, _)| *k)
}

/// Whether `id` is a Temporal constructor native id.
pub(crate) fn is_temporal_ctor_id(id: u16) -> bool {
    (N_TEMPORAL_PLAINDATE..=N_TEMPORAL_ZONEDDATETIME).contains(&id)
}

/// The `(methods, getters)` name tables for a kind, sourced from the per-type
/// module so each type's surface stays in its own file.
fn tables_for(kind: TemporalKind) -> (&'static [&'static str], &'static [&'static str]) {
    use crate::nbexec::{
        temporal_duration as dur, temporal_instant as inst, temporal_plaindate as pd,
        temporal_plaindatetime as pdt, temporal_plainmonthday as pmd, temporal_plaintime as pt,
        temporal_plainyearmonth as pym, temporal_zoneddatetime as zdt,
    };
    match kind {
        TemporalKind::PlainDate => (pd::METHODS, pd::GETTERS),
        TemporalKind::PlainTime => (pt::METHODS, pt::GETTERS),
        TemporalKind::PlainDateTime => (pdt::METHODS, pdt::GETTERS),
        TemporalKind::Duration => (dur::METHODS, dur::GETTERS),
        TemporalKind::Instant => (inst::METHODS, inst::GETTERS),
        TemporalKind::PlainYearMonth => (pym::METHODS, pym::GETTERS),
        TemporalKind::PlainMonthDay => (pmd::METHODS, pmd::GETTERS),
        TemporalKind::ZonedDateTime => (zdt::METHODS, zdt::GETTERS),
    }
}

impl<'a> Interp<'a> {
    /// Installs the `Temporal` namespace object and every type's constructor,
    /// prototype (methods + getters + `Symbol.toStringTag`), into the global
    /// scope. Statics (`from`/`compare`/…) are recognised dynamically in
    /// `call_method` by the constructor's native id, so they need no own props
    /// here beyond what the tests' `verifyProperty` requires (added later).
    pub(crate) fn install_temporal(&mut self) {
        let temporal_ns = self.realm.new_object();
        if let Some(op) = self.object_prototype() {
            self.realm.set_object_proto(temporal_ns, Some(op));
        }
        for (kind, ctor_id) in TEMPORAL_CTOR_IDS {
            let name = kind.type_name();
            let ctor = self.new_named_native(name, ctor_id);
            self.install_fn_name_length(ctor, name, ctor_len(kind));
            let (methods, getters) = tables_for(kind);
            let op = self.object_prototype();
            let proto = self.realm.new_object_with_proto(op);
            // Static methods (`from`/`compare`/…): own non-enumerable function
            // properties of the constructor.
            for &(sname, slen) in statics_for(kind) {
                let name_h = self.realm.new_string(sname);
                let f = self.realm.new_bound_native(N_TEMPORAL_STATIC_FN, name_h);
                self.install_fn_name_length(f, sname, slen);
                self.realm
                    .set_property(ctor, sname, NanBox::handle(f.to_raw()));
                self.realm.mark_hidden(ctor, sname);
            }
            // Prototype methods: each a BoundNative{proto_fn, name}.
            for &m in methods {
                let name_h = self.realm.new_string(m);
                let f = self.realm.new_bound_native(N_TEMPORAL_PROTO_FN, name_h);
                self.install_fn_name_length(f, m, method_len(m));
                self.realm
                    .set_property(proto, m, NanBox::handle(f.to_raw()));
                self.realm.mark_hidden(proto, m);
            }
            // Getter accessors: each a BoundNative{getter, name}.
            for &g in getters {
                let getname = alloc::format!("get {g}");
                let name_h = self.realm.new_string(g);
                let f = self.realm.new_bound_native(N_TEMPORAL_GETTER, name_h);
                self.install_fn_name_length(f, &getname, 0);
                self.realm.define_accessor(
                    proto,
                    g,
                    NanBox::handle(f.to_raw()),
                    NanBox::undefined(),
                );
                self.realm.mark_hidden(proto, g);
            }
            // `Class.prototype[Symbol.toStringTag] = "Temporal.<Type>"`.
            self.install_to_string_tag(proto, &alloc::format!("Temporal.{name}"));
            // `constructor` back-link + prototype wiring (non-enumerable,
            // prototype writable:false / configurable:false).
            self.realm
                .set_hidden_property(proto, "constructor", NanBox::handle(ctor.to_raw()));
            self.realm
                .set_property(ctor, "prototype", NanBox::handle(proto.to_raw()));
            self.realm.mark_hidden(ctor, "prototype");
            self.realm.set_readonly_property(ctor, "prototype");
            self.realm.set_non_configurable_property(ctor, "prototype");
            // Install the constructor as a non-enumerable prop of `Temporal`.
            self.realm
                .set_property(temporal_ns, name, NanBox::handle(ctor.to_raw()));
            self.realm.mark_hidden(temporal_ns, name);
            self.set_temporal_proto(kind, proto);
        }
        // `Temporal[Symbol.toStringTag] = "Temporal"`.
        // `Temporal.Now` — a namespace object of clock methods.
        let now_ns = self.realm.new_object();
        if let Some(op) = self.object_prototype() {
            self.realm.set_object_proto(now_ns, Some(op));
        }
        for (nm, len) in NOW_METHODS {
            let name_h = self.realm.new_string(nm);
            let f = self.realm.new_bound_native(N_TEMPORAL_NOW_FN, name_h);
            self.install_fn_name_length(f, nm, len);
            self.realm
                .set_property(now_ns, nm, NanBox::handle(f.to_raw()));
            self.realm.mark_hidden(now_ns, nm);
        }
        self.install_to_string_tag(now_ns, "Temporal.Now");
        self.realm
            .set_property(temporal_ns, "Now", NanBox::handle(now_ns.to_raw()));
        self.realm.mark_hidden(temporal_ns, "Now");
        self.install_to_string_tag(temporal_ns, "Temporal");
        self.current
            .declare("Temporal", NanBox::handle(temporal_ns.to_raw()));
    }

    /// Records the intrinsic prototype handle for a kind (used by
    /// `GetPrototypeFromConstructor` when building instances).
    fn set_temporal_proto(&mut self, kind: TemporalKind, proto: Handle) {
        let idx = kind as usize;
        if self.temporal_protos.len() <= idx {
            self.temporal_protos.resize(idx + 1, None);
        }
        self.temporal_protos[idx] = Some(proto);
    }

    /// The intrinsic `Temporal.<Type>.prototype`, if installed.
    #[allow(dead_code)] // used by per-type construct logic as it lands
    pub(crate) fn temporal_proto(&self, kind: TemporalKind) -> Option<Handle> {
        self.temporal_protos.get(kind as usize).copied().flatten()
    }

    /// Links a freshly-built Temporal instance to `newTarget.prototype` (subclass)
    /// or the intrinsic prototype, then returns it boxed.
    #[allow(dead_code)] // used by per-type construct logic as it lands
    pub(crate) fn finish_temporal(
        &mut self,
        data: crate::temporal_iso::TemporalData,
        new_target: NanBox,
        callee: NanBox,
    ) -> NanBox {
        let kind = data.kind;
        let h = self.realm.new_temporal(data);
        let default = self.temporal_proto(kind);
        if let Some(proto) = self.instance_proto(new_target, callee, default) {
            self.realm.set_native_proto(h, proto);
        }
        NanBox::handle(h.to_raw())
    }

    /// Routes `new Temporal.<kind>(...)` to the per-type constructor logic.
    pub(crate) fn temporal_construct(
        &mut self,
        kind: TemporalKind,
        args: &[NanBox],
        new_target: NanBox,
        callee: NanBox,
    ) -> Result<NanBox, ExecError> {
        match kind {
            TemporalKind::PlainDate => self.plaindate_construct(args, new_target, callee),
            TemporalKind::PlainTime => self.plaintime_construct(args, new_target, callee),
            TemporalKind::PlainDateTime => self.plaindatetime_construct(args, new_target, callee),
            TemporalKind::Duration => self.duration_construct(args, new_target, callee),
            TemporalKind::Instant => self.instant_construct(args, new_target, callee),
            TemporalKind::PlainYearMonth => self.plainyearmonth_construct(args, new_target, callee),
            TemporalKind::PlainMonthDay => self.plainmonthday_construct(args, new_target, callee),
            TemporalKind::ZonedDateTime => self.zoneddatetime_construct(args, new_target, callee),
        }
    }

    /// Routes an instance method call on a Temporal receiver to the per-type
    /// logic. `data` is the receiver's already-fetched internal record.
    pub(crate) fn temporal_method(
        &mut self,
        this: NanBox,
        data: &crate::temporal_iso::TemporalData,
        method: &str,
        args: &[NanBox],
    ) -> Result<NanBox, ExecError> {
        match data.kind {
            TemporalKind::PlainDate => self.plaindate_method(this, data, method, args),
            TemporalKind::PlainTime => self.plaintime_method(this, data, method, args),
            TemporalKind::PlainDateTime => self.plaindatetime_method(this, data, method, args),
            TemporalKind::Duration => self.duration_method(this, data, method, args),
            TemporalKind::Instant => self.instant_method(this, data, method, args),
            TemporalKind::PlainYearMonth => self.plainyearmonth_method(this, data, method, args),
            TemporalKind::PlainMonthDay => self.plainmonthday_method(this, data, method, args),
            TemporalKind::ZonedDateTime => self.zoneddatetime_method(this, data, method, args),
        }
    }

    /// Routes a getter read on a Temporal receiver to the per-type logic.
    pub(crate) fn temporal_getter(
        &mut self,
        this: NanBox,
        data: &crate::temporal_iso::TemporalData,
        name: &str,
    ) -> Result<NanBox, ExecError> {
        match data.kind {
            TemporalKind::PlainDate => self.plaindate_getter(this, data, name),
            TemporalKind::PlainTime => self.plaintime_getter(this, data, name),
            TemporalKind::PlainDateTime => self.plaindatetime_getter(this, data, name),
            TemporalKind::Duration => self.duration_getter(this, data, name),
            TemporalKind::Instant => self.instant_getter(this, data, name),
            TemporalKind::PlainYearMonth => self.plainyearmonth_getter(this, data, name),
            TemporalKind::PlainMonthDay => self.plainmonthday_getter(this, data, name),
            TemporalKind::ZonedDateTime => self.zoneddatetime_getter(this, data, name),
        }
    }

    /// Routes a static method call (`Temporal.<Type>.from(...)` etc.) to the
    /// per-type logic. `ctor` is the constructor object (the receiver).
    pub(crate) fn temporal_static(
        &mut self,
        kind: TemporalKind,
        ctor: NanBox,
        method: &str,
        args: &[NanBox],
    ) -> Result<Option<NanBox>, ExecError> {
        match kind {
            TemporalKind::PlainDate => self.plaindate_static(ctor, method, args),
            TemporalKind::PlainTime => self.plaintime_static(ctor, method, args),
            TemporalKind::PlainDateTime => self.plaindatetime_static(ctor, method, args),
            TemporalKind::Duration => self.duration_static(ctor, method, args),
            TemporalKind::Instant => self.instant_static(ctor, method, args),
            TemporalKind::PlainYearMonth => self.plainyearmonth_static(ctor, method, args),
            TemporalKind::PlainMonthDay => self.plainmonthday_static(ctor, method, args),
            TemporalKind::ZonedDateTime => self.zoneddatetime_static(ctor, method, args),
        }
    }

    /// Shared helper for the per-type modules: a `TypeError` for an unimplemented
    /// or brand-mismatched Temporal operation.
    pub(crate) fn temporal_todo(&mut self, what: &str) -> ExecError {
        self.type_error(&alloc::format!("Temporal: {what} is not yet implemented"))
    }

    /// Builds a Temporal instance linked to its kind's intrinsic prototype (no
    /// subclass), for engine-produced values (e.g. `Temporal.Now.*`).
    fn build_temporal(&mut self, data: crate::temporal_iso::TemporalData) -> NanBox {
        let kind = data.kind;
        let h = self.realm.new_temporal(data);
        if let Some(p) = self.temporal_proto(kind) {
            self.realm.set_native_proto(h, p);
        }
        NanBox::handle(h.to_raw())
    }

    /// Resolves a `Temporal.Now` time-zone argument to an IANA/offset id string
    /// (undefined → the system default, taken here as `"UTC"`); validates it.
    fn now_tz_arg(&mut self, arg: NanBox) -> Result<alloc::string::String, ExecError> {
        if matches!(arg.unpack(), Unpacked::Undefined) {
            return Ok(alloc::string::String::from("UTC"));
        }
        // A string time-zone id, or an object with a `timeZone`/`id`.
        let s = if self.is_object_value(arg) {
            let h = arg.as_handle().map(Handle::from_raw).unwrap();
            let tzv = self
                .realm
                .get_property(h, "timeZone")
                .unwrap_or(NanBox::undefined());
            self.coerce_to_string(if matches!(tzv.unpack(), Unpacked::Undefined) {
                arg
            } else {
                tzv
            })?
        } else {
            self.coerce_to_string(arg)?
        };
        // Accept UTC, a fixed offset, or a known IANA name.
        if self.now_tz_offset_ns(&s, 0).is_some() {
            Ok(s)
        } else {
            let m = self.new_str(&alloc::format!("invalid time zone: {s}"));
            Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))))
        }
    }

    /// The offset (in nanoseconds east of UTC) for time-zone id `tz` at
    /// `epoch_ns`, or `None` if `tz` is not a recognised zone.
    fn now_tz_offset_ns(&self, tz: &str, epoch_ns: i128) -> Option<i128> {
        if tz == "UTC" || tz == "Etc/UTC" || tz == "Etc/GMT" {
            return Some(0);
        }
        // Fixed offset `±HH[:MM[:SS]]`.
        let b = tz.as_bytes();
        if matches!(b.first(), Some(b'+' | b'-')) {
            let neg = b[0] == b'-';
            let digits: alloc::vec::Vec<u8> = tz[1..].bytes().filter(u8::is_ascii_digit).collect();
            if digits.len() >= 2 {
                let h: i128 = (digits[0] - b'0') as i128 * 10 + (digits[1] - b'0') as i128;
                let m: i128 = if digits.len() >= 4 {
                    (digits[2] - b'0') as i128 * 10 + (digits[3] - b'0') as i128
                } else {
                    0
                };
                let ns = (h * 3600 + m * 60) * crate::temporal_iso::NS_PER_SEC;
                return Some(if neg { -ns } else { ns });
            }
            return None;
        }
        // Named IANA zone via the timezone-data crate.
        let unix = (epoch_ns / crate::temporal_iso::NS_PER_SEC) as i64;
        timezone_data::load_insensitive(tz)
            .ok()
            .map(|z| i128::from(z.lookup(unix).offset) * crate::temporal_iso::NS_PER_SEC)
    }

    /// `Temporal.Now.<method>()` dispatch.
    pub(crate) fn now_method(&mut self, name: &str, args: &[NanBox]) -> Result<NanBox, ExecError> {
        use crate::temporal_iso::{
            TemporalData, TemporalKind, balance_time_from_nanos, epoch_days_to_iso,
        };
        let ms = now_ms();
        let epoch_ns = (ms as i128) * 1_000_000;
        match name {
            "timeZoneId" => {
                let h = self.realm.new_string("UTC");
                Ok(NanBox::handle(h.to_raw()))
            }
            "instant" => Ok(self.build_temporal(TemporalData {
                kind: TemporalKind::Instant,
                epoch_ns,
                ..Default::default()
            })),
            "zonedDateTimeISO" => {
                let tz = self.now_tz_arg(args.first().copied().unwrap_or(NanBox::undefined()))?;
                Ok(self.build_temporal(TemporalData {
                    kind: TemporalKind::ZonedDateTime,
                    epoch_ns,
                    tz: Some(tz),
                    ..Default::default()
                }))
            }
            "plainDateTimeISO" | "plainDateISO" | "plainTimeISO" => {
                let tz = self.now_tz_arg(args.first().copied().unwrap_or(NanBox::undefined()))?;
                let offset = self.now_tz_offset_ns(&tz, epoch_ns).unwrap_or(0);
                let (day, time) = balance_time_from_nanos(epoch_ns + offset);
                let date = epoch_days_to_iso(day);
                let data = match name {
                    "plainDateISO" => TemporalData {
                        kind: TemporalKind::PlainDate,
                        date,
                        ..Default::default()
                    },
                    "plainTimeISO" => TemporalData {
                        kind: TemporalKind::PlainTime,
                        time,
                        ..Default::default()
                    },
                    _ => TemporalData {
                        kind: TemporalKind::PlainDateTime,
                        date,
                        time,
                        ..Default::default()
                    },
                };
                Ok(self.build_temporal(data))
            }
            _ => Err(self.temporal_todo(&alloc::format!("Now.{name}"))),
        }
    }
}
