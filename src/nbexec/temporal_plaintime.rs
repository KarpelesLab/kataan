//! `Temporal.PlainTime` — logic module. A fan-out unit: everything specific to
//! `PlainTime` lives here (its method/getter name tables plus the construct/
//! method/getter/static logic), so it can be implemented independently of the
//! other Temporal types and of the shared wiring in `temporal.rs`.
use super::*;
use crate::temporal_iso::TemporalData;

/// Prototype method names installed on `Temporal.PlainTime.prototype`.
pub(crate) const METHODS: &[&str] = &[];
/// Getter-accessor names installed on `Temporal.PlainTime.prototype`.
pub(crate) const GETTERS: &[&str] = &[];

impl<'a> Interp<'a> {
    /// `new Temporal.PlainTime(...)`.
    pub(crate) fn plaintime_construct(
        &mut self,
        _args: &[NanBox],
        _new_target: NanBox,
        _callee: NanBox,
    ) -> Result<NanBox, ExecError> {
        Err(self.temporal_todo("PlainTime constructor"))
    }

    /// A `Temporal.PlainTime.prototype.<method>()` call.
    pub(crate) fn plaintime_method(
        &mut self,
        _this: NanBox,
        _data: &TemporalData,
        method: &str,
        _args: &[NanBox],
    ) -> Result<NanBox, ExecError> {
        Err(self.temporal_todo(&alloc::format!("PlainTime.prototype.{method}")))
    }

    /// A `Temporal.PlainTime.prototype.<getter>` read.
    pub(crate) fn plaintime_getter(
        &mut self,
        _this: NanBox,
        _data: &TemporalData,
        name: &str,
    ) -> Result<NanBox, ExecError> {
        Err(self.temporal_todo(&alloc::format!("PlainTime getter {name}")))
    }

    /// A `Temporal.PlainTime.<static>()` call. `Ok(None)` = not a recognised static.
    pub(crate) fn plaintime_static(
        &mut self,
        _ctor: NanBox,
        _method: &str,
        _args: &[NanBox],
    ) -> Result<Option<NanBox>, ExecError> {
        Ok(None)
    }
}
