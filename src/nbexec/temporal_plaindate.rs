//! `Temporal.PlainDate` — logic module. A fan-out unit: everything specific to
//! `PlainDate` lives here (its method/getter name tables plus the construct/
//! method/getter/static logic), so it can be implemented independently of the
//! other Temporal types and of the shared wiring in `temporal.rs`.
use super::*;
use crate::temporal_iso::TemporalData;

/// Prototype method names installed on `Temporal.PlainDate.prototype`.
pub(crate) const METHODS: &[&str] = &[];
/// Getter-accessor names installed on `Temporal.PlainDate.prototype`.
pub(crate) const GETTERS: &[&str] = &[];

impl<'a> Interp<'a> {
    /// `new Temporal.PlainDate(...)`.
    pub(crate) fn plaindate_construct(
        &mut self,
        _args: &[NanBox],
        _new_target: NanBox,
        _callee: NanBox,
    ) -> Result<NanBox, ExecError> {
        Err(self.temporal_todo("PlainDate constructor"))
    }

    /// A `Temporal.PlainDate.prototype.<method>()` call.
    pub(crate) fn plaindate_method(
        &mut self,
        _this: NanBox,
        _data: &TemporalData,
        method: &str,
        _args: &[NanBox],
    ) -> Result<NanBox, ExecError> {
        Err(self.temporal_todo(&alloc::format!("PlainDate.prototype.{method}")))
    }

    /// A `Temporal.PlainDate.prototype.<getter>` read.
    pub(crate) fn plaindate_getter(
        &mut self,
        _this: NanBox,
        _data: &TemporalData,
        name: &str,
    ) -> Result<NanBox, ExecError> {
        Err(self.temporal_todo(&alloc::format!("PlainDate getter {name}")))
    }

    /// A `Temporal.PlainDate.<static>()` call. `Ok(None)` = not a recognised static.
    pub(crate) fn plaindate_static(
        &mut self,
        _ctor: NanBox,
        _method: &str,
        _args: &[NanBox],
    ) -> Result<Option<NanBox>, ExecError> {
        Ok(None)
    }
}
