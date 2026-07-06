# Temporal per-type implementation guide (for the fan-out subagents)

You are implementing **one** `Temporal.<Type>` in its own file
`src/nbexec/temporal_<type>.rs`. The shared scaffold is done and committed; you
**only edit your one file** (and, locally, un-skip Temporal to test — see below).
Do NOT touch other `temporal_*.rs` files or the shared `src/nbexec/temporal.rs`.

## The surface is defined by the test tree
The exact set of methods/getters/statics your type must have is the directory
listing of the corpus. Run:
- Statics: `ls vendor/test262/test/built-ins/Temporal/<Type>/` (each `*.js` +
  subdir like `from/`, `compare/` is a static or a top-level test).
- Instance methods + getters: `ls vendor/test262/test/built-ins/Temporal/<Type>/prototype/`
  (each subdir is a method or getter).
Read a handful of the actual test `.js` files to learn exact argument coercion
order, option handling, error types, and `toString` output. The tests ARE the spec.

## Data model / brand
A Temporal instance is `Cell::Temporal(Rc<TemporalData>)`. From
`crate::temporal_iso`:
```rust
pub struct TemporalData { pub kind: TemporalKind, pub date: IsoDate, pub time: IsoTime,
    pub duration: DurationFields, pub epoch_ns: i128, pub calendar: String, pub tz: Option<String> }
pub struct IsoDate { pub year: i32, pub month: u8, pub day: u8 }
pub struct IsoTime { pub hour: u8, pub minute: u8, pub second: u8,
    pub millisecond: u16, pub microsecond: u16, pub nanosecond: u16 }
pub struct DurationFields { years, months, weeks, days, hours, minutes, seconds,
    milliseconds, microseconds, nanoseconds: i64 }   // all i64
pub enum TemporalKind { PlainDate, PlainTime, PlainDateTime, Duration, Instant,
    PlainYearMonth, PlainMonthDay, ZonedDateTime }
```
Only fill the fields your kind uses; leave others at `TemporalData::default()`.

## ISO core API (`crate::temporal_iso`, already unit-tested — USE IT)
`is_leap_year(y)`, `iso_days_in_month(y,m)->u8`, `iso_days_in_year(y)->u16`,
`iso_to_epoch_days(IsoDate)->i64`, `epoch_days_to_iso(i64)->IsoDate`,
`iso_day_of_week(d)->u8` (1=Mon..7=Sun), `iso_day_of_year(d)->u16`,
`iso_week_of_year(d)->(u8 week, i32 year)`, `iso_date_in_range(d)->bool`,
`regulate_iso_date(year:i32, month:i64, day:i64, Overflow)->Option<IsoDate>`,
`regulate_iso_time(h,m,s,ms,us,ns:i64, Overflow)->Option<IsoTime>`,
`time_to_nanos(IsoTime)->i128`, `balance_time_from_nanos(i128)->(i64 daycarry, IsoTime)`,
`add_time(IsoTime, delta_ns:i128)->(i64,IsoTime)`,
`balance_iso_year_month(year:i64, month:i64)->(i32,u8)`,
`add_iso_date(IsoDate, years,months,weeks,days:i64, Overflow)->Option<IsoDate>`,
`difference_iso_date(from,to:IsoDate, largest:Unit)->(y,m,w,d:i64)`,
`compare_iso_date`, `compare_iso_time`, `balance_time_duration(total_ns:i128, largest:Unit)->DurationFields`,
`round_to_increment(x:i128, incr:i128, RoundMode)->i128`,
`pad(v:u64,width)->String`, `format_iso_year(i32)->String`,
`format_fraction(sub_second_ns:u32, precision:Option<u8>)->String`,
`parse_iso_datetime(&str)->Option<ParsedIso>` (fields: date, time, offset_ns, z, tz_name, calendar),
`parse_iso_duration(&str)->Option<DurationFields>`.
`enum Unit { Year..Nanosecond }`, `enum RoundMode {Ceil,Floor,Expand,Trunc,HalfCeil,HalfFloor,HalfExpand,HalfTrunc,HalfEven}`,
`enum Overflow {Constrain, Reject}`. Constants `NS_PER_DAY/HOUR/MINUTE/SEC: i128`,
`MAX/MIN_EPOCH_NS: i128`, `MAX/MIN_EPOCH_DAYS: i64`.
If you need a pure helper not present, ADD it to `src/temporal_iso.rs` with a unit
test (that file is shared but append-only additions rarely conflict; prefer a
`<type>_`-prefixed fn name).

## The four functions you implement (signatures already stubbed in your file)
```rust
fn <type>_construct(&mut self, args:&[NanBox], new_target:NanBox, callee:NanBox) -> Result<NanBox, ExecError>
fn <type>_method(&mut self, this:NanBox, data:&TemporalData, method:&str, args:&[NanBox]) -> Result<NanBox, ExecError>
fn <type>_getter(&mut self, this:NanBox, data:&TemporalData, name:&str) -> Result<NanBox, ExecError>
fn <type>_static(&mut self, ctor:NanBox, method:&str, args:&[NanBox]) -> Result<Option<NanBox>, ExecError>  // Ok(None)=unrecognised
```
Also fill `pub(crate) const METHODS: &[&str]` and `GETTERS: &[&str]` with the
prototype method / getter names (the shared registration installs them; a getter
name must NOT also be in METHODS). Statics are matched by name inside
`<type>_static` (return `Ok(None)` for names you don't handle).

Dispatch is already wired: `new T(..)`→construct; reading `inst.year`→getter;
`inst.add(..)`→method; `T.from(..)`→static. Match on `method`/`name` inside.

## Building results & reading args (engine helpers — all on `self`, i.e. `Interp`)
- Build an instance: `let data = TemporalData { kind: TemporalKind::X, ..Default::default() };`
  then `Ok(self.finish_temporal(data, new_target, callee))` in construct, or for a
  result produced by a method use `Ok(self.finish_temporal(data, NanBox::undefined(), <ctor value>))`.
  To get the intrinsic ctor for a from-produced value, the simplest correct form
  in a method/static is: build `data`, then
  `let h = self.realm.new_temporal(data); if let Some(p) = self.temporal_proto(TemporalKind::X) { self.realm.set_native_proto(h, p); } Ok(NanBox::handle(h.to_raw()))`.
- Integer arg (ToIntegerWithTruncation / ToIntegerOrInfinity): `let n = self.coerce_to_integer_or_infinity(args.get(i).copied().unwrap_or(NanBox::undefined()))?;` (an f64; Symbol/BigInt→TypeError, throwing valueOf propagates). Reject non-finite where the spec's ToIntegerWithTruncation demands (many Temporal ctors require integral+finite → RangeError otherwise; check the tests).
- Number: `self.coerce_to_number(v)?` → NanBox, then `self.realm.to_number(nb)` → f64.
- String: `self.coerce_to_string(v)?` → String.
- Object check: `self.is_object_value(v)`; get a prop: `v.as_handle().map(Handle::from_raw)` then `self.realm.get_property(h, "overflow")` → `Option<NanBox>`; or `self.read_member(h, key)?` (runs getters).
- Results: `NanBox::number(f)`, `NanBox::boolean(b)`, `NanBox::undefined()`, string via `let h = self.realm.new_string(&s); NanBox::handle(h.to_raw())`.
- Errors: `Err(self.type_error("msg"))` / build a RangeError with
  `Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(self.new_str("msg")))))`.
  `valueOf` on every Temporal type MUST throw a TypeError ("use compare() / an
  explicit conversion").
- Read the receiver's own kind/fields from the `data: &TemporalData` param (already
  fetched + brand-checked by the dispatcher). To re-read `this` as a handle:
  `this.as_handle().map(Handle::from_raw)`.

## Options bag (`overflow`, `roundingMode`, `smallestUnit`, `largestUnit`, …)
Many methods take a final `options` argument (an object or undefined). Read a
string option: get the property, `coerce_to_string`, match. `overflow` ∈
{"constrain"(default),"reject"}. Follow GetOption semantics from the tests
(undefined→default; an invalid value→RangeError; reading a non-object non-undefined
options→TypeError). Keep it pragmatic: pass the tests you can, skip the exotic.

## `from(item, options)` pattern
`T.from(x)`: if `x` is already a `Temporal.<Type>` (check `self.realm.temporal_at(h)`
with matching kind) → copy it; if `x` is a string → `parse_iso_datetime`/`parse_iso_duration`;
if `x` is an object with the type's fields → read them (`read_member`) + regulate.
Return a new instance.

## VERIFICATION PROTOCOL (do this — measure, don't guess)
1. Un-skip Temporal LOCALLY: in `tests/test262_official.rs` change the line
   `    "Temporal",` (in SKIP_FEATURES, ~line 74) to `    // "Temporal",`. (This
   line is your only edit outside your type file; it will NOT be merged — it's
   just to let the tests run.)
2. Build: `cargo build --lib 2>&1 | grep -E "error" | head` — fix all errors.
3. Run YOUR type only:
   `KATAAN_TEST262_FILTER=built-ins/Temporal/<Type>/ timeout 600 cargo test --test test262_official official_test262 -- --ignored --nocapture 2>&1 | grep -oE "total=[0-9]+ ran=[0-9]+ pass=[0-9]+ fail=[0-9]+" | head -1`
4. Iterate: read failing tests (they print as `- <path> (<reason>)`), fix, re-run.
   Maximize `pass=`. Aim to get the bulk passing; don't rabbit-hole on exotic edge
   cases (calendar systems other than iso8601, relativeTo, sub-nanosecond).
5. Run `cargo fmt -p kataan` and `cargo clippy --lib` — your file must be clean.

## RETURN (your final message)
- The final `pass=N fail=M` for your type.
- Confirm `cargo build --lib` is clean and `cargo clippy --lib` adds no new warnings.
- A 2-3 line note on what's implemented vs deliberately skipped.
Your file's contents are already on disk in the worktree — leave them there; the
orchestrator collects `src/nbexec/temporal_<type>.rs` from your worktree.

## Scope guard
Calendar is always `"iso8601"` (reject other calendars per the tests, or ignore
non-iso calendar tests). No time zones (that's ZonedDateTime, not your job unless
assigned). Focus on correctness of the common operations; a large pass-count from
the mainstream methods beats perfection on edge cases.
