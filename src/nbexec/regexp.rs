use super::*;

impl<'a> Interp<'a> {
    /// Builds a match-result array from a `Captures` whose spans are **code-unit**
    /// indices into the pre-collected `&[u16]` subject (the native regex model).
    /// Element `i` is capture group `i` (group 0 = whole match), sliced from the
    /// unit buffer and re-encoded to WTF-8 so astral characters and lone
    /// surrogates survive. The result is a real Array (so `Array.isArray`,
    /// `JSON.stringify`, `.length`, and array methods all behave), with `index` /
    /// `input` / `groups` as enumerable own properties. `.index` is a code-unit
    /// index; the `input` argument carries the original subject string.
    #[cfg(feature = "regex")]
    pub(crate) fn regex_match_object_u16(
        &mut self,
        units: &[u16],
        input: NanBox,
        caps: &crate::regex::Captures,
        group_names: &[(usize, String)],
    ) -> NanBox {
        let elems: Vec<NanBox> = caps
            .groups
            .iter()
            .map(|g| match g {
                Some((s, e)) => self.new_str_bytes(u16_slice(units, *s, *e)),
                None => NanBox::undefined(),
            })
            .collect();
        let obj = self.realm.new_array(elems);
        let index = caps.groups.first().and_then(|g| *g).map_or(0, |(s, _)| s);
        self.realm
            .set_property(obj, "index", NanBox::number(index as f64));
        self.realm.set_property(obj, "input", input);
        let groups = if group_names.is_empty() {
            NanBox::undefined()
        } else {
            let g = self.realm.new_object();
            for (idx, name) in group_names {
                let v = match caps.groups.get(*idx).and_then(|x| *x) {
                    Some((s, e)) => self.new_str_bytes(u16_slice(units, s, e)),
                    None => NanBox::undefined(),
                };
                self.realm.set_property(g, name, v);
            }
            NanBox::handle(g.to_raw())
        };
        self.realm.set_property(obj, "groups", groups);
        NanBox::handle(obj.to_raw())
    }

    /// `IsRegExp(value)`: a value with a truthy `@@match` property, or (absent that
    /// property) a RegExp instance. A non-object is not a RegExp.
    pub(crate) fn is_regexp_arg(&mut self, v: NanBox) -> bool {
        let Some(h) = v.as_handle().map(Handle::from_raw) else {
            return false;
        };
        let sym = self.well_known_symbol("match");
        let key = self.member_key(sym);
        if let Some(m) = self.realm.get_property(h, &key)
            && !matches!(m.unpack(), Unpacked::Undefined)
        {
            return self.realm.truthy(m);
        }
        self.realm.regexp_at(h).is_some()
    }
}
