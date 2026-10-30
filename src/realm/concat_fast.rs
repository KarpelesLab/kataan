//! `+` of a short string and an integer (`'k' + i`, `i + 'px'`), the commonest
//! concatenation a program makes: the result is written straight into the
//! inline buffer of one new small rope, with no intermediate digit string and
//! one heap lookup for the string operand.

use super::Realm;
use crate::cell::Cell;
use crate::heap::Handle;
use crate::nanbox::NanBox;
use crate::rope::{Rope, SMALL_CAP};

/// Integers below this magnitude print as plain decimal digits (2^53).
const LIMIT: f64 = 9_007_199_254_740_992.0;

/// `"00" "01" … "99"`: the two decimal digits of each value below 100.
const DIGIT_PAIRS: [u8; 200] = {
    let mut t = [0u8; 200];
    let mut i = 0;
    while i < 100 {
        t[2 * i] = b'0' + (i / 10) as u8;
        t[2 * i + 1] = b'0' + (i % 10) as u8;
        i += 1;
    }
    t
};

impl Realm {
    /// `a + b` when one operand is a flat string cell whose `typeof` is
    /// `"string"` and the other an integer Number of magnitude below 2^53, and
    /// the joined text fits a small rope: that new string. `None`, having done
    /// nothing, for any other operands (the general `+` then runs).
    #[inline]
    pub(crate) fn add_str_int(&mut self, a: NanBox, b: NanBox) -> Option<NanBox> {
        let (s, n, number_first) = match b.as_number() {
            Some(n) => (a, n, false),
            None => (b, a.as_number()?, true),
        };
        let sh = Handle::from_raw(s.as_handle()?);
        if !(n > -LIMIT && n < LIMIT) {
            return None;
        }
        let i = n as i64;
        // (`-0` passes as `0`, which is also how it prints.)
        if i as f64 != n {
            return None;
        }
        let mut u = i.unsigned_abs();
        let width = u.checked_ilog10().map_or(1, |l| l as usize + 1);
        let digits = width + usize::from(i < 0);
        let bytes = self.heap.get(sh)?.as_str()?.as_leaf_bytes()?;
        let len = bytes.len() + digits;
        // A string cell carrying a function marker (or any other auxiliary
        // object) is left to the general path, which decides its `typeof`.
        if len > SMALL_CAP || self.aux_props.contains_key(&sh.to_raw()) {
            return None;
        }
        let rope = Rope::small_with(len, |out| {
            let (num, text) = if number_first {
                let (d, t) = out.split_at_mut(digits);
                (d, t)
            } else {
                let (t, d) = out.split_at_mut(bytes.len());
                (d, t)
            };
            text.copy_from_slice(bytes);
            // Two digits per division, from the right.
            let mut at = num.len();
            while u >= 100 {
                let pair = (u % 100) as usize * 2;
                u /= 100;
                at -= 2;
                num[at..at + 2].copy_from_slice(&DIGIT_PAIRS[pair..pair + 2]);
            }
            if u >= 10 {
                let pair = u as usize * 2;
                at -= 2;
                num[at..at + 2].copy_from_slice(&DIGIT_PAIRS[pair..pair + 2]);
            } else {
                at -= 1;
                num[at] = b'0' + u as u8;
            }
            if i < 0 {
                num[0] = b'-';
            }
        })?;
        Some(NanBox::handle(self.heap.alloc(Cell::Str(rope)).to_raw()))
    }
}
