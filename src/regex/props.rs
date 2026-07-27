//! Unicode property-escape (`\p{…}` / `\P{…}`) name resolution and matching.
//!
//! This module is the data/semantics half of the regex engine's Unicode property
//! escapes (ECMAScript `UnicodePropertyValueExpression`). It maps property and
//! value *names* (canonical names plus the spec's aliases) to a [`PropEscape`]
//! and tests code points for membership.
//!
//! Three flavours of property escape are supported, mirroring the spec:
//!
//! * **Lone binary property** — `\p{Alphabetic}`, `\p{White_Space}`, `\p{ASCII}`,
//!   … : a property name from the ECMAScript "binary Unicode properties" set
//!   (plus the non-binary loners `Any`, `Assigned`, `ASCII`).
//! * **Lone `General_Category` value** — `\p{L}`, `\p{Lu}`, `\p{Letter}`, … : a
//!   general-category value used without an explicit `General_Category=` prefix.
//! * **`name=value`** — `\p{General_Category=Letter}`, `\p{Script=Greek}`,
//!   `\p{Script_Extensions=Latn}` (and the `gc` / `sc` / `scx` aliases).
//!
//! Resolution is **exact** (case-sensitive, no whitespace folding): an unknown
//! property name or value is rejected so the caller raises a `SyntaxError`, as
//! the corpus's many negative parse tests require. Matching is backed by the
//! pure-Rust `intl` crate's Unicode tables when the `intl` feature is on; without
//! it only the categories/properties expressible via `char` methods resolve.

#[cfg(feature = "intl")]
use intl::unicode::Script;

/// A resolved Unicode property escape, ready to test code points against.
#[derive(Clone, Copy)]
pub(crate) enum PropEscape {
    /// A binary property (or the binary-like loners `Any`/`Assigned`/`ASCII`).
    Binary(BinProp),
    /// A `General_Category` value: a single-letter group, a two-letter
    /// subcategory, or one of the synthetic unions (`LC`, `Assigned`).
    Gc(GcSel),
    /// `Script=…` — the code point's resolved `Script` equals this script.
    Script(ScriptId),
    /// `Script_Extensions=…` — this script is in the code point's
    /// `Script_Extensions` set.
    ScriptExt(ScriptId),
}

/// A binary Unicode property recognised by ECMAScript property escapes. Every
/// variant is a valid lone `\p{Name}`; matching is exact where the `intl`
/// tables (or a closed-form rule) provide the data, and otherwise conservative.
#[derive(Clone, Copy)]
pub(crate) enum BinProp {
    Ascii,
    AsciiHexDigit,
    Alphabetic,
    Any,
    Assigned,
    BidiControl,
    BidiMirrored,
    CaseIgnorable,
    Cased,
    ChangesWhenCasefolded,
    ChangesWhenCasemapped,
    ChangesWhenLowercased,
    ChangesWhenNfkcCasefolded,
    ChangesWhenTitlecased,
    ChangesWhenUppercased,
    Dash,
    DefaultIgnorable,
    Deprecated,
    Diacritic,
    Emoji,
    EmojiComponent,
    EmojiModifier,
    EmojiModifierBase,
    EmojiPresentation,
    ExtendedPictographic,
    Extender,
    GraphemeBase,
    GraphemeExtend,
    HexDigit,
    IdsBinaryOperator,
    IdsTrinaryOperator,
    IdContinue,
    IdStart,
    Ideographic,
    JoinControl,
    LogicalOrderException,
    Lowercase,
    Math,
    NoncharacterCodePoint,
    PatternSyntax,
    PatternWhiteSpace,
    QuotationMark,
    Radical,
    RegionalIndicator,
    SentenceTerminal,
    SoftDotted,
    TerminalPunctuation,
    UnifiedIdeograph,
    Uppercase,
    VariationSelector,
    WhiteSpace,
    XidContinue,
    XidStart,
}

/// A `General_Category` selector.
#[derive(Clone, Copy)]
pub(crate) enum GcSel {
    /// A top-level group: one of `L M N P S Z C`.
    Group(u8),
    /// A two-letter subcategory, e.g. `Lu`, `Nd`, `Cs`.
    Sub([u8; 2]),
    /// `LC` / `Cased_Letter` — the union `Lu | Ll | Lt`.
    CasedLetter,
}

/// An opaque script identifier. With the `intl` feature it wraps the crate's
/// `Script` enum; without it (no Unicode tables) script escapes resolve to this
/// zero-sized placeholder that matches nothing — but the *names* still validate,
/// so a script escape never spuriously raises a `SyntaxError`.
#[cfg(feature = "intl")]
#[derive(Clone, Copy)]
pub(crate) struct ScriptId(Script);

#[cfg(not(feature = "intl"))]
#[derive(Clone, Copy)]
pub(crate) struct ScriptId;

/// Resolves a lone `\p{Name}` (no `=value`). Per the spec this is either a
/// binary property name or a `General_Category` value; anything else is `None`
/// (→ `SyntaxError`). Non-binary property *names* used alone (`General_Category`,
/// `Script`, `Script_Extensions`) are intentionally rejected here.
pub(crate) fn resolve_lone(name: &str) -> Option<PropEscape> {
    if let Some(b) = binary_property(name) {
        return Some(PropEscape::Binary(b));
    }
    gc_value(name).map(PropEscape::Gc)
}

/// The seven ECMAScript *properties of strings* (`\p{…}` names valid only in a
/// `v`-mode class, matching multi-code-point strings). All seven are backed by
/// the generated Unicode Emoji tables in [`super::props_emoji`]; the predicate
/// stays separate from the data so the parser can apply the early-error rules
/// (a property of strings may not be negated, nor used without `v`) by name.
#[must_use]
pub(crate) fn is_property_of_strings(name: &str) -> bool {
    matches!(
        name,
        "Basic_Emoji"
            | "Emoji_Keycap_Sequence"
            | "RGI_Emoji"
            | "RGI_Emoji_Flag_Sequence"
            | "RGI_Emoji_Modifier_Sequence"
            | "RGI_Emoji_Tag_Sequence"
            | "RGI_Emoji_ZWJ_Sequence"
    )
}

/// Resolves a *property of strings* name (a `v`-mode `\p{…}` that matches a set
/// of multi-code-point strings) to that set. Each string is a sequence of scalar
/// code points, taken from the generated Unicode Emoji tables.
#[must_use]
pub(crate) fn strings_for_property(name: &str) -> Option<alloc::vec::Vec<alloc::vec::Vec<u32>>> {
    super::props_emoji::sequences(name)
}

/// Resolves a `name=value` property escape. The left-hand side must be one of the
/// non-binary properties `General_Category`/`gc`, `Script`/`sc`, or
/// `Script_Extensions`/`scx`; the value is validated against it. Any other LHS,
/// or an unrecognised value, is `None` (→ `SyntaxError`).
pub(crate) fn resolve_pair(name: &str, value: &str) -> Option<PropEscape> {
    match name {
        "General_Category" | "gc" => gc_value(value).map(PropEscape::Gc),
        "Script" | "sc" => script_id(value).map(PropEscape::Script),
        "Script_Extensions" | "scx" => script_id(value).map(PropEscape::ScriptExt),
        _ => None,
    }
}

/// Maps a binary property name (canonical or alias) to its [`BinProp`]. Exact,
/// case-sensitive matching — `\p{ANY}` / `\p{ lowercase }` must NOT resolve.
fn binary_property(name: &str) -> Option<BinProp> {
    use BinProp::*;
    Some(match name {
        "ASCII" => Ascii,
        "AHex" | "ASCII_Hex_Digit" => AsciiHexDigit,
        "Alpha" | "Alphabetic" => Alphabetic,
        "Any" => Any,
        "Assigned" => Assigned,
        "Bidi_C" | "Bidi_Control" => BidiControl,
        "Bidi_M" | "Bidi_Mirrored" => BidiMirrored,
        "CI" | "Case_Ignorable" => CaseIgnorable,
        "Cased" => Cased,
        "CWCF" | "Changes_When_Casefolded" => ChangesWhenCasefolded,
        "CWCM" | "Changes_When_Casemapped" => ChangesWhenCasemapped,
        "CWL" | "Changes_When_Lowercased" => ChangesWhenLowercased,
        "CWKCF" | "Changes_When_NFKC_Casefolded" => ChangesWhenNfkcCasefolded,
        "CWT" | "Changes_When_Titlecased" => ChangesWhenTitlecased,
        "CWU" | "Changes_When_Uppercased" => ChangesWhenUppercased,
        "Dash" => Dash,
        "DI" | "Default_Ignorable_Code_Point" => DefaultIgnorable,
        "Dep" | "Deprecated" => Deprecated,
        "Dia" | "Diacritic" => Diacritic,
        "Emoji" => Emoji,
        "EComp" | "Emoji_Component" => EmojiComponent,
        "EMod" | "Emoji_Modifier" => EmojiModifier,
        "EBase" | "Emoji_Modifier_Base" => EmojiModifierBase,
        "EPres" | "Emoji_Presentation" => EmojiPresentation,
        "ExtPict" | "Extended_Pictographic" => ExtendedPictographic,
        "Ext" | "Extender" => Extender,
        "Gr_Base" | "Grapheme_Base" => GraphemeBase,
        "Gr_Ext" | "Grapheme_Extend" => GraphemeExtend,
        "Hex" | "Hex_Digit" => HexDigit,
        "IDSB" | "IDS_Binary_Operator" => IdsBinaryOperator,
        "IDST" | "IDS_Trinary_Operator" => IdsTrinaryOperator,
        "IDC" | "ID_Continue" => IdContinue,
        "IDS" | "ID_Start" => IdStart,
        "Ideo" | "Ideographic" => Ideographic,
        "Join_C" | "Join_Control" => JoinControl,
        "LOE" | "Logical_Order_Exception" => LogicalOrderException,
        "Lower" | "Lowercase" => Lowercase,
        "Math" => Math,
        "NChar" | "Noncharacter_Code_Point" => NoncharacterCodePoint,
        "Pat_Syn" | "Pattern_Syntax" => PatternSyntax,
        "Pat_WS" | "Pattern_White_Space" => PatternWhiteSpace,
        "QMark" | "Quotation_Mark" => QuotationMark,
        "Radical" => Radical,
        "RI" | "Regional_Indicator" => RegionalIndicator,
        "STerm" | "Sentence_Terminal" => SentenceTerminal,
        "SD" | "Soft_Dotted" => SoftDotted,
        "Term" | "Terminal_Punctuation" => TerminalPunctuation,
        "UIdeo" | "Unified_Ideograph" => UnifiedIdeograph,
        "Upper" | "Uppercase" => Uppercase,
        "VS" | "Variation_Selector" => VariationSelector,
        "White_Space" | "space" => WhiteSpace,
        "XIDC" | "XID_Continue" => XidContinue,
        "XIDS" | "XID_Start" => XidStart,
        _ => return None,
    })
}

/// Maps a `General_Category` value name (canonical, alias, or short code) to a
/// [`GcSel`]. Covers the seven groups, the thirty subcategories, the synthetic
/// `LC`/`Cased_Letter`, and the POSIX-style aliases ECMAScript recognises
/// (`cntrl`, `digit`, `punct`).
fn gc_value(name: &str) -> Option<GcSel> {
    // Long-form / alias → canonical short code (or a synthetic selector).
    let code: &str = match name {
        "Other" => "C",
        "Letter" => "L",
        "Cased_Letter" | "LC" => return Some(GcSel::CasedLetter),
        "Mark" | "Combining_Mark" => "M",
        "Number" => "N",
        "Punctuation" | "punct" => "P",
        "Symbol" => "S",
        "Separator" => "Z",
        "Uppercase_Letter" => "Lu",
        "Lowercase_Letter" => "Ll",
        "Titlecase_Letter" => "Lt",
        "Modifier_Letter" => "Lm",
        "Other_Letter" => "Lo",
        "Nonspacing_Mark" => "Mn",
        "Spacing_Mark" => "Mc",
        "Enclosing_Mark" => "Me",
        "Decimal_Number" | "digit" => "Nd",
        "Letter_Number" => "Nl",
        "Other_Number" => "No",
        "Connector_Punctuation" => "Pc",
        "Dash_Punctuation" => "Pd",
        "Open_Punctuation" => "Ps",
        "Close_Punctuation" => "Pe",
        "Initial_Punctuation" => "Pi",
        "Final_Punctuation" => "Pf",
        "Other_Punctuation" => "Po",
        "Math_Symbol" => "Sm",
        "Currency_Symbol" => "Sc",
        "Modifier_Symbol" => "Sk",
        "Other_Symbol" => "So",
        "Space_Separator" => "Zs",
        "Line_Separator" => "Zl",
        "Paragraph_Separator" => "Zp",
        "Control" | "cntrl" => "Cc",
        "Format" => "Cf",
        "Surrogate" => "Cs",
        "Private_Use" => "Co",
        "Unassigned" => "Cn",
        other => other,
    };
    const SUBCATS: [&str; 30] = [
        "Lu", "Ll", "Lt", "Lm", "Lo", "Mn", "Mc", "Me", "Nd", "Nl", "No", "Pc", "Pd", "Ps", "Pe",
        "Pi", "Pf", "Po", "Sm", "Sc", "Sk", "So", "Zs", "Zl", "Zp", "Cc", "Cf", "Cs", "Co", "Cn",
    ];
    let b = code.as_bytes();
    match code {
        "L" | "M" | "N" | "P" | "S" | "Z" | "C" => Some(GcSel::Group(b[0])),
        _ if SUBCATS.contains(&code) => Some(GcSel::Sub([b[0], b[1]])),
        _ => None,
    }
}

/// Resolves a `Script`/`Script_Extensions` value name (long name or ISO 15924
/// short code) to a [`ScriptId`]. Exact, case-sensitive matching.
#[cfg(feature = "intl")]
fn script_id(name: &str) -> Option<ScriptId> {
    let s = match name {
        "Adlam" | "Adlm" => Script::Adlam,
        "Ahom" => Script::Ahom,
        "Anatolian_Hieroglyphs" | "Hluw" => Script::AnatolianHieroglyphs,
        "Arabic" | "Arab" => Script::Arabic,
        "Armenian" | "Armn" => Script::Armenian,
        "Avestan" | "Avst" => Script::Avestan,
        "Balinese" | "Bali" => Script::Balinese,
        "Bamum" | "Bamu" => Script::Bamum,
        "Bassa_Vah" | "Bass" => Script::BassaVah,
        "Batak" | "Batk" => Script::Batak,
        "Bengali" | "Beng" => Script::Bengali,
        "Beria_Erfe" | "Berf" => Script::BeriaErfe,
        "Bhaiksuki" | "Bhks" => Script::Bhaiksuki,
        "Bopomofo" | "Bopo" => Script::Bopomofo,
        "Brahmi" | "Brah" => Script::Brahmi,
        "Braille" | "Brai" => Script::Braille,
        "Buginese" | "Bugi" => Script::Buginese,
        "Buhid" | "Buhd" => Script::Buhid,
        "Canadian_Aboriginal" | "Cans" => Script::CanadianAboriginal,
        "Carian" | "Cari" => Script::Carian,
        "Caucasian_Albanian" | "Aghb" => Script::CaucasianAlbanian,
        "Chakma" | "Cakm" => Script::Chakma,
        "Cham" => Script::Cham,
        "Cherokee" | "Cher" => Script::Cherokee,
        "Chorasmian" | "Chrs" => Script::Chorasmian,
        "Common" | "Zyyy" => Script::Common,
        "Coptic" | "Copt" | "Qaac" => Script::Coptic,
        "Cuneiform" | "Xsux" => Script::Cuneiform,
        "Cypriot" | "Cprt" => Script::Cypriot,
        "Cypro_Minoan" | "Cpmn" => Script::CyproMinoan,
        "Cyrillic" | "Cyrl" => Script::Cyrillic,
        "Deseret" | "Dsrt" => Script::Deseret,
        "Devanagari" | "Deva" => Script::Devanagari,
        "Dives_Akuru" | "Diak" => Script::DivesAkuru,
        "Dogra" | "Dogr" => Script::Dogra,
        "Duployan" | "Dupl" => Script::Duployan,
        "Egyptian_Hieroglyphs" | "Egyp" => Script::EgyptianHieroglyphs,
        "Elbasan" | "Elba" => Script::Elbasan,
        "Elymaic" | "Elym" => Script::Elymaic,
        "Ethiopic" | "Ethi" => Script::Ethiopic,
        "Garay" | "Gara" => Script::Garay,
        "Georgian" | "Geor" => Script::Georgian,
        "Glagolitic" | "Glag" => Script::Glagolitic,
        "Gothic" | "Goth" => Script::Gothic,
        "Grantha" | "Gran" => Script::Grantha,
        "Greek" | "Grek" => Script::Greek,
        "Gujarati" | "Gujr" => Script::Gujarati,
        "Gunjala_Gondi" | "Gong" => Script::GunjalaGondi,
        "Gurmukhi" | "Guru" => Script::Gurmukhi,
        "Gurung_Khema" | "Gukh" => Script::GurungKhema,
        "Han" | "Hani" => Script::Han,
        "Hangul" | "Hang" => Script::Hangul,
        "Hanifi_Rohingya" | "Rohg" => Script::HanifiRohingya,
        "Hanunoo" | "Hano" => Script::Hanunoo,
        "Hatran" | "Hatr" => Script::Hatran,
        "Hebrew" | "Hebr" => Script::Hebrew,
        "Hiragana" | "Hira" => Script::Hiragana,
        "Imperial_Aramaic" | "Armi" => Script::ImperialAramaic,
        "Inherited" | "Zinh" | "Qaai" => Script::Inherited,
        "Inscriptional_Pahlavi" | "Phli" => Script::InscriptionalPahlavi,
        "Inscriptional_Parthian" | "Prti" => Script::InscriptionalParthian,
        "Javanese" | "Java" => Script::Javanese,
        "Kaithi" | "Kthi" => Script::Kaithi,
        "Kannada" | "Knda" => Script::Kannada,
        "Katakana" | "Kana" => Script::Katakana,
        "Kawi" => Script::Kawi,
        "Kayah_Li" | "Kali" => Script::KayahLi,
        "Kharoshthi" | "Khar" => Script::Kharoshthi,
        "Khitan_Small_Script" | "Kits" => Script::KhitanSmallScript,
        "Khmer" | "Khmr" => Script::Khmer,
        "Khojki" | "Khoj" => Script::Khojki,
        "Khudawadi" | "Sind" => Script::Khudawadi,
        "Kirat_Rai" | "Krai" => Script::KiratRai,
        "Lao" | "Laoo" => Script::Lao,
        "Latin" | "Latn" => Script::Latin,
        "Lepcha" | "Lepc" => Script::Lepcha,
        "Limbu" | "Limb" => Script::Limbu,
        "Linear_A" | "Lina" => Script::LinearA,
        "Linear_B" | "Linb" => Script::LinearB,
        "Lisu" => Script::Lisu,
        "Lycian" | "Lyci" => Script::Lycian,
        "Lydian" | "Lydi" => Script::Lydian,
        "Mahajani" | "Mahj" => Script::Mahajani,
        "Makasar" | "Maka" => Script::Makasar,
        "Malayalam" | "Mlym" => Script::Malayalam,
        "Mandaic" | "Mand" => Script::Mandaic,
        "Manichaean" | "Mani" => Script::Manichaean,
        "Marchen" | "Marc" => Script::Marchen,
        "Masaram_Gondi" | "Gonm" => Script::MasaramGondi,
        "Medefaidrin" | "Medf" => Script::Medefaidrin,
        "Meetei_Mayek" | "Mtei" => Script::MeeteiMayek,
        "Mende_Kikakui" | "Mend" => Script::MendeKikakui,
        "Meroitic_Cursive" | "Merc" => Script::MeroiticCursive,
        "Meroitic_Hieroglyphs" | "Mero" => Script::MeroiticHieroglyphs,
        "Miao" | "Plrd" => Script::Miao,
        "Modi" => Script::Modi,
        "Mongolian" | "Mong" => Script::Mongolian,
        "Mro" | "Mroo" => Script::Mro,
        "Multani" | "Mult" => Script::Multani,
        "Myanmar" | "Mymr" => Script::Myanmar,
        "Nabataean" | "Nbat" => Script::Nabataean,
        "Nag_Mundari" | "Nagm" => Script::NagMundari,
        "Nandinagari" | "Nand" => Script::Nandinagari,
        "New_Tai_Lue" | "Talu" => Script::NewTaiLue,
        "Newa" => Script::Newa,
        "Nko" | "Nkoo" => Script::Nko,
        "Nushu" | "Nshu" => Script::Nushu,
        "Nyiakeng_Puachue_Hmong" | "Hmnp" => Script::NyiakengPuachueHmong,
        "Ogham" | "Ogam" => Script::Ogham,
        "Ol_Chiki" | "Olck" => Script::OlChiki,
        "Ol_Onal" | "Onao" => Script::OlOnal,
        "Old_Hungarian" | "Hung" => Script::OldHungarian,
        "Old_Italic" | "Ital" => Script::OldItalic,
        "Old_North_Arabian" | "Narb" => Script::OldNorthArabian,
        "Old_Permic" | "Perm" => Script::OldPermic,
        "Old_Persian" | "Xpeo" => Script::OldPersian,
        "Old_Sogdian" | "Sogo" => Script::OldSogdian,
        "Old_South_Arabian" | "Sarb" => Script::OldSouthArabian,
        "Old_Turkic" | "Orkh" => Script::OldTurkic,
        "Old_Uyghur" | "Ougr" => Script::OldUyghur,
        "Oriya" | "Orya" => Script::Oriya,
        "Osage" | "Osge" => Script::Osage,
        "Osmanya" | "Osma" => Script::Osmanya,
        "Pahawh_Hmong" | "Hmng" => Script::PahawhHmong,
        "Palmyrene" | "Palm" => Script::Palmyrene,
        "Pau_Cin_Hau" | "Pauc" => Script::PauCinHau,
        "Phags_Pa" | "Phag" => Script::PhagsPa,
        "Phoenician" | "Phnx" => Script::Phoenician,
        "Psalter_Pahlavi" | "Phlp" => Script::PsalterPahlavi,
        "Rejang" | "Rjng" => Script::Rejang,
        "Runic" | "Runr" => Script::Runic,
        "Samaritan" | "Samr" => Script::Samaritan,
        "Saurashtra" | "Saur" => Script::Saurashtra,
        "Sharada" | "Shrd" => Script::Sharada,
        "Shavian" | "Shaw" => Script::Shavian,
        "Siddham" | "Sidd" => Script::Siddham,
        "Sidetic" | "Sidt" => Script::Sidetic,
        "SignWriting" | "Sgnw" => Script::SignWriting,
        "Sinhala" | "Sinh" => Script::Sinhala,
        "Sogdian" | "Sogd" => Script::Sogdian,
        "Sora_Sompeng" | "Sora" => Script::SoraSompeng,
        "Soyombo" | "Soyo" => Script::Soyombo,
        "Sundanese" | "Sund" => Script::Sundanese,
        "Sunuwar" | "Sunu" => Script::Sunuwar,
        "Syloti_Nagri" | "Sylo" => Script::SylotiNagri,
        "Syriac" | "Syrc" => Script::Syriac,
        "Tagalog" | "Tglg" => Script::Tagalog,
        "Tagbanwa" | "Tagb" => Script::Tagbanwa,
        "Tai_Le" | "Tale" => Script::TaiLe,
        "Tai_Tham" | "Lana" => Script::TaiTham,
        "Tai_Viet" | "Tavt" => Script::TaiViet,
        "Tai_Yo" | "Tayo" => Script::TaiYo,
        "Takri" | "Takr" => Script::Takri,
        "Tamil" | "Taml" => Script::Tamil,
        "Tangsa" | "Tnsa" => Script::Tangsa,
        "Tangut" | "Tang" => Script::Tangut,
        "Telugu" | "Telu" => Script::Telugu,
        "Thaana" | "Thaa" => Script::Thaana,
        "Thai" => Script::Thai,
        "Tibetan" | "Tibt" => Script::Tibetan,
        "Tifinagh" | "Tfng" => Script::Tifinagh,
        "Tirhuta" | "Tirh" => Script::Tirhuta,
        "Todhri" | "Todr" => Script::Todhri,
        "Tolong_Siki" | "Tols" => Script::TolongSiki,
        "Toto" => Script::Toto,
        "Tulu_Tigalari" | "Tutg" => Script::TuluTigalari,
        "Ugaritic" | "Ugar" => Script::Ugaritic,
        "Vai" | "Vaii" => Script::Vai,
        "Vithkuqi" | "Vith" => Script::Vithkuqi,
        "Wancho" | "Wcho" => Script::Wancho,
        "Warang_Citi" | "Wara" => Script::WarangCiti,
        "Yezidi" | "Yezi" => Script::Yezidi,
        "Yi" | "Yiii" => Script::Yi,
        "Zanabazar_Square" | "Zanb" => Script::ZanabazarSquare,
        "Unknown" | "Zzzz" => Script::Unknown,
        _ => return None,
    };
    Some(ScriptId(s))
}

/// Without the Unicode tables there is no script data, but the *names* must still
/// validate (so a script escape is never a spurious `SyntaxError`). We therefore
/// accept any non-empty value and resolve it to a placeholder that matches no
/// code point.
#[cfg(not(feature = "intl"))]
fn script_id(name: &str) -> Option<ScriptId> {
    if name.is_empty() {
        None
    } else {
        Some(ScriptId)
    }
}

impl PropEscape {
    /// Tests whether code point `c` is a member of this property escape. A lone
    /// surrogate (`u32` with no scalar value) belongs to no property except the
    /// `General_Category` `Cs`/`Surrogate`, `Any`, and `Assigned` checks, which
    /// are handled on the raw code point.
    pub(crate) fn matches(self, cp: u32) -> bool {
        match self {
            PropEscape::Binary(b) => b.matches(cp),
            PropEscape::Gc(sel) => sel.matches(cp),
            PropEscape::Script(id) => script_of(cp) == Some(id),
            PropEscape::ScriptExt(id) => script_ext_contains(cp, id),
        }
    }
}

impl PartialEq for ScriptId {
    fn eq(&self, other: &Self) -> bool {
        #[cfg(feature = "intl")]
        {
            self.0 == other.0
        }
        #[cfg(not(feature = "intl"))]
        {
            let _ = other;
            // The placeholder matches nothing, so two placeholders are never
            // "equal" for the purpose of a (always-false) script test.
            false
        }
    }
}

#[cfg(feature = "intl")]
fn script_of(cp: u32) -> Option<ScriptId> {
    Some(ScriptId(intl::unicode::script_u32(cp)))
}

#[cfg(not(feature = "intl"))]
fn script_of(_cp: u32) -> Option<ScriptId> {
    None
}

#[cfg(feature = "intl")]
fn script_ext_contains(cp: u32, id: ScriptId) -> bool {
    intl::unicode::script_extensions_u32(cp).contains(id.0)
}

#[cfg(not(feature = "intl"))]
fn script_ext_contains(_cp: u32, _id: ScriptId) -> bool {
    false
}

impl GcSel {
    fn matches(self, cp: u32) -> bool {
        match self {
            GcSel::Group(g) => gc_group_matches(g, cp),
            GcSel::Sub(code) => gc_sub_matches(code, cp),
            // `Cased_Letter` = Lu | Ll | Lt.
            GcSel::CasedLetter => {
                gc_sub_matches(*b"Lu", cp)
                    || gc_sub_matches(*b"Ll", cp)
                    || gc_sub_matches(*b"Lt", cp)
            }
        }
    }
}

/// Whether `cp` is in the top-level general-category group `g` (`b'L'`, …).
#[cfg(feature = "intl")]
fn gc_group_matches(g: u8, cp: u32) -> bool {
    use intl::unicode::category::Group;
    let want = match g {
        b'L' => Group::Letter,
        b'M' => Group::Mark,
        b'N' => Group::Number,
        b'P' => Group::Punctuation,
        b'S' => Group::Symbol,
        b'Z' => Group::Separator,
        b'C' => Group::Other,
        _ => return false,
    };
    intl::unicode::general_category_u32(cp).group() == want
}

/// Whether `cp` is exactly the two-letter subcategory `code` (e.g. `b"Lu"`).
#[cfg(feature = "intl")]
fn gc_sub_matches(code: [u8; 2], cp: u32) -> bool {
    intl::unicode::general_category_u32(cp).abbr().as_bytes() == code
}

#[cfg(not(feature = "intl"))]
fn gc_group_matches(g: u8, cp: u32) -> bool {
    let Some(c) = char::from_u32(cp) else {
        // Lone surrogate: only the `C` (Other) group claims it.
        return g == b'C';
    };
    match g {
        b'L' => c.is_alphabetic(),
        b'N' => c.is_numeric(),
        b'Z' => c == ' ' || (c.is_whitespace() && !c.is_control()),
        b'C' => c.is_control(),
        b'P' => c.is_ascii_punctuation(),
        _ => false,
    }
}

#[cfg(not(feature = "intl"))]
fn gc_sub_matches(code: [u8; 2], cp: u32) -> bool {
    let Some(c) = char::from_u32(cp) else {
        return &code == b"Cs";
    };
    match &code {
        b"Lu" => c.is_uppercase(),
        b"Ll" => c.is_lowercase(),
        b"Lt" => false,
        b"Lo" => c.is_alphabetic() && !c.is_uppercase() && !c.is_lowercase(),
        b"Nd" => c.is_ascii_digit(),
        b"Cc" => c.is_control(),
        // Finer categories need the Unicode tables (the `intl` feature).
        _ => false,
    }
}

impl BinProp {
    fn matches(self, cp: u32) -> bool {
        use BinProp::*;
        match self {
            // Closed-form properties — no Unicode tables needed, always exact.
            Ascii => cp <= 0x7F,
            AsciiHexDigit => {
                matches!(cp, 0x30..=0x39 | 0x41..=0x46 | 0x61..=0x66)
            }
            Any => cp <= 0x10_FFFF,
            BidiControl => {
                matches!(cp, 0x061C | 0x200E | 0x200F | 0x202A..=0x202E | 0x2066..=0x2069)
            }
            JoinControl => matches!(cp, 0x200C | 0x200D),
            RegionalIndicator => matches!(cp, 0x1F1E6..=0x1F1FF),
            NoncharacterCodePoint => {
                matches!(cp, 0xFDD0..=0xFDEF) || (cp & 0xFFFE) == 0xFFFE && cp <= 0x10_FFFF
            }
            VariationSelector => {
                matches!(cp, 0x180B..=0x180D | 0x180F | 0xFE00..=0xFE0F | 0xE0100..=0xE01EF)
            }
            // Table-backed properties (intl). Without the feature these fall back
            // to the closest `char`-method approximation, or to "no match".
            other => other.matches_table(cp),
        }
    }

    /// Table-backed membership for properties that need Unicode data.
    #[cfg(feature = "intl")]
    fn matches_table(self, cp: u32) -> bool {
        use BinProp::*;
        use intl::unicode as u;
        let Some(c) = char::from_u32(cp) else {
            // Lone surrogates are assigned (Cs) but carry no other property.
            return matches!(self, Assigned);
        };
        match self {
            Alphabetic => u::is_alphabetic(c),
            Assigned => u::is_assigned(c),
            BidiMirrored => u::is_bidi_mirrored(c),
            Cased => {
                u::general_category(c).is_cased_letter() || u::is_lowercase(c) || u::is_uppercase(c)
            }
            ChangesWhenCasefolded => u::changes_when_casefolded(c),
            ChangesWhenCasemapped => u::changes_when_casemapped(c),
            ChangesWhenLowercased => u::changes_when_lowercased(c),
            ChangesWhenTitlecased => u::changes_when_titlecased(c),
            ChangesWhenUppercased => u::changes_when_uppercased(c),
            Dash => u::is_dash(c),
            DefaultIgnorable => u::is_default_ignorable(c),
            Diacritic => u::is_diacritic(c),
            HexDigit => u::is_hex_digit(c),
            Lowercase => u::is_lowercase(c),
            Math => u::is_math(c),
            QuotationMark => u::is_quotation_mark(c),
            Uppercase => u::is_uppercase(c),
            WhiteSpace => u::is_whitespace(c),
            // `XID_Start`/`XID_Continue` are exposed by the `intl` `identifiers`
            // feature, so they resolve exactly.
            XidStart => u::is_xid_start(c),
            XidContinue => u::is_xid_continue(c),
            // Closed-form variants are handled in `matches` and never reach here.
            Ascii
            | AsciiHexDigit
            | Any
            | BidiControl
            | JoinControl
            | RegionalIndicator
            | NoncharacterCodePoint
            | VariationSelector => false,
            // Everything else has no public `intl` predicate and no closed form:
            // served by the generated UCD range tables (same Unicode version as
            // the `intl` tables above).
            other => other.matches_ranges(cp),
        }
    }

    /// Membership via the generated UCD range tables in [`super::props_data`],
    /// for the binary properties `intl` exposes no predicate for. `None` means
    /// this property is not table-backed (it is closed-form or `intl`-backed and
    /// answered by the callers above).
    fn ranges(self) -> Option<&'static [(u32, u32)]> {
        use super::props_data as d;
        use BinProp::*;
        Some(match self {
            CaseIgnorable => d::CASE_IGNORABLE,
            ChangesWhenNfkcCasefolded => d::CHANGES_WHEN_NFKC_CASEFOLDED,
            Deprecated => d::DEPRECATED,
            Emoji => d::EMOJI,
            EmojiComponent => d::EMOJI_COMPONENT,
            EmojiModifier => d::EMOJI_MODIFIER,
            EmojiModifierBase => d::EMOJI_MODIFIER_BASE,
            EmojiPresentation => d::EMOJI_PRESENTATION,
            ExtendedPictographic => d::EXTENDED_PICTOGRAPHIC,
            Extender => d::EXTENDER,
            GraphemeBase => d::GRAPHEME_BASE,
            GraphemeExtend => d::GRAPHEME_EXTEND,
            IdContinue => d::ID_CONTINUE,
            IdStart => d::ID_START,
            IdsBinaryOperator => d::IDS_BINARY_OPERATOR,
            IdsTrinaryOperator => d::IDS_TRINARY_OPERATOR,
            Ideographic => d::IDEOGRAPHIC,
            LogicalOrderException => d::LOGICAL_ORDER_EXCEPTION,
            PatternSyntax => d::PATTERN_SYNTAX,
            PatternWhiteSpace => d::PATTERN_WHITE_SPACE,
            Radical => d::RADICAL,
            SentenceTerminal => d::SENTENCE_TERMINAL,
            SoftDotted => d::SOFT_DOTTED,
            TerminalPunctuation => d::TERMINAL_PUNCTUATION,
            UnifiedIdeograph => d::UNIFIED_IDEOGRAPH,
            _ => return None,
        })
    }

    fn matches_ranges(self, cp: u32) -> bool {
        self.ranges()
            .is_some_and(|r| super::props_data::in_ranges(r, cp))
    }

    #[cfg(not(feature = "intl"))]
    fn matches_table(self, cp: u32) -> bool {
        use BinProp::*;
        let Some(c) = char::from_u32(cp) else {
            return matches!(self, Assigned);
        };
        match self {
            Alphabetic => c.is_alphabetic(),
            Lowercase => c.is_lowercase(),
            Uppercase => c.is_uppercase(),
            WhiteSpace => c.is_whitespace(),
            HexDigit => c.is_ascii_hexdigit(),
            // The generated UCD range tables carry their own data and need no
            // `intl` feature, so they stay exact here too.
            other => other.matches_ranges(cp),
        }
    }
}
