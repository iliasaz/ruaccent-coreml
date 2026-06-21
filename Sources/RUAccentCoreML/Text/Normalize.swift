import Foundation

/// `normalize(text)` — `re.sub(self.normalize, "", text)` (ruaccent.py:24/235).
///
/// The source regex is an adjacent-string-literal **trap** (§1.1): the `""` you think adds
/// ASCII double-quote actually closes one literal and opens the next, so **U+0022 is NOT in
/// the allowed set** while `'` (U+0027) survives. We hardcode the verified 45-char allowed
/// class (§1.3) as an explicit membership test rather than re-deriving it, and DELETE every
/// scalar not in the set (negated class + `re.sub` with `""`). See `docs/SWIFT_TEXT_PIPELINE.md` §1.
enum TextNormalizer {

    /// Cyrillic core ranges plus explicit `ё`/`Ё` (which sit outside а-я/А-Я).
    private static let cyrLowerRange: ClosedRange<UInt32> = 0x0430...0x044F   // а-я
    private static let cyrUpperRange: ClosedRange<UInt32> = 0x0410...0x042F   // А-Я

    /// The exact punctuation/symbol survivors (§1.3) — order irrelevant, membership is all
    /// that matters. NB: ASCII `"` (U+0022) is intentionally absent; `'` (U+0027) is present.
    private static let allowedPunct: Set<Unicode.Scalar> = [
        "\u{2014}",          // — EM DASH
        ".", ",", "!", "?", ":", ";",
        "\u{0027}",          // ' apostrophe (survives)
        "(", ")", "{", "}", "[", "]",
        "\u{00AB}", "\u{00BB}",          // « »
        "\u{201E}", "\u{201C}", "\u{201D}",  // „ “ ”
        "\u{002D}",          // - HYPHEN-MINUS
    ]

    /// True if a scalar is in the allowed (surviving) set.
    @inline(__always)
    static func isAllowed(_ s: Unicode.Scalar) -> Bool {
        let v = s.value
        // ASCII letters a-z / A-Z and digits 0-9.
        if (v >= 0x61 && v <= 0x7A) || (v >= 0x41 && v <= 0x5A) || (v >= 0x30 && v <= 0x39) {
            return true
        }
        // Cyrillic core ranges + ё/Ё.
        if cyrLowerRange.contains(v) || cyrUpperRange.contains(v) { return true }
        if v == 0x0451 || v == 0x0401 { return true }     // ё / Ё
        // \s — Unicode whitespace.
        if ScalarPredicates.isSpace(s) { return true }
        // The explicit punctuation set.
        return allowedPunct.contains(s)
    }

    /// Delete every disallowed scalar; keep order, no substitution.
    static func normalize(_ text: String) -> String {
        var out = String.UnicodeScalarView()
        for s in text.unicodeScalars where isAllowed(s) {
            out.append(s)
        }
        return String(out)
    }
}
