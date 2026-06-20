import Foundation

/// Unicode-scalar predicates that mirror Python `re` Unicode character classes, and a
/// handful of scalar-array string helpers. The whole text pipeline operates on
/// `[Unicode.Scalar]` (Python `str` = code points) — see `docs/SWIFT_TEXT_PIPELINE.md` §2.4/§7.
///
/// All offset/length math elsewhere in this module is in scalar units, NOT grapheme
/// clusters and NOT UTF-16, so that razdel offsets / `str.find` / `len()` / slices match
/// upstream byte-for-byte.
enum ScalarPredicates {

    /// Python `\s` — Unicode whitespace (space, `\t\n\r`, FF, VT, NBSP, Unicode spaces).
    @inline(__always)
    static func isSpace(_ s: Unicode.Scalar) -> Bool {
        s.properties.isWhitespace
    }

    /// Python `\d` — Unicode decimal digit.
    @inline(__always)
    static func isDigit(_ s: Unicode.Scalar) -> Bool {
        s.properties.numericType == .decimal
    }

    /// Python `\w` — `[\p{L}\p{N}_]`-ish: letter, number, or underscore.
    /// Post-normalize input has no combining marks, so the Mn/Mc subtleties of Python `\w`
    /// are moot (§3.3).
    @inline(__always)
    static func isWord(_ s: Unicode.Scalar) -> Bool {
        if s == "_" { return true }
        if s.properties.isAlphabetic { return true }
        switch s.properties.numericType {
        case .some: return true
        case .none: return false
        }
    }

    /// Python `[^\W\d]` — a word char that is not a digit = letter or underscore.
    @inline(__always)
    static func isWordNonDigit(_ s: Unicode.Scalar) -> Bool {
        isWord(s) && !isDigit(s)
    }

    /// Python per-char `str.isupper()` for a single scalar (cased & uppercase).
    @inline(__always)
    static func isUpper(_ s: Unicode.Scalar) -> Bool {
        s.properties.isUppercase
    }
}

/// Lightweight scalar-array helpers replicating the Python string ops the pipeline needs.
extension Array where Element == Unicode.Scalar {

    /// String built from these scalars.
    var asString: String {
        var v = String.UnicodeScalarView()
        v.append(contentsOf: self)
        return String(v)
    }

    /// Python `str.strip()` — drop leading & trailing Unicode whitespace, scalar-wise.
    func pyStrip() -> [Unicode.Scalar] {
        var lo = 0
        var hi = count
        while lo < hi, ScalarPredicates.isSpace(self[lo]) { lo += 1 }
        while hi > lo, ScalarPredicates.isSpace(self[hi - 1]) { hi -= 1 }
        return Array(self[lo..<hi])
    }

    /// Python `text.find(sub, offset)` over scalars: first index ≥ `from` where `sub`
    /// occurs, or `-1`. Mirrors `str.find` (empty `sub` returns `from`).
    func pyFind(_ sub: [Unicode.Scalar], from: Int) -> Int {
        let n = count
        let m = sub.count
        if m == 0 { return Swift.min(from, n) }
        if from > n - m { return -1 }
        var i = Swift.max(0, from)
        while i <= n - m {
            var k = 0
            while k < m, self[i + k] == sub[k] { k += 1 }
            if k == m { return i }
            i += 1
        }
        return -1
    }
}

extension String {
    /// This string as an array of Unicode scalars (the pipeline's working unit).
    var scalarArray: [Unicode.Scalar] { Array(unicodeScalars) }
}
