import Foundation

/// Detects and canonicalizes caller-supplied manual stress for the public `RUAccent.stress` wrapper
/// (the "manual stress always wins" integration contract). This is an explicit ADDITION over upstream
/// `process_all_internal`, whose `normalize` deletes both `"+"` and `U+0301` before any word logic.
///
/// A user mark is recognised at WORD granularity, in two notations:
///   - `"+"` immediately BEFORE a Cyrillic vowel (the internal/upstream form, e.g. `"сл+ово"`);
///   - `U+0301` immediately AFTER a Cyrillic vowel (the combining-acute form, e.g. `"сло́во"`).
///
/// `canonicalize` rewrites every `U+0301`-after-vowel into the internal `"+"`-before-vowel form and
/// reports whether any user mark was present. Downstream, the pipeline runs with `preserveUserPlus`,
/// so the canonical `"+"` survives normalize and `_process_accent` skips that word
/// (`if '+' in word: continue`) — preserving the user's stress verbatim.
enum ManualStress {

    /// U+0301 COMBINING ACUTE ACCENT.
    static let combiningAcute = Unicode.Scalar(0x0301)!

    /// Cyrillic vowels (both cases) — the `count_vowels` set (ruaccent.py:128).
    static let vowels: Set<Unicode.Scalar> = Set(
        "аеёиоуыэюяАЕЁИОУЫЭЮЯ".unicodeScalars)

    /// Canonicalize user marks to the internal `"+"`-before-vowel form.
    ///
    /// - Returns: the rewritten text and `hasUserMark` (true iff at least one recognised user mark
    ///   was found). When `hasUserMark` is false the text is returned with only `U+0301`-after-vowel
    ///   conversions applied (there are none in that case), i.e. effectively unchanged.
    static func canonicalize(_ text: String) -> (text: String, hasUserMark: Bool) {
        let scalars = Array(text.unicodeScalars)
        let n = scalars.count
        var out = String.UnicodeScalarView()
        var hasUserMark = false
        var i = 0
        while i < n {
            let s = scalars[i]
            if s == combiningAcute {
                // A combining acute AFTER a vowel: we've already emitted that vowel; rewrite to a
                // "+" placed BEFORE it. Find the last emitted scalar; if it's a vowel, splice.
                if let last = out.last, vowels.contains(last) {
                    // Remove the just-emitted vowel, emit "+", then the vowel.
                    out.removeLast()
                    out.append("+")
                    out.append(last)
                    hasUserMark = true
                }
                // A stray U+0301 not after a vowel is dropped (normalize would delete it anyway).
                i += 1
                continue
            }
            if s == "+" {
                // A "+" immediately before a Cyrillic vowel is a user mark; keep it. (A "+" not
                // before a vowel — e.g. arithmetic `2+2` — is left as-is here and later stripped by
                // normalizePreservingPlus unless it sits between word chars, matching split_by_words.)
                if i + 1 < n, vowels.contains(scalars[i + 1]) {
                    hasUserMark = true
                }
                out.append(s)
                i += 1
                continue
            }
            out.append(s)
            i += 1
        }
        return (String(out), hasUserMark)
    }
}
