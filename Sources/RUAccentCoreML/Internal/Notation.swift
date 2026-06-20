import Foundation

/// Converts the internal upstream-parity `"+"`-before-vowel notation to the public notations.
///
/// The internal pipeline emits `"+"` immediately BEFORE the stressed vowel (e.g. `"м+ой"`). The
/// public `StressNotation` choices:
///   - `.plusBeforeVowel` — returned verbatim (byte-parity with upstream / the golden oracle).
///   - `.combiningAcute`  — each `"+<vowel>"` becomes `"<vowel>\u{0301}"` (combining acute AFTER
///     the vowel), the chatterbox-coreml / TTS convention. `ё` is already restored by the pipeline
///     and carried through unchanged.
enum Notation {

    /// U+0301 COMBINING ACUTE ACCENT.
    static let combiningAcute = Unicode.Scalar(0x0301)!

    /// Render the internal `"+"`-form `text` into `notation`.
    static func render(_ text: String, as notation: StressNotation) -> String {
        switch notation {
        case .plusBeforeVowel:
            return text
        case .combiningAcute:
            return plusBeforeToCombiningAfter(text)
        }
    }

    /// Move every `"+"` that precedes a scalar to a `U+0301` placed AFTER that scalar.
    /// A `"+"` not followed by a scalar (end of string) is dropped (it marks nothing). Operates on
    /// Unicode scalars so the mark lands immediately after the stressed vowel.
    static func plusBeforeToCombiningAfter(_ text: String) -> String {
        let scalars = Array(text.unicodeScalars)
        var out = String.UnicodeScalarView()
        var i = 0
        let n = scalars.count
        while i < n {
            if scalars[i] == "+" {
                // Emit the following scalar, then the combining acute. Drop a trailing bare '+'.
                if i + 1 < n {
                    out.append(scalars[i + 1])
                    out.append(combiningAcute)
                    i += 2
                } else {
                    i += 1
                }
            } else {
                out.append(scalars[i])
                i += 1
            }
        }
        return String(out)
    }
}
