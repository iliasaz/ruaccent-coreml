import Foundation

/// How a stress mark is written into the returned string.
public enum StressNotation: Sendable {
    /// Combining acute accent `U+0301` placed **after** the stressed vowel
    /// (the chatterbox-coreml / TTS convention, and this package's default).
    /// e.g. `мой` → `мо́й`.
    case combiningAcute
    /// RUAccent's native `+` placed **before** the stressed vowel — for byte-parity
    /// with the upstream Python output. e.g. `мой` → `м+ой`.
    case plusBeforeVowel
}

/// Places Russian lexical stress (ударение) on text.
///
/// This is the integration contract consumed by `chatterbox-coreml` (multilingual TTS):
/// the TTS text-prep step calls `stress(_:)` and feeds the `U+0301`-marked string into
/// the tokenizer. Implementations resolve stress in priority order — explicit manual
/// mark in the input (always wins) → accents dictionary → context homograph model →
/// neural OOV accentor — and restore `ё`. See `docs/MIGRATION_PLAN.md`.
public protocol RussianStressing: AnyObject {
    /// Returns `text` with stress marks placed per `notation`.
    ///
    /// A stress mark already present in the input (a manual `U+0301`, or a `+`) is
    /// **preserved and never overridden** — manual stress always wins.
    func stress(_ text: String, notation: StressNotation) throws -> String
}

public extension RussianStressing {
    /// Convenience: stress with the default `.combiningAcute` notation.
    func stress(_ text: String) throws -> String {
        try stress(text, notation: .combiningAcute)
    }
}

/// Errors surfaced by stress implementations.
public enum RUAccentError: Error, Sendable {
    /// A required model/dictionary resource was missing at the given path.
    case missingResource(String)
    /// A CoreML model failed to load or predict.
    case model(String)
}
