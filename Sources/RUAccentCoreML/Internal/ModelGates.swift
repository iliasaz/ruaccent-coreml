import Foundation

/// The four neural operations the internal orchestrator depends on, abstracted so the
/// pipeline can be driven by the real CoreML runners *or* by deterministic stand-ins in tests
/// (and so `Configuration` can disable individual models — see `Disabled*` below).
///
/// Each protocol mirrors exactly one upstream model entry point used by
/// `process_all_internal` (`ruaccent.py`):
///   - `StressUsageGate`  ← `stress_usage_predictor.predict_stress_usage(sentence)` (M3)
///   - `YoGate`           ← `yo_homograph_model.predict_yo_homographs(lower_sentence)` (M4)
///   - `OmographGate`     ← `omograph_model.classify(texts, hypotheses, num_hypotheses)` (M2)
///   - `AccentGate`       ← `accent_model.put_accent(word)` (M1)
///
/// The orchestrator consumes only the *entity labels* (M3/M4) and the chosen variant / `"+"`-form
/// (M2/M1); it never sees raw logits. This is the §6 alignment-contract boundary: the gates return
/// lists in upstream order, and the orchestrator indexes them positionally **as-is** (no realign).

/// M3 — per-word stress-usage gate. Returns one label per WordPiece word-group, in sentence order.
protocol StressUsageGate {
    /// `predict_stress_usage(sentence)` → `["NO_STRESS" | "PUNCT" | "STRESS", …]` (entity labels).
    func stressUsages(forSentence sentence: String) throws -> [String]
}

/// M4 — per-word yo gate. Returns one label per word-group over the **lowercased** sentence.
protocol YoGate {
    /// `predict_yo_homographs(sentence.lower())` → `["NO_YO" | "PUNCT" | "YO", …]` (entity labels).
    func yoPredictions(forSentence sentence: String) throws -> [String]
}

/// M2 — omograph cross-encoder. Picks one stress-marked variant per omograph.
protocol OmographGate {
    /// `classify(texts, hypotheses, num_hypotheses)` → one chosen variant string per omograph.
    func classify(markedTexts: [String], hypotheses: [String], numHypotheses: [Int]) throws -> [String]
}

/// M1 — per-character OOV accentor. Renders the `"+"`-before-vowel form for one word.
protocol AccentGate {
    /// `put_accent(word)` → the `"+"`-marked word (original casing preserved).
    func putAccent(_ word: String) throws -> String
}

// MARK: - Disabled stand-ins (Configuration toggles)

/// `useHomographModel == false`: the orchestrator skips `_process_omographs` entirely (words left
/// unchanged), so this stand-in is never invoked. Provided so a non-optional gate type exists.
struct DisabledOmographGate: OmographGate {
    func classify(markedTexts: [String], hypotheses: [String], numHypotheses: [Int]) throws -> [String] {
        []  // unreachable when the gate is off (the orchestrator returns early)
    }
}

/// `useNeuralAccentor == false`: `_process_accent`'s OOV branch is suppressed — a word that would
/// have gone to M1 is left unstressed (dict hits still apply). The orchestrator checks the flag and
/// never calls this; provided so a non-optional gate type exists.
struct DisabledAccentGate: AccentGate {
    func putAccent(_ word: String) throws -> String { word }
}

/// `restoreYo == false`: M4 + the yo dicts are skipped. `_process_yo` returns the words unchanged.
struct DisabledYoGate: YoGate {
    func yoPredictions(forSentence sentence: String) throws -> [String] { [] }
}
