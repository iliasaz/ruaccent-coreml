import Foundation

/// Adapters wiring the CoreML runners (Stage 4) to the orchestrator gate protocols
/// (`ModelGates.swift`). Each adapter forwards to its runner and projects to the label/variant
/// strings the orchestrator consumes — so the runners stay model-focused and the orchestrator
/// stays model-agnostic.

/// M3 adapter: `predict_stress_usage` → entity labels (`extract_entities`).
struct StressUsageGateRunner: StressUsageGate {
    let runner: StressUsageRunner
    func stressUsages(forSentence sentence: String) throws -> [String] {
        try runner.predictStressUsage(sentence).map(\.label)
    }
}

/// M4 adapter: `predict_yo_homographs` → entity labels.
struct YoGateRunner: YoGate {
    let runner: YoRunner
    func yoPredictions(forSentence sentence: String) throws -> [String] {
        try runner.predictYoHomographs(sentence).map(\.label)
    }
}

/// M2 adapter: `classify` → chosen variant per omograph.
struct OmographGateRunner: OmographGate {
    let runner: OmographRunner
    func classify(markedTexts: [String], hypotheses: [String], numHypotheses: [Int]) throws -> [String] {
        try runner.classify(markedTexts: markedTexts, hypotheses: hypotheses, numHypotheses: numHypotheses)
    }
}

/// M1 adapter: `put_accent` → `"+"`-form string.
struct AccentGateRunner: AccentGate {
    let runner: AccentRunner
    func putAccent(_ word: String) throws -> String {
        try runner.putAccent(word)
    }
}
