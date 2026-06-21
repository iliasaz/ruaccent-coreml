import Testing
import Foundation
import CoreML
@testable import RUAccentCoreML

/// Stage 4 / ModelRunners parity tests.
///
/// Each runner is fed **golden `input_ids`** directly (decoupled from tokenizer correctness) and
/// its decode is asserted against the golden oracle:
/// - **M1**: `put_accent(word)` == golden `put_accent` (the rendered `"+"`-form). The argmax/score
///   are exercised implicitly by the rendered string (golden carries no per-position labels).
/// - **M3/M4**: per-token softmax → AVERAGE aggregation; aggregated per-word **entity label** is
///   asserted exactly, the **score within ~5e-2** (fp16 on Mac drifts vs onnxruntime fp32, but
///   decisions are Phase-0-stable).
/// - **M2**: per-pair **decision** (argmax of the 2-vector) asserted exactly; `prob_true` within
///   ~5e-2. Plus the `замок` door/castle selection picks the right variant.
///
/// Models (`.mlpackage`) and tokenizer source files are gitignored but present locally; loaded via
/// a `#filePath`-relative path to the repo root. If an artifact is missing the test is skipped
/// (so CI without the ~500 MB artifacts stays green); locally they are present and run.
@Suite struct ModelRunnerTests {

    // MARK: paths

    static let repoRoot: URL = {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }()

    static let nnDir = repoRoot.appendingPathComponent("converter/_work/nn")
    static let coremlDir = repoRoot.appendingPathComponent("converter/_work/coreml")
    static let goldenURL = repoRoot.appendingPathComponent("converter/fixtures/golden.json")

    static let m1TokDir = nnDir.appendingPathComponent("nn_accent")
    static let m3TokDir = nnDir.appendingPathComponent("nn_stress_usage_predictor")
    static let m4TokDir = nnDir.appendingPathComponent("nn_yo_homograph_resolver")
    static let m2TokDir = nnDir.appendingPathComponent("nn_omograph/turbo3.1")

    static let m1Model = coremlDir.appendingPathComponent("M1_accent_fp16.mlpackage")
    static let m2Model = coremlDir.appendingPathComponent("M2_omograph_fp16_pal8.mlpackage")
    static let m3Model = coremlDir.appendingPathComponent("M3_stress_fp16_pal8.mlpackage")
    static let m4Model = coremlDir.appendingPathComponent("M4_yo_fp16.mlpackage")

    static func exists(_ url: URL) -> Bool { FileManager.default.fileExists(atPath: url.path) }

    /// Compile + load a `.mlpackage` (CPU+NE, matching the Phase-0 oracle compute path).
    static func loadModel(_ url: URL) throws -> MLModel {
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        let compiled = try MLModel.compileModel(at: url)
        return try MLModel(contentsOf: compiled, configuration: config)
    }

    // MARK: golden decode

    static let golden: GoldenRunners = {
        guard let data = try? Data(contentsOf: goldenURL),
              let g = try? JSONDecoder().decode(GoldenRunners.self, from: data)
        else { return GoldenRunners() }
        return g
    }()

    static let scoreTol = 5e-2

    // MARK: - M1 accent

    @Test func m1PutAccentMatchesGolden() throws {
        try #require(Self.exists(Self.m1Model), "M1 .mlpackage missing")
        try #require(Self.exists(Self.m1TokDir.appendingPathComponent("vocab.txt")), "M1 vocab.txt missing")
        let model = try Self.loadModel(Self.m1Model)
        let tok = try CharTokenizer(directory: Self.m1TokDir)
        let runner = AccentRunner(model: model, tokenizer: tok)

        let rows = Self.golden.tokenizers.M1_char
        try #require(!rows.isEmpty, "golden M1_char rows missing")
        for row in rows {
            // Feed golden input_ids directly → probs → render onto the original word.
            let probs = try runner.probabilities(inputIds: row.input_ids)
            let rendered = AccentRunner.putAccent(word: row.word, probs: probs)
            #expect(rendered == row.put_accent,
                    "M1 put_accent mismatch for '\(row.word)': got '\(rendered)' want '\(row.put_accent)'")
        }
    }

    @Test func m1PutAccentEndToEnd() throws {
        // Same assertion but driving the tokenizer too (the Assembly-facing path).
        try #require(Self.exists(Self.m1Model), "M1 .mlpackage missing")
        try #require(Self.exists(Self.m1TokDir.appendingPathComponent("vocab.txt")), "M1 vocab.txt missing")
        let model = try Self.loadModel(Self.m1Model)
        let tok = try CharTokenizer(directory: Self.m1TokDir)
        let runner = AccentRunner(model: model, tokenizer: tok)
        for row in Self.golden.tokenizers.M1_char {
            let rendered = try runner.putAccent(row.word)
            #expect(rendered == row.put_accent,
                    "M1 end-to-end put_accent mismatch for '\(row.word)': got '\(rendered)'")
        }
    }

    // MARK: - M3 stress-usage

    @Test func m3EntitiesMatchGolden() throws {
        try #require(Self.exists(Self.m3Model), "M3 .mlpackage missing")
        try #require(Self.exists(Self.m3TokDir.appendingPathComponent("vocab.txt")), "M3 vocab.txt missing")
        let model = try Self.loadModel(Self.m3Model)
        let tok = try WordPieceTokenizer(directory: Self.m3TokDir)
        let runner = StressUsageRunner(model: model, tokenizer: tok)

        let rows = Self.golden.tokenizers.M3_wordpiece
        try #require(!rows.isEmpty, "golden M3_wordpiece rows missing")
        for row in rows {
            // Feed golden input_ids → probs, then aggregate against the golden token layout.
            let probs = try runner.probabilities(inputIds: row.input_ids)
            let entities = WordPieceAggregation.aggregate(
                tokens: row.tokens,
                offsets: row.offset_mapping.map { ($0[0], $0[1]) },
                specialTokensMask: row.special_tokens_mask,
                inputIds: row.input_ids,
                scores: probs,
                sentence: Array(row.text.unicodeScalars),
                id2label: WordPieceAggregation.m3Labels)
            try Self.assertEntities(entities, row.entities, model: "M3", text: row.text)
        }
    }

    // MARK: - M4 yo

    @Test func m4EntitiesMatchGolden() throws {
        try #require(Self.exists(Self.m4Model), "M4 .mlpackage missing")
        try #require(Self.exists(Self.m4TokDir.appendingPathComponent("vocab.txt")), "M4 vocab.txt missing")
        let model = try Self.loadModel(Self.m4Model)
        let tok = try WordPieceTokenizer(directory: Self.m4TokDir)
        let runner = YoRunner(model: model, tokenizer: tok)

        let rows = Self.golden.tokenizers.M4_wordpiece
        try #require(!rows.isEmpty, "golden M4_wordpiece rows missing")
        for row in rows {
            let probs = try runner.probabilities(inputIds: row.input_ids)
            let entities = WordPieceAggregation.aggregate(
                tokens: row.tokens,
                offsets: row.offset_mapping.map { ($0[0], $0[1]) },
                specialTokensMask: row.special_tokens_mask,
                inputIds: row.input_ids,
                scores: probs,
                sentence: Array(row.text.unicodeScalars),
                id2label: WordPieceAggregation.m4Labels)
            try Self.assertEntities(entities, row.entities, model: "M4", text: row.text)
        }
    }

    static func assertEntities(
        _ got: [WordPieceAggregation.WordEntity],
        _ want: [GoldenRunners.Entity],
        model: String, text: String
    ) throws {
        #expect(got.count == want.count, "\(model) entity count mismatch for '\(text)'")
        guard got.count == want.count else { return }
        for (g, w) in zip(got, want) {
            // Label/decision: exact (Phase-0 proved decisions stable under fp16).
            #expect(g.label == w.entity,
                    "\(model) label mismatch for '\(w.word)' in '\(text)': got \(g.label) want \(w.entity)")
            // Score: within tolerance (fp16 Mac drift).
            #expect(abs(g.score - w.score) <= scoreTol,
                    "\(model) score drift for '\(w.word)': got \(g.score) want \(w.score)")
        }
    }

    // MARK: - M2 omograph

    @Test func m2ProbTrueAndDecisionMatchGolden() throws {
        try #require(Self.exists(Self.m2Model), "M2 .mlpackage missing")
        let model = try Self.loadModel(Self.m2Model)
        let tok = try ByteLevelBPETokenizer(directory: Self.m2TokDir)
        let runner = OmographRunner(model: model, tokenizer: tok)

        let rows = Self.golden.tokenizers.M2_bpe
        try #require(!rows.isEmpty, "golden M2_bpe rows missing")
        for row in rows {
            let (_, probTrue, logits) = try runner.probabilities(inputIds: row.input_ids)
            // Decision (argmax of the 2-vector) must match the golden decision exactly.
            let gotDecision = logits[1] > logits[0]
            let wantDecision = row.prob_true > 0.5
            #expect(gotDecision == wantDecision,
                    "M2 decision mismatch for '\(row.word)'/'\(row.hypothesis)': got p1>p0=\(gotDecision) want prob_true=\(row.prob_true)")
            // prob_true within tolerance.
            #expect(abs(probTrue - row.prob_true) <= Self.scoreTol,
                    "M2 prob_true drift for '\(row.word)'/'\(row.hypothesis)': got \(probTrue) want \(row.prob_true)")
        }
    }

    /// The `замок` door/castle selection: pick the variant with the larger `prob_true` per context.
    /// Golden rows 2/3 = "на двери висит замок" (padlock → `зам+ок`); rows 8/9 =
    /// "на горе стоит старинный замок" (castle → `з+амок`). Uses real `marked_text` from golden.
    @Test func m2ZamokSelection() throws {
        try #require(Self.exists(Self.m2Model), "M2 .mlpackage missing")
        let model = try Self.loadModel(Self.m2Model)
        let tok = try ByteLevelBPETokenizer(directory: Self.m2TokDir)
        let runner = OmographRunner(model: model, tokenizer: tok)

        let rows = Self.golden.tokenizers.M2_bpe
        // Find the two замок contexts by source sentence.
        let door = rows.filter { $0.word == "замок" && $0.source_sentence.contains("двери") }
        let castle = rows.filter { $0.word == "замок" && $0.source_sentence.contains("горе") }
        try #require(door.count == 2 && castle.count == 2, "expected 2 замок rows per context")

        // classify() for the door context: both variants scored, max prob_true wins.
        let doorChoice = try runner.classify(
            markedTexts: door.map(\.marked_text),
            hypotheses: door.map(\.hypothesis),
            numHypotheses: [2])
        #expect(doorChoice == ["зам+ок"], "padlock замок should pick зам+ок, got \(doorChoice)")

        let castleChoice = try runner.classify(
            markedTexts: castle.map(\.marked_text),
            hypotheses: castle.map(\.hypothesis),
            numHypotheses: [2])
        #expect(castleChoice == ["з+амок"], "castle замок should pick з+амок, got \(castleChoice)")
    }
}

// MARK: - Golden JSON model (runner-focused fields)

/// A second decode of `golden.json`'s `tokenizers` section, carrying the fields the runner tests
/// need (`put_accent`, entity `score`, `prob_true`, `marked_text`, `source_sentence`) that the
/// tokenizer-test `GoldenTok` omits. Kept separate to avoid touching that file.
struct GoldenRunners: Decodable {
    var tokenizers = Tokenizers()

    struct Tokenizers: Decodable {
        var M1_char: [M1Row] = []
        var M3_wordpiece: [WPRow] = []
        var M4_wordpiece: [WPRow] = []
        var M2_bpe: [M2Row] = []
    }

    struct M1Row: Decodable {
        let word: String
        let input_ids: [Int]
        let put_accent: String
    }

    struct WPRow: Decodable {
        let text: String
        let tokens: [String]
        let input_ids: [Int]
        let offset_mapping: [[Int]]
        let special_tokens_mask: [Int]
        let entities: [Entity]
    }

    struct Entity: Decodable {
        let word: String
        let entity: String
        let score: Double
        let start: Int
        let end: Int
    }

    struct M2Row: Decodable {
        let source_sentence: String
        let word: String
        let marked_text: String
        let hypothesis: String
        let input_ids: [Int]
        let prob_true: Double
    }
}
