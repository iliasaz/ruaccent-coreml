import Testing
import Foundation
import CoreML
@testable import RUAccentCoreML

/// Stage 5 / Assembly — end-to-end golden parity.
///
/// Runs the real internal orchestrator (`InternalPipeline`, wired to the four CoreML runners +
/// dict packs + tokenizers) over every `pipeline` row in `converter/fixtures/golden.json` and
/// asserts the rendered `"+"`-before-vowel output equals the oracle `final` byte-for-byte. For a
/// representative subset it also asserts the per-sentence intermediates (`after_yo`,
/// `after_omographs`, `after_accent`) to catch stage drift the final string could hide.
///
/// Models (`.mlpackage`), dict packs (`.rapack`) and tokenizer source files are gitignored but
/// present locally; loaded via a `#filePath`-relative path to the repo root. If any artifact is
/// missing the suite is skipped (CI without the artifacts stays green).
@Suite struct InternalPipelineTests {

    // MARK: paths

    static let repoRoot: URL = {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }()

    static let nnDir = repoRoot.appendingPathComponent("converter/_work/nn")
    static let coremlDir = repoRoot.appendingPathComponent("converter/_work/coreml")
    static let dictDir = repoRoot.appendingPathComponent("converter/_work/dictpack")
    static let goldenURL = repoRoot.appendingPathComponent("converter/fixtures/golden.json")
    static let fixturesURL = repoRoot.appendingPathComponent("converter/fixtures/fixtures.json")

    static let m1Model = coremlDir.appendingPathComponent("M1_accent_fp16.mlpackage")
    static let m2Model = coremlDir.appendingPathComponent("M2_omograph_fp16_pal8.mlpackage")
    static let m3Model = coremlDir.appendingPathComponent("M3_stress_fp16_pal8.mlpackage")
    static let m4Model = coremlDir.appendingPathComponent("M4_yo_fp16.mlpackage")

    static let m1TokDir = nnDir.appendingPathComponent("nn_accent")
    static let m3TokDir = nnDir.appendingPathComponent("nn_stress_usage_predictor")
    static let m4TokDir = nnDir.appendingPathComponent("nn_yo_homograph_resolver")
    static let m2TokDir = nnDir.appendingPathComponent("nn_omograph/turbo3.1")

    static func exists(_ url: URL) -> Bool { FileManager.default.fileExists(atPath: url.path) }

    static var artifactsPresent: Bool {
        [m1Model, m2Model, m3Model, m4Model, dictDir, goldenURL].allSatisfy(exists)
            && [m1TokDir, m3TokDir, m4TokDir, m2TokDir].allSatisfy(exists)
    }

    static func loadModel(_ url: URL) throws -> MLModel {
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        let compiled = try MLModel.compileModel(at: url)
        return try MLModel(contentsOf: compiled, configuration: config)
    }

    // MARK: shared pipeline (built once)

    /// A real `InternalPipeline` wired to the four runners + dicts. Built lazily and shared across
    /// tests in this suite (model compilation is expensive). `nil` if artifacts are missing.
    nonisolated(unsafe) static let pipeline: InternalPipeline? = {
        guard artifactsPresent else { return nil }
        return try? buildPipeline(configuration: .init())
    }()

    static func buildPipeline(configuration: RUAccent.Configuration) throws -> InternalPipeline {
        let dict = try DictReader(directory: dictDir)
        let stress = StressUsageGateRunner(runner: try StressUsageRunner(
            model: loadModel(m3Model), tokenizer: WordPieceTokenizer(directory: m3TokDir)))
        let yo = YoGateRunner(runner: try YoRunner(
            model: loadModel(m4Model), tokenizer: WordPieceTokenizer(directory: m4TokDir)))
        let omo = OmographGateRunner(runner: try OmographRunner(
            model: loadModel(m2Model), tokenizer: ByteLevelBPETokenizer(directory: m2TokDir)))
        let acc = AccentGateRunner(runner: try AccentRunner(
            model: loadModel(m1Model), tokenizer: CharTokenizer(directory: m1TokDir)))
        return InternalPipeline(
            dict: dict, configuration: configuration,
            stressGate: stress, yoGate: yo, omographGate: omo, accentGate: acc)
    }

    // MARK: golden decode

    static let golden: GoldenPipelineFull = {
        guard let data = try? Data(contentsOf: goldenURL),
              let g = try? JSONDecoder().decode(GoldenPipelineFull.self, from: data)
        else { return GoldenPipelineFull() }
        return g
    }()

    // MARK: - end-to-end parity over all 52 pipeline inputs

    @Test func endToEndGoldenParity() throws {
        try #require(Self.artifactsPresent, "artifacts (models/dicts) missing — skipping")
        let pipeline = try #require(Self.pipeline, "pipeline failed to build")
        let rows = Self.golden.pipeline
        try #require(rows.count == 52, "expected 52 golden pipeline rows, got \(rows.count)")

        var matches = 0
        var misses: [(idx: Int, input: String, got: String, want: String)] = []
        for (idx, row) in rows.enumerated() {
            let got = try pipeline.processAllInternal(row.input)   // "+"-before-vowel form
            if got == row.final {
                matches += 1
            } else {
                misses.append((idx, row.input, got, row.final))
            }
        }
        for m in misses {
            Issue.record("E2E miss [\(m.idx)] \(m.input.debugDescription): got \(m.got.debugDescription) want \(m.want.debugDescription)")
        }
        // Acceptance: >= 50/52, target 52/52. Each miss is recorded above for diagnosis.
        #expect(matches >= 50, "end-to-end golden parity \(matches)/52 (< 50)")
        #expect(matches == 52, "end-to-end golden parity \(matches)/52 (target 52)")
    }

    // MARK: - cross-check against fixtures.json (overlapping subset)

    @Test func fixturesJsonParity() throws {
        try #require(Self.artifactsPresent, "artifacts missing — skipping")
        let pipeline = try #require(Self.pipeline, "pipeline failed to build")
        let data = try #require(try? Data(contentsOf: Self.fixturesURL), "fixtures.json missing")
        let fx = try JSONDecoder().decode(GoldenFixtures.self, from: data)
        var pairs: [(String, String)] = []
        for (_, group) in fx.groups { for p in group { pairs.append((p.input, p.output)) } }
        try #require(!pairs.isEmpty, "fixtures.json had no pairs")

        var matches = 0
        for (input, output) in pairs {
            let got = try pipeline.processAllInternal(input)
            if got == output { matches += 1 }
            else { Issue.record("fixtures miss \(input.debugDescription): got \(got.debugDescription) want \(output.debugDescription)") }
        }
        #expect(matches == pairs.count, "fixtures.json parity \(matches)/\(pairs.count)")
    }

    // MARK: - per-stage intermediates (representative subset)

    /// Indices chosen to exercise each stage: homograph (0,1), yo (18,20), OOV M1 (24),
    /// function-word gate (28), punctuation-run alignment (32, 42), initials (35), multi-vowel
    /// dict hit (33), and the omograph-with-punctuation marked-text path (43).
    static let stageSubset = [0, 1, 18, 20, 24, 28, 32, 33, 35, 42, 43]

    @Test func perStageParity() throws {
        try #require(Self.artifactsPresent, "artifacts missing — skipping")
        let pipeline = try #require(Self.pipeline, "pipeline failed to build")
        for idx in Self.stageSubset {
            let row = Self.golden.pipeline[idx]
            // Reproduce the per-sentence stages directly (the pipeline does this internally; here we
            // drive the same calls so we can assert each intermediate against golden).
            let normalized = TextPipeline.normalize(row.input)
            let sentences = TextPipeline.splitBySentences(normalized)
            #expect(sentences == row.sentences, "[\(idx)] sentences mismatch")
            #expect(sentences.count == row.per_sentence.count, "[\(idx)] sentence count mismatch")

            for (si, sentence) in sentences.enumerated() {
                let gs = row.per_sentence[si]
                let split = TextPipeline.splitByWords(sentence)
                #expect(split.words == gs.words, "[\(idx)] words mismatch")
                #expect(split.remainingText == gs.remaining_text, "[\(idx)] remaining_text mismatch")

                let stressUsages = try pipeline.stressGate.stressUsages(forSentence: sentence)
                #expect(stressUsages == gs.stress_usages, "[\(idx)] stress_usages mismatch (got \(stressUsages))")

                let afterYo = try pipeline.processYo(words: split.words, sentence: sentence)
                #expect(afterYo == gs.after_yo, "[\(idx)] after_yo mismatch (got \(afterYo))")

                let afterOmo = try pipeline.processOmographs(words: afterYo)
                #expect(afterOmo == gs.after_omographs, "[\(idx)] after_omographs mismatch (got \(afterOmo))")

                let afterAcc = try pipeline.processAccent(words: afterOmo, stressUsages: stressUsages)
                #expect(afterAcc == gs.after_accent, "[\(idx)] after_accent mismatch (got \(afterAcc))")

                let rendered = TextPipeline.reassembleSentence(words: afterAcc, remainingText: split.remainingText)
                #expect(rendered == gs.rendered, "[\(idx)] rendered mismatch (got \(rendered.debugDescription))")
            }
        }
    }
}

// MARK: - Golden JSON model (pipeline section, with per-stage intermediates)

struct GoldenPipelineFull: Decodable {
    var pipeline: [Row] = []

    struct Row: Decodable {
        let input: String
        let normalized: String
        let sentences: [String]
        let per_sentence: [Sentence]
        let final: String
    }

    struct Sentence: Decodable {
        let sentence: String
        let words: [String]
        let remaining_text: [String]
        let stress_usages: [String]
        let after_yo: [String]
        let after_omographs: [String]
        let after_accent: [String]
        let rendered: String
    }
}

struct GoldenFixtures: Decodable {
    let groups: [String: [Pair]]
    struct Pair: Decodable { let input: String; let output: String }
}
