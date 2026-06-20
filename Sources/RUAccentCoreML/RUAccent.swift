import CoreML
import Foundation

/// On-device Russian accentuator — the CoreML port of RUAccent.
///
/// Loads the four CoreML models (M1 accent, M2 omograph, M3 stress-usage, M4 yo), the four packed
/// dictionaries, and the tokenizer resources, then runs the upstream `process_all_internal` pipeline
/// (`Internal/InternalPipeline.swift`) behind the `RussianStressing` contract.
///
/// `stress(_:notation:)` layers two things OVER the upstream-parity internal output:
///   1. **manual-stress protection** — a stress the caller already wrote (`"+"` before a vowel, or a
///      `U+0301` after a vowel) is preserved for that word and excluded from re-stressing. This is an
///      explicit addition to upstream (whose `normalize` would delete the mark); it is justified by
///      the integration contract ("manual stress in the input always wins").
///   2. **notation rendering** — the internal `"+"`-before-vowel form is returned verbatim for
///      `.plusBeforeVowel` (golden-parity) or converted to `U+0301`-after-vowel for `.combiningAcute`
///      (the default, the chatterbox-coreml / TTS convention). `ё` is restored by the pipeline.
public final class RUAccent: RussianStressing {

    /// Tuning for which stress sources are active.
    public struct Configuration: Sendable {
        /// Use the neural accentor model (M1) for out-of-dictionary multi-vowel words.
        public var useNeuralAccentor: Bool
        /// Use the context homograph-disambiguation model (M2) for ambiguous words.
        public var useHomographModel: Bool
        /// Restore `ё` from `е` (M4 + the yo dictionaries).
        public var restoreYo: Bool

        public init(useNeuralAccentor: Bool = true,
                    useHomographModel: Bool = true,
                    restoreYo: Bool = true) {
            self.useNeuralAccentor = useNeuralAccentor
            self.useHomographModel = useHomographModel
            self.restoreYo = restoreYo
        }
    }

    private let configuration: Configuration
    private let pipeline: InternalPipeline

    /// Where each model `.mlpackage` and its tokenizer resources live, relative to a model directory.
    /// The defaults match the converter's `_work/coreml` + `_work/nn` layout; a consumer bundles the
    /// `.mlpackage`s and the tokenizer source files (`vocab.txt`, `merges.txt`, …) alongside the packs.
    public struct ResourceLayout: Sendable {
        public var dictDirectory: URL
        public var m1Model: URL, m1TokenizerDir: URL
        public var m2Model: URL, m2TokenizerDir: URL
        public var m3Model: URL, m3TokenizerDir: URL
        public var m4Model: URL, m4TokenizerDir: URL

        public init(dictDirectory: URL,
                    m1Model: URL, m1TokenizerDir: URL,
                    m2Model: URL, m2TokenizerDir: URL,
                    m3Model: URL, m3TokenizerDir: URL,
                    m4Model: URL, m4TokenizerDir: URL) {
            self.dictDirectory = dictDirectory
            self.m1Model = m1Model; self.m1TokenizerDir = m1TokenizerDir
            self.m2Model = m2Model; self.m2TokenizerDir = m2TokenizerDir
            self.m3Model = m3Model; self.m3TokenizerDir = m3TokenizerDir
            self.m4Model = m4Model; self.m4TokenizerDir = m4TokenizerDir
        }
    }

    /// Designated initializer: load every resource from an explicit `ResourceLayout`.
    ///
    /// Models are compiled (`MLModel.compileModel`) and loaded on `cpuAndNeuralEngine` (the Phase-0
    /// oracle compute path). A disabled model (`Configuration`) is **not** loaded — so a caller who
    /// only wants dictionary accenting need not ship those `.mlpackage`s.
    public init(resources: ResourceLayout, configuration: Configuration = .init()) throws {
        self.configuration = configuration

        let dict = try DictReader(directory: resources.dictDirectory)

        // M3 stress-usage gate is always needed (it gates _process_accent for every word).
        let m3 = try RUAccent.loadModel(resources.m3Model)
        let stressRunner = try StressUsageRunner(
            model: m3, tokenizer: WordPieceTokenizer(directory: resources.m3TokenizerDir))
        let stressGate: StressUsageGate = StressUsageGateRunner(runner: stressRunner)

        let yoGate: YoGate
        if configuration.restoreYo {
            let m4 = try RUAccent.loadModel(resources.m4Model)
            let yoRunner = try YoRunner(
                model: m4, tokenizer: WordPieceTokenizer(directory: resources.m4TokenizerDir))
            yoGate = YoGateRunner(runner: yoRunner)
        } else {
            yoGate = DisabledYoGate()
        }

        let omographGate: OmographGate
        if configuration.useHomographModel {
            let m2 = try RUAccent.loadModel(resources.m2Model)
            let omoRunner = try OmographRunner(
                model: m2, tokenizer: ByteLevelBPETokenizer(directory: resources.m2TokenizerDir))
            omographGate = OmographGateRunner(runner: omoRunner)
        } else {
            omographGate = DisabledOmographGate()
        }

        let accentGate: AccentGate
        if configuration.useNeuralAccentor {
            let m1 = try RUAccent.loadModel(resources.m1Model)
            let accentRunner = try AccentRunner(
                model: m1, tokenizer: CharTokenizer(directory: resources.m1TokenizerDir))
            accentGate = AccentGateRunner(runner: accentRunner)
        } else {
            accentGate = DisabledAccentGate()
        }

        self.pipeline = InternalPipeline(
            dict: dict,
            configuration: configuration,
            stressGate: stressGate,
            yoGate: yoGate,
            omographGate: omographGate,
            accentGate: accentGate)
    }

    /// Convenience initializer: derive a `ResourceLayout` from a single `modelDirectory` using the
    /// converter's `_work` filenames. Expects, under `modelDirectory`:
    ///   - `dictpack/` with the four `.rapack`s,
    ///   - `coreml/` with `M{1,2,3,4}_*.mlpackage`,
    ///   - `nn/` with the four tokenizer source dirs.
    /// (A shipping consumer typically constructs `ResourceLayout` directly from its bundle.)
    public convenience init(modelDirectory: URL, configuration: Configuration = .init()) throws {
        let coreml = modelDirectory.appendingPathComponent("coreml")
        let nn = modelDirectory.appendingPathComponent("nn")
        let layout = ResourceLayout(
            dictDirectory: modelDirectory.appendingPathComponent("dictpack"),
            m1Model: coreml.appendingPathComponent("M1_accent_fp16.mlpackage"),
            m1TokenizerDir: nn.appendingPathComponent("nn_accent"),
            m2Model: coreml.appendingPathComponent("M2_omograph_fp16_pal8.mlpackage"),
            m2TokenizerDir: nn.appendingPathComponent("nn_omograph/turbo3.1"),
            m3Model: coreml.appendingPathComponent("M3_stress_fp16_pal8.mlpackage"),
            m3TokenizerDir: nn.appendingPathComponent("nn_stress_usage_predictor"),
            m4Model: coreml.appendingPathComponent("M4_yo_fp16.mlpackage"),
            m4TokenizerDir: nn.appendingPathComponent("nn_yo_homograph_resolver"))
        try self.init(resources: layout, configuration: configuration)
    }

    /// Test/advanced seam: inject a pre-built `InternalPipeline` (real or stubbed gates).
    init(pipeline: InternalPipeline, configuration: Configuration) {
        self.pipeline = pipeline
        self.configuration = configuration
    }

    // MARK: - RussianStressing

    public func stress(_ text: String, notation: StressNotation) throws -> String {
        // 1. Detect & canonicalize caller-supplied manual stress (manual-stress-wins). The user may
        //    write a "+" before a vowel or a U+0301 after a vowel; we canonicalize the latter to the
        //    internal "+"-before form (BEFORE normalize, which deletes U+0301), then preserve those
        //    marks through the pipeline (the word carries "+", so _process_accent skips it). This is
        //    an explicit addition over upstream (whose normalize strips both marks).
        let (canonical, hasUserMark) = ManualStress.canonicalize(text)

        let internalForm = try pipeline.processAllInternal(
            canonical, preserveUserPlus: hasUserMark)

        // 2. Render the internal "+"-before-vowel form into the requested notation.
        return Notation.render(internalForm, as: notation)
    }

    // MARK: - model loading

    /// Compile a `.mlpackage` and load it on CPU + Neural Engine (Phase-0 oracle compute path).
    private static func loadModel(_ url: URL) throws -> MLModel {
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        let compiled = try MLModel.compileModel(at: url)
        return try MLModel(contentsOf: compiled, configuration: config)
    }
}
