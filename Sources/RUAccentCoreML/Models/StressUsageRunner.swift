import CoreML
import Foundation

/// **M3** stress-usage runner — the per-word gate that decides whether each word in a sentence
/// is eligible for accenting (`"STRESS"`) or not (`"NO_STRESS"`/`"PUNCT"`).
///
/// Byte-exact port of `StressUsagePredictorModel.predict_stress_usage` (`stress_usage_model.py`),
/// wrapping `M3_stress_fp16_pal8.mlpackage` (BERT token-classifier, `logits [1, S, 3]`,
/// `id2label = {0:"NO_STRESS", 1:"PUNCT", 2:"STRESS"}`). M3 declares a `token_type_ids` input →
/// feed zeros.
///
/// Flow: WordPiece-tokenize the sentence (`WordPieceTokenizer`) → run the model → per-token
/// softmax → AVERAGE subword aggregation (`WordPieceAggregation`) → one entity per word, in
/// sentence order. The model's flexible input dim has a **minimum of 4** (`shapeRange (4,512)`),
/// so short sentences are padded up to 4 (pad id 0, mask 0) and the logits sliced back to `S`.
final class StressUsageRunner {

    /// Model flexible-shape minimum sequence length (`shapeRange` lower bound on dim 1).
    static let minLength = 4

    private let model: MLModel
    private let tokenizer: WordPieceTokenizer
    private let needsTokenTypeIds: Bool

    /// Load `M3_stress_fp16_pal8.mlpackage` and the M3 WordPiece tokenizer (`vocab.txt`).
    init(modelURL: URL, tokenizerDirectory: URL, configuration: MLModelConfiguration = .init()) throws {
        self.model = try MLModel(contentsOf: modelURL, configuration: configuration)
        self.tokenizer = try WordPieceTokenizer(directory: tokenizerDirectory)
        self.needsTokenTypeIds = model.modelDescription.inputDescriptionsByName["token_type_ids"] != nil
    }

    init(model: MLModel, tokenizer: WordPieceTokenizer) {
        self.model = model
        self.tokenizer = tokenizer
        self.needsTokenTypeIds = model.modelDescription.inputDescriptionsByName["token_type_ids"] != nil
    }

    /// Run the model on `inputIds` and return per-token `[3]` softmax vectors (length == `S`).
    /// Exposed for tests feeding golden `input_ids` directly. M3 feeds zero `token_type_ids`.
    func probabilities(inputIds: [Int]) throws -> [[Double]] {
        try ModelRunnerSupport.tokenClassProbabilities(
            model: model,
            inputIds: inputIds,
            minLength: StressUsageRunner.minLength,
            classes: 3,
            needsTokenTypeIds: needsTokenTypeIds,
            modelName: "M3")
    }

    /// `predict_stress_usage(text)`: tokenize, run, aggregate → one entity per word in order.
    func predictStressUsage(_ sentence: String) throws -> [WordPieceAggregation.WordEntity] {
        let enc = tokenizer.encode(sentence)
        let probs = try probabilities(inputIds: enc.inputIds)
        return WordPieceAggregation.aggregate(
            tokens: enc.tokens,
            offsets: enc.offsetMapping,
            specialTokensMask: enc.specialTokensMask,
            inputIds: enc.inputIds,
            scores: probs,
            sentence: Array(sentence.unicodeScalars),
            id2label: WordPieceAggregation.m3Labels)
    }
}
