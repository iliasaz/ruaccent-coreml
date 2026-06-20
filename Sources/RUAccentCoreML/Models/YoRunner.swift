import CoreML
import Foundation

/// **M4** yo / homograph runner — the per-word gate that flags which words should be checked
/// for a `ё` restoration (`"YO"`) vs left alone (`"NO_YO"`/`"PUNCT"`).
///
/// Byte-exact port of `YoHomographResolverModel.predict_yo_homographs` (`yo_homograph_model.py`),
/// wrapping `M4_yo_fp16.mlpackage` (DistilBERT token-classifier, `logits [1, S, 3]`,
/// `id2label = {0:"NO_YO", 1:"PUNCT", 2:"YO"}`). DistilBERT has **no** `token_type_ids` input.
///
/// The caller lowercases the sentence before tokenizing (`_process_yo` calls `sentence.lower()`);
/// `predictYoHomographs` does the lowercasing so the entity list aligns 1:1 with the words. Flow is
/// identical to M3: WordPiece → run → per-token softmax → AVERAGE aggregation → per-word entities.
/// The model's flexible input dim minimum is 4, so short sentences pad up to 4 and slice back.
final class YoRunner {

    /// Model flexible-shape minimum sequence length.
    static let minLength = 4

    private let model: MLModel
    private let tokenizer: WordPieceTokenizer
    private let needsTokenTypeIds: Bool

    /// Load `M4_yo_fp16.mlpackage` and the M4 WordPiece tokenizer (`vocab.txt`).
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
    /// Exposed for tests feeding golden `input_ids` directly. M4 has no `token_type_ids`.
    func probabilities(inputIds: [Int]) throws -> [[Double]] {
        try ModelRunnerSupport.tokenClassProbabilities(
            model: model,
            inputIds: inputIds,
            minLength: YoRunner.minLength,
            classes: 3,
            needsTokenTypeIds: needsTokenTypeIds,
            modelName: "M4")
    }

    /// `predict_yo_homographs(text)`: lowercase, tokenize, run, aggregate → per-word entities.
    /// The caller in `_process_yo` already lowercases; this mirrors that for a standalone call.
    func predictYoHomographs(_ sentence: String) throws -> [WordPieceAggregation.WordEntity] {
        let lowered = sentence.lowercased()
        let enc = tokenizer.encode(lowered)
        let probs = try probabilities(inputIds: enc.inputIds)
        return WordPieceAggregation.aggregate(
            tokens: enc.tokens,
            offsets: enc.offsetMapping,
            specialTokensMask: enc.specialTokensMask,
            inputIds: enc.inputIds,
            scores: probs,
            sentence: Array(lowered.unicodeScalars),
            id2label: WordPieceAggregation.m4Labels)
    }
}
