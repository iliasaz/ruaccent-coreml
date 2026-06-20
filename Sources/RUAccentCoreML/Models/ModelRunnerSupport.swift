import CoreML
import Foundation

/// Shared inference path for the two WordPiece token-classification heads (M3 stress-usage,
/// M4 yo). Both are `[1, S, C]` logit models with a flexible input dim whose minimum is 4;
/// they differ only in whether they declare `token_type_ids` (M3 yes / M4 no) and in label maps.
enum ModelRunnerSupport {

    /// Run a token-classification model on `inputIds`, padding up to `minLength`, and return the
    /// per-token softmax vectors (length == `inputIds.count`, each of length `classes`).
    ///
    /// - Parameters:
    ///   - model: the loaded CoreML model.
    ///   - inputIds: real token ids (no padding); attention mask is `1` over them, `0` on padding.
    ///   - minLength: model flexible-shape minimum (pad short sequences up to this).
    ///   - classes: logit width (3 for M3/M4).
    ///   - needsTokenTypeIds: feed a zero `token_type_ids` tensor when the model declares one.
    ///   - modelName: for error messages.
    static func tokenClassProbabilities(
        model: MLModel,
        inputIds: [Int],
        minLength: Int,
        classes: Int,
        needsTokenTypeIds: Bool,
        modelName: String
    ) throws -> [[Double]] {
        let s = inputIds.count
        let length = max(s, minLength)
        let attention = [Int](repeating: 1, count: s)

        var features: [String: MLFeatureValue] = [
            "input_ids": MLFeatureValue(multiArray: try CoreMLSupport.int32Array(inputIds, paddedTo: length)),
            "attention_mask": MLFeatureValue(multiArray: try CoreMLSupport.int32Array(attention, paddedTo: length)),
        ]
        if needsTokenTypeIds {
            features["token_type_ids"] = MLFeatureValue(
                multiArray: try CoreMLSupport.int32Array([], paddedTo: length))
        }
        let provider = try MLDictionaryFeatureProvider(dictionary: features)
        let out = try model.prediction(from: provider)
        guard let logits = out.featureValue(for: "logits")?.multiArrayValue else {
            throw RUAccentError.model("\(modelName) produced no `logits` output")
        }

        // Storage order is [1, length, classes]; take the first `s` token rows.
        let flat = CoreMLSupport.doubles(logits)
        var rows: [[Double]] = []
        rows.reserveCapacity(s)
        for i in 0..<s {
            let base = i * classes
            var row = [Double](repeating: 0, count: classes)
            for c in 0..<classes { row[c] = flat[base + c] }
            rows.append(CoreMLSupport.softmax(row))
        }
        return rows
    }
}
