import CoreML
import Foundation

/// **M1** accent runner — the per-character stress placer for out-of-dictionary words.
///
/// Byte-exact port of `AccentModel.put_accent` / `render_stress` (`accent_model.py`), wrapping
/// `M1_accent_fp16.mlpackage`. The model is a RoFormer token-classifier with a **fixed** input
/// shape `[1, 48]` (no flexible range) and a single output `logits [1, 48, 3]`. The Swift caller
/// therefore pads `input_ids`/`attention_mask`/`token_type_ids` to length 48 (pad id 0, mask 0)
/// and reads back only the first `S` rows, where `S` is the real token count.
///
/// Decode (`accent_model.py` lines 22–46):
/// - per position `i` over the `S` tokens (`[bos] … [eos]`): `score = max(softmax(logits[i]))`,
///   `label = argmax(logits[i])` via `id2label = {0:"NO", 1:"STRESS_PRIMARY", 2:"STRESS_SECONDARY"}`;
/// - `render_stress` walks `text = list(word)` (the **original**, non-lowercased word as scalars);
///   when `label ∉ {"NO","STRESS_SECONDARY"}` **and** `score >= 0.55`, it inserts `"+"` **before**
///   `text[i-1]` (the previous scalar — the `[bos]` prefix shifts the model index by one, so a stressed
///   vowel at model index `i` is `word[i-1]`).
final class AccentRunner {

    /// `id2label` from `nn_accent/config.json`.
    static let label2id: [Int: String] = [0: "NO", 1: "STRESS_PRIMARY", 2: "STRESS_SECONDARY"]

    /// The model's fixed sequence length (`input_ids` shape `[1, 48]`).
    static let fixedLength = 48

    /// The render-stress score threshold (`render_stress`: `score >= 0.55`).
    static let scoreThreshold = 0.55

    private let model: MLModel
    private let tokenizer: CharTokenizer

    /// Whether the loaded model declares a `token_type_ids` input (M1 does → feed zeros).
    private let needsTokenTypeIds: Bool

    /// Load `M1_accent_fp16.mlpackage` and the char tokenizer (`vocab.txt`).
    ///
    /// - Parameters:
    ///   - modelURL: path to the `.mlpackage`.
    ///   - tokenizerDirectory: directory containing `vocab.txt` (the M1 `nn_accent/` dir).
    ///   - configuration: CoreML config; defaults to `.all` compute units (overridable for probes).
    init(modelURL: URL, tokenizerDirectory: URL, configuration: MLModelConfiguration = .init()) throws {
        self.model = try MLModel(contentsOf: modelURL, configuration: configuration)
        self.tokenizer = try CharTokenizer(directory: tokenizerDirectory)
        self.needsTokenTypeIds = model.modelDescription.inputDescriptionsByName["token_type_ids"] != nil
    }

    /// Build from a pre-loaded tokenizer (lets Assembly share one tokenizer instance).
    init(model: MLModel, tokenizer: CharTokenizer) {
        self.model = model
        self.tokenizer = tokenizer
        self.needsTokenTypeIds = model.modelDescription.inputDescriptionsByName["token_type_ids"] != nil
    }

    /// Run the model on `inputIds` and return the per-token `[3]` softmax probability vectors,
    /// length == `inputIds.count`. Pads to the model's fixed `[1, 48]` shape and slices back.
    /// Exposed for tests that feed golden `input_ids` directly (decoupled from the tokenizer).
    func probabilities(inputIds: [Int]) throws -> [[Double]] {
        let s = inputIds.count
        precondition(s <= AccentRunner.fixedLength, "M1 word longer than 48 tokens")
        let length = AccentRunner.fixedLength
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
            throw RUAccentError.model("M1 produced no `logits` output")
        }

        // logits storage order is [1, length, 3]; take the first `s` rows.
        let flat = CoreMLSupport.doubles(logits)
        let classes = 3
        var rows: [[Double]] = []
        rows.reserveCapacity(s)
        for i in 0..<s {
            let base = i * classes
            let row = [flat[base], flat[base + 1], flat[base + 2]]
            rows.append(CoreMLSupport.softmax(row))
        }
        return rows
    }

    /// `put_accent(word)`: tokenize the lowercased word, run the model, render `"+"` onto the
    /// original-cased word. Returns the stressed `"+"`-form string.
    func putAccent(_ word: String) throws -> String {
        let enc = tokenizer.encode(word)
        let probs = try probabilities(inputIds: enc.inputIds)
        return AccentRunner.putAccent(word: word, probs: probs)
    }

    /// Pure decode (no model): `render_stress` exactly as upstream — build the output by emitting
    /// a `"+"` immediately before `word[i-1]` for each firing model index, using the **original**
    /// scalar positions (Python indexes the immutable `list(text)` snapshot via `i-1`, prepending
    /// `"+"` to that element). Implemented as a position→prefix map to be insertion-order-safe.
    static func putAccent(word: String, probs: [[Double]]) -> String {
        let scalars = Array(word.unicodeScalars)
        var prefixPlus = [Bool](repeating: false, count: scalars.count)
        for i in 0..<probs.count {
            let row = probs[i]
            let label = label2id[CoreMLSupport.argmax(row)] ?? "NO"
            let score = row.max() ?? 0
            guard label != "NO", label != "STRESS_SECONDARY", score >= scoreThreshold else { continue }
            let target = i - 1
            if target >= 0 && target < scalars.count {
                prefixPlus[target] = true
            }
        }
        var out = String.UnicodeScalarView()
        for (idx, sc) in scalars.enumerated() {
            if prefixPlus[idx] { out.append("+") }
            out.append(sc)
        }
        return String(out)
    }
}
