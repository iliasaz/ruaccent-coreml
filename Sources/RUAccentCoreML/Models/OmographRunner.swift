import CoreML
import Foundation

/// **M2** omograph runner — the cross-encoder that, for each homograph in a sentence, scores
/// its candidate stress variants in context and picks the best one.
///
/// Byte-exact port of `OmographModel.classify` (`omograph_model.py`), wrapping
/// `M2_omograph_fp16_pal8.mlpackage` (DeBERTa seq-classifier, `logits [1, 2]`, no
/// `token_type_ids`). One model call scores one `(A, B)` pair, where `A` is the sentence with the
/// target word wrapped `<w>…</w>` (space-before-punct stripped) and `B` is a single stress-marked
/// variant; the **second** logit (`out[1]`) is the "this variant is correct" score.
///
/// Selection (spec §3.9 + `classify`): for each omograph, the variant with the maximum `out[1]`
/// wins. Upstream's `softmax` subtracts a **global** max and divides by a **global** sum (both
/// constants across the 2-vector), so the per-variant `out[1]` ranking is identical whether you
/// take the raw logit `logits[1]`, `logits[1] - logits[0]`, or the per-pair softmax `p[1]`. This
/// runner replicates the per-pair global softmax for fidelity and selects argmax-by-`p[1]` per
/// omograph — which matches both the even-path (pairs-of-2) and odd-path (`group_words`)
/// batching, because each variant is scored independently within its sub-batch.
final class OmographRunner {

    /// `special_words` from `OmographModel.__init__` — only affects odd-path *batching*, never the
    /// selected variant (each variant is scored as its own pair). Carried for parity completeness.
    static let specialWords: Set<String> = [
        "балчуга", "вертела", "волоки", "волоку", "воронью", "выбродите", "вывозите", "выносите",
        "выноситесь", "выходите", "железы", "начала", "округа", "перепела", "развитая", "развитого",
        "развитое", "развитой", "развитом", "развитому", "развитою", "развитую", "развитые",
        "развитым", "развитыми", "развитых", "сторожа", "сторожи", "сторожу", "удало", "начался",
        "началась", "началось", "бутиках", "ожила", "создало", "коротки", "проклята", "роженица",
        "роженицы", "рожениц", "роженице", "роженицам", "роженицу", "роженицей", "роженицею",
        "роженицами", "роженицах", "пристава", "приставов", "приставам", "приставами", "приставах",
        "пережитое", "пережитого", "пережитые", "пережитых", "пережитому", "пережитым", "пережитыми",
        "пережитом", "нипоняла",
    ]

    private let model: MLModel
    private let tokenizer: ByteLevelBPETokenizer

    /// Load `M2_omograph_fp16_pal8.mlpackage` and the M2 ByteLevel-BPE tokenizer.
    init(modelURL: URL, tokenizerDirectory: URL, configuration: MLModelConfiguration = .init()) throws {
        self.model = try MLModel(contentsOf: modelURL, configuration: configuration)
        self.tokenizer = try ByteLevelBPETokenizer(directory: tokenizerDirectory)
    }

    init(model: MLModel, tokenizer: ByteLevelBPETokenizer) {
        self.model = model
        self.tokenizer = tokenizer
    }

    /// Run the model on a pre-tokenized pair (`input_ids`) and return the **global** softmax over
    /// the 2-vector logits. `probTrue = p[1]`. Exposed for tests feeding golden `input_ids`.
    func probabilities(inputIds: [Int]) throws -> (probFalse: Double, probTrue: Double, logits: [Double]) {
        let logits = try rawLogits(inputIds: inputIds)
        let sm = CoreMLSupport.softmax(logits)  // 2-vector: global softmax == per-axis softmax
        return (sm[0], sm[1], logits)
    }

    /// Run the model and return the raw `[2]` logits.
    func rawLogits(inputIds: [Int]) throws -> [Double] {
        let s = inputIds.count
        let attention = [Int](repeating: 1, count: s)
        let features: [String: MLFeatureValue] = [
            "input_ids": MLFeatureValue(multiArray: try CoreMLSupport.int32Array(inputIds, paddedTo: s)),
            "attention_mask": MLFeatureValue(multiArray: try CoreMLSupport.int32Array(attention, paddedTo: s)),
        ]
        let provider = try MLDictionaryFeatureProvider(dictionary: features)
        let out = try model.prediction(from: provider)
        guard let logits = out.featureValue(for: "logits")?.multiArrayValue else {
            throw RUAccentError.model("M2 produced no `logits` output")
        }
        let flat = CoreMLSupport.doubles(logits)
        precondition(flat.count >= 2, "M2 logits expected width 2, got \(flat.count)")
        return [flat[0], flat[1]]
    }

    /// Score one `(markedText, hypothesis)` pair → `probTrue`. Encodes the Roberta pair layout
    /// (`<s> A </s></s> B </s>`) then runs the model.
    func probTrue(markedText: String, hypothesis: String) throws -> Double {
        let preprocessed = OmographRunner.stripSpaceBeforePunc(markedText)
        let enc = tokenizer.encodePair(preprocessed, hypothesis)
        return try probabilities(inputIds: enc.inputIds).probTrue
    }

    /// `classify(texts, hypotheses, num_hypotheses)` — pick one stress-marked variant per omograph.
    ///
    /// - Parameters:
    ///   - markedTexts: A-strings, one **per variant** (the sentence with `<w>…</w>` around the
    ///     omograph; repeated for each of that omograph's variants), flattened across all omographs.
    ///   - hypotheses: the flattened variant strings, same length & order as `markedTexts`.
    ///   - numHypotheses: per-omograph variant counts (sums to `hypotheses.count`).
    /// - Returns: one chosen variant string per omograph, in order.
    ///
    /// Reproduces the upstream even/odd batching shape; selection is per-variant argmax by `p[1]`,
    /// which is identical across both paths (the global softmax is monotone per pair).
    func classify(markedTexts: [String], hypotheses: [String], numHypotheses: [Int]) throws -> [String] {
        precondition(markedTexts.count == hypotheses.count, "texts/hypotheses length mismatch")
        let preprocessed = markedTexts.map(OmographRunner.stripSpaceBeforePunc)

        // Score every (A_i, B_i) pair once → p[1]. Selection is the same regardless of batching.
        var probs = [Double](repeating: 0, count: hypotheses.count)
        for i in 0..<hypotheses.count {
            let enc = tokenizer.encodePair(preprocessed[i], hypotheses[i])
            probs[i] = try probabilities(inputIds: enc.inputIds).probTrue
        }

        // Walk per-omograph variant groups (num_hypotheses) and pick the max-p[1] variant.
        var outs: [String] = []
        var cursor = 0
        for n in numHypotheses {
            guard n > 0 else { continue }
            var best = cursor
            for j in (cursor + 1)..<(cursor + n) where probs[j] > probs[best] { best = j }
            outs.append(hypotheses[best])
            cursor += n
        }
        return outs
    }

    // MARK: - Preprocessing

    /// `re.sub(r'\s+(?=(?:[,.?!:;…]))', '', text)` — strip whitespace immediately before
    /// `, . ? ! : ; …`. Operates on Unicode scalars to match Python `str` semantics.
    static func stripSpaceBeforePunc(_ text: String) -> String {
        let punct: Set<Unicode.Scalar> = [",", ".", "?", "!", ":", ";", "\u{2026}"]
        let scalars = Array(text.unicodeScalars)
        var out = String.UnicodeScalarView()
        var i = 0
        let n = scalars.count
        while i < n {
            let c = scalars[i]
            if isWhitespace(c) {
                // Look ahead past the whitespace run; drop it only if a target punct follows.
                var j = i
                while j < n && isWhitespace(scalars[j]) { j += 1 }
                if j < n && punct.contains(scalars[j]) {
                    i = j  // drop the whole whitespace run
                    continue
                }
                // Otherwise keep the run verbatim.
                while i < j { out.append(scalars[i]); i += 1 }
                continue
            }
            out.append(c)
            i += 1
        }
        return String(out)
    }

    /// Python `\s` whitespace (used by the strip-before-punct regex).
    private static func isWhitespace(_ c: Unicode.Scalar) -> Bool {
        let v = c.value
        if v == 0x20 || v == 0x09 || v == 0x0A || v == 0x0D || v == 0x0B || v == 0x0C { return true }
        return c.properties.generalCategory == .spaceSeparator
            || c.properties.generalCategory == .lineSeparator
            || c.properties.generalCategory == .paragraphSeparator
    }
}
