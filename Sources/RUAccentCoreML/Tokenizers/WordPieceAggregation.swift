import Foundation

/// Downstream subword AVERAGE aggregation for the M3 / M4 token-classification heads.
///
/// Byte-exact port of `collect_pre_entities` → `aggregate_words` → `aggregate_word("AVERAGE")`
/// in `stress_usage_model.py` / `yo_homograph_model.py`. See `docs/SWIFT_TOKENIZERS.md` §2.7.
///
/// The model produces a **per-token softmax score vector**; aggregation collapses subwords back
/// to whole words and assigns one label per word (the AVERAGE of the per-token softmax vectors,
/// then argmax). The ModelRunners / Assembly stages feed the real softmax vectors here; this file
/// owns only the grouping + averaging + argmax so the word segmentation is testable now against
/// the golden `entities` word list.
enum WordPieceAggregation {

    /// One aggregated word: its label index (`argmax` of the averaged scores), the label string,
    /// the averaged score at that index, and the `(start, end)` scalar span of the leading token.
    struct WordEntity: Equatable {
        var labelIndex: Int
        var label: String
        var score: Double
        var start: Int
        var end: Int
        /// The grouped token strings (incl. `##`), debug-only; the consumer reads `label`.
        var tokens: [String]
    }

    /// A pre-entity (one non-special token kept for aggregation).
    private struct PreEntity {
        var scores: [Double]
        var isSubword: Bool
        var token: String
        var start: Int
        var end: Int
    }

    /// Aggregate per-token softmax vectors into per-word entities.
    ///
    /// - Parameters:
    ///   - tokens: token strings from `WordPieceTokenizer.Encoding.tokens` (incl. specials & `##`).
    ///   - offsets: matching `(start, end)` scalar spans (specials `(0, 0)`).
    ///   - specialTokensMask: 1 at specials (skipped), 0 elsewhere.
    ///   - inputIds: token ids (used only for the `[UNK]` special-case).
    ///   - scores: per-token **softmax** probability vectors, one per token (length `n_labels`).
    ///   - sentence: the original sentence as scalars, for the `[UNK]` `word_ref` slice.
    ///   - id2label: label-index → label string (e.g. `{0:"NO_STRESS",1:"PUNCT",2:"STRESS"}`).
    /// - Returns: one `WordEntity` per word, in sentence order.
    static func aggregate(
        tokens: [String],
        offsets: [(Int, Int)],
        specialTokensMask: [Int],
        inputIds: [Int],
        scores: [[Double]],
        sentence: [Unicode.Scalar],
        id2label: [Int: String]
    ) -> [WordEntity] {
        // --- collect_pre_entities ---
        var preEntities: [PreEntity] = []
        for idx in 0..<tokens.count {
            if specialTokensMask[idx] != 0 { continue }  // CLS/SEP skipped entirely
            let (start, end) = offsets[idx]
            var word = tokens[idx]
            var isSubword: Bool

            if inputIds[idx] == WordPieceTokenizer.unkId {
                // UNK special case: word_ref = sentence[start:end]; not a subword.
                let lo = min(start, sentence.count)
                let hi = min(end, sentence.count)
                word = String(String.UnicodeScalarView(sentence[lo..<hi]))
                isSubword = false
            } else {
                // continuing_subword_prefix == "##" (non-empty) → first branch:
                // is_subword = len(word) != len(word_ref), i.e. the token carries a "##".
                // Equivalent and offset-edge-case-proof: the token string starts with "##".
                let wordRefCount = max(0, end - start)
                isSubword = word.unicodeScalars.count != wordRefCount
            }
            preEntities.append(PreEntity(
                scores: scores[idx], isSubword: isSubword, token: word, start: start, end: end
            ))
        }

        // --- aggregate_words: group leading + its following subwords ---
        var entities: [WordEntity] = []
        var group: [PreEntity] = []
        func flush() {
            guard !group.isEmpty else { return }
            entities.append(WordPieceAggregation.aggregateWord(group, id2label: id2label))
            group = []
        }
        for pe in preEntities {
            if group.isEmpty {
                group.append(pe)
            } else if pe.isSubword {
                group.append(pe)
            } else {
                flush()
                group.append(pe)
            }
        }
        flush()
        return entities
    }

    /// `aggregate_word("AVERAGE")`: element-wise mean of the per-token softmax vectors, then argmax.
    private static func aggregateWord(_ group: [PreEntity], id2label: [Int: String]) -> WordEntity {
        let nLabels = group[0].scores.count
        var sums = [Double](repeating: 0, count: nLabels)
        for pe in group {
            for j in 0..<nLabels { sums[j] += pe.scores[j] }
        }
        let denom = Double(group.count)
        var avg = [Double](repeating: 0, count: nLabels)
        for j in 0..<nLabels { avg[j] = sums[j] / denom }

        var best = 0
        for j in 1..<nLabels where avg[j] > avg[best] { best = j }

        return WordEntity(
            labelIndex: best,
            label: id2label[best] ?? String(best),
            score: avg[best],
            start: group.first!.start,
            end: group.last!.end,
            tokens: group.map(\.token)
        )
    }

    // MARK: - Label maps (from each model's config.json)

    /// M3 stress-usage labels.
    static let m3Labels: [Int: String] = [0: "NO_STRESS", 1: "PUNCT", 2: "STRESS"]
    /// M4 yo labels.
    static let m4Labels: [Int: String] = [0: "NO_YO", 1: "PUNCT", 2: "YO"]
}
