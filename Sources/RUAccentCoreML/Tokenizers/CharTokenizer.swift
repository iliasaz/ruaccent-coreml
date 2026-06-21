import Foundation

/// Char-level tokenizer for **M1** (the accent / RoFormer token-classification model).
///
/// A byte-exact Swift port of the tiny custom `PreTrainedTokenizer` in
/// `ruaccent-src/ruaccent/char_tokenizer.py`. See `docs/SWIFT_TOKENIZERS.md` §1.
///
/// Algorithm (exact):
/// 1. lowercase the input word (`do_lower_case: true`);
/// 2. split into **Unicode scalars** — Python `list(str)` enumerates code points, NOT grapheme
///    clusters, so a combining mark like `U+0301` is its own element;
/// 3. map each scalar through the 45-entry vocab; OOV → `[unk]` (id 1);
/// 4. wrap: `[bos]` (2) + ids + `[eos]` (3);
/// 5. `attention_mask` = all 1s; `token_type_ids` = all 0s (M1's ONNX graph requires the
///    zero-filled tensor — see §0 of the spec).
///
/// The vocab is loaded from `vocab.txt` (45 lines, line index = id). Line 11 is the bare
/// combining-acute `U+0301`; the loader keys it as a one-scalar string.
struct CharTokenizer {

    /// Special-token ids (from `special_tokens_map.json`; these are the lowercase-bracket
    /// strings `[pad]/[unk]/[bos]/[eos]`, NOT BERT's uppercase ones).
    static let padId = 0
    static let unkId = 1
    static let bosId = 2
    static let eosId = 3

    /// scalar (1-code-point string) → id. Built from `vocab.txt` line order.
    private let vocab: [String: Int]

    /// Build from a model directory containing `vocab.txt`.
    init(directory: URL) throws {
        try self.init(vocabFile: directory.appendingPathComponent("vocab.txt"))
    }

    /// Build from an explicit `vocab.txt` URL.
    init(vocabFile url: URL) throws {
        let text = try String(contentsOf: url, encoding: .utf8)
        // Python's tokenizer reads `vocab.txt` line by line (`readlines()`), id = 0-based index.
        // The file ends with a trailing newline → a final empty line that is NOT a real token.
        // Splitting on "\n" and dropping a single trailing empty entry mirrors `vocab_size: 45`.
        var lines = text.components(separatedBy: "\n")
        if lines.last == "" { lines.removeLast() }
        var v = [String: Int](minimumCapacity: lines.count)
        for (i, line) in lines.enumerated() {
            v[line] = i
        }
        self.vocab = v
    }

    /// The output of a single encode (one word, batch size 1, no padding).
    struct Encoding: Equatable {
        var inputIds: [Int]
        var attentionMask: [Int]
        var tokenTypeIds: [Int]
    }

    /// Encode one word per `AccentModel.put_accent` (`lower_word = word.lower()` then `tokenizer`).
    func encode(_ word: String) -> Encoding {
        // `text.lower()` then `list(text)` — iterate scalars, not graphemes.
        let lowered = word.lowercased()
        var ids = [CharTokenizer.bosId]
        for scalar in lowered.unicodeScalars {
            // `_convert_token_to_id`: lowercase the token (no-op for a single scalar) then
            // `vocab.get(token, [unk])`.
            let key = String(scalar)
            ids.append(vocab[key] ?? CharTokenizer.unkId)
        }
        ids.append(CharTokenizer.eosId)
        let n = ids.count
        return Encoding(
            inputIds: ids,
            attentionMask: [Int](repeating: 1, count: n),
            tokenTypeIds: [Int](repeating: 0, count: n)
        )
    }
}
