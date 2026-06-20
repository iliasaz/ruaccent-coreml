import Testing
import Foundation
@testable import RUAccentCoreML

/// Byte-exact tokenizer parity tests (Stage 2 / Tokenizers).
///
/// Each family iterates the golden `tokenizers` rows in `converter/fixtures/golden.json` and
/// asserts `input_ids` / masks / offsets / `special_tokens_mask` (as applicable) match
/// byte-for-byte. Tokenizer source files (`vocab.txt`, `tokenizer.json`, `merges.txt`, …) are
/// loaded from `converter/_work/nn/<model>/` via a `#filePath`-relative path to the repo root
/// (gitignored but present locally).
@Suite struct TokenizerTests {

    // MARK: paths

    /// Repo root: <root>/Tests/RUAccentCoreMLTests/TokenizerTests.swift → up 3.
    static let repoRoot: URL = {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }()

    static let nnDir = repoRoot.appendingPathComponent("converter/_work/nn")
    static let goldenURL = repoRoot.appendingPathComponent("converter/fixtures/golden.json")

    static let m1Dir = nnDir.appendingPathComponent("nn_accent")
    static let m3Dir = nnDir.appendingPathComponent("nn_stress_usage_predictor")
    static let m4Dir = nnDir.appendingPathComponent("nn_yo_homograph_resolver")
    static let m2Dir = nnDir.appendingPathComponent("nn_omograph/turbo3.1")

    static func present(_ dir: URL, _ file: String) -> Bool {
        FileManager.default.fileExists(atPath: dir.appendingPathComponent(file).path)
    }

    // MARK: golden decode

    static let golden: GoldenTok = {
        guard let data = try? Data(contentsOf: goldenURL),
              let g = try? JSONDecoder().decode(GoldenTok.self, from: data)
        else { return GoldenTok() }
        return g
    }()

    // MARK: - M1 CharTokenizer

    @Test func charTokenizerMatchesGolden() throws {
        try #require(Self.present(Self.m1Dir, "vocab.txt"), "M1 vocab.txt missing")
        let tok = try CharTokenizer(directory: Self.m1Dir)
        let rows = Self.golden.tokenizers.M1_char
        try #require(!rows.isEmpty, "golden M1_char rows missing")
        for row in rows {
            let enc = tok.encode(row.word)
            #expect(enc.inputIds == row.input_ids, "M1 input_ids mismatch for \(row.word)")
            #expect(enc.attentionMask == row.attention_mask, "M1 attention_mask mismatch for \(row.word)")
            if let tt = row.token_type_ids {
                #expect(enc.tokenTypeIds == tt, "M1 token_type_ids mismatch for \(row.word)")
            }
        }
    }

    /// Pin the OOV / latin row explicitly: `t`,`s` are latin and not in vocab → `[unk]`=1.
    @Test func charTokenizerOOVLatin() throws {
        try #require(Self.present(Self.m1Dir, "vocab.txt"))
        let tok = try CharTokenizer(directory: Self.m1Dir)
        // From the spec §1.3: "test" -> [2, 1, 10, 1, 1, 3] (t,s -> unk=1; e -> 10).
        #expect(tok.encode("test").inputIds == [2, 1, 10, 1, 1, 3])
        // U+0301 combining acute is its own vocab entry (id 11), reachable via list-of-scalars.
        #expect(tok.encode("\u{0301}").inputIds == [2, 11, 3])
    }

    // MARK: - M3 WordPiece

    @Test func wordPieceM3MatchesGolden() throws {
        try #require(Self.present(Self.m3Dir, "vocab.txt"), "M3 vocab.txt missing")
        let tok = try WordPieceTokenizer(directory: Self.m3Dir)
        let rows = Self.golden.tokenizers.M3_wordpiece
        try #require(!rows.isEmpty, "golden M3 rows missing")
        for row in rows {
            let enc = tok.encode(row.text)
            #expect(enc.inputIds == row.input_ids, "M3 input_ids mismatch: \(row.text)")
            #expect(enc.attentionMask == row.attention_mask, "M3 attention_mask mismatch: \(row.text)")
            #expect(enc.specialTokensMask == row.special_tokens_mask, "M3 stm mismatch: \(row.text)")
            #expect(enc.tokens == row.tokens, "M3 tokens mismatch: \(row.text)")
            let offs = enc.offsetMapping.map { [$0.0, $0.1] }
            #expect(offs == row.offset_mapping, "M3 offsets mismatch: \(row.text)")
            // M3 emits token_type_ids = zeros.
            if let tt = row.token_type_ids {
                #expect(enc.tokenTypeIds == tt, "M3 token_type_ids mismatch: \(row.text)")
            }
        }
    }

    // MARK: - M4 WordPiece

    @Test func wordPieceM4MatchesGolden() throws {
        try #require(Self.present(Self.m4Dir, "vocab.txt"), "M4 vocab.txt missing")
        let tok = try WordPieceTokenizer(directory: Self.m4Dir)
        let rows = Self.golden.tokenizers.M4_wordpiece
        try #require(!rows.isEmpty, "golden M4 rows missing")
        for row in rows {
            let enc = tok.encode(row.text)
            #expect(enc.inputIds == row.input_ids, "M4 input_ids mismatch: \(row.text)")
            #expect(enc.attentionMask == row.attention_mask, "M4 attention_mask mismatch: \(row.text)")
            #expect(enc.specialTokensMask == row.special_tokens_mask, "M4 stm mismatch: \(row.text)")
            #expect(enc.tokens == row.tokens, "M4 tokens mismatch: \(row.text)")
            let offs = enc.offsetMapping.map { [$0.0, $0.1] }
            #expect(offs == row.offset_mapping, "M4 offsets mismatch: \(row.text)")
            // M4 has no token_type_ids in golden (null) — caller zero-fills only if needed.
        }
    }

    // MARK: - WordPiece word-grouping (aggregation segmentation)

    /// The aggregated-entity WORD SEGMENTATION (the `words` list) must match golden for both
    /// M3 and M4. The entity *labels* depend on real model logits (validated later); here we feed
    /// **synthetic one-hot scores derived from the golden entity** so the grouping is exercised
    /// without a model, and assert the per-word segmentation (word count + spans + reconstructed
    /// word from offsets) equals golden.
    @Test func wordPieceAggregationGroupingM3() throws {
        try #require(Self.present(Self.m3Dir, "vocab.txt"))
        let tok = try WordPieceTokenizer(directory: Self.m3Dir)
        try assertGroupingMatches(tok: tok, rows: Self.golden.tokenizers.M3_wordpiece,
                                  labels: WordPieceAggregation.m3Labels)
    }

    @Test func wordPieceAggregationGroupingM4() throws {
        try #require(Self.present(Self.m4Dir, "vocab.txt"))
        let tok = try WordPieceTokenizer(directory: Self.m4Dir)
        try assertGroupingMatches(tok: tok, rows: Self.golden.tokenizers.M4_wordpiece,
                                  labels: WordPieceAggregation.m4Labels)
    }

    /// For each row, build per-token softmax vectors that are one-hot for the golden word's
    /// label (so AVERAGE+argmax reproduces the golden label), aggregate, and assert the number
    /// of words, their `(start,end)` spans, and the word text (sliced from the original sentence
    /// by scalar offsets) match golden.
    private func assertGroupingMatches(
        tok: WordPieceTokenizer, rows: [GoldenTok.WPRow], labels: [Int: String]
    ) throws {
        let label2id = Dictionary(uniqueKeysWithValues: labels.map { ($1, $0) })
        for row in rows {
            let enc = tok.encode(row.text)
            let sentenceScalars = Array(row.text.unicodeScalars)
            // Map each real token to the golden entity whose [start,end] covers its offset; use
            // that entity's label to make a one-hot score vector. Specials get a zero vector.
            var scores = [[Double]]()
            for idx in 0..<enc.tokens.count {
                if enc.specialTokensMask[idx] != 0 {
                    scores.append([0, 0, 0]); continue
                }
                let (s, _) = enc.offsetMapping[idx]
                let ent = row.entities.first { $0.start <= s && s < $0.end }
                let lid = ent.flatMap { label2id[$0.entity] } ?? 0
                var v = [0.0, 0.0, 0.0]
                v[lid] = 1.0
                scores.append(v)
            }
            let words = WordPieceAggregation.aggregate(
                tokens: enc.tokens, offsets: enc.offsetMapping,
                specialTokensMask: enc.specialTokensMask, inputIds: enc.inputIds,
                scores: scores, sentence: sentenceScalars, id2label: labels
            )
            // Word count matches golden entity count.
            #expect(words.count == row.entities.count, "group count mismatch: \(row.text)")
            for (w, e) in zip(words, row.entities) {
                #expect(w.start == e.start && w.end == e.end,
                        "span mismatch for '\(e.word)' in '\(row.text)': got (\(w.start),\(w.end)) want (\(e.start),\(e.end))")
                // Reconstruct the word text from the original sentence by the aggregated span;
                // it must equal the golden entity word.
                let lo = min(w.start, sentenceScalars.count)
                let hi = min(w.end, sentenceScalars.count)
                let recon = String(String.UnicodeScalarView(sentenceScalars[lo..<hi]))
                #expect(recon == e.word, "word text mismatch in '\(row.text)': got '\(recon)' want '\(e.word)'")
                // Label round-trips from the one-hot we fed.
                #expect(w.label == e.entity, "label mismatch for '\(e.word)' in '\(row.text)'")
            }
        }
    }

    // MARK: - M2 ByteLevel-BPE

    @Test func byteLevelBPEM2MatchesGolden() throws {
        try #require(Self.present(Self.m2Dir, "vocab.json"), "M2 vocab.json missing")
        try #require(Self.present(Self.m2Dir, "merges.txt"), "M2 merges.txt missing")
        let tok = try ByteLevelBPETokenizer(directory: Self.m2Dir)
        let rows = Self.golden.tokenizers.M2_bpe
        try #require(!rows.isEmpty, "golden M2 rows missing")
        for row in rows {
            // classify() feeds tokenizer(preprocessed_text, hypothesis) as a pair.
            let enc = tok.encodePair(row.preprocessed_text, row.hypothesis)
            #expect(enc.inputIds == row.input_ids,
                    "M2 input_ids mismatch for word='\(row.word)' hyp='\(row.hypothesis)':\n  got  \(enc.inputIds)\n  want \(row.input_ids)")
            #expect(enc.attentionMask == row.attention_mask,
                    "M2 attention_mask mismatch for hyp='\(row.hypothesis)'")
        }
    }

    /// Pin the GPT-2 byte→unicode table at a few known points (spec §3.4).
    @Test func byteEncoderKnownPoints() {
        let t = ByteLevelBPETokenizer.makeByteEncoder()
        #expect(t[0x20] == Unicode.Scalar(0x0120))  // space → 'Ġ'
        #expect(t[0xD0] == Unicode.Scalar(0x00D0))  // 'Ð'
        #expect(t[0x0A] == Unicode.Scalar(0x010A))  // newline → 'Ċ'
        #expect(t[0x21] == Unicode.Scalar(0x0021))  // '!' maps to itself
    }
}

// MARK: - Golden JSON model (tokenizers section)

struct GoldenTok: Decodable {
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
        let attention_mask: [Int]
        let token_type_ids: [Int]?
    }

    struct WPRow: Decodable {
        let text: String
        let tokens: [String]
        let input_ids: [Int]
        let attention_mask: [Int]
        let offset_mapping: [[Int]]
        let special_tokens_mask: [Int]
        let token_type_ids: [Int]?
        let entities: [Entity]
    }

    struct Entity: Decodable {
        let word: String
        let entity: String
        let start: Int
        let end: Int
    }

    struct M2Row: Decodable {
        let word: String
        let hypothesis: String
        let preprocessed_text: String
        let input_ids: [Int]
        let attention_mask: [Int]
    }
}
