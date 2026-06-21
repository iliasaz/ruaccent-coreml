import Foundation

/// WordPiece tokenizer for **M3** (stress-usage / Bert) and **M4** (yo-homograph / DistilBert).
///
/// Byte-exact Swift port of the `tokenizers` fast pipeline encoded in each model's
/// `tokenizer.json`: `BertNormalizer` → `BertPreTokenizer` → `WordpieceTokenizer` →
/// `TemplateProcessing`. See `docs/SWIFT_TOKENIZERS.md` §2.
///
/// M3 and M4 share identical normalizer / pre-tokenizer / WordPiece / post-processor settings;
/// they differ only in `vocab` and in whether `token_type_ids` is emitted (M3 yes / M4 no — the
/// caller decides; this type always *computes* the zero vector and the caller drops it for M4).
///
/// All offsets are **Unicode scalar** indices into the input sentence (Python `str` indices),
/// matching the fast tokenizer's char offsets for the length-preserving normalization these
/// inputs undergo.
struct WordPieceTokenizer {

    static let padId = 0
    static let unkId = 1
    static let clsId = 2
    static let sepId = 3

    static let continuingPrefix = "##"
    static let maxInputCharsPerWord = 100

    /// token string → id (from `vocab.txt`, line index = id).
    private let vocab: [String: Int]

    /// Build from a model directory containing `vocab.txt`.
    init(directory: URL) throws {
        try self.init(vocabFile: directory.appendingPathComponent("vocab.txt"))
    }

    /// Build from an explicit `vocab.txt` URL.
    init(vocabFile url: URL) throws {
        let text = try String(contentsOf: url, encoding: .utf8)
        var lines = text.components(separatedBy: "\n")
        if lines.last == "" { lines.removeLast() }
        var v = [String: Int](minimumCapacity: lines.count)
        for (i, line) in lines.enumerated() {
            v[line] = i
        }
        self.vocab = v
    }

    // MARK: - Output

    struct Encoding: Equatable {
        var inputIds: [Int]
        var attentionMask: [Int]
        var tokenTypeIds: [Int]
        /// `(start, end)` scalar spans into the original sentence; specials get `(0, 0)`.
        var offsetMapping: [(Int, Int)]
        var specialTokensMask: [Int]
        /// Token strings (incl. `##` for continuation pieces, `[CLS]`/`[SEP]` for specials).
        var tokens: [String]

        static func == (lhs: Encoding, rhs: Encoding) -> Bool {
            lhs.inputIds == rhs.inputIds
                && lhs.attentionMask == rhs.attentionMask
                && lhs.tokenTypeIds == rhs.tokenTypeIds
                && lhs.specialTokensMask == rhs.specialTokensMask
                && lhs.tokens == rhs.tokens
                && lhs.offsetMapping.count == rhs.offsetMapping.count
                && zip(lhs.offsetMapping, rhs.offsetMapping).allSatisfy { $0 == $1 }
        }
    }

    // MARK: - Pre-token (post-normalization span)

    /// A whitespace/punctuation-delimited chunk and its scalar offset span into the
    /// **normalized** sentence (== original for length-preserving normalization).
    struct PreToken {
        var scalars: [Unicode.Scalar]
        var start: Int  // scalar index into the normalized sentence
        var end: Int
    }

    // MARK: - Encode

    /// Encode a single sentence (batch size 1, no padding). `token_type_ids` is always computed
    /// as zeros; the M4 caller ignores it (DistilBERT ONNX has no such input).
    func encode(_ sentence: String) -> Encoding {
        let normalized = WordPieceTokenizer.normalize(sentence)
        let preTokens = WordPieceTokenizer.preTokenize(normalized)

        var ids: [Int] = [WordPieceTokenizer.clsId]
        var tokens: [String] = ["[CLS]"]
        var offsets: [(Int, Int)] = [(0, 0)]
        var stm: [Int] = [1]

        for pt in preTokens {
            let pieces = wordPiece(pt)
            for piece in pieces {
                ids.append(piece.id)
                tokens.append(piece.token)
                offsets.append((piece.start, piece.end))
                stm.append(0)
            }
        }

        ids.append(WordPieceTokenizer.sepId)
        tokens.append("[SEP]")
        offsets.append((0, 0))
        stm.append(1)

        let n = ids.count
        return Encoding(
            inputIds: ids,
            attentionMask: [Int](repeating: 1, count: n),
            tokenTypeIds: [Int](repeating: 0, count: n),
            offsetMapping: offsets,
            specialTokensMask: stm,
            tokens: tokens
        )
    }

    // MARK: - WordPiece (greedy longest-match)

    private struct Piece {
        var id: Int
        var token: String
        var start: Int  // scalar offset into the normalized sentence
        var end: Int
    }

    /// Greedy longest-match-first WordPiece over one pre-token, carrying scalar offsets.
    /// The `##` prefix is part of the stored token string but is NOT counted in the offsets
    /// (which index the original text, which has no `##`). Mirrors `WordpieceTokenizer.tokenize`.
    private func wordPiece(_ pt: PreToken) -> [Piece] {
        let chars = pt.scalars
        let count = chars.count

        // max_input_chars_per_word (100): the whole pre-token → one [UNK].
        if count > WordPieceTokenizer.maxInputCharsPerWord {
            return [Piece(id: WordPieceTokenizer.unkId, token: "[UNK]", start: pt.start, end: pt.end)]
        }

        var pieces: [Piece] = []
        var start = 0
        var isBad = false
        while start < count {
            var end = count
            var curId: Int? = nil
            var curToken: String? = nil
            while start < end {
                var substr = String(String.UnicodeScalarView(chars[start..<end]))
                if start > 0 { substr = WordPieceTokenizer.continuingPrefix + substr }
                if let id = vocab[substr] {
                    curId = id
                    curToken = substr
                    break
                }
                end -= 1
            }
            if curId == nil {
                isBad = true
                break
            }
            pieces.append(Piece(
                id: curId!,
                token: curToken!,
                start: pt.start + start,
                end: pt.start + end
            ))
            start = end
        }

        if isBad {
            // The entire word is unknown: discard collected pieces, emit one [UNK].
            return [Piece(id: WordPieceTokenizer.unkId, token: "[UNK]", start: pt.start, end: pt.end)]
        }
        return pieces
    }

    // MARK: - Step 1: BertNormalizer

    /// `BertNormalizer(clean_text: true, handle_chinese_chars: true, lowercase: false,
    /// strip_accents: null→false)`. Length-preserving for Cyrillic/Latin/digits (apart from
    /// dropped control chars and CJK space-wrapping); accents are NOT stripped; no lowercasing.
    static func normalize(_ text: String) -> String {
        var out = String.UnicodeScalarView()
        for c in text.unicodeScalars {
            // clean_text: drop null, replacement, and control chars (except \t\n\r → space).
            let v = c.value
            if v == 0x0 || v == 0xFFFD || isControl(c) {
                continue
            }
            if isWhitespace(c) {
                out.append(" ")  // any whitespace → single ASCII space (no run collapsing)
                continue
            }
            // handle_chinese_chars: wrap CJK ideographs with surrounding spaces.
            if isChineseChar(v) {
                out.append(" ")
                out.append(c)
                out.append(" ")
                continue
            }
            // strip_accents = false, lowercase = false → pass through unchanged.
            out.append(c)
        }
        return String(out)
    }

    // MARK: - Step 2: BertPreTokenizer

    /// Whitespace + punctuation split with scalar-offset tracking (= BasicTokenizer splitting).
    /// Whitespace is consumed; every punctuation char becomes its own pre-token; runs of
    /// letters/digits stay glued (WordPiece splits them later).
    static func preTokenize(_ normalized: String) -> [PreToken] {
        let scalars = Array(normalized.unicodeScalars)
        var result: [PreToken] = []
        var i = 0
        let n = scalars.count
        var current: [Unicode.Scalar] = []
        var currentStart = 0

        func flush(_ endIndex: Int) {
            if !current.isEmpty {
                result.append(PreToken(scalars: current, start: currentStart, end: endIndex))
                current = []
            }
        }

        while i < n {
            let c = scalars[i]
            if isWhitespace(c) {
                flush(i)
                i += 1
                continue
            }
            if isPunctuation(c) {
                flush(i)
                result.append(PreToken(scalars: [c], start: i, end: i + 1))
                i += 1
                continue
            }
            if current.isEmpty { currentStart = i }
            current.append(c)
            i += 1
        }
        flush(n)
        return result
    }

    // MARK: - Character predicates (BERT semantics)

    /// BERT `_is_control`: true for Unicode category C* except `\t\n\r` (handled as whitespace).
    static func isControl(_ c: Unicode.Scalar) -> Bool {
        let v = c.value
        if v == 0x09 || v == 0x0A || v == 0x0D { return false }
        // Categories Cc, Cf, Cs, Co, Cn (others). Swift exposes these via Unicode properties.
        switch c.properties.generalCategory {
        case .control, .format, .surrogate, .privateUse, .unassigned:
            return true
        default:
            return false
        }
    }

    /// BERT `_is_whitespace`: `\t \n \r`, ASCII space, and Unicode `Zs` separators (NBSP etc.).
    static func isWhitespace(_ c: Unicode.Scalar) -> Bool {
        let v = c.value
        if v == 0x20 || v == 0x09 || v == 0x0A || v == 0x0D { return true }
        return c.properties.generalCategory == .spaceSeparator
    }

    /// BERT `_is_punctuation`: ASCII ranges 33–47, 58–64, 91–96, 123–126, OR any Unicode P* category.
    static func isPunctuation(_ c: Unicode.Scalar) -> Bool {
        let v = c.value
        if (v >= 33 && v <= 47) || (v >= 58 && v <= 64)
            || (v >= 91 && v <= 96) || (v >= 123 && v <= 126) {
            return true
        }
        switch c.properties.generalCategory {
        case .connectorPunctuation, .dashPunctuation, .openPunctuation, .closePunctuation,
             .initialPunctuation, .finalPunctuation, .otherPunctuation:
            return true
        default:
            return false
        }
    }

    /// Standard BERT CJK ideograph ranges (for `handle_chinese_chars`).
    static func isChineseChar(_ v: UInt32) -> Bool {
        (v >= 0x4E00 && v <= 0x9FFF)
            || (v >= 0x3400 && v <= 0x4DBF)
            || (v >= 0x20000 && v <= 0x2A6DF)
            || (v >= 0x2A700 && v <= 0x2B73F)
            || (v >= 0x2B740 && v <= 0x2B81F)
            || (v >= 0x2B820 && v <= 0x2CEAF)
            || (v >= 0xF900 && v <= 0xFAFF)
            || (v >= 0x2F800 && v <= 0x2FA1F)
    }
}
