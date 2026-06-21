import Foundation

/// ByteLevel-BPE (Roberta / GPT-2 style) tokenizer for **M2** (omograph / DeBERTa seq-clf).
///
/// Byte-exact Swift port of the fast `RobertaTokenizer` pipeline: added-token extraction →
/// GPT-2 pre-tokenize regex (`add_prefix_space: false`) → GPT-2 byte→unicode mapping → BPE
/// merges → `RobertaProcessing` post-processor. See `docs/SWIFT_TOKENIZERS.md` §3.
///
/// Vocab layout: base BPE `vocab.json` (ids 0..50255, incl. the 5 specials at 0..4) +
/// `added_tokens.json` (ids 50256..60257, the stress-marked word forms + `<w>`/`</w>`). Added
/// tokens — including `<w>`(60256)/`</w>`(60257) and any hypothesis variant present in the added
/// set — are matched as whole tokens **before** byte-level BPE; their literal chars are never
/// byte-BPE'd.
///
/// `classify()` feeds the model `tokenizer(marked_text, hypothesis)` as a pair → the
/// `RobertaProcessing` layout `<s> A </s></s> B </s>` (a **double** `</s>` between A and B).
final class ByteLevelBPETokenizer {

    static let bosId = 0   // <s> = cls
    static let padId = 1   // <pad>
    static let eosId = 2   // </s> = sep
    static let unkId = 3   // <unk>
    static let maskId = 4  // <mask>

    /// Base byte-mapped symbol string → id (from `vocab.json`).
    private let vocab: [String: Int]
    /// Added-token literal string → id (from `added_tokens.json`, incl. `<w>`/`</w>`).
    private let addedTokens: [String: Int]
    /// Added tokens bucketed by **first scalar**, each bucket sorted by scalar length descending,
    /// so greedy longest-match at a position only scans candidates sharing its first scalar.
    private let addedByFirst: [Unicode.Scalar: [(scalars: [Unicode.Scalar], id: Int)]]
    /// BPE pair rank: "A B" merge → 0-based priority (lower = merged first).
    private let bpeRanks: [String: Int]
    /// byte (0..255) → mapped unicode scalar (GPT-2 `bytes_to_unicode`).
    private let byteEncoder: [Unicode.Scalar]

    /// GPT-2 / ByteLevel pre-tokenize regex (`use_regex: true`).
    private let preRegex: NSRegularExpression

    /// Build from the M2 model directory (`vocab.json`, `merges.txt`, `added_tokens.json`).
    convenience init(directory: URL) throws {
        try self.init(
            vocabFile: directory.appendingPathComponent("vocab.json"),
            mergesFile: directory.appendingPathComponent("merges.txt"),
            addedTokensFile: directory.appendingPathComponent("added_tokens.json")
        )
    }

    init(vocabFile: URL, mergesFile: URL, addedTokensFile: URL) throws {
        // vocab.json: { "symbolString": id }
        let vocabData = try Data(contentsOf: vocabFile)
        let vocabRaw = try JSONSerialization.jsonObject(with: vocabData) as? [String: Int] ?? [:]
        self.vocab = vocabRaw

        // added_tokens.json: { "literal": id }
        let addedData = try Data(contentsOf: addedTokensFile)
        let addedRaw = try JSONSerialization.jsonObject(with: addedData) as? [String: Int] ?? [:]
        self.addedTokens = addedRaw
        var byFirst = [Unicode.Scalar: [(scalars: [Unicode.Scalar], id: Int)]]()
        for (literal, id) in addedRaw {
            let scalars = Array(literal.unicodeScalars)
            guard let first = scalars.first else { continue }
            byFirst[first, default: []].append((scalars: scalars, id: id))
        }
        for key in byFirst.keys {
            byFirst[key]!.sort { $0.scalars.count > $1.scalars.count }
        }
        self.addedByFirst = byFirst

        // merges.txt: "#version" header on line 1, then "A B" rules (0-based rank by order).
        let mergesText = try String(contentsOf: mergesFile, encoding: .utf8)
        var ranks = [String: Int]()
        var rank = 0
        for line in mergesText.split(separator: "\n", omittingEmptySubsequences: true) {
            if line.hasPrefix("#") { continue }  // skip "#version: 0.2"
            ranks[String(line)] = rank
            rank += 1
        }
        self.bpeRanks = ranks

        self.byteEncoder = ByteLevelBPETokenizer.makeByteEncoder()

        // GPT-2 regex. NSRegularExpression supports \p{L}, \p{N}, \s, and lookahead.
        let pattern = "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"
        self.preRegex = try NSRegularExpression(pattern: pattern, options: [])
    }

    // MARK: - Output

    struct Encoding: Equatable {
        var inputIds: [Int]
        var attentionMask: [Int]
    }

    // MARK: - GPT-2 byte→unicode table

    /// Build the 256-entry GPT-2 `bytes_to_unicode` map.
    static func makeByteEncoder() -> [Unicode.Scalar] {
        // Printable bytes that map to themselves.
        var visible: [Int] = []
        visible.append(contentsOf: Array(0x21...0x7E))
        visible.append(contentsOf: Array(0xA1...0xAC))
        visible.append(contentsOf: Array(0xAE...0xFF))
        let visibleSet = Set(visible)

        var table = [Unicode.Scalar](repeating: Unicode.Scalar(0), count: 256)
        for b in visible { table[b] = Unicode.Scalar(UInt32(b))! }
        var n = 0
        for b in 0..<256 where !visibleSet.contains(b) {
            table[b] = Unicode.Scalar(UInt32(256 + n))!
            n += 1
        }
        return table
    }

    // MARK: - Single-sequence encode (byte-BPE → ids, no specials)

    /// Encode one text into byte-BPE ids (no `<s>`/`</s>`). Added tokens (incl. `<w>`/`</w>`
    /// and hypothesis variants) are extracted as whole tokens before byte-level BPE.
    func encodeToIds(_ text: String) -> [Int] {
        var ids: [Int] = []
        for span in splitAddedTokens(text) {
            switch span {
            case .added(let id):
                ids.append(id)
            case .text(let s):
                ids.append(contentsOf: byteLevelEncode(s))
            }
        }
        return ids
    }

    /// Encode a single sequence A: `<s> A </s>`.
    func encodeSingle(_ text: String) -> Encoding {
        var ids = [ByteLevelBPETokenizer.bosId]
        ids.append(contentsOf: encodeToIds(text))
        ids.append(ByteLevelBPETokenizer.eosId)
        return Encoding(inputIds: ids, attentionMask: [Int](repeating: 1, count: ids.count))
    }

    /// Encode a pair `(A, B)`: `<s> A </s></s> B </s>` (RobertaProcessing — double `</s>`).
    /// This is exactly what `OmographModel.classify` feeds: `tokenizer(marked_text, hypothesis)`.
    func encodePair(_ a: String, _ b: String) -> Encoding {
        var ids = [ByteLevelBPETokenizer.bosId]
        ids.append(contentsOf: encodeToIds(a))
        ids.append(ByteLevelBPETokenizer.eosId)
        ids.append(ByteLevelBPETokenizer.eosId)
        ids.append(contentsOf: encodeToIds(b))
        ids.append(ByteLevelBPETokenizer.eosId)
        return Encoding(inputIds: ids, attentionMask: [Int](repeating: 1, count: ids.count))
    }

    // MARK: - Added-token extraction

    private enum Span {
        case added(Int)
        case text(String)
    }

    /// Greedily split out added tokens (matched as whole literals, longest-first) from `text`,
    /// leaving residual text spans for byte-level BPE. Mirrors the fast tokenizer's order:
    /// added/special token extraction happens before byte-level pre-tokenization.
    private func splitAddedTokens(_ text: String) -> [Span] {
        let scalars = Array(text.unicodeScalars)
        let n = scalars.count
        var spans: [Span] = []
        var i = 0
        var literalStart = 0

        func flushLiteral(_ upTo: Int) {
            if upTo > literalStart {
                spans.append(.text(String(String.UnicodeScalarView(scalars[literalStart..<upTo]))))
            }
        }

        while i < n {
            var matched = false
            if let bucket = addedByFirst[scalars[i]] {
                for entry in bucket {  // sorted longest-first → greedy longest-match
                    let len = entry.scalars.count
                    if i + len > n { continue }
                    var eq = true
                    for k in 1..<len where scalars[i + k] != entry.scalars[k] {
                        eq = false
                        break
                    }
                    if eq {
                        flushLiteral(i)
                        spans.append(.added(entry.id))
                        i += len
                        literalStart = i
                        matched = true
                        break
                    }
                }
            }
            if !matched { i += 1 }
        }
        flushLiteral(n)
        return spans
    }

    // MARK: - Byte-level pre-tokenize + BPE

    /// For a residual text span: GPT-2 regex split → byte-map each piece → BPE → ids.
    private func byteLevelEncode(_ text: String) -> [Int] {
        var ids: [Int] = []
        let ns = text as NSString
        let full = NSRange(location: 0, length: ns.length)
        preRegex.enumerateMatches(in: text, options: [], range: full) { match, _, _ in
            guard let m = match else { return }
            let piece = ns.substring(with: m.range)
            // Byte-map: UTF-8 bytes → mapped unicode scalars.
            var mapped = String.UnicodeScalarView()
            for byte in Array(piece.utf8) {
                mapped.append(byteEncoder[Int(byte)])
            }
            let symbol = String(mapped)
            for tok in bpe(symbol) {
                // unk_token is null and all 256 byte symbols exist; every symbol resolves.
                ids.append(vocab[tok] ?? ByteLevelBPETokenizer.unkId)
            }
        }
        return ids
    }

    /// Standard GPT-2 BPE over the byte-mapped char string of one regex piece.
    private func bpe(_ token: String) -> [String] {
        var word = token.unicodeScalars.map { String($0) }
        if word.count < 2 { return word }

        while true {
            // Find the adjacent pair with the lowest merge rank.
            var bestRank = Int.max
            var bestIndex = -1
            for i in 0..<(word.count - 1) {
                let pair = word[i] + " " + word[i + 1]
                if let r = bpeRanks[pair], r < bestRank {
                    bestRank = r
                    bestIndex = i
                }
            }
            if bestIndex < 0 { break }  // no mergeable pair remains

            // Merge ALL occurrences of the best pair (GPT-2 merges left-to-right in one pass).
            let first = word[bestIndex]
            let second = word[bestIndex + 1]
            var newWord: [String] = []
            var i = 0
            while i < word.count {
                if i < word.count - 1 && word[i] == first && word[i + 1] == second {
                    newWord.append(first + second)
                    i += 2
                } else {
                    newWord.append(word[i])
                    i += 1
                }
            }
            word = newWord
            if word.count == 1 { break }
        }
        return word
    }
}
