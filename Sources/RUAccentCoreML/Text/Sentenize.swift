import Foundation

/// Faithful Swift port of `razdel.sentenize` (the segmenter actually used by RUAccent via
/// `TextPreprocessor.split_by_sentences`; `text_split.py` is dead code). All offsets/lengths
/// are **Unicode scalars** (Python `str` indices). See `docs/SWIFT_TEXT_PIPELINE.md` §2.
///
/// Pipeline: split (regex delimiters → atoms + `SentSplit`s) → segment (JOIN/SPLIT rule loop)
/// → post (`strip`) → `find_substrings` (relocate each chunk in the source).
enum Sentenize {

    /// A located sentence span: scalar offsets `[start, stop)` into the source + the
    /// stripped body. Mirrors razdel `Substring(start, stop, text)`.
    struct Span: Equatable {
        var start: Int
        var stop: Int
        var text: [Unicode.Scalar]
    }

    // MARK: - Constants (punct.py / sentenize.py)

    private static let endings: Set<Unicode.Scalar> = [".", "?", "!", "\u{2026}"]            // .?!…
    private static let dashes: Set<Unicode.Scalar> = ["\u{2011}", "\u{2013}", "\u{2014}", "\u{2212}", "\u{002D}"]  // ‑–—−-
    private static let openQuotes: Set<Unicode.Scalar> = ["\u{00AB}", "\u{201C}", "\u{2018}"]  // «“‘
    private static let closeQuotes: Set<Unicode.Scalar> = ["\u{00BB}", "\u{201D}", "\u{2019}"]  // »”’
    private static let genericQuotes: Set<Unicode.Scalar> = ["\u{0022}", "\u{201E}", "\u{0027}"]  // "„'
    private static var quotes: Set<Unicode.Scalar> { openQuotes.union(closeQuotes).union(genericQuotes) }
    private static let closeBrackets: Set<Unicode.Scalar> = [")", "]", "}"]
    /// `DELIMITERS = ENDINGS + ';' + GENERIC_QUOTES + CLOSE_QUOTES + CLOSE_BRACKETS` (14 chars).
    private static let delimiters: Set<Unicode.Scalar> = [
        ".", "?", "!", "\u{2026}", ";", "\u{0022}", "\u{201E}", "\u{0027}",
        "\u{00BB}", "\u{201D}", "\u{2019}", ")", "]", "}",
    ]
    private static let bulletChars: Set<Unicode.Scalar> = Set("§абвгдеabcdef".unicodeScalars)
    private static let bulletBounds: Set<Unicode.Scalar> = [".", ")"]
    private static let bulletSize = 20
    private static let window = 10

    // MARK: - Splitter

    /// One element of the alternating `[atom, split, atom, split, …, atom]` stream.
    private enum Part {
        case atom([Unicode.Scalar])
        case split(SentSplit)
    }

    /// Smiley matcher: `[=:;]-?[)(]{1,3}` anchored at `pos`. Returns the scalar count of the
    /// match, or 0 if no match. (`SMILES` in punct.py.)
    private static func smileyLength(_ s: [Unicode.Scalar], at pos: Int) -> Int {
        var i = pos
        let n = s.count
        guard i < n else { return 0 }
        // [=:;]
        let head = s[i]
        guard head == "=" || head == ":" || head == ";" else { return 0 }
        i += 1
        // -?
        if i < n, s[i] == "-" { i += 1 }
        // [)(]{1,3}
        var paren = 0
        while i < n, paren < 3, (s[i] == ")" || s[i] == "(") {
            i += 1
            paren += 1
        }
        return paren >= 1 ? (i - pos) : 0
    }

    /// Find the next delimiter match (smiley OR single delimiter char) at or after `pos`.
    /// Returns `(start, stop)` scalar bounds of the match, left-to-right non-overlapping.
    private static func nextDelimiter(_ s: [Unicode.Scalar], from pos: Int) -> (start: Int, stop: Int)? {
        var i = pos
        let n = s.count
        while i < n {
            // Smiley is tried first at each position (regex alternation order).
            let sm = smileyLength(s, at: i)
            if sm > 0 { return (i, i + sm) }
            if delimiters.contains(s[i]) { return (i, i + 1) }
            i += 1
        }
        return nil
    }

    /// `SentSplitter.__call__` (sentenize.py:309). Yields alternating atoms & splits.
    private static func split(_ text: [Unicode.Scalar]) -> [Part] {
        var parts: [Part] = []
        var previous = 0
        var cursor = 0
        let n = text.count
        while let m = nextDelimiter(text, from: cursor) {
            let start = m.start, stop = m.stop
            parts.append(.atom(Array(text[previous..<start])))
            let left = Array(text[Swift.max(0, start - window)..<start])
            let delimiter = Array(text[start..<stop])
            let right = Array(text[stop..<Swift.min(n, stop + window)])
            parts.append(.split(SentSplit(left: left, delimiter: delimiter, right: right)))
            previous = stop
            cursor = stop
        }
        parts.append(.atom(Array(text[previous..<n])))
        return parts
    }

    // MARK: - SentSplit (cached fields, sentenize.py:245)

    /// Carries the delimiter + left/right 10-scalar windows + accumulated buffer, with the
    /// lazily-derived token fields the rules consult.
    private final class SentSplit {
        let left: [Unicode.Scalar]
        let delimiter: [Unicode.Scalar]
        let right: [Unicode.Scalar]
        var buffer: [Unicode.Scalar] = []

        init(left: [Unicode.Scalar], delimiter: [Unicode.Scalar], right: [Unicode.Scalar]) {
            self.left = left
            self.delimiter = delimiter
            self.right = right
        }

        /// `SPACE_PREFIX = ^\s` on `right`.
        var rightSpacePrefix: Bool {
            guard let f = right.first else { return false }
            return ScalarPredicates.isSpace(f)
        }

        /// `SPACE_SUFFIX = \s$` on `left`.
        var leftSpaceSuffix: Bool {
            guard let l = left.last else { return false }
            return ScalarPredicates.isSpace(l)
        }

        /// `FIRST_TOKEN = ^\s*([^\W\d]+|\d+|[^\w\s])` on `right`.
        var rightToken: [Unicode.Scalar]? {
            firstToken(right)
        }

        /// `LAST_TOKEN = ([^\W\d]+|\d+|[^\w\s])\s*$` on `left`.
        var leftToken: [Unicode.Scalar]? {
            lastToken(left)
        }

        /// `PAIR_SOKR = (\w)\s*\.\s*(\w)\s*$` on `left` → `(a, b)`.
        var leftPairSokr: (Unicode.Scalar, Unicode.Scalar)? {
            pairSokr(left)
        }

        /// `WORD = ([^\W\d]+|\d+)` first match anywhere in `right`.
        var rightWord: [Unicode.Scalar]? {
            firstWord(right)
        }

        /// `TOKEN.findall(buffer)`.
        var bufferTokens: [[Unicode.Scalar]] {
            allTokens(buffer)
        }
    }

    // MARK: - Token regex ports (all `re.U`)

    /// Match a single `TOKEN` starting exactly at `i`: a run of letter/underscore (non-digit),
    /// OR a run of digits, OR exactly one non-word non-space scalar. Returns end index or nil.
    private static func tokenMatch(_ s: [Unicode.Scalar], at i: Int) -> Int? {
        let n = s.count
        guard i < n else { return nil }
        let c = s[i]
        if ScalarPredicates.isWordNonDigit(c) {
            var j = i + 1
            while j < n, ScalarPredicates.isWordNonDigit(s[j]) { j += 1 }
            return j
        }
        if ScalarPredicates.isDigit(c) {
            var j = i + 1
            while j < n, ScalarPredicates.isDigit(s[j]) { j += 1 }
            return j
        }
        if !ScalarPredicates.isWord(c) && !ScalarPredicates.isSpace(c) {
            return i + 1   // single non-word non-space char
        }
        return nil
    }

    /// `FIRST_TOKEN`: skip leading whitespace, then first `TOKEN`.
    private static func firstToken(_ s: [Unicode.Scalar]) -> [Unicode.Scalar]? {
        var i = 0
        let n = s.count
        while i < n, ScalarPredicates.isSpace(s[i]) { i += 1 }
        guard let end = tokenMatch(s, at: i) else { return nil }
        return Array(s[i..<end])
    }

    /// `LAST_TOKEN`: last `TOKEN` allowing trailing whitespace. Implemented by scanning all
    /// tokens and taking the last whose match reaches the trailing-whitespace tail.
    private static func lastToken(_ s: [Unicode.Scalar]) -> [Unicode.Scalar]? {
        // Trailing whitespace boundary: `\s*$`.
        var end = s.count
        while end > 0, ScalarPredicates.isSpace(s[end - 1]) { end -= 1 }
        guard end > 0 else { return nil }
        // The last TOKEN is the maximal token ending exactly at `end`.
        // Determine its start by walking back per the TOKEN alternatives.
        let last = s[end - 1]
        if ScalarPredicates.isWordNonDigit(last) {
            var start = end - 1
            while start > 0, ScalarPredicates.isWordNonDigit(s[start - 1]) { start -= 1 }
            return Array(s[start..<end])
        }
        if ScalarPredicates.isDigit(last) {
            var start = end - 1
            while start > 0, ScalarPredicates.isDigit(s[start - 1]) { start -= 1 }
            return Array(s[start..<end])
        }
        // single non-word non-space char
        return [last]
    }

    /// `WORD`: first `([^\W\d]+|\d+)` anywhere.
    private static func firstWord(_ s: [Unicode.Scalar]) -> [Unicode.Scalar]? {
        let n = s.count
        var i = 0
        while i < n {
            let c = s[i]
            if ScalarPredicates.isWordNonDigit(c) {
                var j = i + 1
                while j < n, ScalarPredicates.isWordNonDigit(s[j]) { j += 1 }
                return Array(s[i..<j])
            }
            if ScalarPredicates.isDigit(c) {
                var j = i + 1
                while j < n, ScalarPredicates.isDigit(s[j]) { j += 1 }
                return Array(s[i..<j])
            }
            i += 1
        }
        return nil
    }

    /// `PAIR_SOKR = (\w)\s*\.\s*(\w)\s*$`: trailing `a . b` (with optional internal spaces).
    private static func pairSokr(_ s: [Unicode.Scalar]) -> (Unicode.Scalar, Unicode.Scalar)? {
        var i = s.count
        // trailing \s*
        while i > 0, ScalarPredicates.isSpace(s[i - 1]) { i -= 1 }
        // (\w)
        guard i > 0, ScalarPredicates.isWord(s[i - 1]) else { return nil }
        let b = s[i - 1]
        i -= 1
        // \s*
        while i > 0, ScalarPredicates.isSpace(s[i - 1]) { i -= 1 }
        // \.
        guard i > 0, s[i - 1] == "." else { return nil }
        i -= 1
        // \s*
        while i > 0, ScalarPredicates.isSpace(s[i - 1]) { i -= 1 }
        // (\w)
        guard i > 0, ScalarPredicates.isWord(s[i - 1]) else { return nil }
        let a = s[i - 1]
        return (a, b)
    }

    /// `TOKEN.findall`: all tokens left-to-right (skipping whitespace).
    private static func allTokens(_ s: [Unicode.Scalar]) -> [[Unicode.Scalar]] {
        var out: [[Unicode.Scalar]] = []
        var i = 0
        let n = s.count
        while i < n {
            if ScalarPredicates.isSpace(s[i]) { i += 1; continue }
            if let end = tokenMatch(s, at: i) {
                out.append(Array(s[i..<end]))
                i = end
            } else {
                i += 1   // a \w-but-not-token char (none post-classification) — advance defensively
            }
        }
        return out
    }

    // MARK: - Rule helpers

    /// Python `str.islower()`: ≥1 cased char and no uppercase/titlecase cased char.
    private static func pyIsLower(_ t: [Unicode.Scalar]) -> Bool {
        var sawCased = false
        for c in t {
            if c.properties.isUppercase { return false }
            if c.properties.isLowercase { sawCased = true }
            else if c.properties.isCased { return false }   // titlecase / other cased non-lower
        }
        return sawCased
    }

    private static func isLowerAlpha(_ t: [Unicode.Scalar]) -> Bool {
        // token.isalpha() and token.islower().
        guard !t.isEmpty else { return false }
        return isAlphaToken(t) && pyIsLower(t)
    }

    private static func isDigitToken(_ t: [Unicode.Scalar]) -> Bool {
        guard !t.isEmpty else { return false }
        return t.allSatisfy { ScalarPredicates.isDigit($0) }
    }

    private static func isAlphaToken(_ t: [Unicode.Scalar]) -> Bool {
        guard !t.isEmpty else { return false }
        return t.allSatisfy { $0.properties.isAlphabetic }
    }

    /// `is_sokr(token)`: digit → true; non-alpha (punct) → true; else token.islower().
    private static func isSokr(_ t: [Unicode.Scalar]?) -> Bool {
        guard let t else { return false }   // no right token → not a sokr context
        if isDigitToken(t) { return true }
        if !isAlphaToken(t) { return true } // punct
        return pyIsLower(t)                 // lower alpha
    }

    private static func lowered(_ t: [Unicode.Scalar]) -> [Unicode.Scalar] {
        t.asString.lowercased().scalarArray
    }

    private static let roman: Set<Unicode.Scalar> = Set("IVXML".unicodeScalars)

    /// `ROMAN = ^[IVXML]+$`.
    private static func isRoman(_ t: [Unicode.Scalar]) -> Bool {
        !t.isEmpty && t.allSatisfy { roman.contains($0) }
    }

    private static func isBullet(_ t: [Unicode.Scalar]) -> Bool {
        if isDigitToken(t) { return true }
        if t.count == 1, bulletBounds.contains(t[0]) { return true }   // t in ".)"
        // `token.lower() in BULLET_CHARS`: BULLET_CHARS is single chars, so only a
        // length-1 lowercased token can match.
        let lo = lowered(t)
        if lo.count == 1, bulletChars.contains(lo[0]) { return true }
        if isRoman(t) { return true }
        return false
    }

    /// `close_bound`: split (abstain → default SPLIT) iff left_token ∈ ENDINGS, else JOIN.
    private static func closeBound(_ split: SentSplit) -> Action? {
        if let lt = split.leftToken, lt.count == 1, endings.contains(lt[0]) {
            return nil
        }
        return .join
    }

    // MARK: - Rules (sentenize.py:331)

    private enum Action { case join, split }

    /// Run rules in order (sentenize.py:331); first non-nil decides. nil from all → SPLIT.
    private static func decide(_ split: SentSplit) -> Bool {
        let rules: [(SentSplit) -> Action?] = [
            emptySide, noSpacePrefix, lowerRight, delimiterRight,
            sokrLeft, insidePairSokr, initialsLeft,
            listItem,
            closeQuote, closeBracket,
            dashRight,
        ]
        for rule in rules {
            if let a = rule(split) {
                return a == .join
            }
        }
        return false   // no rule fired → SPLIT
    }

    private static func emptySide(_ s: SentSplit) -> Action? {
        (s.leftToken == nil || s.rightToken == nil) ? .join : nil
    }

    private static func noSpacePrefix(_ s: SentSplit) -> Action? {
        !s.rightSpacePrefix ? .join : nil
    }

    private static func lowerRight(_ s: SentSplit) -> Action? {
        if let rt = s.rightToken, isLowerAlpha(rt) { return .join }
        return nil
    }

    private static func delimiterRight(_ s: SentSplit) -> Action? {
        guard let rt = s.rightToken else { return nil }
        if rt.count == 1, genericQuotes.contains(rt[0]) { return nil }   // abstain
        if rt.count == 1, delimiters.contains(rt[0]) { return .join }
        // SMILE_PREFIX = ^\s* SMILES  on `right`
        var i = 0
        while i < s.right.count, ScalarPredicates.isSpace(s.right[i]) { i += 1 }
        if smileyLength(s.right, at: i) > 0 { return .join }
        return nil
    }

    private static func sokrLeft(_ s: SentSplit) -> Action? {
        guard s.delimiter == ["."] else { return nil }
        let right = s.rightToken
        if let (a, b) = s.leftPairSokr {
            let key = [lowered([a]).asString, lowered([b]).asString]
            if Sokr.headPairSokrs.contains(key) { return .join }
            if Sokr.pairSokrs.contains(key) {
                if isSokr(right) { return .join }
                return nil
            }
        }
        guard let lt = s.leftToken else { return nil }
        let left = lowered(lt).asString
        if Sokr.headSokrs.contains(left) { return .join }
        if Sokr.sokrs.contains(left), isSokr(right) { return .join }
        return nil
    }

    private static func insidePairSokr(_ s: SentSplit) -> Action? {
        guard s.delimiter == ["."] else { return nil }
        guard let lt = s.leftToken, let rt = s.rightToken else { return nil }
        let key = [lowered(lt).asString, lowered(rt).asString]
        return Sokr.pairSokrs.contains(key) ? .join : nil
    }

    private static func initialsLeft(_ s: SentSplit) -> Action? {
        guard s.delimiter == ["."] else { return nil }
        guard let lt = s.leftToken else { return nil }
        // left.isupper() and len(left) == 1
        if lt.count == 1, lt[0].properties.isUppercase { return .join }
        if Sokr.initials.contains(lowered(lt).asString) { return .join }
        return nil
    }

    private static func listItem(_ s: SentSplit) -> Action? {
        guard s.delimiter.count == 1, bulletBounds.contains(s.delimiter[0]) else { return nil }
        if s.buffer.count > bulletSize { return nil }
        let toks = s.bufferTokens
        if toks.allSatisfy({ isBullet($0) }) { return .join }
        return nil
    }

    private static func closeQuote(_ s: SentSplit) -> Action? {
        guard s.delimiter.count == 1 else { return nil }
        let d = s.delimiter[0]
        guard quotes.contains(d) else { return nil }
        if closeQuotes.contains(d) { return closeBound(s) }
        if genericQuotes.contains(d) {
            if !s.leftSpaceSuffix { return closeBound(s) }
            return .join
        }
        return nil
    }

    private static func closeBracket(_ s: SentSplit) -> Action? {
        guard s.delimiter.count == 1, closeBrackets.contains(s.delimiter[0]) else { return nil }
        return closeBound(s)
    }

    private static func dashRight(_ s: SentSplit) -> Action? {
        guard let rt = s.rightToken, rt.count == 1, dashes.contains(rt[0]) else { return nil }
        guard let rw = s.rightWord else { return nil }
        return isLowerAlpha(rw) ? .join : nil
    }

    // MARK: - Segment + post + find_substrings

    /// `Segmenter.segment` (base.py:27): consume the alternating parts, JOIN/SPLIT per rules.
    private static func segment(_ parts: [Part]) -> [[Unicode.Scalar]] {
        var out: [[Unicode.Scalar]] = []
        guard case .atom(let first)? = parts.first else { return out }
        var buffer = first
        var i = 1
        while i < parts.count {
            guard case .split(let sp) = parts[i] else { break }
            // next atom (the right side) — always present (atoms bracket splits).
            i += 1
            let right: [Unicode.Scalar]
            if i < parts.count, case .atom(let a) = parts[i] { right = a } else { right = [] }
            i += 1
            sp.buffer = buffer
            if decide(sp) {
                buffer = buffer + sp.delimiter + right
            } else {
                out.append(buffer + sp.delimiter)
                buffer = right
            }
        }
        out.append(buffer)
        return out
    }

    /// `find_substrings` (substring.py:14): relocate each stripped chunk with a moving offset.
    private static func findSubstrings(_ chunks: [[Unicode.Scalar]], in text: [Unicode.Scalar]) -> [Span] {
        var spans: [Span] = []
        var offset = 0
        for chunk in chunks {
            let start = text.pyFind(chunk, from: offset)
            let s = start < 0 ? offset : start   // defensive; razdel always finds the chunk
            let stop = s + chunk.count
            spans.append(Span(start: s, stop: stop, text: chunk))
            offset = stop
        }
        return spans
    }

    // MARK: - Public entry

    /// `sentenize(text)` → ordered spans. `text` is the **scalar array** of the source.
    static func sentenize(_ text: [Unicode.Scalar]) -> [Span] {
        let parts = split(text)
        let chunks = segment(parts).map { $0.pyStrip() }   // post: strip()
        return findSubstrings(chunks, in: text)
    }

    /// Convenience over a `String`.
    static func sentenize(_ text: String) -> [Span] {
        sentenize(text.scalarArray)
    }
}
