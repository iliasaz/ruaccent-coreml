import Foundation

/// Non-neural text handling for the RUAccent pipeline: normalize, sentence segmentation
/// (via `Sentenize`), word splitting, post-processing, and sentence reassembly. Byte-exact
/// port per `docs/SWIFT_TEXT_PIPELINE.md`. Assembly drives these around the neural stages.
///
/// All string math is on Unicode scalars (Python `str` indices) — §2.4/§7.
enum TextPipeline {

    // MARK: - normalize (§1)

    /// `re.sub(self.normalize, "", text)` — delete every disallowed scalar.
    static func normalize(_ text: String) -> String {
        TextNormalizer.normalize(text)
    }

    /// Like `normalize`, but additionally keeps a `"+"` that sits BETWEEN two word chars (a genuine
    /// word-internal manual-stress mark, the form `split_by_words`'s `\w*(?:\+\w+)*` carries). Used
    /// only by the public manual-stress-wins wrapper (an addition over upstream — upstream's normalize
    /// always strips `"+"`). A standalone or boundary `"+"` (e.g. in `2+2`) is still deleted.
    static func normalizePreservingPlus(_ text: String) -> String {
        let scalars = text.scalarArray
        var out = String.UnicodeScalarView()
        let n = scalars.count
        for i in 0..<n {
            let s = scalars[i]
            if s == "+" {
                let prevIsWord = i > 0 && ScalarPredicates.isWord(scalars[i - 1])
                let nextIsWord = i + 1 < n && ScalarPredicates.isWord(scalars[i + 1])
                if prevIsWord && nextIsWord { out.append(s) }
                // else: drop (matches upstream normalize, which has no '+').
            } else if TextNormalizer.isAllowed(s) {
                out.append(s)
            }
        }
        return String(out)
    }

    // MARK: - split_by_sentences (§2.1)

    /// `TextPreprocessor.split_by_sentences` — the lossless stitching wrapper around razdel
    /// `sentenize`. Returns sentence strings whose concatenation reproduces `string` exactly:
    /// the inter-sentence gap is reattached to the FOLLOWING sentence, and the trailing tail
    /// to the last. (text_preprocessor.py:22.)
    static func splitBySentences(_ string: String) -> [String] {
        let scalars = string.scalarArray
        let spans = Sentenize.sentenize(scalars)
        if spans.isEmpty { return [] }

        var result: [[Unicode.Scalar]] = []
        var prevStop = 0   // sentinel Substring(0,0,"") → prev.stop == 0
        for span in spans {
            if prevStop != span.start {
                let gap = Array(scalars[prevStop..<span.start])
                result.append(gap + span.text)
            } else {
                result.append(span.text)
            }
            prevStop = span.stop
        }
        // Final tail.
        let lastStop = spans[spans.count - 1].stop
        result[result.count - 1] += Array(scalars[lastStop..<scalars.count])
        return result.map { $0.asString }
    }

    // MARK: - split_by_words (§3)

    /// Result of `split_by_words`: word tokens (length n) and inter-word gap buckets
    /// (length n+1, the off-by-one). For the empty-words case `words == []` and
    /// `remainingText == ["", ""]` (§3.7).
    struct WordSplit: Equatable {
        var words: [String]
        var remainingText: [String]
    }

    /// `TextPreprocessor.split_by_words` (text_preprocessor.py:6).
    ///
    /// 1. `" - " → " ~ "` (Python `str.replace`, all non-overlapping, left→right).
    /// 2. tokenize `string.lower()` with `\w*(?:\+\w+)*|[^\w\s]+` — words (incl. embedded `+`)
    ///    and grouped punctuation runs; whitespace is the gap. Slice spans from ORIGINAL case.
    /// 3. drop empty (`\w*`-zero-width) matches, re-bucket gaps so `len(rem) == len(words)+1`.
    static func splitByWords(_ string: String) -> WordSplit {
        let swapped = replaceSpacedHyphen(string.scalarArray)   // " - " → " ~ "
        let lower = swapped.asString.lowercased().scalarArray   // lower for span-finding only

        // NB: lowercasing must be length-preserving for spans to line up with `swapped`.
        // Post-normalize Russian/Latin lowers 1:1 (§3.1). If it ever diverged we'd desync;
        // we operate on the lowercased array for matches but slice `swapped` at the same idx.
        let matches = tokenize(lower)

        // remaining_text = inter-match gaps from ORIGINAL (swapped) string.
        var remainingGaps: [[Unicode.Scalar]] = []
        if matches.count >= 2 {
            for k in 0..<(matches.count - 1) {
                let l = matches[k], r = matches[k + 1]
                remainingGaps.append(Array(swapped[l.end..<r.start]))
            }
        }

        // words sliced from swapped (original case), including empty matches.
        let words = matches.map { Array(swapped[$0.start..<$0.end]) }
        let wordsMask = words.indices.filter { !words[$0].isEmpty }

        let validWords = wordsMask.map { words[$0].asString }

        if wordsMask.isEmpty {
            return WordSplit(words: [], remainingText: ["", ""])
        }

        // remaining_text_res: bucket 0 = gaps before first word; bucket k = gaps strictly
        // between word k-1 and k; last = gaps after last word.
        func joinGaps(_ range: Range<Int>) -> String {
            var acc: [Unicode.Scalar] = []
            for i in range where i >= 0 && i < remainingGaps.count { acc += remainingGaps[i] }
            return acc.asString
        }

        var res: [String] = [joinGaps(0..<wordsMask[0])]
        for k in 0..<(wordsMask.count - 1) {
            let l = wordsMask[k], r = wordsMask[k + 1]
            res.append(joinGaps((l + 1)..<r))
        }
        res.append(joinGaps((wordsMask[wordsMask.count - 1] + 1)..<remainingGaps.count))

        return WordSplit(words: validWords, remainingText: res)
    }

    /// Replace every non-overlapping `" - "` (space, hyphen-minus, space) with `" ~ "`,
    /// scanning left→right (Python `str.replace` semantics).
    private static func replaceSpacedHyphen(_ s: [Unicode.Scalar]) -> [Unicode.Scalar] {
        var out: [Unicode.Scalar] = []
        out.reserveCapacity(s.count)
        var i = 0
        let n = s.count
        while i < n {
            if i + 2 < n, s[i] == " ", s[i + 1] == "-", s[i + 2] == " " {
                out.append(" "); out.append("~"); out.append(" ")
                i += 3
            } else {
                out.append(s[i])
                i += 1
            }
        }
        return out
    }

    private struct Match { var start: Int; var end: Int }

    /// `re.finditer(r"\w*(?:\+\w+)*|[^\w\s]+")` over scalars. The first alternative is tried
    /// at every position and CAN match the empty string (zero-width) at non-word/non-`+`
    /// boundaries — those empty matches are emitted (they become the dropped `""` words).
    private static func tokenize(_ s: [Unicode.Scalar]) -> [Match] {
        var out: [Match] = []
        let n = s.count
        var pos = 0
        while pos <= n {
            // Alternative 1: \w*(?:\+\w+)*
            var i = pos
            while i < n, ScalarPredicates.isWord(s[i]) { i += 1 }   // \w*
            // (?:\+\w+)* — each group: a '+' followed by ≥1 \w
            while i < n, s[i] == "+" {
                var j = i + 1
                var cnt = 0
                while j < n, ScalarPredicates.isWord(s[j]) { j += 1; cnt += 1 }
                if cnt >= 1 { i = j } else { break }   // trailing bare '+' not absorbed
            }
            if i > pos {
                out.append(Match(start: pos, end: i))
                pos = i
                continue
            }
            // Alternative 1 matched empty at `pos`. Try alternative 2: [^\w\s]+
            if pos < n, !ScalarPredicates.isWord(s[pos]), !ScalarPredicates.isSpace(s[pos]) {
                var j = pos
                while j < n, !ScalarPredicates.isWord(s[j]), !ScalarPredicates.isSpace(s[j]) { j += 1 }
                out.append(Match(start: pos, end: j))   // non-empty punctuation run
                pos = j
                continue
            }
            // Both alternatives are zero-width here (whitespace or end). The regex engine
            // still records the empty alternative-1 match, then advances by one.
            out.append(Match(start: pos, end: pos))
            if pos == n { break }
            pos += 1
        }
        return out
    }

    // MARK: - Reassembly (§5)

    /// `process_all_internal` reassembly (line 247): zip gaps with words, then APPEND the
    /// trailing gap (`remaining_text[-1]`) that zip drops, then `delete_spaces_before_punc`.
    ///
    /// `processedWords` has length n; `remainingText` length n+1.
    static func reassembleSentence(words processedWords: [String], remainingText: [String]) -> String {
        var out = ""
        let n = processedWords.count
        for i in 0..<n {
            // zip stops at the shorter list (processedWords); remainingText[i] exists for i<n+1.
            out += remainingText[i] + processedWords[i]
        }
        out += remainingText[remainingText.count - 1]   // the appended trailing gap
        return deleteSpacesBeforePunc(out)
    }

    /// `_process_yo`-style reassembly (ruaccent.py:228): zip WITHOUT the trailing-gap append.
    /// (The ё-only public method drops the trailing gap; kept for fidelity per §5.)
    static func reassembleYoSentence(words processedWords: [String], remainingText: [String]) -> String {
        var out = ""
        let n = Swift.min(processedWords.count, remainingText.count)
        for i in 0..<n {
            out += remainingText[i] + processedWords[i]
        }
        return out
    }

    // MARK: - Post-processing helpers (§4)

    /// `delete_spaces_before_punc` (ruaccent.py:137). Literal per-char loop; single-pass
    /// `replace` (one space removed per occurrence, runs NOT collapsed); `-` additionally
    /// removes a following space; `~`→`-` restore runs LAST (so `" ~ "` → `" - "`, spaces kept).
    static func deleteSpacesBeforePunc(_ text: String) -> String {
        // 30 chars: ASCII punctuation minus '+' and '~'.
        let punc = "!\"#$%&'()*,./:;<=>?@[\\]^_`{|}-"
        var t = text
        for char in punc {
            if char == "-" {
                t = t.replacingOccurrences(of: " -", with: "-")
                t = t.replacingOccurrences(of: "- ", with: "-")
            }
            t = t.replacingOccurrences(of: " \(char)", with: String(char))
        }
        return t.replacingOccurrences(of: "~", with: "-")
    }

    /// `count_vowels` (ruaccent.py:127) — Cyrillic vowels only, both cases.
    static let vowels: Set<Character> = Set("аеёиоуыэюяАЕЁИОУЫЭЮЯ")
    static func countVowels(_ text: String) -> Int {
        text.reduce(0) { $0 + (vowels.contains($1) ? 1 : 0) }
    }

    /// `has_punctuation` (ruaccent.py:131) — full ASCII punctuation block incl. `+` and `~`.
    static let punctuation: Set<Character> = Set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
    static func hasPunctuation(_ text: String) -> Bool {
        text.contains { punctuation.contains($0) }
    }

    /// `fix_capital(source, target)` (text_postprocessor.py:1). Transfer source's per-scalar
    /// capitalization onto target; if scalar lengths differ, return target unchanged.
    static func fixCapital(source: String, target: String) -> String {
        let src = source.scalarArray
        let tgt = target.scalarArray
        guard src.count == tgt.count else { return target }
        var out = String.UnicodeScalarView()
        for (m, t) in zip(src, tgt) {
            if ScalarPredicates.isUpper(m) {
                out.append(contentsOf: String(t).uppercased().unicodeScalars)
            } else {
                out.append(contentsOf: String(t).lowercased().unicodeScalars)
            }
        }
        return String(out)
    }
}
