import Testing
import Foundation
@testable import RUAccentCoreML

/// Stage 3 / TextPipeline parity tests.
///
/// Drives the golden `pipeline` rows (52 inputs) in `converter/fixtures/golden.json`:
/// asserts `normalize(input) == normalized`, the sentence split == `sentences`, and for each
/// sentence `words`/`remaining_text` == the golden per-sentence stages. Also covers the §2.5
/// razdel sentenize behavior corpus and the §4 post-processing helpers.
///
/// ## M3-vs-words ALIGNMENT risk (DOCUMENTED, NOT FIXED — §6)
/// `split_by_words` groups a consecutive punctuation run `[^\w\s]+` into ONE token, but the M3
/// BERT WordPiece tokenizer emits each punctuation char as its own "word" group. So a sentence
/// with a punctuation run of length ≥2 (e.g. `?!`, `...`) yields `len(stress_usages) > len(words)`
/// and every real word after the run is paired with a SHIFTED M3 label — `замок` after `?!` is
/// silently left unstressed. The `" - "`→`" ~ "` swap adds another drift source (M3 sees `-`,
/// `split_by_words` sees `~`). The Swift port REPLICATES this positional pairing as-is for upstream
/// parity; `_process_accent`/`_process_yo` index `stress_usages[i]`/`yo_predictions[i]` against
/// `words[i]` positionally with no re-alignment. Do NOT "fix" it. (See SWIFT_TEXT_PIPELINE.md §6.)
@Suite struct TextPipelineTests {

    // MARK: paths

    /// Repo root: <root>/Tests/RUAccentCoreMLTests/TextPipelineTests.swift → up 3.
    static let repoRoot: URL = {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }()

    static let goldenURL = repoRoot.appendingPathComponent("converter/fixtures/golden.json")

    static let golden: GoldenPipeline = {
        guard let data = try? Data(contentsOf: goldenURL),
              let g = try? JSONDecoder().decode(GoldenPipeline.self, from: data)
        else { return GoldenPipeline() }
        return g
    }()

    // MARK: - Golden-driven: normalize

    @Test func normalizeMatchesGoldenForAllInputs() throws {
        let rows = Self.golden.pipeline
        try #require(!rows.isEmpty, "golden pipeline rows missing")
        var matched = 0
        for row in rows {
            let got = TextPipeline.normalize(row.input)
            #expect(got == row.normalized, "normalize mismatch for input \(row.input.debugDescription): got \(got.debugDescription) expected \(row.normalized.debugDescription)")
            if got == row.normalized { matched += 1 }
        }
        #expect(matched == rows.count, "normalize matched \(matched)/\(rows.count)")
    }

    // MARK: - Golden-driven: sentence split

    @Test func sentenceSplitMatchesGoldenForAllInputs() throws {
        let rows = Self.golden.pipeline
        try #require(!rows.isEmpty)
        var matched = 0
        for row in rows {
            // Upstream sentenizes the NORMALIZED text (process_all_internal:235→236).
            let got = TextPipeline.splitBySentences(row.normalized)
            #expect(got == row.sentences, "sentence split mismatch for \(row.input.debugDescription): got \(got) expected \(row.sentences)")
            if got == row.sentences { matched += 1 }
            // Losslessness: the wrapper reproduces the normalized input exactly.
            #expect(got.joined() == row.normalized, "split_by_sentences not lossless for \(row.input.debugDescription)")
        }
        #expect(matched == rows.count, "sentence split matched \(matched)/\(rows.count)")
    }

    // MARK: - Golden-driven: word split (per sentence)

    @Test func wordSplitMatchesGoldenForAllSentences() throws {
        let rows = Self.golden.pipeline
        try #require(!rows.isEmpty)
        var sentenceCount = 0
        var matched = 0
        for row in rows {
            for ps in row.per_sentence {
                sentenceCount += 1
                let split = TextPipeline.splitByWords(ps.sentence)
                #expect(split.words == ps.words, "words mismatch for sentence \(ps.sentence.debugDescription): got \(split.words) expected \(ps.words)")
                #expect(split.remainingText == ps.remaining_text, "remaining_text mismatch for sentence \(ps.sentence.debugDescription): got \(split.remainingText) expected \(ps.remaining_text)")
                if split.words == ps.words && split.remainingText == ps.remaining_text { matched += 1 }
                // Off-by-one contract.
                if !split.words.isEmpty {
                    #expect(split.remainingText.count == split.words.count + 1,
                            "off-by-one violated for \(ps.sentence.debugDescription)")
                }
            }
        }
        #expect(matched == sentenceCount, "word split matched \(matched)/\(sentenceCount) sentences")
    }

    // MARK: - Reassembly round-trip (identity case)

    /// For the trivial identity case (words unchanged), reassembly reconstructs the sentence
    /// with `" - "`→`" ~ "` restored and spaces-before-punc cleaned — i.e. it round-trips a
    /// sentence whose words contain no `" - "`/leading-spaces-before-punc to the same string.
    @Test func reassembleRoundTripsIdentity() throws {
        let rows = Self.golden.pipeline
        try #require(!rows.isEmpty)
        for row in rows {
            for ps in row.per_sentence {
                let split = TextPipeline.splitByWords(ps.sentence)
                guard !split.words.isEmpty else { continue }
                let reassembled = TextPipeline.reassembleSentence(words: split.words, remainingText: split.remainingText)
                // delete_spaces_before_punc is idempotent on a normalized sentence with single
                // inter-word spaces; for these golden sentences it reproduces the input.
                #expect(reassembled == ps.sentence,
                        "identity reassembly mismatch for \(ps.sentence.debugDescription): got \(reassembled.debugDescription)")
            }
        }
    }

    // MARK: - §1.4 normalize membership table

    @Test func normalizeMembershipTable() {
        // Deleted vs kept, the surprising cases.
        #expect(TextPipeline.normalize("a\"b") == "ab")                  // U+0022 deleted
        #expect(TextPipeline.normalize("a'b") == "a'b")                  // U+0027 kept
        #expect(TextPipeline.normalize("a…b") == "ab")                   // … deleted
        #expect(TextPipeline.normalize("a...b") == "a...b")              // ... kept
        #expect(TextPipeline.normalize("к+оса") == "коса")              // + deleted
        #expect(TextPipeline.normalize("a–b") == "ab")                  // EN DASH deleted
        #expect(TextPipeline.normalize("a−b") == "ab")                  // MINUS SIGN deleted
        #expect(TextPipeline.normalize("a—b") == "a—b")                 // EM DASH kept
        #expect(TextPipeline.normalize("a-b") == "a-b")                 // HYPHEN-MINUS kept
        #expect(TextPipeline.normalize("ма\u{0301}р") == "мар")        // combining acute deleted
        #expect(TextPipeline.normalize("наїв") == "нав")               // Ukr ї silently dropped
        #expect(TextPipeline.normalize("café") == "caf")               // é deleted
        #expect(TextPipeline.normalize("«»„“”") == "«»„“”")             // guillemets/curly doubles kept
        #expect(TextPipeline.normalize("‘’") == "")                     // curly singles deleted
        #expect(TextPipeline.normalize("a\u{00A0}b") == "a\u{00A0}b")  // NBSP survives (\s)
    }

    // MARK: - §2.5 sentenize behavior corpus

    struct SpanTriple: Equatable {
        let start: Int, stop: Int, text: String
        init(_ start: Int, _ stop: Int, _ text: String) { self.start = start; self.stop = stop; self.text = text }
    }

    @Test func sentenizeBehaviorCorpus() {
        func spans(_ s: String) -> [SpanTriple] {
            Sentenize.sentenize(s).map { SpanTriple($0.start, $0.stop, $0.text.asString) }
        }
        // [start, stop) scalar spans + stripped body.
        #expect(spans("Привет мир. Как дела?") == [SpanTriple(0, 11, "Привет мир."), SpanTriple(12, 21, "Как дела?")])
        #expect(spans("Это т.е. пример.") == [SpanTriple(0, 16, "Это т.е. пример.")])
        #expect(spans("В 1996 г. он родился. Потом уехал.") == [SpanTriple(0, 21, "В 1996 г. он родился."), SpanTriple(22, 34, "Потом уехал.")])
        #expect(spans("А.С. Пушкин писал стихи.") == [SpanTriple(0, 24, "А.С. Пушкин писал стихи.")])
        #expect(spans("Цена 5 руб. за штуку. Дорого!") == [SpanTriple(0, 21, "Цена 5 руб. за штуку."), SpanTriple(22, 29, "Дорого!")])
        #expect(spans("Список: 1. Первое. 2. Второе.") == [SpanTriple(0, 10, "Список: 1."), SpanTriple(11, 18, "Первое."), SpanTriple(19, 29, "2. Второе.")])
        #expect(spans("Что?! Невероятно... Да!") == [SpanTriple(0, 5, "Что?!"), SpanTriple(6, 19, "Невероятно..."), SpanTriple(20, 23, "Да!")])
        #expect(spans("Сравни ст. 5 ч. 2 закона.").count == 1)
        #expect(spans("Дж. Р. Р. Толкин — автор.").count == 1)
        #expect(spans("Смайл :) был тут. И тут.") == [SpanTriple(0, 17, "Смайл :) был тут."), SpanTriple(18, 24, "И тут.")])
        #expect(spans("Точка.Без пробела.После.").count == 1)
        #expect(spans("  Пробелы  в  начале.  Конец.  ") == [SpanTriple(2, 21, "Пробелы  в  начале."), SpanTriple(23, 29, "Конец.")])
        #expect(spans("Да. Да. Да.") == [SpanTriple(0, 3, "Да."), SpanTriple(4, 7, "Да."), SpanTriple(8, 11, "Да.")])
        // closing-guillemet split
        #expect(spans("Он сказал: «Привет.» И ушёл.") == [SpanTriple(0, 20, "Он сказал: «Привет.»"), SpanTriple(21, 28, "И ушёл.")])
    }

    @Test func splitBySentencesLosslessAndGapAttachment() {
        // Leading/trailing whitespace preserved; gap attaches to FOLLOWING sentence.
        let s = "  Пробелы  в  начале.  Конец.  "
        let parts = TextPipeline.splitBySentences(s)
        #expect(parts == ["  Пробелы  в  начале.", "  Конец.  "])
        #expect(parts.joined() == s)

        let s2 = "Привет мир. Как дела?"
        let p2 = TextPipeline.splitBySentences(s2)
        #expect(p2 == ["Привет мир.", " Как дела?"])
        #expect(p2.joined() == s2)

        // Empty / single.
        #expect(TextPipeline.splitBySentences("") == [""])
        #expect(TextPipeline.splitBySentences("Один.Два.Три.") == ["Один.Два.Три."])
    }

    // MARK: - §3.4 split_by_words worked examples

    @Test func splitByWordsWorkedExamples() {
        func w(_ s: String) -> [String] { TextPipeline.splitByWords(s).words }
        func r(_ s: String) -> [String] { TextPipeline.splitByWords(s).remainingText }

        #expect(w("Привет мир.") == ["Привет", "мир", "."])
        #expect(r("Привет мир.") == ["", " ", "", ""])

        #expect(w("  Как дела?") == ["Как", "дела", "?"])
        #expect(r("  Как дела?") == ["  ", " ", "", ""])

        #expect(w("что-то и кто-то") == ["что", "-", "то", "и", "кто", "-", "то"])
        #expect(r("что-то и кто-то") == ["", "", "", " ", " ", "", "", ""])

        // " - " -> " ~ " (the ~ is its own token).
        #expect(w("слово - тире") == ["слово", "~", "тире"])
        #expect(r("слово - тире") == ["", " ", " ", ""])

        // manual stress kept whole; trailing bare '+' separate.
        #expect(w("к+оса уже+") == ["к+оса", "уже", "+"])
        #expect(r("к+оса уже+") == ["", " ", "", ""])

        #expect(w("Цена 5 руб.") == ["Цена", "5", "руб", "."])
        #expect(r("Цена 5 руб.") == ["", " ", " ", "", ""])

        #expect(w("Привет, как дела?") == ["Привет", ",", "как", "дела", "?"])
        #expect(r("Привет, как дела?") == ["", "", " ", " ", "", ""])

        // grouped punctuation runs.
        #expect(w("Что?! Невероятно...") == ["Что", "?!", "Невероятно", "..."])
        #expect(r("Что?! Невероятно...") == ["", "", " ", "", ""])

        #expect(w("...") == ["..."])
        #expect(r("...") == ["", ""])

        // empty edge case.
        #expect(w("   ") == [])
        #expect(r("   ") == ["", ""])

        #expect(w("a.b.c") == ["a", ".", "b", ".", "c"])
        #expect(r("a.b.c") == ["", "", "", "", "", ""])
    }

    /// Adversarial `split_by_words` cases (oracle generated from upstream
    /// `TextPreprocessor.split_by_words`): empty/whitespace, bare/grouped `+`, the `" - "` swap
    /// incl. overlapping `" - - "`, leading guillemet, bracketed clause, underscore words.
    @Test func splitByWordsAdversarialOracle() {
        func check(_ s: String, _ words: [String], _ rem: [String]) {
            let split = TextPipeline.splitByWords(s)
            #expect(split.words == words, "words for \(s.debugDescription): got \(split.words)")
            #expect(split.remainingText == rem, "remaining for \(s.debugDescription): got \(split.remainingText)")
        }
        check("", [], ["", ""])
        check("   ", [], ["", ""])
        check("...", ["..."], ["", ""])
        check("a", ["a"], ["", ""])
        check("+", ["+"], ["", ""])
        check("++", ["++"], ["", ""])
        check(" - ", ["~"], [" ", " "])
        check(" - - ", ["~", "-"], [" ", " ", " "])           // non-overlapping replace
        check("«Цитата».", ["«", "Цитата", "»."], ["", "", "", ""])
        check("Текст (в скобках).", ["Текст", "(", "в", "скобках", ")."], ["", " ", "", " ", "", ""])
        check("кто - то", ["кто", "~", "то"], ["", " ", " ", ""])
        check("a+b+c", ["a+b+c"], ["", ""])                   // embedded '+' kept whole
        check("word+", ["word", "+"], ["", "", ""])           // trailing bare '+' separate
        check("_x_", ["_x_"], ["", ""])                       // underscore is \w
        check("5 - 6", ["5", "~", "6"], ["", " ", " ", ""])
    }

    /// Reconstruction contract: interleave gaps + words reproduces the `~`-substituted input.
    @Test func splitByWordsReconstructs() {
        func check(_ s: String) {
            let split = TextPipeline.splitByWords(s)
            guard !split.words.isEmpty else { return }
            var rebuilt = ""
            for i in 0..<split.words.count {
                rebuilt += split.remainingText[i] + split.words[i]
            }
            rebuilt += split.remainingText[split.words.count]
            let swapped = s.replacingOccurrences(of: " - ", with: " ~ ")
            #expect(rebuilt == swapped, "reconstruction mismatch for \(s.debugDescription)")
        }
        for s in ["Привет мир.", "  Как дела?", "что-то и кто-то", "слово - тире",
                  "к+оса уже+", "Цена 5 руб.", "Что?! Невероятно...", "a.b.c"] {
            check(s)
        }
    }

    // MARK: - §4 post-processing helpers

    @Test func deleteSpacesBeforePunc() {
        #expect(TextPipeline.deleteSpacesBeforePunc("слово ,") == "слово,")
        #expect(TextPipeline.deleteSpacesBeforePunc("слово  ,") == "слово ,")   // one space only
        #expect(TextPipeline.deleteSpacesBeforePunc("a - b") == "a-b")          // literal hyphen: both sides
        #expect(TextPipeline.deleteSpacesBeforePunc("a ~ b") == "a - b")        // tilde restored last, spaces kept
        #expect(TextPipeline.deleteSpacesBeforePunc("x  -  y") == "x- y")       // asymmetric collapse
        #expect(TextPipeline.deleteSpacesBeforePunc("нет ~ да") == "нет - да")
    }

    @Test func countVowelsAndHasPunctuation() {
        #expect(TextPipeline.countVowels("замок") == 2)
        #expect(TextPipeline.countVowels("ёжик") == 2)
        #expect(TextPipeline.countVowels("ыкий") == 2)
        #expect(TextPipeline.countVowels("test123") == 0)   // Latin/digits don't count
        #expect(TextPipeline.countVowels("крк") == 0)

        #expect(TextPipeline.hasPunctuation("к+оса") == true)   // '+' counts here
        #expect(TextPipeline.hasPunctuation("сло~во") == true)  // '~' counts here
        #expect(TextPipeline.hasPunctuation("замок") == false)
        #expect(TextPipeline.hasPunctuation("дом.") == true)
    }

    @Test func fixCapital() {
        #expect(TextPipeline.fixCapital(source: "Все", target: "всё") == "Всё")
        #expect(TextPipeline.fixCapital(source: "ВСЕ", target: "всё") == "ВСЁ")
        #expect(TextPipeline.fixCapital(source: "все", target: "ВСЁ") == "всё")
        // length mismatch (scalars) → passthrough.
        #expect(TextPipeline.fixCapital(source: "Их", target: "ихний") == "ихний")
    }

    // MARK: - §6 alignment-divergence corpus (parity: replicate, don't fix)

    /// These sentences expose the M3-vs-words length/order divergence. We assert the
    /// `split_by_words` side (the only deterministic, non-neural part) matches upstream
    /// `split_by_words`, which is the foundation of the buggy positional pairing. The M3
    /// grouping (which produces MORE tokens for punctuation runs) is the neural stage's
    /// concern; here we pin that `split_by_words` does NOT merge the run, so the drift exists.
    @Test func alignmentDivergenceWordSplitParity() {
        func w(_ s: String) -> [String] { TextPipeline.splitByWords(s).words }
        // punctuation run grouped into ONE token by split_by_words (M3 would emit each char).
        #expect(w("Что?! Невероятно...") == ["Что", "?!", "Невероятно", "..."])
        #expect(w("Да!!! Нет???") == ["Да", "!!!", "Нет", "???"])
        #expect(w("Привет... мир") == ["Привет", "...", "мир"])
        #expect(w("Ого?! замок открыт.") == ["Ого", "?!", "замок", "открыт", "."])
        // " - " swap: M3 sees '-' but split_by_words emits '~' (another drift source).
        #expect(w("кто - то идёт") == ["кто", "~", "то", "идёт"])
    }
}

// MARK: - Golden JSON model (pipeline section)

struct GoldenPipeline: Decodable {
    var pipeline: [Row] = []

    struct Row: Decodable {
        let input: String
        let normalized: String
        let sentences: [String]
        let per_sentence: [PerSentence]
        let final: String
    }

    struct PerSentence: Decodable {
        let sentence: String
        let words: [String]
        let remaining_text: [String]
    }
}
