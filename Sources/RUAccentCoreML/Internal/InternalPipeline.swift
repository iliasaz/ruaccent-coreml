import Foundation

/// The internal orchestrator — a byte-exact port of `RUAccent.process_all_internal`
/// (`ruaccent-src/ruaccent/ruaccent.py:234`), producing the upstream-parity **`"+"`-before-vowel**
/// form. The public `RUAccent.stress(_:notation:)` wraps this with manual-stress protection and
/// notation conversion (see `RUAccent.swift`).
///
/// Data flow (each step cites the upstream line):
/// ```
/// text = normalize(text)                                    # §1
/// sentences = split_by_sentences(text)                      # §2
/// for sentence in sentences:
///     words, remaining = split_by_words(sentence)           # §3
///     if words.isEmpty: emit "".join(remaining); continue   # :240
///     stress_usages = M3.predict_stress_usage(sentence)     # :243  (entity labels)
///     words = _process_yo(words, sentence)                  # :244  (M4 + yo dicts)
///     words = _process_omographs(words)                     # :245  (M2 + omographs dict)
///     words = _process_accent(words, stress_usages)         # :246  (accents dict / M1)
///     emit reassemble(words, remaining)                     # :247–248
/// return "".join(outputs)                                   # :251
/// ```
///
/// **Alignment contract (§6):** `stress_usages[i]`/`yo_predictions[i]` are indexed positionally
/// against `words[i]` **as-is**, replicating upstream's buggy pairing (e.g. the `замок`-after-`?!`
/// miss). We do NOT realign by word text — that would break golden parity.
struct InternalPipeline {

    var dict: DictReader
    let configuration: RUAccent.Configuration

    let stressGate: StressUsageGate
    let yoGate: YoGate
    let omographGate: OmographGate
    let accentGate: AccentGate

    /// `letters_accent = {'о': '+о', 'О': '+О'}` (ruaccent.py:39), layered into `accents` ABOVE the
    /// dict in upstream `load()`. We apply it here as part of the accent lookup (Assembly already
    /// folds `custom_dict` into `dict` via `addAccentOverrides`).
    static let lettersAccent: [String: String] = ["о": "+о", "О": "+О"]

    init(dict: DictReader,
         configuration: RUAccent.Configuration,
         stressGate: StressUsageGate,
         yoGate: YoGate,
         omographGate: OmographGate,
         accentGate: AccentGate) {
        self.dict = dict
        self.configuration = configuration
        self.stressGate = stressGate
        self.yoGate = yoGate
        self.omographGate = omographGate
        self.accentGate = accentGate
        // letters_accent sits ABOVE the accents pack (upstream applies it last in load()).
        self.dict.addAccentOverrides(InternalPipeline.lettersAccent)
    }

    // MARK: - process_all_internal (:234)

    /// Full pipeline producing the upstream `"+"`-before-vowel string.
    ///
    /// - Parameter preserveUserPlus: when true, the normalize pass keeps `"+"` that the caller put
    ///   between word chars (the manual-stress-wins wrapper canonicalizes user marks to this form
    ///   first). The `"+"` then flows through `split_by_words` and `_process_accent` skips the word
    ///   (`if '+' in word: continue`), so the user's stress is preserved verbatim. With false (the
    ///   default, golden-parity path) `"+"` is stripped by normalize exactly like upstream.
    func processAllInternal(_ text: String, preserveUserPlus: Bool = false) throws -> String {
        let normalized = preserveUserPlus
            ? TextPipeline.normalizePreservingPlus(text)
            : TextPipeline.normalize(text)
        let sentences = TextPipeline.splitBySentences(normalized)
        var outputs: [String] = []
        for sentence in sentences {
            let split = TextPipeline.splitByWords(sentence)
            if split.words.isEmpty {
                outputs.append(split.remainingText.joined())   // :241
                continue
            }
            var words = split.words

            // M3 stress-usage gate (entity labels, upstream order).
            let stressUsages = try stressGate.stressUsages(forSentence: sentence)

            words = try processYo(words: words, sentence: sentence)
            words = try processOmographs(words: words)
            words = try processAccent(words: words, stressUsages: stressUsages)

            let processed = TextPipeline.reassembleSentence(
                words: words, remainingText: split.remainingText)
            outputs.append(processed)
        }
        return outputs.joined()
    }

    // MARK: - _process_yo (:152)

    /// `_process_yo(words, sentence)` — restore `ё` via `yo_words`, then (for `yo_predictions[i]=="YO"`)
    /// via `yo_homographs`, re-applying the original word's casing with `fix_capital`. The M4 gate is
    /// only consulted when `'е'` appears in the lowercased sentence (upstream guard, :156). When
    /// `restoreYo == false`, the whole step is skipped (words unchanged).
    func processYo(words inputWords: [String], sentence: String) throws -> [String] {
        guard configuration.restoreYo else { return inputWords }
        var words = inputWords
        let lowerSentence = sentence.lowercased()

        var yoPredictions: [String] = []
        if lowerSentence.unicodeScalars.contains("е") {
            yoPredictions = try yoGate.yoPredictions(forSentence: sentence)
        }

        for i in 0..<words.count {
            let word = words[i]
            let lowerWord = word.lowercased()
            // words[i] = fix_capital(word, yo_words.get(lower_word, word))
            let yoWordForm = dict.yoWord(lowerWord) ?? word
            words[i] = TextPipeline.fixCapital(source: word, target: yoWordForm)
            // if yo_predictions and yo_predictions[i] == "YO": words[i] = fix_capital(word, yo_homographs.get(lower_word, word))
            if !yoPredictions.isEmpty, i < yoPredictions.count, yoPredictions[i] == "YO" {
                let yoHomoForm = dict.yoHomograph(lowerWord) ?? word
                words[i] = TextPipeline.fixCapital(source: word, target: yoHomoForm)
            }
        }
        return words
    }

    // MARK: - _process_omographs (:167)

    /// `_process_omographs(text)` — for each word that is an omographs-dict key, build the marked
    /// `(A, B)` batch exactly as upstream and let M2 pick the variant. When `useHomographModel ==
    /// false`, the step is skipped entirely (words unchanged). Note the dict lookup uses
    /// `omographs.get(word)` — upstream passes the *current* (possibly ё-restored, cased) word, **not**
    /// lowercased; `OmographPack.variants` lowercases internally, matching `.get(word.lower())` only
    /// because the dict keys are lowercase. We replicate `.get(word)` semantics via the façade.
    func processOmographs(words inputWords: [String]) throws -> [String] {
        guard configuration.useHomographModel else { return inputWords }
        var words = inputWords

        // Pass 1: find omographs (in order), collect their variants + positions.
        struct Found { let word: String; let variants: [String]; let position: Int }
        var found: [Found] = []
        for (i, word) in words.enumerated() {
            if let variants = dict.omographVariants(word) {
                found.append(Found(word: word, variants: variants, position: i))
            }
        }
        if found.isEmpty { return words }

        // Build the flattened batch: per omograph, one marked-text copy per variant.
        // hypotheses_batch = flatten(variants); num_hypotheses = [variants.count …].
        var markedTexts: [String] = []
        var hypotheses: [String] = []
        var numHypotheses: [Int] = []
        for o in found {
            // t = words.copy(); t[pos] = ' <w>'+t[pos]+'</w> '; marked = delete_spaces_before_punc(" ".join(t))
            var t = words
            t[o.position] = " <w>" + t[o.position] + "</w> "
            let marked = TextPipeline.deleteSpacesBeforePunc(t.joined(separator: " "))
            for _ in 0..<o.variants.count {
                markedTexts.append(marked)
            }
            hypotheses.append(contentsOf: o.variants)
            numHypotheses.append(o.variants.count)
        }

        let chosen = try omographGate.classify(
            markedTexts: markedTexts, hypotheses: hypotheses, numHypotheses: numHypotheses)
        // Replace each omograph's word with its chosen variant (cls_batch is per-omograph, in order).
        for (idx, o) in found.enumerated() where idx < chosen.count {
            words[o.position] = chosen[idx]
        }
        return words
    }

    // MARK: - _process_accent (:203)

    /// `_process_accent(text, stress_usages)`:
    /// - skip any word already containing `"+"` (`if '+' in word: continue`);
    /// - only act when `stress_usages[i] == "STRESS"` (positional, §6);
    /// - accents-dict hit (`stressed != lower_word`) OR a word with punctuation OR ≤1 vowel → remap
    ///   the dict's `"+"` positions onto the original-cased word (the finditer remap, :214–218);
    /// - else (dict miss, no punctuation, >1 vowel) → M1 `put_accent` (when `useNeuralAccentor`).
    func processAccent(words inputWords: [String], stressUsages: [String]) throws -> [String] {
        var words = inputWords
        for i in 0..<words.count {
            let word = words[i]
            if word.unicodeScalars.contains("+") { continue }            // :206
            // Upstream indexes stress_usages[i] directly; out-of-range would IndexError. We mirror
            // the *effective* behavior parity-safely: a missing label is treated as non-STRESS (skip).
            guard i < stressUsages.count else { continue }
            guard stressUsages[i] == "STRESS" else { continue }           // :208

            let lowerWord = word.lowercased()
            let stressedWord = dict.accent(lowerWord) ?? lowerWord        // accents.get(lower, lower)

            if stressedWord == lowerWord
                && !TextPipeline.hasPunctuation(lowerWord)
                && TextPipeline.countVowels(lowerWord) > 1 {
                // Dict miss, eligible → M1 (when enabled). When the accentor is off, leave unstressed.
                if configuration.useNeuralAccentor {
                    words[i] = try accentGate.putAccent(word)
                }
            } else {
                // Dict hit (or punctuation / ≤1 vowel) → remap the dict "+" onto original case.
                words[i] = InternalPipeline.remapDictPlus(originalWord: word, stressedForm: stressedWord)
            }
        }
        return words
    }

    /// Port of the finditer remap (`_process_accent` :214–218): insert a `"+"` before the original
    /// word's scalar at each scalar position where `stressedForm` carries a `"+"`. Casing is taken
    /// from `originalWord` (the dict value is lowercase). Operates on Unicode scalars (Python `str`).
    ///
    /// Upstream:
    /// ```
    /// match = re.finditer(r'\+', stressed_word)        # positions of '+' in the dict value
    /// word_fixed = list(word)
    /// for j, e in enumerate(list(match)):
    ///     word_fixed = word_fixed[:e.start()+j] + ["+"] + list(word)[e.end()-1:]
    /// "".join(word_fixed)
    /// ```
    /// For a single-char `+` match `e.end()-1 == e.start()`, so each iteration splices a `"+"` before
    /// the original scalar that was at `stressed`-index `e.start()`. `+ j` shifts past the `j` plusses
    /// already inserted. Replicated literally below.
    static func remapDictPlus(originalWord word: String, stressedForm stressed: String) -> String {
        let stressedScalars = Array(stressed.unicodeScalars)
        let wordScalars = Array(word.unicodeScalars)
        // Positions (in `stressed`) of each '+'.
        var plusPositions: [Int] = []
        for (k, sc) in stressedScalars.enumerated() where sc == "+" { plusPositions.append(k) }
        if plusPositions.isEmpty { return word }

        var wordFixed = wordScalars
        for (j, start) in plusPositions.enumerated() {
            // word_fixed[:start+j] + ["+"] + word[start:]   (e.end()-1 == start; uses ORIGINAL word tail)
            let cut = start + j
            let head = cut <= wordFixed.count ? Array(wordFixed[0..<cut]) : wordFixed
            let tailStart = start
            let tail = tailStart <= wordScalars.count ? Array(wordScalars[tailStart...]) : []
            wordFixed = head + [Unicode.Scalar(UInt8(0x2B))] + tail
        }
        var view = String.UnicodeScalarView()
        view.append(contentsOf: wordFixed)
        return String(view)
    }
}
