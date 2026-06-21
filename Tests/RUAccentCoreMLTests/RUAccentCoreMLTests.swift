import Testing
import Foundation
import CoreML
@testable import RUAccentCoreML

/// Stage 5 / public API — notation rendering + manual-stress-wins.
///
/// The pure notation conversions need no models; the end-to-end notation + manual-stress cases
/// reuse the real pipeline from `InternalPipelineTests` (built once, shared) and exercise the public
/// `RUAccent.stress` wrapper. If artifacts are missing those tests are skipped.
@Suite struct RUAccentCoreMLTests {

    // MARK: - Notation conversion (pure, no models)

    @Test func combiningAcutePlacesMarkAfterVowel() {
        // Internal "+"-before-vowel -> combining acute U+0301 AFTER the vowel.
        #expect(Notation.render("м+ой", as: .combiningAcute) == "мо\u{0301}й")     // мо́й
        #expect(Notation.render("двер+и", as: .combiningAcute) == "двери\u{0301}") // двери́
        #expect(Notation.render("+это", as: .combiningAcute) == "э\u{0301}то")     // э́то
    }

    @Test func plusBeforeVowelIsVerbatim() {
        // .plusBeforeVowel returns the internal form unchanged (golden-parity).
        #expect(Notation.render("м+ой", as: .plusBeforeVowel) == "м+ой")
        #expect(Notation.render("на двер+и вис+ит зам+ок", as: .plusBeforeVowel) == "на двер+и вис+ит зам+ок")
    }

    @Test func combiningAcutePreservesYo() {
        // ё is restored by the pipeline; notation conversion carries it through untouched.
        #expect(Notation.render("тёплый", as: .combiningAcute) == "тёплый")
        #expect(Notation.render("наш+ёл", as: .combiningAcute) == "нашё\u{0301}л")  // нашё́л
    }

    @Test func combiningAcuteDropsTrailingBarePlus() {
        // A "+" with no following scalar marks nothing → dropped.
        #expect(Notation.render("abc+", as: .combiningAcute) == "abc")
    }

    // MARK: - ManualStress.canonicalize (pure)

    @Test func canonicalizeDetectsPlusBeforeVowel() {
        let (text, mark) = ManualStress.canonicalize("сл+ово")
        #expect(text == "сл+ово")
        #expect(mark)
    }

    @Test func canonicalizeRewritesCombiningAcuteToPlus() {
        // "сло́во" (о + U+0301) -> "сл+ово" with the "+" before the stressed vowel.
        let (text, mark) = ManualStress.canonicalize("сло\u{0301}во")
        #expect(text == "сл+ово")
        #expect(mark)
    }

    @Test func canonicalizeNoMarkLeavesTextUnchanged() {
        let (text, mark) = ManualStress.canonicalize("слово")
        #expect(text == "слово")
        #expect(!mark)
    }

    @Test func canonicalizeIgnoresArithmeticPlus() {
        // "2+2" — the "+" is not before a Cyrillic vowel → not a user mark.
        let (text, mark) = ManualStress.canonicalize("2+2=4")
        #expect(text == "2+2=4")
        #expect(!mark)
    }

    // MARK: - end-to-end via public RUAccent (manual-stress-wins + notation)

    /// Build a public `RUAccent` over the real pipeline (shared from InternalPipelineTests).
    static func makeRUAccent() -> RUAccent? {
        guard let pipeline = InternalPipelineTests.pipeline else { return nil }
        return RUAccent(pipeline: pipeline, configuration: .init())
    }

    @Test func defaultNotationIsCombiningAcute() throws {
        try #require(InternalPipelineTests.artifactsPresent, "artifacts missing — skipping")
        let r = try #require(Self.makeRUAccent())
        // The protocol default overload uses .combiningAcute.
        #expect(try r.stress("мой") == r.stress("мой", notation: .combiningAcute))
        #expect(try r.stress("мой") == "мо\u{0301}й")   // мо́й
    }

    @Test func plusBeforeVowelMatchesGolden() throws {
        try #require(InternalPipelineTests.artifactsPresent, "artifacts missing — skipping")
        let r = try #require(Self.makeRUAccent())
        // The upstream-parity form, for a homograph-in-context sentence.
        #expect(try r.stress("на двери висит замок", notation: .plusBeforeVowel) == "на двер+и вис+ит зам+ок")
    }

    @Test func manualPlusStressWins() throws {
        try #require(InternalPipelineTests.artifactsPresent, "artifacts missing — skipping")
        let r = try #require(Self.makeRUAccent())
        // User wrote "сл+ово": that word's mark is preserved (the word carries '+', so
        // _process_accent skips it). The rest of the sentence is stressed normally.
        let out = try r.stress("сл+ово тут", notation: .plusBeforeVowel)
        let words = out.split(separator: " ").map(String.init)
        #expect(words.first == "сл+ово", "manual '+' on слово not preserved: \(out)")
    }

    @Test func manualCombiningAcuteStressWins() throws {
        try #require(InternalPipelineTests.artifactsPresent, "artifacts missing — skipping")
        let r = try #require(Self.makeRUAccent())
        // User wrote "сло́во" (combining acute). Preserve it, render back to combining acute.
        let out = try r.stress("сло\u{0301}во тут", notation: .combiningAcute)
        let words = out.split(separator: " ").map(String.init)
        #expect(words.first == "сло\u{0301}во", "manual combining acute on слово not preserved: \(out)")
    }

    @Test func manualStressOnHomographOverridesModel() throws {
        try #require(InternalPipelineTests.artifactsPresent, "artifacts missing — skipping")
        let r = try #require(Self.makeRUAccent())
        // "за́мок" (castle) — user forces the castle reading even though door-lock context exists.
        // Without the user mark, "на двери висит замок" yields "зам+ок"; with "за́мок" the user wins.
        let out = try r.stress("на двери висит за\u{0301}мок", notation: .plusBeforeVowel)
        let last = out.split(separator: " ").map(String.init).last
        #expect(last == "з+амок", "user-forced з+амок not preserved: \(out)")
    }
}
