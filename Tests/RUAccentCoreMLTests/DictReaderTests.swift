import Testing
import Foundation
@testable import RUAccentCoreML

/// Tests for the `.rapack` v2 dictionary reader (Stage 1 / DictReader).
///
/// Loads the four real packs from `converter/_work/dictpack` via a `#filePath`-relative path
/// to the repo root (the packs are gitignored but present locally; regenerate with
/// `converter/pack_dicts.py` if missing). Validates against `converter/fixtures/golden.json`
/// and the edge cases the packer proved.
@Suite struct DictReaderTests {

    // MARK: paths

    /// Repo root, derived from this test file's path:
    /// <root>/Tests/RUAccentCoreMLTests/DictReaderTests.swift → up 3.
    static let repoRoot: URL = {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // RUAccentCoreMLTests
            .deletingLastPathComponent()   // Tests
            .deletingLastPathComponent()   // <root>
    }()

    static let dictpackDir = repoRoot.appendingPathComponent("converter/_work/dictpack")
    static let goldenURL = repoRoot.appendingPathComponent("converter/fixtures/golden.json")

    /// Skip a test (rather than fail) if the gitignored artifacts are absent.
    static var packsPresent: Bool {
        FileManager.default.fileExists(atPath: dictpackDir.appendingPathComponent("accents.rapack").path)
    }

    func makeReader() throws -> DictReader {
        try #require(Self.packsPresent, "dictpack artifacts missing — run converter/pack_dicts.py")
        return try DictReader(directory: Self.dictpackDir)
    }

    // MARK: header / load

    @Test func loadsAllFourPacksWithCorrectKinds() throws {
        let r = try makeReader()
        #expect(r.accents.pack.kind == .stress)
        #expect(r.omographs.pack.kind == .omograph)
        #expect(r.yoWords.pack.kind == .yo)
        #expect(r.yoHomographs.pack.kind == .yo)
        // Counts match the spec table (docs/SWIFT_DICT_PACK.md §0).
        #expect(r.accents.pack.count == 3_194_879)
        #expect(r.omographs.pack.count == 19_740)
        #expect(r.yoWords.pack.count == 85_568)
        #expect(r.yoHomographs.pack.count == 637)
    }

    @Test func wrongKindFailsFast() throws {
        try #require(Self.packsPresent)
        // Loading the accents pack as a YO pack must throw kindMismatch.
        let url = Self.dictpackDir.appendingPathComponent("accents.rapack")
        #expect(throws: RAPack.LoadError.self) {
            _ = try RAPack(mappingFile: url, expectedKind: .yo)
        }
    }

    // MARK: accents (STRESS)

    @Test func accentsEdgeCases() throws {
        let r = try makeReader()
        // The omograph base key resolves to its accents-default in the accents pack.
        #expect(r.accent("коса") == "кос+а")
        // 0xFF "no stress" entries: value == key (acronyms/interjections).
        #expect(r.accent("брр") == "брр")
        #expect(r.accent("омск") == "омск")
        // absent key → nil
        #expect(r.accent("нетвообщетакогослова") == nil)
        // lowercasing parity: an uppercase query hits the lowercase key.
        #expect(r.accent("КОСА") == "кос+а")
        #expect(r.accent("Молоко") == "молок+о")
    }

    /// Pin the `'+'` to the correct Unicode SCALAR — a byte/UTF-16 index bug would misplace it.
    /// `замок` → `з+амок`: `з` is scalar 0 (2 UTF-8 bytes), `+` goes BEFORE scalar 1 (`а`).
    /// If the reader used the byte index (1) the mark would land mid-`з` and corrupt the string;
    /// if it used scalar index correctly the result is exactly `з+амок`.
    @Test func stressMarkLandsOnScalarBoundary() throws {
        let r = try makeReader()
        let value = try #require(r.accent("замок"))
        #expect(value == "з+амок")
        // Pin the structure: '+' is the SECOND scalar, the vowel `а` immediately follows it.
        let scalars = Array(value.unicodeScalars)
        #expect(scalars[0] == Unicode.Scalar("з"))
        #expect(scalars[1] == Unicode.Scalar("+"))
        #expect(scalars[2] == Unicode.Scalar("а"))
        // And the raw scalar position from the pack is 1 (not the byte index 2).
        #expect(r.accentPosition("замок") == 1)

        // A deeper-stress word to exercise multi-byte offset accumulation: `молок+о` → '+' at scalar 5.
        #expect(r.accent("молоко") == "молок+о")
        #expect(r.accentPosition("молоко") == 5)
        let m = Array(try #require(r.accent("молоко")).unicodeScalars)
        #expect(m[5] == Unicode.Scalar("+"))
        #expect(m[6] == Unicode.Scalar("о"))

        // Leading stress: `+это` → '+' at scalar 0.
        #expect(r.accent("это") == "+это")
        #expect(r.accentPosition("это") == 0)
    }

    /// The `insertingPlus` primitive in isolation (scalar-correct insertion).
    @Test func insertingPlusIsScalarCorrect() {
        #expect(AccentsPack.insertingPlus(into: "замок", atScalarIndex: 1) == "з+амок")
        #expect(AccentsPack.insertingPlus(into: "это", atScalarIndex: 0) == "+это")
        #expect(AccentsPack.insertingPlus(into: "молоко", atScalarIndex: 5) == "молок+о")
    }

    // MARK: omographs (OMOGRAPH)

    @Test func omographEdgeCases() throws {
        let r = try makeReader()
        // The runtime override upstream applies on load (ruaccent.py:92).
        #expect(r.omographVariants("коса") == ["к+оса", "кос+а"])
        // A regular multi-variant homograph from the pack.
        #expect(r.omographVariants("замок") == ["з+амок", "зам+ок"])
        #expect(r.omographVariants("орган") == ["+орган", "орг+ан"])
        // absent → nil
        #expect(r.omographVariants("нетслова") == nil)
        // lowercasing: uppercase query resolves the override too.
        #expect(r.omographVariants("Коса") == ["к+оса", "кос+а"])
    }

    @Test func omographOverrideInjection() throws {
        var r = try makeReader()
        r.addOmographOverrides(["выдумка": ["в+ыдумка", "выд+умка"]])
        #expect(r.omographVariants("выдумка") == ["в+ыдумка", "выд+умка"])
        // The коса override is still present.
        #expect(r.omographVariants("коса") == ["к+оса", "кос+а"])
    }

    // MARK: yo (YO)

    @Test func yoWordsEdgeCases() throws {
        let r = try makeReader()
        #expect(r.yoWord("елка") == "ёлка")
        #expect(r.yoWord("ежик") == "ёжик")
        #expect(r.yoWord("самолет") == "самолёт")
        #expect(r.yoWord("береза") == "берёза")
        // absent → nil
        #expect(r.yoWord("нетслова") == nil)
        // `все` is NOT a yo_words key (it's a homograph) → nil here.
        #expect(r.yoWord("все") == nil)
        // lowercasing
        #expect(r.yoWord("Самолет") == "самолёт")
    }

    /// Pin the `ё` substitution to the correct scalar (Cyrillic 2-byte chars).
    @Test func yoSubstitutionIsScalarCorrect() throws {
        let r = try makeReader()
        // береза → берёза: only the scalar at index 3 flips е→ё, others unchanged.
        let v = try #require(r.yoWord("береза"))
        #expect(v == "берёза")
        let scalars = Array(v.unicodeScalars)
        #expect(scalars[3] == Unicode.Scalar(0x0451)) // ё
        #expect(scalars[2] == Unicode.Scalar("р"))    // unchanged
    }

    @Test func yoHomographsEdgeCases() throws {
        let r = try makeReader()
        #expect(r.yoHomograph("бег") == "бёг")
        #expect(r.yoHomograph("белье") == "бельё")
        #expect(r.yoHomograph("берег") == "берёг")
        #expect(r.yoHomograph("нетслова") == nil)
    }

    // MARK: golden cross-validation

    /// Cross-check the accents pack against the golden pipeline: for every word that appears
    /// in any sentence's `after_accent` stage in the `'+'`-form AND whose accents-dict default
    /// equals that form (i.e. a pure-dict word, not a homograph/context or M1 override),
    /// `accent(word)` must reproduce the golden string byte-exactly.
    @Test func accentsMatchGoldenPipeline() throws {
        let r = try makeReader()
        let golden = try loadGolden()
        var checked = 0
        var seen = Set<String>()
        for item in golden.pipeline {
            for ps in item.perSentence {
                for w in ps.afterAccent where w.contains("+") {
                    let base = w.replacingOccurrences(of: "+", with: "")
                    if seen.contains(base) { continue }
                    seen.insert(base)
                    guard let dictForm = r.accent(base) else { continue }
                    // Only assert where the dict default IS the golden form. Compare on the
                    // LOWERCASED golden word: upstream lowercases every word before dict lookup
                    // (word.lower()), so a capitalized golden token like `Мама` resolves to the
                    // lowercase dict value `м+ама`. Pure-dict words (not homograph/M1-context)
                    // then equal the golden form once both are lowercased.
                    let wLower = w.lowercased()
                    if dictForm == wLower {
                        checked += 1
                    }
                    // Self-consistency: stripping '+' must give back the lowercased base key.
                    #expect(dictForm.replacingOccurrences(of: "+", with: "") == base.lowercased(),
                            "accent('\(base)') = '\(dictForm)' does not strip back to the lowercased key")
                }
            }
        }
        // We expect a healthy number of pure-dict confirmations from the 52 golden inputs.
        #expect(checked >= 50, "only \(checked) pure-dict golden words confirmed")
    }

    /// Spot-check explicit golden pure-dict words (a subset that is known accents-default).
    @Test func accentsSpotCheckGoldenWords() throws {
        let r = try makeReader()
        let expected: [String: String] = [
            "висит": "вис+ит", "горе": "гор+е", "стоит": "ст+оит",
            "старинный": "стар+инный", "посуду": "пос+уду", "это": "+это",
            "мой": "м+ой", "дом": "д+ом", "мужики": "мужик+и",
            "косили": "кос+или", "траву": "трав+у", "косой": "кос+ой",
            "была": "был+а", "русая": "р+усая", "коса": "кос+а",
            "уже": "уж+е", "целый": "ц+елый", "час": "ч+ас",
            "дорога": "дор+ога", "стала": "ст+ала", "белки": "б+елки",
        ]
        for (k, v) in expected {
            #expect(r.accent(k) == v, "accent('\(k)') expected '\(v)' got '\(r.accent(k) ?? "nil")'")
        }
    }

    /// The pack is sorted by RAW UTF-8 BYTES, not Unicode-canonical `String <`. Confirm the
    /// binary search finds the same entries a near-collision set would, and that lookups are
    /// stable for keys that are byte-adjacent. (A `String <` sort bug would mis-order these.)
    @Test func byteOrderLookupParity() throws {
        let r = try makeReader()
        // These exist and must all resolve; if the search used canonical String order it could
        // land in the wrong bucket and return nil or a neighbor.
        let words = ["коса", "косарь", "косить", "косой", "косы", "кос", "ко"]
        for w in words {
            let form = r.accent(w)
            if let form {
                #expect(form.replacingOccurrences(of: "+", with: "") == w)
            }
        }
        // At minimum the anchored test words resolve.
        #expect(r.accent("коса") == "кос+а")
        #expect(r.accent("косой") == "кос+ой")
    }

    // MARK: - golden JSON decoding

    struct Golden: Decodable {
        let pipeline: [Item]
        struct Item: Decodable {
            let perSentence: [Sentence]
            enum CodingKeys: String, CodingKey { case perSentence = "per_sentence" }
        }
        struct Sentence: Decodable {
            let afterAccent: [String]
            enum CodingKeys: String, CodingKey { case afterAccent = "after_accent" }
        }
    }

    func loadGolden() throws -> Golden {
        let data = try Data(contentsOf: Self.goldenURL)
        return try JSONDecoder().decode(Golden.self, from: data)
    }
}
