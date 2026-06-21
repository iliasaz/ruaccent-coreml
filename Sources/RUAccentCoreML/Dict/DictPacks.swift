import Foundation

// Typed façades over `RAPack` + the `DictReader` aggregate the Assembly stage consumes.
//
// All lookups take a `word` and lowercase it with `String.lowercased()` (Python `str.lower()`
// parity for the Cyrillic/Latin alphabets in these dicts, including `Ё`→`ё`). The comparison
// key fed to the pack is `word.lowercased().utf8` as raw bytes — see docs/SWIFT_DICT_PACK.md §3.

/// The combining acute used by the public API. NOT used inside the packs: pack values use the
/// upstream `'+'`-before-vowel notation. Notation conversion happens in the Assembly stage.
let kStressPlus: Character = "+"

// MARK: - lowercasing + index helper

/// Find the global entry index for a word, lowercasing then comparing raw UTF-8 bytes.
/// Centralised so every façade lowercases identically.
private func lookupIndex(_ pack: RAPack, _ word: String) -> Int? {
    // `withUTF8` requires a mutable, contiguous string; lowercase first (Python str.lower()).
    var lowered = word.lowercased()
    return lowered.withUTF8 { utf8 in
        pack.index(ofUTF8: utf8)
    }
}

// MARK: - AccentsPack (KIND_STRESS)

/// `accents` lookups. The on-disk payload is a 1-byte **scalar index** where `'+'` goes
/// (`0xFF` = no stress → value == key). Exposes both the raw scalar index and the ready
/// `'+'`-form string (upstream `accents.get(word.lower())`).
struct AccentsPack {
    let pack: RAPack

    init(mappingFile url: URL) throws {
        self.pack = try RAPack(mappingFile: url, expectedKind: .stress)
    }
    init(pack: RAPack) { self.pack = pack }

    /// Scalar index where the stress mark belongs, or nil if the word is absent.
    /// Returns `nil` also is impossible to distinguish here from "present, no stress" — use
    /// `stressedForm` for the upstream `.get()` semantics (which returns the key unchanged
    /// for the 0xFF case). This method returns nil ONLY when the key is absent; a present
    /// "no stress" key returns `nil` index but `stressedForm` returns the key.
    func stressPosition(of word: String) -> Int? {
        guard let i = lookupIndex(pack, word) else { return nil }
        let b = pack.stressByte(at: i)
        if b == RAPack.noStress { return nil }
        return Int(b)
    }

    /// Upstream `accents.get(word.lower()) -> String?`: the `'+'`-before-vowel form, or nil if
    /// the key is absent. For the rare 0xFF ("no stress") entries the value is the key unchanged.
    /// The `'+'` is inserted **before the Unicode scalar at the stored scalar index**.
    func stressedForm(of word: String) -> String? {
        guard let i = lookupIndex(pack, word) else { return nil }
        let key = pack.key(at: i)
        let b = pack.stressByte(at: i)
        if b == RAPack.noStress { return key }
        return AccentsPack.insertingPlus(into: key, atScalarIndex: Int(b))
    }

    /// Insert `'+'` BEFORE the Unicode scalar at `scalarIndex` (upstream notation).
    /// Operates on `String.unicodeScalars` so Cyrillic (2 UTF-8 bytes / 1 UTF-16 unit) is
    /// placed correctly; a byte- or UTF-16-index bug would misplace the mark.
    static func insertingPlus(into key: String, atScalarIndex scalarIndex: Int) -> String {
        let scalars = key.unicodeScalars
        let cut = scalars.index(scalars.startIndex, offsetBy: scalarIndex)
        var out = String.UnicodeScalarView()
        out.append(contentsOf: scalars[..<cut])
        out.append(Unicode.Scalar(UInt8(0x2B))) // '+'
        out.append(contentsOf: scalars[cut...])
        return String(out)
    }
}

// MARK: - YoPack (KIND_YO)

/// `yo_words` / `yo_homographs` lookups. Payload is a list of scalar indices where `е`→`ё`
/// (`Е`→`Ё` likewise, defensively). Upstream `yo_words.get(word.lower()) -> String?`.
struct YoPack {
    let pack: RAPack

    init(mappingFile url: URL) throws {
        self.pack = try RAPack(mappingFile: url, expectedKind: .yo)
    }
    init(pack: RAPack) { self.pack = pack }

    /// The `ё`-restored form, or nil if the word is absent.
    func resolve(_ word: String) -> String? {
        guard let i = lookupIndex(pack, word) else { return nil }
        let key = pack.key(at: i)
        let subs = pack.yoSubstitutions(at: i)
        if subs.isEmpty { return key }
        var scalars = Array(key.unicodeScalars)
        for p in subs {
            // е (U+0435) → ё (U+0451); Е (U+0415) → Ё (U+0401). Mirrors the packer's assert.
            switch scalars[p] {
            case Unicode.Scalar(0x0435): scalars[p] = Unicode.Scalar(0x0451)! // е → ё
            case Unicode.Scalar(0x0415): scalars[p] = Unicode.Scalar(0x0401)! // Е → Ё
            default: break // packer guarantees this never happens; leave untouched defensively
            }
        }
        var view = String.UnicodeScalarView()
        view.append(contentsOf: scalars)
        return String(view)
    }
}

// MARK: - OmographPack (KIND_OMOGRAPH)

/// `omographs` lookups. Payload is the verbatim variant strings. Upstream
/// `omographs.get(word.lower()) -> list[str]?`.
struct OmographPack {
    let pack: RAPack
    /// Runtime overrides layered ABOVE the pack (mirrors upstream `self.omographs.update(...)`).
    /// Consulted first; keys are already lowercase.
    private var overrides: [String: [String]]

    init(mappingFile url: URL, overrides: [String: [String]] = OmographPack.builtinOverrides) throws {
        self.pack = try RAPack(mappingFile: url, expectedKind: .omograph)
        self.overrides = overrides
    }
    init(pack: RAPack, overrides: [String: [String]] = OmographPack.builtinOverrides) {
        self.pack = pack
        self.overrides = overrides
    }

    /// Upstream runtime mutation `ruaccent.py:92`:
    /// `self.omographs.update({"коса": ["к+оса", "кос+а"]})` applied AFTER loading the gz.
    /// (The pack ships this override baked in too, but we replicate it here so the façade
    /// matches upstream `.get()` semantics regardless of the pack build.)
    static let builtinOverrides: [String: [String]] = [
        "коса": ["к+оса", "кос+а"]
    ]

    /// The list of `'+'`-form variants, or nil if the word is absent.
    func variants(of word: String) -> [String]? {
        let lowered = word.lowercased()
        if let o = overrides[lowered] { return o }
        guard let i = lookupIndex(pack, lowered) else { return nil }
        return pack.omographVariants(at: i)
    }

    /// Add/replace a runtime override (e.g. caller-supplied `custom_homographs`).
    mutating func setOverride(_ word: String, _ variants: [String]) {
        overrides[word.lowercased()] = variants
    }
}

// MARK: - DictReader façade

/// The aggregate the Assembly stage consults. Loads all four packs and exposes lookups
/// equivalent to the upstream `RUAccent` instance dicts:
///   - `accents.get(word.lower())`        → `accent(_:)` / `accentPosition(_:)`
///   - `omographs.get(word.lower())`      → `omographVariants(_:)`
///   - `yo_words.get(word.lower())`       → `yoWord(_:)`
///   - `yo_homographs.get(word.lower())`  → `yoHomograph(_:)`
///
/// Layering note: `custom_dict` and `letters_accent` (`{"о":"+о","О":"+О"}`) are applied by the
/// Assembly stage ABOVE this reader (upstream applies them after the dict in `load()`), NOT here.
/// Use `customAccents` (consulted before the accents pack) when the Assembly stage wants to inject
/// those, and `OmographPack.setOverride` for `custom_homographs`.
struct DictReader {
    let accents: AccentsPack
    let yoWords: YoPack
    let yoHomographs: YoPack
    private(set) var omographs: OmographPack

    /// Caller-supplied accent overrides consulted BEFORE the accents pack (mirrors upstream
    /// `accents.update(custom_dict)` / `letters_accent` ordering). Keys are lowercased on insert.
    /// Values are full `'+'`-form strings, e.g. `["о": "+о"]`.
    private var customAccents: [String: String] = [:]

    /// Default on-disk filenames inside a dict-pack directory.
    enum File: String {
        case accents = "accents.rapack"
        case omographs = "omographs.rapack"
        case yoWords = "yo_words.rapack"
        case yoHomographs = "yo_homographs.rapack"
    }

    /// Load all four packs from a directory containing the `.rapack` files.
    init(directory: URL) throws {
        self.accents = try AccentsPack(mappingFile: directory.appendingPathComponent(File.accents.rawValue))
        self.omographs = try OmographPack(mappingFile: directory.appendingPathComponent(File.omographs.rawValue))
        self.yoWords = try YoPack(mappingFile: directory.appendingPathComponent(File.yoWords.rawValue))
        self.yoHomographs = try YoPack(mappingFile: directory.appendingPathComponent(File.yoHomographs.rawValue))
    }

    /// Inject pre-built packs (testing / custom bundling).
    init(accents: AccentsPack, omographs: OmographPack, yoWords: YoPack, yoHomographs: YoPack) {
        self.accents = accents
        self.omographs = omographs
        self.yoWords = yoWords
        self.yoHomographs = yoHomographs
    }

    // MARK: upstream-equivalent lookups

    /// `accents.get(word.lower()) -> String?` — the `'+'`-form value, or nil.
    /// Consults `customAccents` first (the Assembly-layered overrides), then the pack.
    func accent(_ word: String) -> String? {
        if !customAccents.isEmpty, let c = customAccents[word.lowercased()] { return c }
        return accents.stressedForm(of: word)
    }

    /// Raw scalar stress index from the pack (nil if absent or marked no-stress).
    func accentPosition(_ word: String) -> Int? {
        accents.stressPosition(of: word)
    }

    /// `omographs.get(word.lower()) -> [String]?`.
    func omographVariants(_ word: String) -> [String]? {
        omographs.variants(of: word)
    }

    /// `yo_words.get(word.lower()) -> String?`.
    func yoWord(_ word: String) -> String? {
        yoWords.resolve(word)
    }

    /// `yo_homographs.get(word.lower()) -> String?`.
    func yoHomograph(_ word: String) -> String? {
        yoHomographs.resolve(word)
    }

    // MARK: Assembly-layer override injection

    /// Layer caller-supplied accent overrides (custom_dict / letters_accent) ABOVE the pack.
    /// Values are full `'+'`-form strings; keys are lowercased. Mirrors upstream ordering.
    mutating func addAccentOverrides(_ overrides: [String: String]) {
        for (k, v) in overrides { customAccents[k.lowercased()] = v }
    }

    /// Layer caller-supplied homograph overrides (custom_homographs) ABOVE the pack.
    mutating func addOmographOverrides(_ overrides: [String: [String]]) {
        for (k, v) in overrides { omographs.setOverride(k, v) }
    }
}
