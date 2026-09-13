import Testing
import Foundation
@testable import RUAccentCoreML

/// Opt-in network test for the HF download path (mirrors chatterbox's `HubDownloadTests`).
/// Disabled by default — it needs network. Run with:
///
///   RUACCENT_HUB_TEST=1 \
///     swift test --filter ModelRepositoryTests
///
/// It downloads the bundle from `iliasaz/ruaccent-coreml`, loads `RUAccent` from the snapshot
/// directory, and checks a few `fixtures.json` cases — proving the on-device bundle is complete
/// and wired correctly (the exhaustive 52/52 parity is covered by the local-artifact tests).
@Suite struct ModelRepositoryTests {
    static var hubTestEnabled: Bool {
        ProcessInfo.processInfo.environment["RUACCENT_HUB_TEST"] == "1"
    }

    // input -> expected upstream-parity (`+`-before-vowel) rendering, from fixtures.json.
    static let cases: [(String, String)] = [
        ("на двери висит замок", "на двер+и вис+ит зам+ок"),
        ("это мой дом", "+это м+ой д+ом"),
        ("я мою посуду", "я м+ою пос+уду"),
    ]

    @Test(.enabled(if: hubTestEnabled))
    func downloadAndStressMatchesFixtures() async throws {
        let dir = try await ModelRepository.download { p in
            // progress 0...1 — printed so a long first download is visible in the log.
            if p >= 0.999 { print("RUAccent HF download complete") }
        }
        // The snapshot dir must have the modelDirectory layout (coreml/ + dictpack/ + nn/).
        #expect(FileManager.default.fileExists(atPath: dir.appending(path: "dictpack/accents.rapack").path))
        #expect(ModelRepository.incompleteMLPackage(in: dir) == nil, "a downloaded .mlpackage is incomplete")

        let r = try RUAccent(modelDirectory: dir)
        for (input, expected) in Self.cases {
            let out = try r.stress(input, notation: .plusBeforeVowel)
            #expect(out == expected, "‘\(input)’ → ‘\(out)’ (expected ‘\(expected)’)")
        }
    }
}
