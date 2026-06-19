import Testing
import Foundation
@testable import RUAccentCoreML

@Suite struct RUAccentCoreMLTests {

    @Test func scaffoldBuildsAndPassesThrough() async throws {
        let r = try await RUAccent()                 // dictionary/passthrough-only
        #expect(try r.stress("тест") == "тест")      // scaffold: identity until Phase 4/5
    }

    @Test func defaultNotationIsCombiningAcute() async throws {
        // The protocol's default overload uses .combiningAcute (TTS convention).
        let r = try await RUAccent()
        #expect(try r.stress("мой") == r.stress("мой", notation: .combiningAcute))
    }

    // Phase 5: replace these with parity assertions vs Python RUAccent on the Russian
    // fixture set (мой→мо́й, русский, тёплый, ёжик, йога, homographs with context).
}
