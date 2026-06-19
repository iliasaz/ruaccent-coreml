import Foundation

/// On-device Russian accentuator — the CoreML port of RUAccent.
///
/// **Status: scaffold.** The interface and resolution policy are fixed; the CoreML
/// accentor/homograph models, the packed dictionaries, and the Russian preprocessing
/// are implemented across the phases in `docs/MIGRATION_PLAN.md`. Until then `stress`
/// is a passthrough so the package builds and integrates from day one.
public final class RUAccent: RussianStressing {

    /// Tuning for which stress sources are active.
    public struct Configuration: Sendable {
        /// Use the neural accentor model for out-of-dictionary words (Phase 1).
        public var useNeuralAccentor: Bool
        /// Use the context homograph-disambiguation model for ambiguous words (Phase 2).
        public var useHomographModel: Bool
        /// Restore `ё` from `е` where the dictionary indicates it.
        public var restoreYo: Bool

        public init(useNeuralAccentor: Bool = true,
                    useHomographModel: Bool = true,
                    restoreYo: Bool = true) {
            self.useNeuralAccentor = useNeuralAccentor
            self.useHomographModel = useHomographModel
            self.restoreYo = restoreYo
        }
    }

    private let configuration: Configuration
    // Phase 1/2/3: CoreML accentor + homograph models, and the loaded dictionaries.

    /// - Parameter modelDirectory: folder containing the CoreML `.mlpackage`(s) +
    ///   packed dictionary. `nil` builds a dictionary/passthrough-only instance
    ///   (the OOV/stepping-stone fallback).
    public init(modelDirectory: URL? = nil,
                configuration: Configuration = .init()) async throws {
        self.configuration = configuration
        // TODO(Phase 1/2/3): load accentor + homograph CoreML models and the packed
        // dictionary from `modelDirectory` (or bundled resources). Validate on device
        // via repro/ (ANEProbe-style) before trusting any model output.
    }

    public func stress(_ text: String, notation: StressNotation) throws -> String {
        // TODO(Phase 4/5): manual-mark passthrough → accents dict → homograph(context)
        // → neural(OOV) → ё restoration → emit per `notation`.
        // Scaffold: identity (no marks added).
        return text
    }
}
