// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "RUAccentCoreML",
    platforms: [
        .macOS(.v15),
        .iOS(.v18),
    ],
    products: [
        .library(name: "RUAccentCoreML", targets: ["RUAccentCoreML"]),
    ],
    dependencies: [
        // On-device download of the model + dictionary + tokenizer bundle from the
        // (private) HF repo `iliasaz/ruaccent-coreml`, mirroring how chatterbox-coreml
        // hosts `iliasaz/chatterbox-turbo-coreml`. Keeps THIS package binary-free
        // (no committed weights/dicts); see docs/MIGRATION_PLAN.md + ModelRepository.swift.
        .package(url: "https://github.com/huggingface/swift-transformers.git", from: "1.3.0"),
    ],
    targets: [
        // Pure-Swift on-device Russian stress (accentuation). The CoreML models, packed
        // `.rapack` dictionaries, and tokenizer files are NOT committed here — they ship in
        // the private HF bundle and download on-device via `ModelRepository` (swift-transformers
        // `Hub`), or are supplied by the consumer via `RUAccent(modelDirectory:)`.
        .target(
            name: "RUAccentCoreML",
            dependencies: [
                .product(name: "Hub", package: "swift-transformers"),
            ]
        ),
        .testTarget(name: "RUAccentCoreMLTests", dependencies: ["RUAccentCoreML"]),
    ]
)
