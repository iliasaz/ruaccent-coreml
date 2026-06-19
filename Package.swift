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
    targets: [
        // Pure-Swift on-device Russian stress (accentuation).
        // Phase 3 adds packed dictionaries here via `resources: [.copy("Resources/...")]`.
        // The CoreML models ship in the consumer's model bundle, not inside the package
        // (keep the package small + binary-free; see docs/MIGRATION_PLAN.md).
        .target(name: "RUAccentCoreML"),
        .testTarget(name: "RUAccentCoreMLTests", dependencies: ["RUAccentCoreML"]),
    ]
)
