// swift-tools-version: 6.0
import PackageDescription

// RuaccentProbe — load each converted .mlpackage and check parity + latency across ALL CoreML
// compute units. Named distinctly from chatterbox-coreml's ANEProbe so the two harnesses' logs
// are never conflated. Runs on this Apple Silicon Mac; the same CoreML logic deploys to an iOS app
// for true iPhone/iPad validation (see README).
let package = Package(
    name: "RuaccentProbe",
    platforms: [.macOS(.v15)],
    targets: [
        .executableTarget(name: "RuaccentProbe", path: "Sources/RuaccentProbe"),
    ]
)
