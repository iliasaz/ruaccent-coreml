// swift-tools-version: 6.0
import PackageDescription

// ANEProbe — load each converted .mlpackage and check CPU-vs-NeuralEngine parity + latency
// against the exported oracles. Runs on this Apple Silicon Mac (uses the Mac ANE); the same
// CoreML logic deploys to an iOS app for true iPhone/iPad ANE validation (see README).
let package = Package(
    name: "ANEProbe",
    platforms: [.macOS(.v15)],
    targets: [
        .executableTarget(name: "ANEProbe", path: "Sources/ANEProbe"),
    ]
)
