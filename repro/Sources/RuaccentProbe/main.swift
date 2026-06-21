import CoreML
import Foundation

// RuaccentProbe: for each converted model, run the exported oracle cases under EVERY CoreML
// compute unit (.cpuOnly, .cpuAndGPU, .cpuAndNeuralEngine, .all) and report, per unit:
//   - argmax/decision vs the onnxruntime oracle (correctness)
//   - max |Δ| vs the .cpuOnly baseline (fp16 / accelerator drift)
//   - average prediction latency
// On this Mac it exercises the Mac GPU + Neural Engine; the same code runs in an iOS app on device.
// Named RuaccentProbe (not ANEProbe) so it never conflates with chatterbox-coreml's ANEProbe logs.

struct OracleCase: Decodable {
    let label: String
    let real_len: Int
    let input_ids: [Int]
    let attention_mask: [Int]
    let token_type_ids: [Int]?
    let expected_argmax: [Int]
}
struct ModelOracle: Decodable {
    let mlpackage: String
    let inputs: [String]
    let seq_fixed: Int?
    let cases: [OracleCase]
}

func die(_ m: String) -> Never { FileHandle.standardError.write(Data((m + "\n").utf8)); exit(1) }

let args = CommandLine.arguments
guard args.count >= 3 else { die("usage: RuaccentProbe <oracles.json> <coreml_dir> [model]") }
let oraclesURL = URL(fileURLWithPath: args[1])
let coremlDir = URL(fileURLWithPath: args[2])
let onlyModel = args.count >= 4 ? args[3] : nil

let UNITS: [(String, MLComputeUnits)] = [
    ("cpu", .cpuOnly), ("gpu", .cpuAndGPU), ("ane", .cpuAndNeuralEngine), ("all", .all),
]
let TIMED_RUNS = 5

let oracles = try JSONDecoder().decode([String: ModelOracle].self, from: Data(contentsOf: oraclesURL))

func multiArray(_ v: [Int]) throws -> MLMultiArray {
    let a = try MLMultiArray(shape: [1, NSNumber(value: v.count)], dataType: .int32)
    for (i, x) in v.enumerated() { a[i] = NSNumber(value: Int32(x)) }
    return a
}

func argmaxes(_ logits: MLMultiArray, realLen: Int) -> [Int] {
    let shape = logits.shape.map { $0.intValue }
    if shape.count == 3 {
        let (S, C) = (shape[1], shape[2])
        var out = [Int]()
        for t in 0..<min(realLen, S) {
            var best = 0; var bestv = -Double.infinity
            for c in 0..<C { let v = logits[[0, NSNumber(value: t), NSNumber(value: c)]].doubleValue; if v > bestv { bestv = v; best = c } }
            out.append(best)
        }
        return out
    } else {
        let C = shape.last!
        var best = 0; var bestv = -Double.infinity
        for c in 0..<C { let v = logits[[0, NSNumber(value: c)]].doubleValue; if v > bestv { bestv = v; best = c } }
        return [best]
    }
}

func maxAbsDiff(_ a: MLMultiArray, _ b: MLMultiArray) -> Double {
    var m = 0.0
    for i in 0..<a.count { m = max(m, abs(a[i].doubleValue - b[i].doubleValue)) }
    return m
}

func load(_ url: URL, _ units: MLComputeUnits) throws -> MLModel {
    let cfg = MLModelConfiguration(); cfg.computeUnits = units
    let compiled = try MLModel.compileModel(at: url)
    return try MLModel(contentsOf: compiled, configuration: cfg)
}

func provider(_ c: OracleCase, _ inputs: [String]) throws -> MLDictionaryFeatureProvider {
    var d: [String: MLMultiArray] = ["input_ids": try multiArray(c.input_ids),
                                     "attention_mask": try multiArray(c.attention_mask)]
    if inputs.contains("token_type_ids") { d["token_type_ids"] = try multiArray(c.token_type_ids ?? Array(repeating: 0, count: c.input_ids.count)) }
    return try MLDictionaryFeatureProvider(dictionary: d)
}

// Returns (logits, avg-latency-ms) over TIMED_RUNS after one warmup.
func predict(_ model: MLModel, _ p: MLFeatureProvider) throws -> (MLMultiArray, Double) {
    _ = try model.prediction(from: p)
    var last: MLMultiArray? = nil
    let t0 = Date()
    for _ in 0..<TIMED_RUNS {
        let out = try model.prediction(from: p)
        last = out.featureValue(for: "logits")?.multiArrayValue
    }
    let ms = Date().timeIntervalSince(t0) * 1000 / Double(TIMED_RUNS)
    guard let logits = last else { die("no 'logits' output") }
    return (logits, ms)
}

print("RuaccentProbe — parity + latency across all CoreML compute units")
print("(units: \(UNITS.map { $0.0 }.joined(separator: ", ")); baseline = cpu; Δ = max|unit−cpu| logits)\n")

var grandOK = true
for (model, oracle) in oracles.sorted(by: { $0.key < $1.key }) {
    if let only = onlyModel, only != model { continue }
    let pkg = coremlDir.appendingPathComponent(oracle.mlpackage)
    guard FileManager.default.fileExists(atPath: pkg.path) else { print("\(model): SKIP (missing \(oracle.mlpackage))\n"); continue }
    print("=== \(model)  [\(oracle.mlpackage)] ===")

    var models: [(String, MLModel)] = []
    do { for (n, u) in UNITS { models.append((n, try load(pkg, u))) } }
    catch { print("  load failed: \(error)\n"); grandOK = false; continue }

    var oracleHits = Dictionary(uniqueKeysWithValues: UNITS.map { ($0.0, 0) })
    var latSum = Dictionary(uniqueKeysWithValues: UNITS.map { ($0.0, 0.0) })
    var modelOK = true
    for c in oracle.cases {
        var line = String(format: "  %-20@", c.label as NSString)
        var baseline: MLMultiArray? = nil
        for (n, m) in models {
            do {
                let p = try provider(c, oracle.inputs)
                let (logits, ms) = try predict(m, p)
                let am = argmaxes(logits, realLen: c.real_len)
                let okOracle = am == c.expected_argmax
                if okOracle { oracleHits[n]! += 1 }
                latSum[n]! += ms
                modelOK = modelOK && okOracle
                if n == "cpu" { baseline = logits }
                let delta = (n == "cpu") ? 0.0 : maxAbsDiff(logits, baseline ?? logits)
                let dStr = (n == "cpu") ? "      " : String(format: "Δ%.0e", delta)
                line += String(format: " | %@:%@ %5.2fms %@", n, okOracle ? "OK " : "FAIL", ms, dStr)
            } catch { line += " | \(n):ERR"; modelOK = false }
        }
        print(line)
    }
    let nC = oracle.cases.count
    let summary = UNITS.map { (n, _) in String(format: "%@ %d/%d %.2fms", n, oracleHits[n]!, nC, latSum[n]! / Double(max(nC, 1))) }.joined(separator: " | ")
    print("  summary(oracle-match, avg-lat): \(summary)")
    print("  -> \(modelOK ? "PASS" : "CHECK")\n")
    grandOK = grandOK && modelOK
}
print(grandOK ? "ALL MODELS: every compute unit matches the oracle ✅" : "Some compute unit diverged from the oracle ⚠️")
