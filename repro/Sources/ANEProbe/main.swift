import CoreML
import Foundation

// ANEProbe: for each converted model, run the exported oracle cases under .cpuOnly and
// .cpuAndNeuralEngine, and report (a) ANE-vs-CPU argmax parity, (b) argmax vs the onnxruntime
// oracle, (c) max |Δ| between CPU and ANE logits, (d) per-prediction latency.
// On this Mac it exercises the Mac Neural Engine; the same code runs in an iOS app on device.

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
guard args.count >= 3 else { die("usage: ANEProbe <oracles.json> <coreml_dir> [model]") }
let oraclesURL = URL(fileURLWithPath: args[1])
let coremlDir = URL(fileURLWithPath: args[2])
let onlyModel = args.count >= 4 ? args[3] : nil

let oracles = try JSONDecoder().decode([String: ModelOracle].self, from: Data(contentsOf: oraclesURL))

func multiArray(_ v: [Int]) throws -> MLMultiArray {
    let a = try MLMultiArray(shape: [1, NSNumber(value: v.count)], dataType: .int32)
    for (i, x) in v.enumerated() { a[i] = NSNumber(value: Int32(x)) }
    return a
}

// argmax over the last dim of a [1, S, C] (token-cls) or [1, C] (binary) logits MultiArray,
// restricted to `realLen` tokens. Returns the predicted class per token.
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

func predict(_ model: MLModel, _ p: MLFeatureProvider) throws -> (MLMultiArray, Double) {
    let t0 = Date()
    let out = try model.prediction(from: p)
    let dt = Date().timeIntervalSince(t0) * 1000
    guard let logits = out.featureValue(for: "logits")?.multiArrayValue else { die("no 'logits' output") }
    return (logits, dt)
}

print("ANEProbe — CPU vs Neural Engine parity + latency (Mac ANE; same code runs on iOS device)\n")
var grandOK = true
for (model, oracle) in oracles.sorted(by: { $0.key < $1.key }) {
    if let only = onlyModel, only != model { continue }
    let pkg = coremlDir.appendingPathComponent(oracle.mlpackage)
    guard FileManager.default.fileExists(atPath: pkg.path) else { print("\(model): SKIP (missing \(oracle.mlpackage))"); continue }
    print("=== \(model)  [\(oracle.mlpackage)] ===")
    let cpu: MLModel, ane: MLModel
    do { cpu = try load(pkg, .cpuOnly); ane = try load(pkg, .cpuAndNeuralEngine) }
    catch { print("  load failed: \(error)"); grandOK = false; continue }
    var modelOK = true
    for c in oracle.cases {
        do {
            let p = try provider(c, oracle.inputs)
            // warm + timed
            _ = try predict(ane, p)
            let (lc, _) = try predict(cpu, p)
            let (la, tA) = try predict(ane, p)
            let amCPU = argmaxes(lc, realLen: c.real_len)
            let amANE = argmaxes(la, realLen: c.real_len)
            let cpuVsExp = amCPU == c.expected_argmax
            let aneVsCPU = amANE == amCPU
            let diff = maxAbsDiff(lc, la)
            modelOK = modelOK && cpuVsExp && aneVsCPU
            print(String(format: "  %-22@ cpu==oracle:%@  ane==cpu:%@  |Δcpu,ane|=%.2e  ane:%.1fms",
                         c.label as NSString, cpuVsExp ? "OK " : "FAIL", aneVsCPU ? "OK " : "FAIL", diff, tA))
        } catch { print("  \(c.label): prediction error \(error)"); modelOK = false }
    }
    print("  -> \(modelOK ? "PASS" : "CHECK")\n")
    grandOK = grandOK && modelOK
}
print(grandOK ? "ALL MODELS: CPU==oracle and ANE==CPU ✅" : "Some checks need attention ⚠️")
