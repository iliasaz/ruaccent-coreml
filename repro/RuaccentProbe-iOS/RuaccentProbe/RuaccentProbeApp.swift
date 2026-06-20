import SwiftUI
import CoreML
import OSLog

// On-device (iPhone/iPad) RuaccentProbe: for each converted model, run the bundled oracle cases
// under all four CoreML compute units and report, via os.Logger (privacy:.public so idevicesyslog
// captures it) + a report file in Documents:
//   - argmax/decision vs the onnxruntime oracle (correctness on ANE)
//   - max |Δ| vs the .cpuOnly baseline (fp16/accelerator drift)
//   - latency
//   - MLComputePlan per-op device placement (how much actually lands on the Neural Engine)
// Named RuaccentProbe so it never conflates with chatterbox-coreml's ANEProbe logs.

let log = Logger(subsystem: "com.iliasaz.ruaccentprobe", category: "probe")

@main
struct RuaccentProbeApp: App {
    var body: some Scene { WindowGroup { ContentView() } }
}

struct ContentView: View {
    @State private var status = "starting…"
    var body: some View {
        VStack(spacing: 14) {
            Text("RuaccentProbe").font(.title2).bold()
            Text(status).font(.footnote).monospaced().multilineTextAlignment(.leading)
            Button("Run again") { Task { status = await runProbe() } }
        }.padding().task { status = await runProbe() }
    }
}

struct OracleCase: Decodable {
    let label: String, real_len: Int, input_ids: [Int], attention_mask: [Int]
    let token_type_ids: [Int]?, expected_argmax: [Int]
}
struct ModelOracle: Decodable {
    let mlpackage: String, inputs: [String], seq_fixed: Int?, cases: [OracleCase]
}

let UNITS: [(String, MLComputeUnits)] = [
    ("cpu", .cpuOnly), ("gpu", .cpuAndGPU), ("ane", .cpuAndNeuralEngine), ("all", .all),
]

func multiArray(_ v: [Int]) -> MLMultiArray {
    let a = try! MLMultiArray(shape: [1, NSNumber(value: v.count)], dataType: .int32)
    for (i, x) in v.enumerated() { a[i] = NSNumber(value: Int32(x)) }
    return a
}
func provider(_ c: OracleCase, _ inputs: [String]) -> MLDictionaryFeatureProvider {
    var d: [String: MLMultiArray] = ["input_ids": multiArray(c.input_ids),
                                     "attention_mask": multiArray(c.attention_mask)]
    if inputs.contains("token_type_ids") { d["token_type_ids"] = multiArray(c.token_type_ids ?? Array(repeating: 0, count: c.input_ids.count)) }
    return try! MLDictionaryFeatureProvider(dictionary: d)
}
func argmaxes(_ logits: MLMultiArray, realLen: Int) -> [Int] {
    let s = logits.shape.map { $0.intValue }
    if s.count == 3 {
        let (S, C) = (s[1], s[2]); var out = [Int]()
        for t in 0..<min(realLen, S) {
            var best = 0; var bv = -Double.infinity
            for c in 0..<C { let v = logits[[0, NSNumber(value: t), NSNumber(value: c)]].doubleValue; if v > bv { bv = v; best = c } }
            out.append(best)
        }
        return out
    }
    let C = s.last!; var best = 0; var bv = -Double.infinity
    for c in 0..<C { let v = logits[[0, NSNumber(value: c)]].doubleValue; if v > bv { bv = v; best = c } }
    return [best]
}
func maxAbsDiff(_ a: MLMultiArray, _ b: MLMultiArray) -> Double {
    var m = 0.0; for i in 0..<a.count { m = max(m, abs(a[i].doubleValue - b[i].doubleValue)) }; return m
}
func findModelc(prefix: String) -> URL? {
    guard let root = Bundle.main.resourceURL else { return nil }
    let en = FileManager.default.enumerator(at: root, includingPropertiesForKeys: nil)
    var hit: URL? = nil
    while let u = en?.nextObject() as? URL {
        if u.pathExtension == "mlmodelc" && u.deletingPathExtension().lastPathComponent.hasPrefix(prefix) { hit = u; break }
    }
    return hit
}

// per-op placement via MLComputePlan (how many ops the planner sends to ANE/GPU/CPU)
func placement(_ url: URL) async -> String {
    let cfg = MLModelConfiguration(); cfg.computeUnits = .all
    do {
        let plan = try await MLComputePlan.load(contentsOf: url, configuration: cfg)
        guard case let .program(program) = plan.modelStructure else { return "n/a" }
        var ane = 0, gpu = 0, cpu = 0, other = 0
        func walk(_ b: MLModelStructure.Program.Block) {
            for op in b.operations {
                if op.operatorName == "const" { for bb in op.blocks { walk(bb) }; continue }
                switch plan.deviceUsage(for: op)?.preferred {
                case .some(.neuralEngine): ane += 1
                case .some(.gpu): gpu += 1
                case .some(.cpu): cpu += 1
                default: other += 1
                }
                for bb in op.blocks { walk(bb) }
            }
        }
        for (_, fn) in program.functions { walk(fn.block) }
        let tot = ane + gpu + cpu + other
        return "ane=\(ane) gpu=\(gpu) cpu=\(cpu) other=\(other) (ane \(tot > 0 ? ane * 100 / tot : 0)%)"
    } catch { return "plan-failed: \(error.localizedDescription)" }
}

@MainActor
func runProbe() async -> String {
    var report = "RuaccentProbe on-device — \(UIDevice.current.name) iOS \(UIDevice.current.systemVersion)\n"
    log.notice("RP: start device=\(UIDevice.current.name, privacy: .public) ios=\(UIDevice.current.systemVersion, privacy: .public)")
    guard let oURL = Bundle.main.url(forResource: "oracles", withExtension: "json"),
          let oracles = try? JSONDecoder().decode([String: ModelOracle].self, from: Data(contentsOf: oURL)) else {
        log.error("RP: oracles.json missing"); return "oracles.json missing"
    }
    var grandOK = true
    for (model, oracle) in oracles.sorted(by: { $0.key < $1.key }) {
        guard let url = findModelc(prefix: model) else { log.error("RP: \(model, privacy: .public) mlmodelc missing"); report += "\(model): MISSING\n"; grandOK = false; continue }
        log.notice("RP: model=\(model, privacy: .public) loading \(url.lastPathComponent, privacy: .public)")
        let place = await placement(url)        // async (and first cold-compile on ANE)
        log.notice("RP: model=\(model, privacy: .public) placement \(place, privacy: .public)")
        report += "=== \(model) [\(url.lastPathComponent)]  placement \(place) ===\n"

        // load one model per compute unit
        var models: [(String, MLModel)] = []
        var loadFail = false
        for (n, u) in UNITS {
            let cfg = MLModelConfiguration(); cfg.computeUnits = u
            do { models.append((n, try MLModel(contentsOf: url, configuration: cfg))) }
            catch { log.error("RP: \(model, privacy: .public) cu=\(n, privacy: .public) load FAILED \(error.localizedDescription, privacy: .public)"); loadFail = true }
        }
        if loadFail { grandOK = false }
        var modelOK = true
        for c in oracle.cases {
            var line = "  \(c.label):"
            var baseline: MLMultiArray? = nil
            for (n, m) in models {
                let p = provider(c, oracle.inputs)
                do {
                    _ = try await m.prediction(from: p)            // warm
                    let t0 = Date()
                    let out = try await m.prediction(from: p)
                    let ms = Date().timeIntervalSince(t0) * 1000
                    guard let lg = out.featureValue(for: "logits")?.multiArrayValue else { line += " \(n):NOLOGITS"; modelOK = false; continue }
                    if n == "cpu" { baseline = lg }
                    let am = argmaxes(lg, realLen: c.real_len)
                    let ok = am == c.expected_argmax
                    let d = (n == "cpu") ? 0.0 : maxAbsDiff(lg, baseline ?? lg)
                    if !ok { modelOK = false }
                    line += String(format: " %@:%@ %.1fms%@", n, ok ? "OK" : "FAIL", ms, (n == "cpu") ? "" : String(format: " Δ%.0e", d))
                } catch { line += " \(n):ERR"; modelOK = false }
            }
            log.notice("RP: \(line, privacy: .public)")
            report += line + "\n"
        }
        log.notice("RP: model=\(model, privacy: .public) -> \(modelOK ? "PASS" : "CHECK", privacy: .public)")
        report += "  -> \(modelOK ? "PASS" : "CHECK")\n"
        grandOK = grandOK && modelOK
    }
    let verdict = grandOK ? "ALL MODELS PASS on device ✅" : "Some checks need attention ⚠️"
    log.notice("RP: DONE \(verdict, privacy: .public)")
    report += verdict + "\n"
    if let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
        try? report.write(to: docs.appendingPathComponent("ruaccent_probe_report.txt"), atomically: true, encoding: .utf8)
    }
    return verdict
}
