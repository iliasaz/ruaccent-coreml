import CoreML
import Foundation

/// Shared CoreML plumbing for the four model runners (Stage 4 / ModelRunners).
///
/// Each `.mlpackage` declares its inputs as **int32** `MLMultiArray` of shape `[1, S]`
/// (`input_ids`, `attention_mask`, and — for M1/M3 — a zero-filled `token_type_ids`; M2/M4
/// have no such input). The single output is a float16 `MLMultiArray` named `logits`. These
/// helpers build the int32 feature arrays, read the logits back as `Double`, and implement the
/// two softmax variants the upstream models use (per-row / global). All decode math is in
/// `Double` so the Swift side never re-introduces fp16 rounding beyond the model's own.
enum CoreMLSupport {

    /// Build a `[1, length]` int32 `MLMultiArray` from `values` (the first `values.count` entries),
    /// zero-padding the tail to `length`. Used to pad short sequences up to a model's minimum
    /// flexible-shape length (M3/M4 require ≥ 4) or M1's fixed length (48).
    static func int32Array(_ values: [Int], paddedTo length: Int) throws -> MLMultiArray {
        precondition(values.count <= length, "values longer than padded length")
        let arr = try MLMultiArray(shape: [1, NSNumber(value: length)], dataType: .int32)
        let ptr = arr.dataPointer.bindMemory(to: Int32.self, capacity: length)
        for i in 0..<length {
            ptr[i] = i < values.count ? Int32(values[i]) : 0
        }
        return arr
    }

    /// Read an `MLMultiArray` of logits into a flat `[Double]` in storage order. The runners
    /// reshape this themselves against the known `[1, S, C]` / `[1, C]` layout.
    static func doubles(_ array: MLMultiArray) -> [Double] {
        let count = array.count
        var out = [Double](repeating: 0, count: count)
        switch array.dataType {
        case .float16:
            // No direct Float16→Double bulk path on all SDKs; go element-wise via Float.
            for i in 0..<count { out[i] = Double(truncating: array[i]) }
        case .float32, .double:
            for i in 0..<count { out[i] = Double(truncating: array[i]) }
        default:
            for i in 0..<count { out[i] = Double(truncating: array[i]) }
        }
        return out
    }

    /// Per-row numerically-stable softmax over the last axis of an `[n, c]` logit matrix.
    /// Matches the `predict_*` softmax in `stress_usage_model.py` / `accent_model.py`
    /// (`exp(x - max) / sum`), applied independently per token.
    static func softmaxRows(_ rows: [[Double]]) -> [[Double]] {
        rows.map { softmax($0) }
    }

    /// Numerically-stable softmax over a single vector (`exp(x - max) / sum`).
    static func softmax(_ x: [Double]) -> [Double] {
        guard let m = x.max() else { return x }
        var exps = [Double](repeating: 0, count: x.count)
        var sum = 0.0
        for i in 0..<x.count {
            let e = Foundation.exp(x[i] - m)
            exps[i] = e
            sum += e
        }
        if sum == 0 { return exps }
        for i in 0..<x.count { exps[i] /= sum }
        return exps
    }

    /// Argmax of a vector (first max wins, matching NumPy `argmax`).
    static func argmax(_ x: [Double]) -> Int {
        var best = 0
        for i in 1..<x.count where x[i] > x[best] { best = i }
        return best
    }
}
