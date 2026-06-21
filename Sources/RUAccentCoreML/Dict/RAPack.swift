import Foundation

/// Low-level reader for a `.rapack` v2 dictionary pack.
///
/// A pack is an mmap-able, dependency-free little-endian blob holding a sorted set of
/// keys (bucketed front-coding) plus a kind-specific payload region. See
/// `docs/SWIFT_DICT_PACK.md` §2 for the byte format; this type is a faithful Swift port
/// of the pure-Python `Pack` reader in `converter/pack_dicts.py`.
///
/// Keys are compared as **raw UTF-8 bytes** (`word.utf8`), never as `String`, because the
/// pack is sorted by UTF-8 byte order and `String <` would diverge (it is Unicode-canonical).
///
/// The reader holds the mapping alive via a retained `Data` (mapped with `.mappedIfSafe`),
/// and performs every read through `loadUnaligned` — the format is **not** naturally aligned.
struct RAPack {

    // MARK: Kinds (header byte 5)

    enum Kind: UInt8 {
        case stress = 1     // accents: payload = 1 flat byte stress position (0xFF = none)
        case omograph = 2   // omographs: payload = list of full variant strings
        case yo = 3         // yo_words / yo_homographs: payload = list of substitution positions
    }

    enum LoadError: Error {
        case tooSmall(Int)
        case badMagic
        case badVersion(UInt8)
        case sizeMismatch(declared: UInt32, mapped: Int)
        case kindMismatch(expected: UInt8, actual: UInt8)
    }

    // Sentinel for "no stress" in a STRESS payload byte.
    static let noStress: UInt8 = 0xFF
    static let magic: [UInt8] = [0x52, 0x41, 0x50, 0x4B] // "RAPK"
    static let version: UInt8 = 2

    // Retained backing store (keeps the mmap alive for the lifetime of the reader).
    private let data: Data

    let kind: Kind
    let bucket: Int          // entries per front-coding bucket (== 16)
    let count: Int
    let nBuckets: Int
    let bucketDirOff: Int
    let keyRegionOff: Int
    let payloadOff: Int
    let payloadBlobOff: Int

    /// mmap a `.rapack` and validate its header.
    /// - Parameter expectedKind: the kind this pack must declare (fails fast otherwise).
    init(mappingFile url: URL, expectedKind: Kind) throws {
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        try self.init(data: data, expectedKind: expectedKind)
    }

    /// Validate and parse the 40-byte header over an already-loaded `Data`.
    init(data: Data, expectedKind: Kind) throws {
        let mappedLength = data.count
        guard mappedLength >= 40 else { throw LoadError.tooSmall(mappedLength) }

        // Read the whole header through unaligned loads.
        let header = data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> Result<(UInt8, Int, Int, Int, Int, Int, Int, Int, UInt32), LoadError> in
            // magic
            for i in 0..<4 where raw[i] != RAPack.magic[i] {
                return .failure(.badMagic)
            }
            let version = raw[4]
            guard version == RAPack.version else { return .failure(.badVersion(version)) }
            let kindByte = raw[5]
            guard kindByte == expectedKind.rawValue else {
                return .failure(.kindMismatch(expected: expectedKind.rawValue, actual: kindByte))
            }
            let bucket = Int(raw[6])
            let count = Int(raw.loadUnaligned(fromByteOffset: 8, as: UInt32.self))
            let nBuckets = Int(raw.loadUnaligned(fromByteOffset: 12, as: UInt32.self))
            let bucketDirOff = Int(raw.loadUnaligned(fromByteOffset: 16, as: UInt32.self))
            let keyRegionOff = Int(raw.loadUnaligned(fromByteOffset: 20, as: UInt32.self))
            let payloadOff = Int(raw.loadUnaligned(fromByteOffset: 24, as: UInt32.self))
            let payloadBlobOff = Int(raw.loadUnaligned(fromByteOffset: 28, as: UInt32.self))
            let fileSize = raw.loadUnaligned(fromByteOffset: 32, as: UInt32.self)
            return .success((kindByte, bucket, count, nBuckets, bucketDirOff, keyRegionOff, payloadOff, payloadBlobOff, fileSize))
        }

        let (kindByte, bucket, count, nBuckets, bucketDirOff, keyRegionOff, payloadOff, payloadBlobOff, fileSize) = try header.get()
        guard Int(fileSize) == mappedLength else {
            throw LoadError.sizeMismatch(declared: fileSize, mapped: mappedLength)
        }

        self.data = data
        self.kind = Kind(rawValue: kindByte)! // guaranteed == expectedKind by the check above
        self.bucket = bucket
        self.count = count
        self.nBuckets = nBuckets
        self.bucketDirOff = bucketDirOff
        self.keyRegionOff = keyRegionOff
        self.payloadOff = payloadOff
        self.payloadBlobOff = payloadBlobOff
    }

    // MARK: - Primitive reads

    /// Read a `u32` little-endian at an absolute byte offset (unaligned-safe).
    private static func u32(_ raw: UnsafeRawBufferPointer, _ off: Int) -> UInt32 {
        UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: off, as: UInt32.self))
    }

    /// Read an unsigned LEB128 varint starting at `pos`; returns (value, newPos).
    /// Mirrors `read_uvarint` in the packer.
    private static func uvarint(_ raw: UnsafeRawBufferPointer, _ pos: Int) -> (UInt64, Int) {
        var shift: UInt64 = 0
        var result: UInt64 = 0
        var p = pos
        while true {
            let b = raw[p]
            p += 1
            result |= UInt64(b & 0x7F) << shift
            if (b & 0x80) == 0 { return (result, p) }
            shift += 7
        }
    }

    // MARK: - Bucket / key reconstruction (mirrors the Python Pack)

    /// Byte offset (absolute) of bucket `bi`'s anchor record.
    private func bucketAnchorOffset(_ raw: UnsafeRawBufferPointer, _ bi: Int) -> Int {
        keyRegionOff + Int(RAPack.u32(raw, bucketDirOff + 4 * bi))
    }

    /// Compare the anchor key of bucket `bi` against `target` (UTF-8 bytes).
    /// Returns negative/zero/positive like `memcmp`. Does NOT reconstruct anything.
    private func compareAnchor(_ raw: UnsafeRawBufferPointer, _ bi: Int, _ target: UnsafeBufferPointer<UInt8>) -> Int {
        var pos = bucketAnchorOffset(raw, bi)
        let (klen64, p) = RAPack.uvarint(raw, pos)
        pos = p
        let klen = Int(klen64)
        let n = min(klen, target.count)
        var i = 0
        while i < n {
            let a = raw[pos + i]
            let b = target[i]
            if a != b { return a < b ? -1 : 1 }
            i += 1
        }
        if klen == target.count { return 0 }
        return klen < target.count ? -1 : 1
    }

    /// Find the global entry index for `target` (already-lowercased UTF-8 bytes), or nil.
    /// 1) binary-search bucket anchors for the rightmost anchor <= target;
    /// 2) replay that bucket's front-coded keys, comparing each.
    func index(ofUTF8 target: UnsafeBufferPointer<UInt8>) -> Int? {
        data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> Int? in
            // 1. binary search: rightmost bucket whose anchor <= target
            var lo = 0
            var hi = nBuckets - 1
            var cand = -1
            while lo <= hi {
                let mid = (lo + hi) >> 1
                if compareAnchor(raw, mid, target) <= 0 {
                    cand = mid
                    lo = mid + 1
                } else {
                    hi = mid - 1
                }
            }
            if cand < 0 { return nil }

            // 2. replay the bucket
            let base = cand * bucket
            let end = min(base + bucket, count)
            var pos = bucketAnchorOffset(raw, cand)
            let (klen64, p0) = RAPack.uvarint(raw, pos)
            pos = p0
            let klen = Int(klen64)
            // current key bytes buffer; keys are <= 112 bytes here but support general.
            var prev = [UInt8](repeating: 0, count: klen)
            for i in 0..<klen { prev[i] = raw[pos + i] }
            pos += klen

            if RAPack.equal(prev, target) { return base }
            if RAPack.greater(prev, target) { return nil }

            var j = 1
            while base + j < end {
                let (shared64, p1) = RAPack.uvarint(raw, pos); pos = p1
                let (suflen64, p2) = RAPack.uvarint(raw, pos); pos = p2
                let shared = Int(shared64)
                let suflen = Int(suflen64)
                var cur = [UInt8]()
                cur.reserveCapacity(shared + suflen)
                cur.append(contentsOf: prev[0..<shared])
                for k in 0..<suflen { cur.append(raw[pos + k]) }
                pos += suflen
                if RAPack.equal(cur, target) { return base + j }
                if RAPack.greater(cur, target) { return nil } // sorted ⇒ early out
                prev = cur
                j += 1
            }
            return nil
        }
    }

    // MARK: - Payload decode

    /// STRESS payload byte for global index `i` (0xFF = no stress).
    func stressByte(at i: Int) -> UInt8 {
        data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> UInt8 in raw[payloadOff + i] }
    }

    /// YO substitution scalar positions for global index `i`.
    func yoSubstitutions(at i: Int) -> [Int] {
        data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> [Int] in
            let a = payloadBlobOff + Int(RAPack.u32(raw, payloadOff + 4 * i))
            let n = Int(raw[a])
            var out = [Int](); out.reserveCapacity(n)
            for t in 0..<n { out.append(Int(raw[a + 1 + t])) }
            return out
        }
    }

    /// OMOGRAPH verbatim variant strings for global index `i`.
    func omographVariants(at i: Int) -> [String] {
        data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> [String] in
            var a = payloadBlobOff + Int(RAPack.u32(raw, payloadOff + 4 * i))
            let (nvar64, a0) = RAPack.uvarint(raw, a); a = a0
            let nvar = Int(nvar64)
            var out = [String](); out.reserveCapacity(nvar)
            for _ in 0..<nvar {
                let (blen64, a1) = RAPack.uvarint(raw, a); a = a1
                let blen = Int(blen64)
                let buf = UnsafeRawBufferPointer(rebasing: raw[a ..< a + blen])
                out.append(String(decoding: buf, as: UTF8.self))
                a += blen
            }
            return out
        }
    }

    /// Reconstruct the (lowercased) key string at global index `i`. Used to derive
    /// STRESS / YO values, which are the key with edits applied. Replays the key's bucket.
    func key(at i: Int) -> String {
        data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> String in
            let bi = i / bucket
            let base = bi * bucket
            var pos = bucketAnchorOffset(raw, bi)
            let (klen64, p0) = RAPack.uvarint(raw, pos); pos = p0
            let klen = Int(klen64)
            var prev = [UInt8](repeating: 0, count: klen)
            for k in 0..<klen { prev[k] = raw[pos + k] }
            pos += klen
            if i == base {
                return String(decoding: prev, as: UTF8.self)
            }
            var idx = base + 1
            var cur = prev
            while idx <= i {
                let (shared64, p1) = RAPack.uvarint(raw, pos); pos = p1
                let (suflen64, p2) = RAPack.uvarint(raw, pos); pos = p2
                let shared = Int(shared64)
                let suflen = Int(suflen64)
                cur = []
                cur.reserveCapacity(shared + suflen)
                cur.append(contentsOf: prev[0..<shared])
                for k in 0..<suflen { cur.append(raw[pos + k]) }
                pos += suflen
                prev = cur
                idx += 1
            }
            return String(decoding: cur, as: UTF8.self)
        }
    }

    // MARK: - byte comparison helpers ([UInt8] vs UnsafeBufferPointer<UInt8>)

    private static func equal(_ a: [UInt8], _ b: UnsafeBufferPointer<UInt8>) -> Bool {
        guard a.count == b.count else { return false }
        for i in 0..<a.count where a[i] != b[i] { return false }
        return true
    }

    /// `a > b` in raw UTF-8 byte (lexicographic) order.
    private static func greater(_ a: [UInt8], _ b: UnsafeBufferPointer<UInt8>) -> Bool {
        let n = min(a.count, b.count)
        var i = 0
        while i < n {
            if a[i] != b[i] { return a[i] > b[i] }
            i += 1
        }
        return a.count > b.count
    }
}
