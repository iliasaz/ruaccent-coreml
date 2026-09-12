import Foundation
import Hub
import os

/// Downloads the RUAccentCoreML on-device bundle (the four CoreML models, the packed
/// `.rapack` dictionaries, and the tokenizer files) from the **HF repo**
/// `iliasaz/ruaccent-coreml`, then hands the local directory to `RUAccent(modelDirectory:)`.
///
/// Mirrors `chatterbox-coreml`'s `ModelRepository` (`iliasaz/chatterbox-turbo-coreml`): large
/// artifacts live on the Hub, not in git, so this package stays binary-free. Downloads use
/// swift-transformers `HubApi.snapshot`; the cache base is `HF_HOME/hub` (same layout as the
/// Python `huggingface_hub`). The repo tree (`coreml/`, `dictpack/`, `nn/`) is exactly the
/// layout `RUAccent(modelDirectory:)` expects, so the snapshot dir is usable as-is.
public enum ModelRepository {
    public static let defaultRepoId = "iliasaz/ruaccent-coreml"

    private static let log = Logger(subsystem: "com.iliasaz.ruaccentcoreml", category: "download")

    /// Glob patterns for the full runtime bundle: the four CoreML models under `coreml/`, the
    /// four `.rapack` dictionaries under `dictpack/`, and the tokenizer files under `nn/`. Both
    /// the directory entry (`coreml/*`) and its recursive contents (`coreml/**/*`) are matched,
    /// mirroring chatterbox's `.mlpackage/*` + `.mlpackage/**/*` form.
    public static let runtimeGlobs: [String] = [
        "coreml/*", "coreml/**/*",
        "dictpack/*",
        "nn/*", "nn/**/*",
    ]

    /// A file every valid bundle directory contains (used to discover an existing download).
    private static let markerFile = "dictpack/accents.rapack"

    // MARK: - Base resolution

    /// The hub cache base for a given HF_HOME root: `<hfHome>/hub`.
    public static func base(forHFHome hfHome: URL) -> URL {
        hfHome.appending(component: "hub")
    }

    /// The hub cache base inferred from the environment, or `nil` if none resolve:
    /// `HF_HUB_CACHE`, then `HF_HOME` + `/hub`, then `~/.cache/huggingface/hub`.
    public static func environmentDownloadBase() -> URL? {
        let env = ProcessInfo.processInfo.environment
        if let c = env["HF_HUB_CACHE"], !c.isEmpty {
            return URL(fileURLWithPath: c, isDirectory: true)
        }
        if let h = env["HF_HOME"], !h.isEmpty {
            return URL(fileURLWithPath: h, isDirectory: true).appending(component: "hub")
        }
        if let home = env["HOME"], !home.isEmpty {
            return URL(fileURLWithPath: home, isDirectory: true)
                .appending(component: ".cache").appending(component: "huggingface").appending(component: "hub")
        }
        return nil
    }

    private static func resolvedBase() -> URL {
        environmentDownloadBase()
            ?? FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
                .appending(component: "huggingface")
    }

    // MARK: - Discovery

    /// Searches the known on-disk layouts under `base` and returns the first directory that
    /// actually contains the bundle. Recognizes the swift-transformers layout
    /// (`<base>/models/<repoId>`), a plain `--local-dir` download (`<base>/<repoId>`), and the
    /// Python `huggingface_hub` cache (`<base>/models--<org>--<name>/snapshots/<hash>`).
    public static func existingModelDirectory(in base: URL, repoId: String = defaultRepoId) -> URL? {
        let fm = FileManager.default
        func hasBundle(_ dir: URL) -> Bool {
            fm.fileExists(atPath: dir.appending(path: markerFile).path)
        }
        var candidates: [URL] = [
            base.appending(component: "models").appending(path: repoId),
            base.appending(path: repoId),
        ]
        let cacheName = "models--" + repoId.replacingOccurrences(of: "/", with: "--")
        let snapshots = base.appending(path: cacheName).appending(component: "snapshots")
        if let subdirs = try? fm.contentsOfDirectory(
            at: snapshots, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles]
        ) {
            candidates.append(contentsOf: subdirs)
        }
        return candidates.first(where: hasBundle)
    }

    /// Deletes the downloaded snapshot (if any) so the next `download` is a clean re-fetch.
    @discardableResult
    public static func removeDownload(hfHome: URL? = nil, repoId: String = defaultRepoId) throws -> Bool {
        let searchBase = hfHome.map { Self.base(forHFHome: $0) } ?? resolvedBase()
        guard let dir = existingModelDirectory(in: searchBase, repoId: repoId),
              FileManager.default.fileExists(atPath: dir.path) else { return false }
        try FileManager.default.removeItem(at: dir)
        return true
    }

    /// Minimum plausible size (bytes) for a real CoreML `weight.bin`. A missing file is 0; an
    /// unresolved Git-LFS pointer is ~130 bytes. The smallest real weight here is M1 accent at
    /// ~1.08 MB, so a 100 KB floor cleanly separates "intact" from "broken".
    private static let minWeightBytes = 100_000

    /// Returns the name of the first `coreml/*.mlpackage` whose `weight.bin` is missing or
    /// implausibly small (interrupted download / LFS pointer), or `nil` if all look intact.
    static func incompleteMLPackage(in dir: URL) -> String? {
        let fm = FileManager.default
        let coreml = dir.appending(component: "coreml")
        guard let entries = try? fm.contentsOfDirectory(atPath: coreml.path) else { return nil }
        for name in entries.sorted() where name.hasSuffix(".mlpackage") {
            let weight = coreml.appending(path: name)
                .appending(path: "Data/com.apple.CoreML/weights/weight.bin")
            let size = ((try? fm.attributesOfItem(atPath: weight.path))?[.size] as? Int) ?? 0
            if size < minWeightBytes { return "coreml/\(name)" }
        }
        return nil
    }

    // MARK: - Download

    /// Downloads (or reuses the cache of) the runtime bundle and returns the local directory to
    /// pass to `RUAccent(modelDirectory:)`.
    ///
    /// - Parameters:
    ///   - repoId: the HF model repo (default: the public `iliasaz/ruaccent-coreml`).
    ///   - hfHome: when provided, treated as `HF_HOME`; the snapshot is written under `<hfHome>/hub`.
    ///   - hfToken: an explicit HF access token (only needed for a private repo; the default is public).
    ///     Resolution order: this parameter → `HF_TOKEN` env → `HUGGING_FACE_HUB_TOKEN` env →
    ///     (via `HubApi`) the standard CLI token files (`$HF_HOME/token`, `~/.cache/huggingface/token`).
    ///   - matching: file globs to fetch (default: the full `runtimeGlobs`).
    ///   - progress: optional 0...1 progress callback.
    public static func download(
        repoId: String = defaultRepoId,
        hfHome: URL? = nil,
        hfToken: String? = nil,
        matching globs: [String] = runtimeGlobs,
        progress: (@Sendable (Double) -> Void)? = nil
    ) async throws -> URL {
        let env = ProcessInfo.processInfo.environment
        let resolvedToken = hfToken ?? env["HF_TOKEN"] ?? env["HUGGING_FACE_HUB_TOKEN"]
        let downloadBase: URL? = hfHome.map { base(forHFHome: $0) } ?? environmentDownloadBase()

        let api = HubApi(downloadBase: downloadBase, hfToken: resolvedToken)
        let repo = Hub.Repo(id: repoId)
        func fetch() async throws -> URL {
            try await api.snapshot(from: repo, matching: globs) { p in
                progress?(p.fractionCompleted)
            }
        }
        var dir = try await fetch()

        // Self-heal: a truncated weight.bin compiles to a broken model. If any package looks
        // incomplete (interrupted/partial download), drop the snapshot and re-fetch once.
        if let bad = incompleteMLPackage(in: dir) {
            log.notice("\(bad, privacy: .public) is incomplete (missing/truncated weight); forcing clean re-download")
            _ = try? removeDownload(hfHome: hfHome, repoId: repoId)
            dir = try await fetch()
            if let stillBad = incompleteMLPackage(in: dir) {
                throw RUAccentError.missingResource("\(stillBad): incomplete model package after clean re-download")
            }
        }
        log.debug("repo=\(repoId, privacy: .public) dir=\(dir.path, privacy: .public)")
        return dir
    }
}
