import SwiftUI

// The Rust smoke test's C entry points (ios-host/smoke/src/lib.rs).
@_silgen_name("ios_smoke_run")
func ios_smoke_run() -> UnsafeMutablePointer<CChar>?

@_silgen_name("ios_smoke_free")
func ios_smoke_free(_ ptr: UnsafeMutablePointer<CChar>?)

@_silgen_name("ios_smoke_available_memory_mib")
func ios_smoke_available_memory_mib() -> Double

/// The build's `com.apple.developer.kernel.increased-memory-limit` setting, `YES` or `NO`. Stamped
/// into the Info.plist from the `SMOKE_INCREASED_MEMORY_LIMIT` build setting, which is switched in
/// lockstep with `CODE_SIGN_ENTITLEMENTS` (see project.yml / run_smoke.sh `--memory-entitlement`).
let increasedMemoryLimit =
    Bundle.main.object(forInfoDictionaryKey: "SWSmokeIncreasedMemoryLimit") as? String ?? "unknown"

/// Log `os_proc_available_memory()` at LAUNCH, before the report runs anything large.
///
/// This is the per-app cap measurement (sc-16780): sampled at launch the remaining headroom is
/// effectively the cap. Written to Documents/launch-memory.txt as well as the log, so it survives
/// even if the report that follows is jetsam-killed, and exported to the Rust side so the report
/// names the entitlement setting it ran under.
func logLaunchMemory() {
    let mib = ios_smoke_available_memory_mib()
    setenv("IOS_SMOKE_INCREASED_MEMORY_LIMIT", increasedMemoryLimit, 1)
    let line = String(
        format: "launch: os_proc_available_memory = %.0f MiB, increased-memory-limit = %@\n",
        mib, increasedMemoryLimit)
    NSLog("%@", line)
    if let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
        try? line.write(to: dir.appendingPathComponent("launch-memory.txt"),
                        atomically: true, encoding: .utf8)
    }
}

/// Runs the on-device MLX smoke test and returns its report.
///
/// The Rust side never returns an error code — it renders every outcome into the report string,
/// whose first line is `SMOKE: PASS` or `SMOKE: FAIL`. That keeps this bridge trivial and makes
/// the result equally readable by a human and by an XCTest.
func runSmokeTest() -> String {
    guard let raw = ios_smoke_run() else {
        return "SMOKE: FAIL\n  [XX] ios_smoke_run returned null"
    }
    defer { ios_smoke_free(raw) }
    return String(cString: raw)
}

@main
struct SmokeApp: App {
    init() { logLaunchMemory() }

    var body: some Scene {
        WindowGroup { ContentView() }
    }
}

struct ContentView: View {
    @State private var report: String = ""
    @State private var running = false

    private var passed: Bool { report.hasPrefix("SMOKE: PASS") }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("MLX on-device smoke test")
                .font(.headline)

            if running {
                ProgressView("Dispatching Metal kernels…")
                    .frame(maxWidth: .infinity)
            } else if !report.isEmpty {
                Label(passed ? "PASS" : "FAIL", systemImage: passed ? "checkmark.seal.fill" : "xmark.octagon.fill")
                    .foregroundStyle(passed ? .green : .red)
                    .font(.title2.bold())
            }

            ScrollView {
                Text(report.isEmpty ? "Not run yet." : report)
                    // Monospaced so the aligned [ok]/[XX] column stays readable, and selectable
                    // so a failure detail can be copied off the device.
                    .font(.system(.footnote, design: .monospaced))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }

            Button(running ? "Running…" : "Run smoke test") { run() }
                .buttonStyle(.borderedProminent)
                .disabled(running)
                .frame(maxWidth: .infinity)
        }
        .padding()
        // Run on appear so `xcrun devicectl device process launch` gets a result with no tapping.
        .onAppear(perform: run)
    }

    private func run() {
        guard !running else { return }
        running = true
        // Off the main thread: the first Metal library load plus kernel dispatch takes long
        // enough to hang the UI, and a hung main thread risks a watchdog kill.
        DispatchQueue.global(qos: .userInitiated).async {
            let result = runSmokeTest()
            print(result)
            writeReport(result)
            DispatchQueue.main.async {
                report = result
                running = false
            }
        }
    }
}

/// Persist the report into the app's Documents container.
///
/// `devicectl device process launch --console` does not reliably capture a GUI app's stdout, and
/// `log collect` needs root — so a file is the only dependable way to get the result off the
/// device. `scripts/ios/run_smoke.sh` pulls it with `devicectl device copy from`, which is also
/// what makes the check assertable in CI rather than a human reading the screen.
private func writeReport(_ text: String) {
    guard let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
    else { return }
    try? text.write(to: dir.appendingPathComponent("smoke-report.txt"),
                    atomically: true,
                    encoding: .utf8)
}
