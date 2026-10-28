"""Select a loadable native Rust proc-macro SDK/linker pair before model builds."""

import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile


MACRO_SOURCE = '''extern crate proc_macro;
#[proc_macro]
pub fn marker(_: proc_macro::TokenStream) -> proc_macro::TokenStream {
    "pub const SDK_LINK_PROBE: usize = 1;".parse().unwrap()
}
'''
CONSUMER_SOURCE = '''sdk_link_probe::marker!();
const _: [(); 1] = [(); SDK_LINK_PROBE];
'''


def sdk_directories(selected):
    # CLT can coexist with selected Xcode (and carry an SDK its bundled LLD can read).
    # These are installation roots, never a fixed user path or required SDK version.
    return [selected.parent, Path("/Library/Developer/CommandLineTools/SDKs"),
            *sorted(Path("/Applications").glob(
                "Xcode*.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs"))]


def sdk_candidates(env, *, root):
    """Prefer xcrun's selection, then installed alternatives; explicit SDKROOT is binding."""
    explicit = env.get("SDKROOT")
    if explicit:
        sdk = Path(explicit)
        if not sdk.is_absolute() or not sdk.is_dir():
            raise ValueError("explicit SDKROOT is not an existing absolute macOS SDK directory")
        return [sdk]
    selected = Path(subprocess.run(
        ["xcrun", "--sdk", "macosx", "--show-sdk-path"], cwd=root, env=env,
        check=True, capture_output=True, text=True, encoding="utf-8", timeout=30).stdout.strip())
    if not selected.is_absolute() or not selected.is_dir():
        raise ValueError("xcrun did not select an existing absolute macOS SDK directory")
    alternatives = [sdk for directory in sdk_directories(selected) for sdk in directory.glob("MacOSX*.sdk")]
    alternatives.sort(key=lambda sdk: tuple(map(int, re.findall(r"\d+", sdk.name))), reverse=True)
    result = []
    for sdk in [selected, *alternatives]:
        if sdk.is_dir() and sdk.resolve() not in result:
            result.append(sdk.resolve())
    return result


def probe_sdk(rustc, env, *, root, sdk):
    """Link a tiny dylib, then ask rustc to load and expand its real proc macro.

    The consumer emits metadata only. No Cargo, model, native MLX library or GPU
    is involved, and no SDK file or installed toolchain is modified.
    """
    flags = env["CARGO_ENCODED_RUSTFLAGS"].split("\x1f")
    commands = []
    result = {"sdk": str(sdk), "passed": False, "commands": commands,
              "source_sha256": hashlib.sha256((MACRO_SOURCE + CONSUMER_SOURCE).encode()).hexdigest(),
              "deployment_target": env.get("MACOSX_DEPLOYMENT_TARGET")}
    with tempfile.TemporaryDirectory(prefix="model-proof-sdk-") as temporary:
        work = Path(temporary)
        macro = work / "probe.rs"
        consumer = work / "consumer.rs"
        library = work / "libsdk_link_probe.dylib"
        macro.write_text(MACRO_SOURCE, encoding="utf-8")
        consumer.write_text(CONSUMER_SOURCE, encoding="utf-8")
        invocations = (
            ("link-proc-macro", [*rustc, str(macro), "--edition=2021", "--crate-name", "sdk_link_probe",
                                 "--crate-type", "proc-macro", "-o", str(library), *flags, "-C", "strip=none"]),
            ("load-and-expand-proc-macro", [*rustc, str(consumer), "--edition=2021", "--crate-name", "sdk_link_consumer",
                                            "--crate-type", "lib", "--emit=metadata", "-o", str(work / "consumer.rmeta"),
                                            "--extern", "sdk_link_probe=" + str(library), *flags]),
        )
        for stage, argv in invocations:
            try:
                command = subprocess.run(argv, cwd=root, env=dict(env, SDKROOT=str(sdk)),
                                         capture_output=True, text=True, encoding="utf-8", timeout=30)
            except subprocess.TimeoutExpired:
                commands.append({"stage": stage, "timeout_seconds": 30})
                return result
            commands.append({"stage": stage, "exit": command.returncode,
                             "stderr_sha256": hashlib.sha256(command.stderr.encode()).hexdigest(),
                             "stderr_head": command.stderr[:4000],
                             "stderr_tail": command.stderr[-2000:]})
            if command.returncode:
                return result
        result["passed"] = True
    return result


def select_sdk(rustc, env, *, root):
    attempts = []
    for sdk in sdk_candidates(env, root=root):
        attempt = probe_sdk(rustc, env, root=root, sdk=sdk)
        attempts.append(attempt)
        if attempt["passed"]:
            identities = {}
            for name in ("SDKSettings.json", "usr/lib/libSystem.tbd"):
                path = sdk / name
                if path.is_file():
                    identities[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            return str(sdk), {"explicit_sdkroot": env.get("SDKROOT"), "attempts": attempts,
                              "selected_sdk_realpath": str(sdk.resolve()),
                              "selected_sdk_file_sha256": identities}
    kind = "explicit SDKROOT" if env.get("SDKROOT") else "installed macOS SDKs"
    raise ValueError("no loadable proc-macro SDK/linker pair for " + kind + ": " + json.dumps(attempts))
