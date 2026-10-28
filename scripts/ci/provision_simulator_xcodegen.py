#!/usr/bin/env python3
"""CI_IMPACT: manual simulator lane only; 4,286,070-byte pinned tool download, no global install.

Official provenance: https://github.com/yonaskolb/XcodeGen/releases/tag/2.46.0
The release asset API publishes the SHA-256 below. Preserve the adjacent preset
resource bundle; copying only the executable would break project generation.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import stat
import subprocess
import urllib.request
import zipfile

VERSION = '2.46.0'
URL = 'https://github.com/yonaskolb/XcodeGen/releases/download/2.46.0/xcodegen.artifactbundle.zip'
SHA256 = 'ef6d0a23bfb7393387f98e321ffd78a487231172e2e78c48d3c26275c263fd0c'
ARCHIVE_BYTES = 4_286_070
BUNDLE = 'xcodegen.artifactbundle'
BINARY = BUNDLE + '/xcodegen-2.46.0-macosx/bin/xcodegen'


def install(archive: bytes, destination: Path) -> Path:
    if len(archive) != ARCHIVE_BYTES or hashlib.sha256(archive).hexdigest() != SHA256:
        raise ValueError('XcodeGen release size or SHA-256 mismatch')
    if destination.exists():
        raise ValueError('fresh job-local XcodeGen directory required')
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        total = 0
        names = set()
        for member in bundle.infolist():
            path = PurePosixPath(member.filename)
            total += member.file_size
            if (path.is_absolute() or '..' in path.parts or path.parts[0] != BUNDLE
                    or stat.S_ISLNK(member.external_attr >> 16)
                    or member.file_size > 20 * 1024**2 or total > 32 * 1024**2
                    or member.filename in names):
                raise ValueError('unexpected XcodeGen archive member')
            names.add(member.filename)
        info = json.loads(bundle.read(BUNDLE + '/info.json'))['artifacts']['xcodegen']
        if (info['version'] != VERSION or info['type'] != 'executable'
                or info['variants'] != [{'path': 'xcodegen-2.46.0-macosx/bin/xcodegen',
                                        'supportedTriples': ['x86_64-apple-macosx', 'arm64-apple-macosx']}]):
            raise ValueError('XcodeGen artifact does not declare the pinned native executable')
        if BINARY not in names:
            raise ValueError('XcodeGen executable missing')
        destination.mkdir(mode=0o700)
        bundle.extractall(destination)
    binary = destination / BINARY
    binary.chmod(0o755)
    return binary


def probe(binary: Path) -> dict:
    result = subprocess.run([str(binary), '--version'], check=True, capture_output=True,
                            text=True, encoding='utf-8', timeout=30)
    if result.stdout.strip() != 'Version: ' + VERSION:
        raise ValueError('XcodeGen executable reported an unexpected version')
    return {'version': VERSION, 'release_url': URL, 'archive_sha256': SHA256,
            'archive_bytes': ARCHIVE_BYTES, 'binary': str(binary),
            'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
            'version_stdout': result.stdout.strip(), 'host_arch': platform.machine()}


def main() -> None:
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        raise ValueError('native Apple-silicon XcodeGen prerequisite required')
    evidence = Path(os.environ['RUNNER_TEMP']) / 'ios-simulator-sana'
    evidence.mkdir(exist_ok=True)
    destination = Path(os.environ['RUNNER_TEMP']) / ('xcodegen-' + VERSION + '-' + os.environ['GITHUB_RUN_ID'] + '-' + os.environ['GITHUB_RUN_ATTEMPT'])
    with urllib.request.urlopen(URL, timeout=120) as response:
        archive = response.read(ARCHIVE_BYTES + 1)
    binary = install(archive, destination)
    receipt = probe(binary)
    (evidence / 'xcodegen-prerequisite.json').write_text(json.dumps(receipt, sort_keys=True) + '\n', encoding='utf-8')
    with Path(os.environ['GITHUB_PATH']).open('a', encoding='utf-8') as output:
        output.write(str(binary.parent) + '\n')
    print(json.dumps(receipt, sort_keys=True))


if __name__ == '__main__':
    main()
