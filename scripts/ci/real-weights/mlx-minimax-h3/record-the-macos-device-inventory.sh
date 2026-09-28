sysctl -n hw.model hw.memsize | awk '
  NR == 1 { print "host.hw_model=" $0 }
  NR == 2 { printf "host.hw_memsize=%.2f GiB (%s bytes, physical RAM)\n", $0 / 1073741824, $0 }'
if ! command -v swift > /dev/null 2>&1; then
  echo "metal.recommendedMaxWorkingSetSize=<not captured — swift is not on this runner's PATH>"
  exit 0
fi
swift - <<'SWIFT'
import Foundation
import Metal

if let device = MTLCreateSystemDefaultDevice() {
    let bytes = device.recommendedMaxWorkingSetSize
    print("metal.device=\(device.name)")
    print(String(format: "metal.recommendedMaxWorkingSetSize=%.2f GiB (%llu bytes, NOT hw.memsize)",
                 Double(bytes) / 1073741824.0, bytes))
} else {
    print("metal.recommendedMaxWorkingSetSize=<not captured — no default Metal device>")
}
SWIFT
