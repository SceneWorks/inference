set -o pipefail
mkdir -p "$STARVECTOR_TERMINAL_DIR/inventory" "$STARVECTOR_TERMINAL_DIR/hooks"
python3.12 scripts/release/verify_model_snapshot.py --model starvector-1b-im2svg --snapshot "$STARVECTOR_1B_SNAPSHOT" --inventory-output "$STARVECTOR_TERMINAL_DIR/inventory/starvector-1b-inventory.json"
python3.12 scripts/release/verify_model_snapshot.py --model starvector-8b-im2svg --snapshot "$STARVECTOR_8B_SNAPSHOT" --inventory-output "$STARVECTOR_TERMINAL_DIR/inventory/starvector-8b-inventory.json"
shasum -a 256 "$STARVECTOR_TERMINAL_DIR/inventory/starvector-1b-inventory.json" "$STARVECTOR_TERMINAL_DIR/inventory/starvector-8b-inventory.json" | tee "$STARVECTOR_TERMINAL_DIR/inventory-sha256.txt"
