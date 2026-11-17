if ! xcrun --find metal >/dev/null 2>&1; then
  sudo xcodebuild -downloadComponent MetalToolchain
fi
xcrun metal --version || true
