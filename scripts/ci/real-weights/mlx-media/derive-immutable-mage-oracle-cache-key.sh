snapshot_revision() {
  local snapshot="$1"
  local marker="$snapshot/.sceneworks-model-revision"
  local revision
  if [[ -f "$marker" ]]; then
    revision="$(tr -d '\r\n' < "$marker")"
  else
    revision="$(basename "$(cd "$snapshot" && pwd -P)")"
  fi
  if [[ ! "$revision" =~ ^[0-9a-f]{40}$ ]]; then
    echo "Snapshot does not resolve to an immutable 40-hex revision: $snapshot" >&2
    return 1
  fi
  printf '%s' "$revision"
}
producer="$MAGE_ORACLE_PRODUCER"
if [[ ! "$producer" =~ ^[0-9a-f]{64}$ ]]; then
  echo "Oracle producer fingerprint is missing or malformed: $producer" >&2
  exit 1
fi
key="mage-oracles-v1-$(snapshot_revision "$MAGE_SNAPSHOT")-$(snapshot_revision "$MAGE_EDIT_SNAPSHOT")-$(snapshot_revision "$MAGE_EDIT_BASE_SNAPSHOT")-$(snapshot_revision "$MAGE_EDIT_TURBO_SNAPSHOT")-$producer"
echo "key=$key" >> "$GITHUB_OUTPUT"
