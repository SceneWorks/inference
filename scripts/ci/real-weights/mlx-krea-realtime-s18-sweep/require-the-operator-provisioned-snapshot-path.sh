if [[ -z "$KREA_REALTIME_SNAPSHOT" ]]; then
  echo "Repository variable KREA_REALTIME_SNAPSHOT is unset; this lane cannot resolve its snapshot path" >&2
  exit 1
fi
if [[ "$KREA_REALTIME_SNAPSHOT" != /* ]]; then
  echo "KREA_REALTIME_SNAPSHOT must be an absolute path, got: $KREA_REALTIME_SNAPSHOT" >&2
  exit 1
fi
