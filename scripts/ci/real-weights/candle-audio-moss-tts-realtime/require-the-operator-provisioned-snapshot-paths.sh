for v in MOSS_TTS_REALTIME_SNAPSHOT MOSS_AUDIO_TOKENIZER_SNAPSHOT WHISPER_SNAPSHOT CHATTERBOX_SNAPSHOT; do
  if [[ -z "${!v}" ]]; then
    echo "Repository variable $v is unset; this lane cannot resolve its snapshot path" >&2
    exit 1
  fi
  if [[ "${!v}" != /* ]]; then
    echo "$v must be an absolute path, got: ${!v}" >&2
    exit 1
  fi
done
