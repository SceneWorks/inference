if [[ -z "$MINIMAX_H3_VIDEO_VAE_REFERENCE" ]]; then
  echo "Repository variable MINIMAX_H3_VIDEO_VAE_REFERENCE is unset; the sc-18740 layout gate cannot run and this lane refuses to report green without it" >&2
  exit 1
fi
if [[ "$MINIMAX_H3_VIDEO_VAE_REFERENCE" != /* ]]; then
  echo "MINIMAX_H3_VIDEO_VAE_REFERENCE must be an absolute path, got: $MINIMAX_H3_VIDEO_VAE_REFERENCE" >&2
  exit 1
fi
if [[ ! -s "$MINIMAX_H3_VIDEO_VAE_REFERENCE" ]]; then
  echo "MINIMAX_H3_VIDEO_VAE_REFERENCE names no non-empty file on this runner: $MINIMAX_H3_VIDEO_VAE_REFERENCE" >&2
  exit 1
fi
