if [[ -z "$MINIMAX_H3_VIDEO_VAE_ENCODE_REFERENCE" ]]; then
  echo "Repository variable MINIMAX_H3_VIDEO_VAE_ENCODE_REFERENCE is unset; the sc-19445 encoder level-indexing gate cannot run and this lane refuses to report green without it" >&2
  exit 1
fi
if [[ "$MINIMAX_H3_VIDEO_VAE_ENCODE_REFERENCE" != /* ]]; then
  echo "MINIMAX_H3_VIDEO_VAE_ENCODE_REFERENCE must be an absolute path, got: $MINIMAX_H3_VIDEO_VAE_ENCODE_REFERENCE" >&2
  exit 1
fi
if [[ ! -s "$MINIMAX_H3_VIDEO_VAE_ENCODE_REFERENCE" ]]; then
  echo "MINIMAX_H3_VIDEO_VAE_ENCODE_REFERENCE names no non-empty file on this runner: $MINIMAX_H3_VIDEO_VAE_ENCODE_REFERENCE" >&2
  exit 1
fi
