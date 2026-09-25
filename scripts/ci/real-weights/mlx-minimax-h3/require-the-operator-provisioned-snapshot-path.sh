if [[ -z "$MINIMAX_H3_SNAPSHOT" ]]; then
  echo "Repository variable MINIMAX_H3_SNAPSHOT is unset; this lane cannot resolve its snapshot path" >&2
  exit 1
fi
if [[ "$MINIMAX_H3_SNAPSHOT" != /* ]]; then
  echo "MINIMAX_H3_SNAPSHOT must be an absolute path, got: $MINIMAX_H3_SNAPSHOT" >&2
  exit 1
fi
