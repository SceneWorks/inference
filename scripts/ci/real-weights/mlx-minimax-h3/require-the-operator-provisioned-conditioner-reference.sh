if [[ -z "$MINIMAX_H3_TE_REFERENCE" ]]; then
  echo "Repository variable MINIMAX_H3_TE_REFERENCE is unset; the sc-18741 conditioner gate cannot run and this lane refuses to report green without it" >&2
  exit 1
fi
if [[ "$MINIMAX_H3_TE_REFERENCE" != /* ]]; then
  echo "MINIMAX_H3_TE_REFERENCE must be an absolute path, got: $MINIMAX_H3_TE_REFERENCE" >&2
  exit 1
fi
if [[ ! -s "$MINIMAX_H3_TE_REFERENCE" ]]; then
  echo "MINIMAX_H3_TE_REFERENCE names no non-empty file on this runner: $MINIMAX_H3_TE_REFERENCE" >&2
  exit 1
fi
