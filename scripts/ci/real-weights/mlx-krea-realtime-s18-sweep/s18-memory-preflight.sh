# WxH[@q<bits>] — the KV tier suffix is sc-17807's, and the preflight has to see
# it or it prices the bf16 sweep while a q8 one runs and refuses rows that now fit.
if [[ ! "$KREA_S18_GEOMETRY" =~ ^[0-9]+x[0-9]+(@q[0-9]+)?$ ]]; then
  echo "::error::krea_s18_geometry='$KREA_S18_GEOMETRY' must be WxH[@q<bits>], e.g. 832x480 or 640x384@q8" >&2
  exit 1
fi
geom="${KREA_S18_GEOMETRY%%@*}"
if [[ "$KREA_S18_GEOMETRY" == *@q* ]]; then
  export KREA_S18_KV_BITS="${KREA_S18_GEOMETRY##*@q}"
fi
KREA_SMOKE_W="${geom%x*}" KREA_SMOKE_H="${geom#*x}" \
  python3.12 scripts/ci/s18_memory_preflight.py
