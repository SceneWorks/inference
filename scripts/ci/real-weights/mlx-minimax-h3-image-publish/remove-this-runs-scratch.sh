if [[ -n "${PUBLISH_SCRATCH:-}" && "$PUBLISH_SCRATCH" == */sw-7104-publish-* ]]; then
  rm -rf "$PUBLISH_SCRATCH"
fi
