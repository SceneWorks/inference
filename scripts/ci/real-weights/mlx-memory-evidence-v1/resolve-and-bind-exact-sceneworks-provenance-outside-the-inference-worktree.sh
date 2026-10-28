if [[ ! "$REQUESTED_SCENEWORKS_REVISION" =~ ^[0-9a-f]{40}$ ]]; then
  echo "sceneworks_revision must be an exact lowercase commit SHA" >&2
  exit 1
fi
git init "$SCENEWORKS_PROVENANCE_ROOT"
git -C "$SCENEWORKS_PROVENANCE_ROOT" remote add origin https://github.com/SceneWorks/SceneWorks.git
git -C "$SCENEWORKS_PROVENANCE_ROOT" fetch --depth=1 -- origin "$REQUESTED_SCENEWORKS_REVISION"
git -C "$SCENEWORKS_PROVENANCE_ROOT" checkout --detach FETCH_HEAD
resolved="$(git -C "$SCENEWORKS_PROVENANCE_ROOT" rev-parse HEAD)"
if [[ ! "$resolved" =~ ^[0-9a-f]{40}$ || "$resolved" != "$REQUESTED_SCENEWORKS_REVISION" ]]; then
  echo "sceneworks_revision did not resolve to the exact requested commit" >&2
  exit 1
fi
test -z "$(git -C "$SCENEWORKS_PROVENANCE_ROOT" status --porcelain --untracked-files=normal)"
echo "SCENEWORKS_REVISION=$resolved" >> "$GITHUB_ENV"
