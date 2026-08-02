#!/usr/bin/env python3
"""Build a deterministic runtime manifest, SPDX SBOM, and source archive."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import gzip
import hashlib
import io
import json
import re
import subprocess
import tomllib
from pathlib import Path
from typing import Any
from urllib.parse import quote


TAG_PATTERN = re.compile(r"^runtime-(\d{4})\.(\d{2})\.(\d+)(?:-rc\.(\d+))?$")
REPOSITORY = "https://github.com/SceneWorks/inference"
TOOL_NAME = "SceneWorks inference release builder"

# The committed model-weight-licence manifest (sc-13332, schema 3 since sc-16663). A separate axis
# from the SPDX crate SBOM: the licence facts for every artifact the shipped providers load. The Rust
# catalog ship-gate (`candle-audio-catalog::every_shipped_provider_has_a_weight_license`) keeps this
# file complete and in sync with the provider registry; the release tooling emits it beside the SBOM
# so a consumer reads exactly one file for its end-product licences page.
#
# DISCLOSURE ONLY. Nothing this validator checks gates or withholds anything — it checks that the
# facts a consumer will display are present and structurally sound.
MODEL_LICENSES_SOURCE = "release/model-weight-licenses.json"
MODEL_LICENSES_DIRECTORY = "release"
# Every catalog's manifest is discovered by shape rather than by an enumerated list of filenames, so
# a catalog landing its manifest is picked up with no edit here — and, more importantly, a catalog
# that names its file something this module did not predict cannot be silently *skipped*, which would
# ship a release whose licences page is missing a whole backend while every gate stayed green.
# Discovery is intersected with git's tracked set (`discover_model_license_sources`); the shape is
# still the only rule, so landing a manifest remains "commit the file", with no edit here.
MODEL_LICENSES_GLOB = "model-weight-licenses*.json"
MODEL_LICENSES_KIND = "model-weight-licenses-json"
MODEL_LICENSES_SCHEMA = 3
REQUIRED_FAMILY_FIELDS = ("id", "spdx_id", "name", "text_url")
REQUIRED_COMPONENT_FIELDS = ("component", "source_url", "declared", "family", "retrieved")

# Each section, the field its rows are keyed by, and its singular label — in the order the manifest
# carries them.
MODEL_LICENSES_SECTIONS = (
    ("families", "id", "family"),
    ("components", "component", "component"),
    ("providers", "provider_id", "provider"),
)

# The payload fields each `LicenseTerm` variant carries, mirroring `LicenseTerm::to_json` in
# `crates/contracts/gen-core/src/license.rs`. Enumerated rather than open-ended for the same reason
# the Rust match is exhaustive: a term shape this table does not know would be sorted and deduped as
# though it were a bare variant, so two distinct terms could collapse into one inside a `terms` array
# — silently, and only in the derived view a consumer actually joins over. Adding a Rust variant must
# therefore break here, loudly, rather than be tolerated.
LICENSE_TERM_PAYLOADS: dict[str, tuple[str, ...]] = {
    "attribution_required": (),
    "notice_file_required": (),
    "non_commercial_weights": (),
    "non_commercial_outputs": (),
    "gated_access": (),
    "revenue_ceiling": ("amount_usd", "boundary"),
    "registration_required": ("contact",),
    "acceptable_use_policy": ("url",),
    "deployer_obligation": ("text",),
    "downstream_license_copy": ("family",),
    "downstream_restrictions": ("family",),
}
# The two payloads that are `Option<&str>` in Rust: `None` records "the licence names this and gives
# no address" and is a different disclosure from a blank string, which is a malformed row.
OPTIONAL_TERM_ADDRESSES = {"registration_required": "contact", "acceptable_use_policy": "url"}
FLOW_DOWN_TERMS = ("downstream_license_copy", "downstream_restrictions")
CEILING_BOUNDARIES = ("exclusive", "inclusive")


def _is_blank(value: Any) -> bool:
    """Whether `value` carries no information — mirroring `license::is_blank`.

    Whitespace-only counts: `"   "` satisfies `!is_empty()` while rendering as nothing on a licences
    page, so it is rejected exactly like an absent value. A non-string is blank by definition here;
    the field checks report the type problem as an absence, which is the actionable message either
    way.
    """
    return not isinstance(value, str) or not value.strip()


def _is_iso_date(value: Any) -> bool:
    """Whether `value` is an ISO `YYYY-MM-DD` **calendar** date — mirroring `license::is_iso_date`.

    Calendar-accurate, not merely range-checked: month lengths and the Gregorian leap rule apply, so
    `2026-02-31`, `2026-04-31` and `2026-02-29` are rejected while `2024-02-29` and `2000-02-29` are
    accepted and `1900-02-29` is not. Year `0000` is rejected.

    Hand-rolled rather than delegated to `datetime.date.fromisoformat`, deliberately: that function
    *raises* on `2026-02-31` (and accepts extended forms such as `2026-01-01T00:00:00Z` on 3.11+), so
    leaning on it would answer a malformed row with a `ValueError` traceback instead of a message
    naming the row — and would disagree with the Rust gate about which strings are dates.
    """
    if not isinstance(value, str) or len(value) != 10 or value[4] != "-" or value[7] != "-":
        return False
    digits = value[:4] + value[5:7] + value[8:]
    if not digits.isdigit() or not digits.isascii():
        return False
    year, month, day = int(value[:4]), int(value[5:7]), int(value[8:])
    if year == 0 or not 1 <= month <= 12 or day == 0:
        return False
    leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
    if month in (1, 3, 5, 7, 8, 10, 12):
        return day <= 31
    if month in (4, 6, 9, 11):
        return day <= 30
    return day <= (29 if leap else 28)


def _term_sort_key(term: dict[str, Any]) -> tuple[str, int, str]:
    """The order the serialized union is emitted in — mirroring `LicenseTerm::sort_key`.

    Keyed on the stable `term` tag, **not** on variant declaration order, so a future variant
    inserted mid-`enum` cannot reorder an already-committed manifest. The middle slot carries an
    optional payload's *presence*, not only its contents: `None` and `""` are different values and
    flattening both to `""` would let the dedup below drop one of them.
    """
    tag = term["term"]
    if tag == "revenue_ceiling":
        return (tag, int(term["amount_usd"]), str(term["boundary"]))
    if tag in OPTIONAL_TERM_ADDRESSES:
        address = term.get(OPTIONAL_TERM_ADDRESSES[tag])
        return (tag, int(address is not None), address or "")
    if tag == "deployer_obligation":
        return (tag, 0, str(term["text"]))
    if tag in FLOW_DOWN_TERMS:
        return (tag, 0, str(term["family"]))
    return (tag, 0, "")


def _validate_term(term: Any, context: str) -> None:
    """Assert one serialized `LicenseTerm` carries exactly the payload its tag declares."""
    if not isinstance(term, dict) or not isinstance(term.get("term"), str):
        raise RuntimeError(f"model-licenses {context} carries a malformed term {term!r}")
    tag = term["term"]
    payload = LICENSE_TERM_PAYLOADS.get(tag)
    if payload is None:
        raise RuntimeError(
            f"model-licenses {context} carries an unknown licence term {tag!r}; the release "
            "validator models every LicenseTerm variant explicitly so a new one cannot be sorted "
            "and deduped as if it had no payload"
        )
    expected = {"term", *payload}
    if set(term) != expected:
        raise RuntimeError(
            f"model-licenses {context} term {tag!r} carries fields {sorted(term)}, expected "
            f"{sorted(expected)}"
        )
    if tag == "revenue_ceiling":
        amount = term["amount_usd"]
        if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
            raise RuntimeError(
                f"model-licenses {context} revenue_ceiling amount_usd must be a whole number of "
                f"US dollars, not {amount!r}"
            )
        if term["boundary"] not in CEILING_BOUNDARIES:
            raise RuntimeError(
                f"model-licenses {context} revenue_ceiling boundary {term['boundary']!r} is not "
                f"one of {list(CEILING_BOUNDARIES)}"
            )
    elif tag in OPTIONAL_TERM_ADDRESSES:
        field = OPTIONAL_TERM_ADDRESSES[tag]
        address = term[field]
        if address is not None and _is_blank(address):
            raise RuntimeError(
                f"model-licenses {context} term {tag!r} carries an empty {field!r}; a licence that "
                "names no address is recorded as null, not as an empty string"
            )
    elif tag == "deployer_obligation" and _is_blank(term["text"]):
        raise RuntimeError(f"model-licenses {context} deployer_obligation has no text")
    elif tag in FLOW_DOWN_TERMS and _is_blank(term["family"]):
        raise RuntimeError(f"model-licenses {context} term {tag!r} names no family")


def _provider_term_union(
    provider: dict[str, Any],
    components_by_key: dict[str, dict[str, Any]],
    families_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Recompute a provider's derived term union — mirroring `license::provider_terms`.

    Two sources feed it: every term of the family each resolved component row normalizes to, and
    `gated_access` for each resolved row whose `gated` is set. A gated row contributes its
    `gated_access` even when its `family` fails to resolve, because gating is a property of the
    checkpoint rather than of the family and losing it to an unrelated defect would under-report.
    A component key that does not resolve contributes nothing (the caller rejects that separately).

    Sorted by `_term_sort_key`, then consecutive duplicates dropped — the Rust `sort_by` + `dedup`
    pair. Two flow-down duties naming different families stay two elements, and two ceilings at the
    same amount with different boundary readings stay two elements: a union that collapsed either
    would show one obligation where the catalog carries several.
    """
    terms: list[dict[str, Any]] = []
    for key in provider["components"]:
        row = components_by_key.get(key)
        if row is None:
            continue
        if row.get("gated") is True:
            terms.append({"term": "gated_access"})
        family = families_by_id.get(row.get("family"))
        if family is not None:
            terms.extend(copy.deepcopy(term) for term in family["terms"])
    terms.sort(key=_term_sort_key)
    union: list[dict[str, Any]] = []
    for term in terms:
        if not union or union[-1] != term:
            union.append(term)
    return union


def validate_model_weight_licenses(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Assert a schema-3 model-weight-licence manifest is present, complete and self-consistent.

    Three fact sections. ``families`` are the reviewed upstream licence texts; ``components`` are the
    loaded artifacts, each pointing at a family and carrying its own provenance (``source_url`` /
    ``declared`` / ``retrieved``); ``providers`` map a registry id onto component keys and carry the
    **derived** term union. Keys are unique within each section, and no section may be empty.

    This is the Python mirror of ``license_table_conformance_errors`` — it runs the same rules the
    Rust catalog ship-gate runs, over the *merged* table (see :func:`merge_model_weight_licenses`),
    which is where a collision between two catalogs first becomes visible at all. Per section:

    * **families** — ids non-blank and unique; ``spdx_id`` / ``name`` / ``text_url`` populated; terms
      state facts about the *text* only, so ``gated_access`` is rejected here and belongs on the
      component row; a flow-down term names its own family, so the text a consumer resolves through
      it is the text that imposed the duty; and an optional address is ``null`` or a real address,
      never a blank string.
    * **components** — keys non-blank and unique; ``source_url`` / ``declared`` populated; every
      ``family`` resolves to a family row **in the same document**; ``retrieved`` is a real ISO
      calendar date; and a family imposing ``attribution_required`` implies a non-blank
      ``attribution``.
    * **providers** — ids non-blank and unique; at least one component, none listed twice, all
      resolving; and the emitted ``terms`` array equals the union recomputed here from the component
      and family rows.

    That last check is the load-bearing one. The derived view is what a consumer joins over, and
    nothing else in the document would notice if it stopped agreeing with the rows it claims to
    summarize — a hand edit, a stale emitter, or a merge that took a term from the wrong catalog all
    look identical to a structural check. :func:`_provider_term_union` therefore mirrors
    ``provider_terms`` exactly, down to sort order and dedup semantics.

    **Disclosure only.** Nothing checked here gates, withholds or degrades anything at runtime. This
    is a build-time completeness check on metadata we author ourselves: it fails *our* release when
    *our* licence table is malformed, so that a consumer is never handed an incomplete or
    self-contradicting set of facts to display. It never decides anything from a licence term.

    Raises ``RuntimeError`` on the first problem found. Shared by the builder (emit-time) and the
    verifier (bundle-time) so the two cannot drift.

    Known and accepted loss, recorded so it is not rediscovered as a surprise: schema 2 carried a
    free-text ``restriction`` string per row and schema 3 has no field for it, by decision in
    sc-16661/16662 rather than by oversight. Almost all of that prose was legal *conclusion*
    ("Non-commercial only", "SceneWorks is non-commercial, so the weights are usable"), which the
    disclosure-only rule excludes on purpose and which the typed terms now express as facts. One
    sentence was genuine disclosure and is not represented anywhere in schema 3: MMAudio's note that
    its checkpoints were trained on AudioSet/VGGSound/Freesound/AudioCaps/WavCaps, whose dataset
    terms a downstream user must honour. That is training-data provenance, a different axis from the
    licence of the artifact, so it does not belong in an existing ``LicenseTerm``. If a consumer is
    ever shown to need it, it wants its own field and its own story — do not smuggle it back as
    free text.
    """
    if document.get("kind") != "model-weight-licenses":
        raise RuntimeError("model-licenses manifest has the wrong kind")
    schema = document.get("schema_version")
    if schema != MODEL_LICENSES_SCHEMA:
        raise RuntimeError(
            f"model-licenses manifest is schema {schema!r}, expected {MODEL_LICENSES_SCHEMA}"
        )

    def section(name: str) -> list[dict[str, Any]]:
        rows = document.get(name)
        if not isinstance(rows, list) or not rows:
            raise RuntimeError(f"model-licenses manifest lists no {name}")
        return rows

    def require(rows: list[dict[str, Any]], fields: tuple[str, ...], label: str, key: str) -> None:
        seen: set[str] = set()
        for row in rows:
            for field in fields:
                if _is_blank(row.get(field)):
                    raise RuntimeError(
                        f"model-licenses {label} {row.get(key)!r} is missing {field!r}"
                    )
            if "commercial_use" in row:
                raise RuntimeError(
                    f"model-licenses {label} {row[key]!r} carries commercial_use, a schema-2 legal "
                    "conclusion removed in schema 3"
                )
            identifier = row[key]
            if identifier in seen:
                raise RuntimeError(f"duplicate model-licenses {label} {identifier!r}")
            seen.add(identifier)

    families = section("families")
    components = section("components")
    providers = section("providers")
    require(families, REQUIRED_FAMILY_FIELDS, "family", "id")
    require(components, REQUIRED_COMPONENT_FIELDS, "component", "component")
    require(providers, ("provider_id",), "provider", "provider_id")

    families_by_id: dict[str, dict[str, Any]] = {}
    for family in families:
        identifier = family["id"]
        terms = family.get("terms")
        if not isinstance(terms, list):
            raise RuntimeError(f"model-licenses family {identifier!r} has no terms array")
        for term in terms:
            _validate_term(term, f"family {identifier!r}")
            if term["term"] == "gated_access":
                raise RuntimeError(
                    f"model-licenses family {identifier!r} declares gated_access, a per-checkpoint "
                    "distribution fact recorded on the component row, not a licence term"
                )
            if term["term"] in FLOW_DOWN_TERMS and term["family"] != identifier:
                raise RuntimeError(
                    f"model-licenses family {identifier!r} declares a {term['term']} term naming "
                    f"{term['family']!r}; a flow-down term must name its own family so the text it "
                    "points at is the text that imposed the duty"
                )
        families_by_id[identifier] = family

    components_by_key: dict[str, dict[str, Any]] = {}
    for row in components:
        key = row["component"]
        if not isinstance(row.get("gated"), bool):
            raise RuntimeError(f"model-licenses component {key!r} gated must be boolean")
        if not _is_iso_date(row["retrieved"]):
            raise RuntimeError(
                f"model-licenses component {key!r} has a non-ISO retrieved date "
                f"{row['retrieved']!r}; expected a real YYYY-MM-DD calendar date"
            )
        # `Some("")` is not an attribution: it satisfies nothing, and treating it as present would
        # let an obligation ship unrecorded behind a placeholder. Absent is spelled `null`.
        attribution = row.get("attribution")
        if attribution is not None and _is_blank(attribution):
            raise RuntimeError(
                f"model-licenses component {key!r} records an empty attribution string; an absent "
                "attribution is recorded as null"
            )
        family = families_by_id.get(row["family"])
        if family is None:
            raise RuntimeError(
                f"model-licenses component {key!r} references unknown licence family "
                f"{row['family']!r}"
            )
        requires_attribution = any(
            term["term"] == "attribution_required" for term in family["terms"]
        )
        if requires_attribution and attribution is None:
            raise RuntimeError(
                f"model-licenses component {key!r} resolves to {family['id']!r}, which requires "
                "attribution, but records none"
            )
        components_by_key[key] = row

    for provider in providers:
        identifier = provider["provider_id"]
        keys = provider.get("components")
        if not isinstance(keys, list) or not keys:
            raise RuntimeError(f"model-licenses provider {identifier!r} maps to no components")
        seen_keys: set[str] = set()
        for key in keys:
            if _is_blank(key):
                raise RuntimeError(
                    f"model-licenses provider {identifier!r} lists an empty component key"
                )
            if key in seen_keys:
                raise RuntimeError(
                    f"model-licenses provider {identifier!r} lists component {key!r} twice"
                )
            seen_keys.add(key)
            if key not in components_by_key:
                raise RuntimeError(
                    f"model-licenses provider {identifier!r} references unknown component {key!r}"
                )
        emitted = provider.get("terms")
        if not isinstance(emitted, list):
            raise RuntimeError(
                f"model-licenses provider {identifier!r} has no derived terms array"
            )
        for term in emitted:
            _validate_term(term, f"provider {identifier!r}")
        expected = _provider_term_union(provider, components_by_key, families_by_id)
        if emitted != expected:
            raise RuntimeError(
                f"model-licenses provider {identifier!r} carries a derived term union that does "
                "not match its components; the union is derived, never hand-authored\n"
                f"  emitted:    {json.dumps(emitted, ensure_ascii=False)}\n"
                f"  recomputed: {json.dumps(expected, ensure_ascii=False)}"
            )
    return providers


def render_model_licenses(document: dict[str, Any]) -> str:
    """Serialize a merged manifest exactly as ``component_licenses_manifest_json`` would.

    Two-space indent, keys in insertion order (**not** sorted — the section and field order is the
    Rust emitter's and a consumer reads it), non-ASCII left as itself, trailing newline. Round-trips
    the committed audio manifest byte for byte, which is what lets the aggregate stay comparable to
    a single catalog's output.
    """
    return json.dumps(document, indent=2, ensure_ascii=False) + "\n"


def _row_disagreement(
    label: str,
    key: str,
    first_source: str,
    first_row: dict[str, Any],
    second_source: str,
    second_row: dict[str, Any],
) -> RuntimeError:
    """The error two catalogs recording one key differently produces — with both rows in full.

    A bare "mismatch" would leave whoever hits this diffing two multi-hundred-row JSON files by
    hand, so the message carries the field names that differ *and* both rows as the catalogs emitted
    them. Which one is wrong is a question about the upstream licence, not one this tool can answer.
    """
    absent = object()
    differing = sorted(
        field
        for field in set(first_row) | set(second_row)
        if first_row.get(field, absent) != second_row.get(field, absent)
    )
    return RuntimeError(
        f"model-licenses {label} {first_row[key]!r} is recorded differently by two catalogs; a "
        f"{label} row describes the artifact itself, so every catalog that loads it must record "
        f"the same row. Differing fields: {', '.join(differing)}\n"
        f"  {first_source}:\n"
        f"    {json.dumps(first_row, ensure_ascii=False, sort_keys=True)}\n"
        f"  {second_source}:\n"
        f"    {json.dumps(second_row, ensure_ascii=False, sort_keys=True)}"
    )


def merge_model_weight_licenses(
    sources: list[tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    """Aggregate one schema-3 manifest per catalog into the single document the bundle ships.

    The bundle carries **one** file so a consumer reads one thing for its end-product licences page,
    but the table is produced by three composition roots — the Candle audio catalog and the two
    media catalogs (sc-16665/16666/16667). Rows repeat across them by design: a licence is a
    property of the checkpoint, and MLX and Candle load the same checkpoints, so the shared
    component table is authored once and both media catalogs point at it.

    A key appearing in more than one source must therefore carry an identical row, and disagreement
    fails the build rather than picking a winner — a first-wins merge would make the shipped licence
    facts a function of filename order. See :func:`_row_disagreement` for the message.

    The rule is applied to all three sections, not only components. Families repeat the most (every
    catalog carrying an Apache-2.0 row carries the same family), and a provider id registered by two
    catalogs against different components is a genuine registry collision — one that is invisible
    until the merge, which is why the conformance pass runs over the merged table and not over each
    manifest in turn.

    Equality is by content, not by byte: JSON object key order is not meaningful, and both media
    manifests come out of the same Rust emitter anyway.

    Sections are sorted by key on the way out, matching the Rust emitter, so the aggregate is
    deterministic regardless of which order the catalogs were discovered in.
    """
    if not sources:
        raise RuntimeError("no model-weight-licence manifest was found")

    names = [name for name, _, _ in MODEL_LICENSES_SECTIONS]
    merged: dict[str, dict[str, dict[str, Any]]] = {name: {} for name in names}
    origin: dict[str, dict[str, str]] = {name: {} for name in names}

    for source, document in sources:
        if not isinstance(document, dict) or document.get("kind") != "model-weight-licenses":
            raise RuntimeError(f"{source} is not a model-licenses manifest")
        schema = document.get("schema_version")
        if schema != MODEL_LICENSES_SCHEMA:
            raise RuntimeError(
                f"{source} is schema {schema!r}, expected {MODEL_LICENSES_SCHEMA}"
            )
        for name, key, label in MODEL_LICENSES_SECTIONS:
            rows = document.get(name)
            # An empty section is legal in a *source*: a media catalog may add provider rows against
            # families the shared table already carries. The merged table is what may not be empty,
            # and `validate_model_weight_licenses` checks that.
            if not isinstance(rows, list):
                raise RuntimeError(f"{source} lists no {name}")
            for row in rows:
                if not isinstance(row, dict) or _is_blank(row.get(key)):
                    raise RuntimeError(f"{source} carries a {label} row with no {key!r}")
                identifier = row[key]
                previous = merged[name].get(identifier)
                if previous is None:
                    merged[name][identifier] = row
                    origin[name][identifier] = source
                elif previous != row:
                    raise _row_disagreement(
                        label, key, origin[name][identifier], previous, source, row
                    )

    document = {"schema_version": MODEL_LICENSES_SCHEMA, "kind": "model-weight-licenses"}
    for name, _, _ in MODEL_LICENSES_SECTIONS:
        document[name] = [merged[name][key] for key in sorted(merged[name])]
    return document


def _tracked_release_manifests(root: Path) -> set[Path]:
    """Every path git has under version control in the licence-manifest directory.

    Asked of git rather than derived from the shape, so this function contributes no second opinion
    about *which* filenames count — `MODEL_LICENSES_GLOB` stays the only rule for that.
    """
    try:
        listing = str(run("git", "ls-files", "-z", "--", MODEL_LICENSES_DIRECTORY, cwd=root))
    except (subprocess.CalledProcessError, FileNotFoundError, NotADirectoryError) as error:
        raise RuntimeError(
            f"model-weight-license discovery needs a git checkout at {root}: the manifests that "
            "ship must be the manifests the source archive carries, which only git can answer"
        ) from error
    return {root / name for name in listing.split("\0") if name}


def discover_model_license_sources(root: Path) -> list[Path]:
    """Every committed catalog licence manifest, in a stable order.

    Discovery is by filename shape (``MODEL_LICENSES_GLOB``) rather than by an enumerated list, so a
    catalog that lands its manifest is aggregated with no edit to this module. That is the safe
    direction: an enumerated list that did not predict a filename would *skip* that catalog, and a
    release whose licences page is missing an entire backend would pass every gate. Landing a
    manifest is therefore exactly "commit the file" (sc-16665/16666/16667) — no tooling edit.

    **Committed**, not merely present. The shape match is intersected with git's tracked set because
    the two other statements the release makes about these rows are both answered by git and neither
    can see an untracked file: the dirty gate runs ``git status --untracked-files=no``, and the
    source archive comes from ``git archive <revision>``. An untracked
    ``release/model-weight-licenses-media.json`` merged off the filesystem would put licence rows in
    the shipped table that appear in no commit and in no archive, while ``release.dirty`` still read
    ``false`` and ``release.revision`` named a commit that does not contain them. For a document
    whose entire purpose is provenance, rows with no provenance are the wrong failure — and a
    ``git mergetool`` leftover such as ``model-weight-licenses_BACKUP_123.json`` matches the shape
    too, which is how stale rows would arrive without anyone choosing them.

    An untracked match is refused rather than skipped, because silently dropping it is the same
    missing-a-whole-backend release the glob exists to prevent — the operator must decide. ``git
    add`` is the whole escape hatch: staging makes the file tracked, and the dirty gate then reports
    it, so the build refuses until ``--allow-dirty`` is passed and ``release.dirty`` records the
    truth. Either route leaves the manifest describing what actually shipped.

    The audio manifest is required because it is committed today; the media manifests are not yet,
    and their absence must not fail a release (sc-16665/16666/16667).
    """
    matches = sorted(
        (root / MODEL_LICENSES_DIRECTORY).glob(MODEL_LICENSES_GLOB), key=lambda path: path.name
    )
    tracked = _tracked_release_manifests(root)
    untracked = [path for path in matches if path not in tracked]
    if untracked:
        names = ", ".join(path.relative_to(root).as_posix() for path in untracked)
        raise RuntimeError(
            f"model-weight-license manifest is untracked: {names}; the shipped licence table is "
            "built only from files git carries, because the release's dirty gate and source "
            "archive cannot see an untracked file and would describe a table it does not contain. "
            "Commit it (or `git add` it and pass --allow-dirty) to ship those rows, or delete it "
            "if it is a leftover"
        )
    # Every match is tracked past that raise, so the glob's own order is the merge order.
    paths = matches
    if (root / MODEL_LICENSES_SOURCE) not in paths:
        raise RuntimeError(f"model-weight-license manifest is absent: {MODEL_LICENSES_SOURCE}")
    return paths


def load_model_weight_licenses(root: Path) -> tuple[dict[str, Any], list[str]]:
    """Discover, merge and validate every catalog licence manifest under `root`.

    Returns the merged document and the source paths it came from. The single entry point for both
    the builder and any caller that needs the aggregate, so discovery, merge and conformance cannot
    be run in a different combination somewhere else.
    """
    sources: list[tuple[str, dict[str, Any]]] = []
    for path in discover_model_license_sources(root):
        label = path.relative_to(root).as_posix()
        try:
            sources.append((label, json.loads(path.read_text(encoding="utf-8"))))
        except json.JSONDecodeError as error:
            raise RuntimeError(f"{label} is not valid JSON: {error}") from error
    document = merge_model_weight_licenses(sources)
    validate_model_weight_licenses(document)
    return document, [label for label, _ in sources]


def run(*args: str, cwd: Path, binary: bool = False) -> bytes | str:
    # git and cargo emit UTF-8 on every platform; decoding with the locale encoding instead
    # fails on a non-UTF-8 host such as Windows (cp1252) over non-ASCII package metadata.
    result = subprocess.run(
        list(args),
        check=True,
        cwd=cwd,
        capture_output=True,
        text=not binary,
        encoding=None if binary else "utf-8",
    )
    return result.stdout


def validate_tag(tag: str) -> re.Match[str]:
    match = TAG_PATTERN.fullmatch(tag)
    if not match:
        raise ValueError(
            "release tag must match runtime-YYYY.MM.patch or "
            "runtime-YYYY.MM.patch-rc.N"
        )
    month = int(match.group(2))
    if month not in range(1, 13):
        raise ValueError("release tag month must be between 01 and 12")
    return match


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def spdx_id(name: str, version: str, source: str | None) -> str:
    identity = f"{name}\0{version}\0{source or 'workspace'}".encode()
    suffix = hashlib.sha256(identity).hexdigest()[:12]
    safe_name = re.sub(r"[^A-Za-z0-9.-]", "-", name)
    return f"SPDXRef-Package-{safe_name}-{suffix}"


def source_download_location(package: dict[str, Any]) -> str:
    source = package.get("source")
    if not source:
        return "NOASSERTION"
    if source.startswith("registry+"):
        return (
            f"https://crates.io/api/v1/crates/{quote(package['name'])}/"
            f"{quote(package['version'])}/download"
        )
    if source.startswith("git+"):
        return source.removeprefix("git+").split("#", 1)[0]
    return "NOASSERTION"


def package_key(package: dict[str, Any]) -> tuple[str, str, str | None]:
    return package["name"], package["version"], package.get("source")


def resolve_lock_dependency(
    dependency: str, packages_by_name: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    match = re.fullmatch(r"(\S+)(?: (\S+)(?: \((.+)\))?)?", dependency)
    if not match:
        raise RuntimeError(f"cannot parse Cargo.lock dependency {dependency!r}")
    name, version, source = match.groups()
    candidates = packages_by_name.get(name, [])
    if version:
        candidates = [package for package in candidates if package["version"] == version]
    if source:
        candidates = [package for package in candidates if package.get("source") == source]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Cargo.lock dependency {dependency!r} resolved to {len(candidates)} packages"
        )
    return candidates[0]


def build_spdx(
    *,
    tag: str,
    revision: str,
    created: str,
    metadata: dict[str, Any],
    lock: dict[str, Any],
) -> dict[str, Any]:
    metadata_packages = {
        (package["name"], package["version"], package.get("source")): package
        for package in metadata["packages"]
    }
    packages = sorted(
        lock.get("package", []),
        key=lambda package: (
            package["name"],
            package["version"],
            package.get("source") or "",
        ),
    )
    metadata_ids = {
        package["id"]: spdx_id(*package_key(package)) for package in metadata["packages"]
    }

    spdx_packages: list[dict[str, Any]] = []
    for package in packages:
        key = (package["name"], package["version"], package.get("source"))
        metadata_package = metadata_packages.get(key, {})
        item: dict[str, Any] = {
            "name": package["name"],
            "SPDXID": spdx_id(*key),
            "versionInfo": package["version"],
            "downloadLocation": source_download_location(package),
            "filesAnalyzed": False,
            "licenseConcluded": "NOASSERTION",
            "licenseDeclared": metadata_package.get("license") or "NOASSERTION",
            "copyrightText": "NOASSERTION",
        }
        checksum = package.get("checksum")
        if checksum:
            item["checksums"] = [{"algorithm": "SHA256", "checksumValue": checksum}]
        spdx_packages.append(item)

    relationships = [
        {
            "spdxElementId": "SPDXRef-DOCUMENT",
            "relationshipType": "DESCRIBES",
            "relatedSpdxElement": metadata_ids[member],
        }
        for member in sorted(metadata["workspace_members"])
    ]
    packages_by_name: dict[str, list[dict[str, Any]]] = {}
    for package in packages:
        packages_by_name.setdefault(package["name"], []).append(package)
    for package in packages:
        for dependency in sorted(package.get("dependencies", [])):
            resolved = resolve_lock_dependency(dependency, packages_by_name)
            relationships.append(
                {
                    "spdxElementId": spdx_id(*package_key(package)),
                    "relationshipType": "DEPENDS_ON",
                    "relatedSpdxElement": spdx_id(*package_key(resolved)),
                }
            )

    return {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": f"SceneWorks inference {tag}",
        "documentNamespace": f"{REPOSITORY}/spdx/{tag}/{revision}",
        "creationInfo": {"created": created, "creators": [f"Tool: {TOOL_NAME}"]},
        "packages": spdx_packages,
        "relationships": relationships,
    }


def workspace_packages(metadata: dict[str, Any], root: Path) -> list[dict[str, str]]:
    members = set(metadata["workspace_members"])
    packages = []
    for package in metadata["packages"]:
        if package["id"] not in members:
            continue
        manifest = Path(package["manifest_path"])
        packages.append(
            {
                "name": package["name"],
                "version": package["version"],
                "license": package.get("license") or "NOASSERTION",
                "manifest": manifest.relative_to(root).as_posix(),
            }
        )
    return sorted(packages, key=lambda package: package["name"])


def backend_pin(metadata: dict[str, Any], names: set[str]) -> dict[str, Any]:
    matches = [package for package in metadata["packages"] if package["name"] in names]
    sources = sorted({package.get("source") for package in matches if package.get("source")})
    if len(sources) != 1:
        raise RuntimeError(f"expected one source for {sorted(names)}, found {sources}")
    source = sources[0]
    revision = source.rsplit("#", 1)[1] if "#" in source else None
    return {
        "packages": sorted(package["name"] for package in matches),
        "source": source,
        "revision": revision,
    }


def gzip_archive(tar_content: bytes, timestamp: int) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=timestamp) as archive:
        archive.write(tar_content)
    return output.getvalue()


def write_json(path: Path, content: dict[str, Any]) -> None:
    path.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_release(
    *,
    root: Path,
    output: Path,
    tag: str,
    source_ref: str,
    offline: bool,
    allow_dirty: bool,
    skip_archive: bool,
) -> dict[str, Any]:
    match = validate_tag(tag)
    root = root.resolve()
    output.mkdir(parents=True, exist_ok=True)

    status = str(run("git", "status", "--porcelain", "--untracked-files=no", cwd=root))
    dirty = bool(status.strip())
    if dirty and not allow_dirty:
        raise RuntimeError("release inputs are dirty; commit them or pass --allow-dirty")

    revision = str(run("git", "rev-parse", f"{source_ref}^{{commit}}", cwd=root)).strip()
    timestamp = int(str(run("git", "show", "-s", "--format=%ct", revision, cwd=root)).strip())
    created = dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )

    metadata_command = ["cargo", "metadata", "--locked", "--format-version", "1"]
    if offline:
        metadata_command.append("--offline")
    metadata = json.loads(str(run(*metadata_command, cwd=root)))
    lock_path = root / "Cargo.lock"
    lock_bytes = lock_path.read_bytes()
    lock = tomllib.loads(lock_bytes.decode())
    toolchain = tomllib.loads((root / "rust-toolchain.toml").read_text(encoding="utf-8"))

    sbom_name = f"{tag}.spdx.json"
    sbom_path = output / sbom_name
    spdx = build_spdx(
        tag=tag, revision=revision, created=created, metadata=metadata, lock=lock
    )
    write_json(sbom_path, spdx)

    # Emit the model-weight-license manifest beside the SBOM (sc-13332, aggregated since sc-16664).
    # Every committed catalog manifest is merged into the one file a consumer reads, and the merged
    # table is validated here — a component two catalogs disagree about, an unresolvable family, a
    # provider-id collision that only exists once the catalogs are combined, or a derived term union
    # that no longer matches its components all fail the build.
    #
    # Rendered rather than byte-copied: an aggregate has no single source file to copy, and rendering
    # also makes the artifact independent of the builder's checkout line endings, which a verbatim
    # copy of a text file was not. For a single source the rendered bytes are the committed bytes.
    licenses_document, licenses_sources = load_model_weight_licenses(root)
    print(f"model-weight licences: merged {', '.join(licenses_sources)}")
    model_licenses_name = f"{tag}.model-licenses.json"
    model_licenses_path = output / model_licenses_name
    model_licenses_path.write_bytes(render_model_licenses(licenses_document).encode("utf-8"))

    artifacts = [
        {
            "name": sbom_name,
            "kind": "spdx-2.3-json",
            "sha256": sha256_file(sbom_path),
            "bytes": sbom_path.stat().st_size,
        },
        {
            "name": model_licenses_name,
            "kind": MODEL_LICENSES_KIND,
            "sha256": sha256_file(model_licenses_path),
            "bytes": model_licenses_path.stat().st_size,
        },
    ]
    if not skip_archive:
        archive_name = f"{tag}.tar.gz"
        archive_path = output / archive_name
        tar_content = run(
            "git",
            "archive",
            "--format=tar",
            f"--prefix=inference-{tag}/",
            revision,
            cwd=root,
            binary=True,
        )
        assert isinstance(tar_content, bytes)
        archive_path.write_bytes(gzip_archive(tar_content, timestamp))
        artifacts.insert(
            0,
            {
                "name": archive_name,
                "kind": "source-tar-gzip",
                "sha256": sha256_file(archive_path),
                "bytes": archive_path.stat().st_size,
            },
        )

    packages = workspace_packages(metadata, root)
    manifest = {
        "schema_version": 1,
        "release": {
            "tag": tag,
            "candidate": match.group(4) is not None,
            "repository": REPOSITORY,
            "revision": revision,
            "source_date": created,
            "dirty": dirty,
        },
        "toolchain": {
            "rust": toolchain["toolchain"]["channel"],
            "profile": toolchain["toolchain"].get("profile"),
            "components": sorted(toolchain["toolchain"].get("components", [])),
        },
        "lockfile": {
            "version": lock["version"],
            "sha256": sha256_bytes(lock_bytes),
            "package_count": len(lock.get("package", [])),
        },
        "workspace": {"package_count": len(packages), "packages": packages},
        "backends": {
            "mlx": backend_pin(metadata, {"pmetal-mlx-rs", "pmetal-mlx-sys"}),
            "candle": backend_pin(
                metadata,
                {"candle-core", "candle-nn", "candle-transformers", "candle-flash-attn"},
            ),
        },
        "artifacts": artifacts,
    }
    manifest_path = output / "runtime-manifest.json"
    write_json(manifest_path, manifest)

    checksum_paths = [manifest_path, *[output / artifact["name"] for artifact in artifacts]]
    checksums = "".join(
        f"{sha256_file(path)}  {path.name}\n" for path in sorted(checksum_paths)
    )
    (output / "SHA256SUMS").write_text(checksums, encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--source-ref", default="HEAD")
    parser.add_argument("--output", type=Path, default=Path("dist/release"))
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--skip-archive", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    manifest = build_release(
        root=root,
        output=args.output,
        tag=args.tag,
        source_ref=args.source_ref,
        offline=args.offline,
        allow_dirty=args.allow_dirty,
        skip_archive=args.skip_archive,
    )
    print(
        f"release bundle: {manifest['release']['tag']} "
        f"({manifest['workspace']['package_count']} workspace packages, "
        f"{manifest['lockfile']['package_count']} locked packages)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
