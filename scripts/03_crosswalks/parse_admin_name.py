"""
======================================================================

parse_admin_name.py

Purpose
-------
Config-driven parsing of the FEWS composite admin_name string into
canonical hierarchy / livelihood-zone columns, per country.

Replaces a fixed `parts.columns = [...]` assignment, which assumes a
single component count and crashes on countries (Kenya, Somalia) that
carry several formats in one dataset.

Two classes of format are handled:

1. UNAMBIGUOUS - a component count maps to exactly one field layout.
   Ethiopia (4) and Kenya (4) are of this kind.

2. AMBIGUOUS - a component count maps to SEVERAL candidate layouts
   that are positionally identical and can only be told apart by
   testing field content against reference vocabularies built from
   the FSC geography. Kenya's 5-component records are of this kind:

       Format 2: "Central Highlands, High Potential Zone, Buuri, Meru, Kenya"
                 -> fields 0+1 REJOIN into one LZNAME
       Format 3: "Eastern Pastoral Zone, Liboi, Dadaab, Garissa, Kenya"
                 -> field 0 is LZNAME, field 1 is ADMIN3

   The classifier tests each candidate and requires EXACTLY ONE match.
   Zero matches or multiple matches raise, with samples. This is
   deliberate: the "exactly one" property was verified empirically on
   the Feb-2026 snapshot (0 unclassified, 0 ambiguous of 19,965), but
   that proof is dated. The World Bank file changes; this assertion is
   what converts a future change into a loud failure rather than
   silently dropped rows.

CRITICAL - comma handling
-------------------------
Format 2 rejoins its two LZ fields with ", " (comma-space) because the
FSC LZNAME value literally contains a comma:

    "central highlands, high potential zone"

canonicalize_text() deliberately does NOT strip commas. If a future
edit adds "," to the punctuation replacements in canonicalize.py, all
Format 2 rows will silently stop matching. Do not remove comma
handling from the canonicalizer.

Author:
Robert Odingo MSc Thesis

======================================================================
"""

from __future__ import annotations

import pandas as pd

from canonicalize import canonicalize_text


# ---------------------------------------------------------------------
# Reference vocabularies
# ---------------------------------------------------------------------

def build_fsc_vocabularies(
    fsc: pd.DataFrame,
    fields: tuple[str, ...] = ("LZNAME", "ADMIN3"),
) -> dict[str, set[str]]:
    """
    Build canonical lookup vocabularies from an ALREADY-CANONICALIZED
    FSC frame. Used by the five-component classifier.

    The empty string is excluded from every vocabulary: "" represents
    a missing value, and allowing it to match would let a blank field
    satisfy a pattern it should not.
    """

    vocab = {}

    for field in fields:
        if field not in fsc.columns:
            vocab[field] = set()
            continue
        values = set(fsc[field].astype(str))
        values.discard("")
        vocab[field] = values

    return vocab


# ---------------------------------------------------------------------
# Candidate evaluation
# ---------------------------------------------------------------------

def _candidate_fields(parts: list[str], spec: dict) -> dict[str, str]:
    """
    Apply one candidate layout to a split row, returning a field->value
    mapping with any rejoined fields assembled.
    """

    layout = spec["layout"]

    fields = {
        label: canonicalize_text(parts[i])
        for i, label in enumerate(layout)
    }

    # Rejoin split fields, e.g. LZNAME = LZNAME_A + ", " + LZNAME_B.
    # The comma is load-bearing: FSC LZNAME contains one.
    for target, sources in spec.get("rejoin", {}).items():
        raw = ", ".join(parts[layout.index(s)].strip() for s in sources)
        fields[target] = canonicalize_text(raw)

    return fields


def _candidate_matches(fields: dict, spec: dict, vocab: dict) -> bool:
    """
    Does this candidate layout produce field values consistent with the
    FSC reference vocabularies?

    A candidate matches when every vocabulary-checkable field it
    produces is present in the corresponding vocabulary. LZNAME is
    always checked. ADMIN3 is checked only when the layout produces it.
    """

    lzname = fields.get("LZNAME", "")
    if lzname not in vocab.get("LZNAME", set()):
        return False

    if "ADMIN3" in spec["layout"]:
        admin3 = fields.get("ADMIN3", "")
        if admin3 not in vocab.get("ADMIN3", set()):
            return False

    return True


def classify_five_component(
    parts: list[str],
    candidates: dict,
    vocab: dict,
) -> tuple[str, dict]:
    """
    Classify one ambiguous row. Returns (candidate_name, fields).

    Raises ValueError if zero or more than one candidate matches.
    """

    hits = []

    for name, spec in candidates.items():
        fields = _candidate_fields(parts, spec)
        if _candidate_matches(fields, spec, vocab):
            hits.append((name, fields))

    if len(hits) != 1:
        raise ValueError(
            f"Expected exactly 1 matching format, got {len(hits)} "
            f"({[h[0] for h in hits]}) for row {parts!r}"
        )

    return hits[0]


# ---------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------

def parse_fews_admin_name(
    fews: pd.DataFrame,
    country_config: dict,
    fsc_canonical: pd.DataFrame | None = None,
    admin_name_col: str = "admin_name",
    report: bool = True,
) -> pd.DataFrame:
    """
    Parse admin_name into canonical field columns per the country's
    config. Returns a copy of `fews` with one column per join-key field.

    fsc_canonical
        An already-canonicalized FSC frame. REQUIRED when the country's
        config sets requires_classifier=True (Kenya, Somalia), since the
        classifier needs the reference vocabularies. Ignored otherwise.

    Raises on any component count not declared in the config, and on
    any ambiguous row that does not match exactly one candidate layout.
    """

    fews = fews.copy()

    unambiguous = country_config.get("admin_name_formats", {})
    ambiguous = country_config.get("ambiguous_formats", {})
    needs_classifier = country_config.get("requires_classifier", False)

    if needs_classifier and fsc_canonical is None:
        raise ValueError(
            "This country's config requires the vocabulary classifier, "
            "so fsc_canonical must be supplied."
        )

    vocab = (
        build_fsc_vocabularies(fsc_canonical)
        if fsc_canonical is not None
        else {}
    )

    split = fews[admin_name_col].str.split(",")
    counts = split.apply(len)

    right_anchored = country_config.get("right_anchored_format")

    declared = set(unambiguous) | set(ambiguous)

    if right_anchored:
        # A right-anchored country parses from the END of the string:
        # fixed trailing fields, with everything remaining at the front
        # rejoined into the leading field. Any component count at or
        # above the minimum is therefore valid, so no count is
        # "undeclared" provided it is long enough.
        min_len = len(right_anchored["trailing"]) + 1
        undeclared = {c for c in counts.unique() if c < min_len}
    else:
        undeclared = set(counts.unique()) - declared

    if undeclared:
        examples = {
            int(c): fews.loc[counts == c, admin_name_col]
                        .drop_duplicates().head(3).tolist()
            for c in sorted(undeclared)
        }
        raise ValueError(
            f"admin_name has undeclared component counts "
            f"{sorted(int(c) for c in undeclared)}. "
            f"Declared: {sorted(declared)}. Samples: {examples}"
        )

    # Output columns: every join-key field.
    key_fields = country_config["join_key"]
    for field in key_fields:
        fews[field] = ""

    tally = {}

    # --- right-anchored parsing (whole country) ---
    #
    # Used where the LEADING field may itself contain the delimiter
    # (Somalia livelihood-zone names contain commas). Parsing from the
    # right is unambiguous: the trailing fields are fixed-position, and
    # everything before them rejoins into the leading field.
    if right_anchored:
        trailing = right_anchored["trailing"]   # e.g. [ADMIN2, ADMIN1, COUNTRY]
        leading = right_anchored["leading"]     # e.g. "LZNAME"
        n_trail = len(trailing)

        def _parse_right(parts):
            out = {}
            for offset, label in enumerate(reversed(trailing)):
                out[label] = canonicalize_text(parts[-(offset + 1)])
            out[leading] = canonicalize_text(
                ", ".join(x.strip() for x in parts[:-n_trail])
            )
            return out

        fieldmaps = split.apply(_parse_right)

        for field in key_fields:
            fews[field] = fieldmaps.apply(lambda d: d.get(field, "")).values

        for c, n in counts.value_counts().items():
            tally[f"{int(c)}-component (right-anchored)"] = int(n)

        if report:
            print("\nParse tally (unit-months by format)")
            print("-----------------------------------")
            for k in sorted(tally):
                print(f"{k:32s} : {tally[k]:,}")
            print(f"{'TOTAL':32s} : {sum(tally.values()):,}")

        return fews

    # --- unambiguous counts ---
    for count, layout in unambiguous.items():
        rows = counts == count
        if not rows.any():
            continue

        for field in key_fields:
            if field in layout:
                idx = layout.index(field)
                fews.loc[rows, field] = (
                    split[rows].apply(lambda p: canonicalize_text(p[idx])).values
                )
            # fields absent from this layout stay "" (e.g. ADMIN3)

        tally[f"{count}-component"] = int(rows.sum())

    # --- ambiguous counts: classify row by row ---
    for count, candidates in ambiguous.items():
        rows = counts == count
        if not rows.any():
            continue

        classified = split[rows].apply(
            lambda p: classify_five_component(p, candidates, vocab)
        )

        names = classified.apply(lambda t: t[0])
        fieldmaps = classified.apply(lambda t: t[1])

        for field in key_fields:
            fews.loc[rows, field] = fieldmaps.apply(
                lambda d: d.get(field, "")
            ).values

        for name, n in names.value_counts().items():
            tally[f"{count}-component/{name}"] = int(n)

    if report:
        print("\nParse tally (unit-months by format)")
        print("-----------------------------------")
        for k in sorted(tally):
            print(f"{k:32s} : {tally[k]:,}")
        print(f"{'TOTAL':32s} : {sum(tally.values()):,}")

    return fews