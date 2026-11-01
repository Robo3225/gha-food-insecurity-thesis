from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process


# =====================================================================
# Configuration
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve()

while PROJECT_ROOT.name != "gee_project":
    if PROJECT_ROOT.parent == PROJECT_ROOT:
        raise RuntimeError(
            "Could not locate project root 'gee_project'."
        )
    PROJECT_ROOT = PROJECT_ROOT.parent

INPUT_DIR = PROJECT_ROOT / "outputs" / "diagnostics"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "diagnostics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================================
# Helper functions
# =====================================================================

def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def validate_columns(df: pd.DataFrame, required: list[str], name: str):
    missing = sorted(set(required) - set(df.columns))

    if missing:
        raise ValueError(
            f"{name} is missing required columns: {missing}"
        )


def hierarchy_string(df: pd.DataFrame) -> pd.Series:
    """
    Build a standardized hierarchy string for fuzzy comparison.
    """

    return (
        df["ADMIN1"].str.strip()
        + " | "
        + df["ADMIN2"].str.strip()
        + " | "
        + df["ADMIN3"].str.strip()
    )


# =====================================================================
# Load diagnostics
# =====================================================================

section("ETHIOPIA UNMATCHED INVESTIGATION")

print("Loading unmatched FEWS reporting units...")
fews = pd.read_csv(INPUT_DIR / "ethiopia_fews_only.csv")

print("Loading unmatched FSC reporting units...")
fsc = pd.read_csv(INPUT_DIR / "ethiopia_fsc_only.csv")

required_fews = [
    "admin_code",
    "ADMIN1",
    "ADMIN2",
    "ADMIN3",
]

required_fsc = [
    "FNID",
    "ADMIN1",
    "ADMIN2",
    "ADMIN3",
]

validate_columns(fews, required_fews, "FEWS diagnostics")
validate_columns(fsc, required_fsc, "FSC diagnostics")

print("✓ Required columns verified.\n")

print(f"FEWS unmatched : {len(fews)}")
print(f"FSC unmatched  : {len(fsc)}")

# =====================================================================
# Profile unmatched reporting units
# =====================================================================

section("PROFILING UNMATCHED REPORTING UNITS")


def profile_level(
    df: pd.DataFrame,
    column: str,
    dataset_name: str,
    output_name: str,
) -> None:
    """
    Generate frequency table for a single administrative level.
    """

    profile = (
        df[column]
        .value_counts(dropna=False)
        .rename_axis(column)
        .reset_index(name="count")
    )

    profile.to_csv(
        OUTPUT_DIR / output_name,
        index=False,
    )

    print(f"\n{dataset_name} - {column}")
    print(profile.head(10).to_string(index=False))


# ---------------------------------------------------------------------
# FEWS profiles
# ---------------------------------------------------------------------

profile_level(
    fews,
    "ADMIN1",
    "FEWS",
    "fews_unmatched_admin1_counts.csv",
)

profile_level(
    fews,
    "ADMIN2",
    "FEWS",
    "fews_unmatched_admin2_counts.csv",
)

profile_level(
    fews,
    "ADMIN3",
    "FEWS",
    "fews_unmatched_admin3_counts.csv",
)


# ---------------------------------------------------------------------
# FSC profiles
# ---------------------------------------------------------------------

profile_level(
    fsc,
    "ADMIN1",
    "FSC",
    "fsc_unmatched_admin1_counts.csv",
)

profile_level(
    fsc,
    "ADMIN2",
    "FSC",
    "fsc_unmatched_admin2_counts.csv",
)

profile_level(
    fsc,
    "ADMIN3",
    "FSC",
    "fsc_unmatched_admin3_counts.csv",
)

print("\n✓ Administrative profiles generated.")

# =====================================================================
# Generate candidate matches
# =====================================================================

section("GENERATING CANDIDATE MATCHES")

# Build hierarchy strings used for fuzzy comparison
fews = fews.copy()
fsc = fsc.copy()

fews["hierarchy"] = hierarchy_string(fews)
fsc["hierarchy"] = hierarchy_string(fsc)

candidate_matches = []

for _, fews_row in fews.iterrows():

    # --------------------------------------------------------------
    # Prefer candidates within the same ADMIN1 region
    # --------------------------------------------------------------

    regional_candidates = fsc[
        fsc["ADMIN1"] == fews_row["ADMIN1"]
    ]

    search_scope = "ADMIN1"

    if regional_candidates.empty:
        regional_candidates = fsc
        search_scope = "COUNTRY"

    choices = regional_candidates["hierarchy"].tolist()

    match = process.extractOne(
        fews_row["hierarchy"],
        choices,
        scorer=fuzz.WRatio,
    )

    if match is None:
        continue

    matched_hierarchy, similarity, _ = match

    matched_row = regional_candidates[
        regional_candidates["hierarchy"] == matched_hierarchy
    ].iloc[0]

    candidate_matches.append(
        {
            "admin_code": fews_row["admin_code"],
            "FEWS_ADMIN1": fews_row["ADMIN1"],
            "FEWS_ADMIN2": fews_row["ADMIN2"],
            "FEWS_ADMIN3": fews_row["ADMIN3"],
            "FNID": matched_row["FNID"],
            "FSC_ADMIN1": matched_row["ADMIN1"],
            "FSC_ADMIN2": matched_row["ADMIN2"],
            "FSC_ADMIN3": matched_row["ADMIN3"],
            "similarity": round(similarity, 2),
            "search_scope": search_scope,
        }
    )

candidate_matches = pd.DataFrame(candidate_matches)

candidate_matches = candidate_matches.sort_values(
    "similarity",
    ascending=False,
)
# ---------------------------------------------------------------------
# Split high-confidence and review candidates
# ---------------------------------------------------------------------

SIMILARITY_THRESHOLD = 90

high_confidence = candidate_matches[
    candidate_matches["similarity"] >= SIMILARITY_THRESHOLD
].copy()

review_candidates = candidate_matches[
    candidate_matches["similarity"] < SIMILARITY_THRESHOLD
].copy()

high_confidence.to_csv(
    OUTPUT_DIR / "candidate_matches.csv",
    index=False,
)

review_rows = []

for _, fews_row in review_candidates.iterrows():

    search_space = fsc[
        fsc["ADMIN1"] == fews_row["FEWS_ADMIN1"]
    ]

    if search_space.empty:
        search_space = fsc

    matches = process.extract(
        hierarchy_string(
            pd.DataFrame([{
                "ADMIN1": fews_row["FEWS_ADMIN1"],
                "ADMIN2": fews_row["FEWS_ADMIN2"],
                "ADMIN3": fews_row["FEWS_ADMIN3"],
            }])
        ).iloc[0],
        search_space["hierarchy"].tolist(),
        scorer=fuzz.WRatio,
        limit=3,
    )

    for rank, match in enumerate(matches, start=1):

        matched_hierarchy, similarity, _ = match

        matched_row = search_space[
            search_space["hierarchy"] == matched_hierarchy
        ].iloc[0]

        review_rows.append(
            {
                "admin_code": fews_row["admin_code"],
                "Rank": rank,
                "Similarity": similarity,
                "Suggested_FNID": matched_row["FNID"],
                "Suggested_ADMIN1": matched_row["ADMIN1"],
                "Suggested_ADMIN2": matched_row["ADMIN2"],
                "Suggested_ADMIN3": matched_row["ADMIN3"],
            }
        )

review_candidates = pd.DataFrame(review_rows)

review_candidates.to_csv(
    OUTPUT_DIR / "candidate_matches_review.csv",
    index=False,
)

print(
    f"✓ High-confidence matches : {len(high_confidence)}"
)

if review_candidates.empty:
    print("✓ Review candidates       : 0")
else:
    print(
        f"✓ Review candidates       : "
        f"{review_candidates['admin_code'].nunique()}"
    )

# =====================================================================
# Investigation summary
# =====================================================================

section("INVESTIGATION SUMMARY")

summary = []

summary.append(
    ("FEWS unmatched", len(fews))
)

summary.append(
    ("FSC unmatched", len(fsc))
)

summary.append(
    (
        "High-confidence matches",
        len(high_confidence),
    )
)

if review_candidates.empty:
    review_count = 0
else:
    review_count = review_candidates["admin_code"].nunique()

summary.append(
    (
        "Review required",
        review_count,
    )
)

summary.append(
    (
        "Average similarity",
        round(
            candidate_matches["similarity"].mean(),
            2,
        ),
    )
)

summary.append(
    (
        "Median similarity",
        round(
            candidate_matches["similarity"].median(),
            2,
        ),
    )
)

summary.append(
    (
        "Minimum similarity",
        round(
            candidate_matches["similarity"].min(),
            2,
        ),
    )
)

summary.append(
    (
        "Maximum similarity",
        round(
            candidate_matches["similarity"].max(),
            2,
        ),
    )
)

summary_df = pd.DataFrame(
    summary,
    columns=[
        "Metric",
        "Value",
    ],
)

print(summary_df.to_string(index=False))

summary_df.to_csv(
    OUTPUT_DIR / "investigation_summary.csv",
    index=False,
)

with open(
    OUTPUT_DIR / "investigation_summary.txt",
    "w",
) as f:

    f.write(
        "ETHIOPIA UNMATCHED INVESTIGATION\n"
    )

    f.write("=" * 50)

    f.write("\n\n")

    for metric, value in summary:

        f.write(
            f"{metric:<30}{value}\n"
        )

print("\n✓ Investigation summary saved.")