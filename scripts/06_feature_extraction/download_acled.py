"""
Download ACLED event data for Kenya, Ethiopia, and Somalia
for the period 2018-01-01 through 2023-12-31.

Purpose:
    Reproducible ACLED Research-tier data acquisition.

Outputs:
    data/external/acled/raw/acled_ke_eth_so_2018_2023_raw.csv
    data/external/acled/processed/acled_ke_eth_so_2018_2023.csv
    data/external/acled/metadata/acled_export_metadata.txt

Authentication:
    ACLED OAuth using credentials supplied through environment variables.

Required environment variables:
    ACLED_USERNAME
    ACLED_PASSWORD

Author: Robert Odingo MSc Thesis
"""

from pathlib import Path
from io import StringIO
from datetime import datetime, timezone
import os
import time

import pandas as pd
import requests


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DIR = BASE_DIR / "data" / "external" / "acled" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "external" / "acled" / "processed"
METADATA_DIR = BASE_DIR / "data" / "external" / "acled" / "metadata"

RAW_FILE = RAW_DIR / "acled_ke_eth_so_2018_2023_raw.csv"
PROCESSED_FILE = (
    PROCESSED_DIR / "acled_ke_eth_so_2018_2023.csv"
)
METADATA_FILE = (
    METADATA_DIR / "acled_export_metadata.txt"
)

TOKEN_URL = "https://acleddata.com/oauth/token"
API_URL = "https://acleddata.com/api/acled/read"

COUNTRIES = [
    "Kenya",
    "Ethiopia",
    "Somalia",
]

START_DATE = "2018-01-01"
END_DATE = "2023-12-31"

# Fields deliberately retained in the raw research export.
#
# The five fields requested for the eventual dataset are:
#   event_date
#   latitude
#   longitude
#   fatalities
#   country
#
# Additional fields are retained for provenance, validation,
# and possible later spatial/event-type analysis.
FIELDS = [
    "event_id_cnty",
    "event_date",
    "latitude",
    "longitude",
    "fatalities",
    "country",
    "event_type",
    "sub_event_type",
]

PAGE_SIZE = 5000

REQUEST_TIMEOUT = 120

MAX_RETRIES = 5


# ============================================================
# HELPERS
# ============================================================

def ensure_directories():
    """Create output directories if they do not already exist."""

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)


def get_credentials():
    """
    Read ACLED credentials from environment variables.

    This avoids putting your ACLED password directly into
    the Python source code.
    """

    username = os.getenv("ACLED_USERNAME")
    password = os.getenv("ACLED_PASSWORD")

    if not username:
        raise RuntimeError(
            "ACLED_USERNAME environment variable is not set."
        )

    if not password:
        raise RuntimeError(
            "ACLED_PASSWORD environment variable is not set."
        )

    return username, password


def get_access_token(username, password):
    """
    Authenticate with ACLED using OAuth password grant.
    """

    payload = {
        "username": username,
        "password": password,
        "grant_type": "password",
        "client_id": "acled",
        "scope": "authenticated",
    }

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
    }

    print("Authenticating with ACLED...")

    response = requests.post(
        TOKEN_URL,
        headers=headers,
        data=payload,
        timeout=REQUEST_TIMEOUT,
    )

    if response.status_code != 200:
        raise RuntimeError(
            "ACLED authentication failed.\n"
            f"HTTP status: {response.status_code}\n"
            f"Response: {response.text[:1000]}"
        )

    token_data = response.json()

    if "access_token" not in token_data:
        raise RuntimeError(
            "ACLED authentication response did not contain "
            "an access token."
        )

    print("Authentication successful.")

    return token_data["access_token"]


def request_page(access_token, cursor):
    """
    Request one page of ACLED data.

    Uses cursor-based pagination rather than page-based
    pagination.
    """

    countries_query = "|".join(COUNTRIES)
    fields_query = "|".join(FIELDS)

    params = {
        "_format": "csv",
        "country": countries_query,
        "event_date": f"{START_DATE}|{END_DATE}",
        "event_date_where": "BETWEEN",
        "fields": fields_query,
        "limit": PAGE_SIZE,
        "cursor": cursor,
    }

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            response = requests.get(
                API_URL,
                params=params,
                headers=headers,
                timeout=REQUEST_TIMEOUT,
            )

            # Successful request
            if response.status_code == 200:
                return response

            # Authentication failure
            if response.status_code in (401, 403):
                raise RuntimeError(
                    "ACLED authentication/authorization failed.\n"
                    f"HTTP status: {response.status_code}\n"
                    f"Response: {response.text[:1000]}"
                )

            print(
                f"Request failed "
                f"(attempt {attempt}/{MAX_RETRIES}): "
                f"HTTP {response.status_code}"
            )

        except requests.RequestException as exc:

            print(
                f"Request error "
                f"(attempt {attempt}/{MAX_RETRIES}): {exc}"
            )

        if attempt < MAX_RETRIES:

            wait_seconds = 2 ** attempt

            print(
                f"Waiting {wait_seconds} seconds before retry..."
            )

            time.sleep(wait_seconds)

    raise RuntimeError(
        "ACLED request failed after maximum retries."
    )


def download_all_events(access_token):
    """
    Retrieve all matching ACLED events using cursor pagination.
    """

    all_pages = []

    cursor = 0
    page_number = 1
    total_rows = 0

    while True:

        print()
        print(
            f"Downloading ACLED page {page_number} "
            f"(cursor={cursor})..."
        )

        response = request_page(
            access_token=access_token,
            cursor=cursor,
        )

        # Read CSV response.
        page_df = pd.read_csv(
            StringIO(response.text)
        )

        rows_this_page = len(page_df)

        total_rows += rows_this_page

        print(
            f"  Rows received: {rows_this_page:,}"
        )

        print(
            f"  Total rows so far: {total_rows:,}"
        )

        if rows_this_page > 0:
            all_pages.append(page_df)

        # ACLED returns the next cursor for CSV
        # responses in the X-Next-Cursor header.
        next_cursor = response.headers.get(
            "X-Next-Cursor"
        )

        # No next cursor = finished.
        if not next_cursor:

            print()
            print(
                "No X-Next-Cursor returned. "
                "Download complete."
            )

            break

        # Some responses may explicitly return null.
        if str(next_cursor).lower() == "null":

            print()
            print(
                "ACLED returned a null next cursor. "
                "Download complete."
            )

            break

        # Guard against a non-advancing cursor (would loop forever).
        if str(next_cursor) == str(cursor):
            print()
            print("Cursor did not advance; stopping to avoid an infinite loop.")
            break

        cursor = next_cursor
        page_number += 1

        # Hard safety cap (KE+ET+SO 2018-2023 is well under this).
        if page_number > 500:
            print()
            print("Reached page cap (500); stopping. Verify completeness.")
            break

    if not all_pages:

        raise RuntimeError(
            "ACLED returned no data."
        )

    df = pd.concat(
        all_pages,
        ignore_index=True,
    )

    return df


def validate_data(df):
    """
    Validate the downloaded ACLED dataset.
    """

    print()
    print("=" * 70)
    print("VALIDATION")
    print("=" * 70)

    expected_columns = set(FIELDS)

    actual_columns = set(df.columns)

    missing_columns = expected_columns - actual_columns

    if missing_columns:

        raise RuntimeError(
            "Downloaded data is missing expected columns: "
            f"{sorted(missing_columns)}"
        )

    print("✓ Required columns present")

    # Dates
    df["event_date"] = pd.to_datetime(
        df["event_date"],
        errors="coerce",
    )

    invalid_dates = df["event_date"].isna().sum()

    if invalid_dates:

        raise RuntimeError(
            f"Found {invalid_dates:,} invalid event dates."
        )

    print("✓ Event dates valid")

    # Date range
    min_date = df["event_date"].min()
    max_date = df["event_date"].max()

    print(
        f"  Actual date range: "
        f"{min_date.date()} → {max_date.date()}"
    )

    requested_start = pd.Timestamp(START_DATE)
    requested_end = pd.Timestamp(END_DATE)

    if min_date < requested_start:
        raise RuntimeError(
            "Data contains events before requested start date."
        )

    if max_date > requested_end:
        raise RuntimeError(
            "Data contains events after requested end date."
        )

    print("✓ Date range correct")

    # Countries
    countries = sorted(
        df["country"].dropna().unique()
    )

    print(
        f"  Countries returned: {countries}"
    )

    expected_countries = set(COUNTRIES)

    if set(countries) != expected_countries:

        print(
            "WARNING: Returned country set differs from "
            "requested country set."
        )

    else:

        print("✓ Correct countries returned")

    # Coordinates
    missing_lat = df["latitude"].isna().sum()
    missing_lon = df["longitude"].isna().sum()

    print(
        f"  Missing latitude: {missing_lat:,}"
    )

    print(
        f"  Missing longitude: {missing_lon:,}"
    )

    # Fatalities
    df["fatalities"] = pd.to_numeric(
        df["fatalities"],
        errors="coerce",
    )

    missing_fatalities = df["fatalities"].isna().sum()

    print(
        f"  Missing fatalities: {missing_fatalities:,}"
    )

    # Duplicate event IDs
    duplicate_ids = (
        df["event_id_cnty"]
        .duplicated()
        .sum()
    )

    print(
        f"  Duplicate event IDs: {duplicate_ids:,}"
    )

    if duplicate_ids > 0:

        print(
            "WARNING: Duplicate ACLED event IDs detected."
        )

    # Country counts
    print()
    print("Events by country:")
    print(
        df["country"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # Year counts
    print()
    print("Events by year:")
    print(
        df["event_date"]
        .dt.year
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("✓ Validation completed")


def save_outputs(df):
    """
    Save raw and reduced datasets.
    """

    # Keep date in ISO format.
    df["event_date"] = pd.to_datetime(
        df["event_date"]
    ).dt.strftime("%Y-%m-%d")

    # Sort for reproducibility.
    df = df.sort_values(
        by=[
            "country",
            "event_date",
            "event_id_cnty",
        ]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Raw research export
    # --------------------------------------------------------

    df.to_csv(
        RAW_FILE,
        index=False,
    )

    print()
    print(f"Raw dataset saved to:")
    print(f"  {RAW_FILE}")

    # --------------------------------------------------------
    # Reduced five-column dataset
    # --------------------------------------------------------

    requested_columns = [
        "event_date",
        "latitude",
        "longitude",
        "fatalities",
        "country",
    ]

    reduced_df = df[requested_columns].copy()

    reduced_df.to_csv(
        PROCESSED_FILE,
        index=False,
    )

    print()
    print(
        "Five-column dataset saved to:"
    )
    print(
        f"  {PROCESSED_FILE}"
    )

    return df


def write_metadata(df):
    """
    Save provenance information about the export.
    """

    download_time = datetime.now(
        timezone.utc
    ).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )

    lines = [
        "ACLED RESEARCH EXPORT METADATA",
        "=" * 70,
        "",
        f"Download timestamp: {download_time}",
        "",
        "Source: ACLED",
        "Access tier: Research",
        "",
        "Countries:",
        *[f"  - {country}" for country in COUNTRIES],
        "",
        f"Start date: {START_DATE}",
        f"End date: {END_DATE}",
        "",
        "Primary API endpoint:",
        API_URL,
        "",
        "Fields retained in raw export:",
        *[f"  - {field}" for field in FIELDS],
        "",
        "Reduced dataset fields:",
        "  - event_date",
        "  - latitude",
        "  - longitude",
        "  - fatalities",
        "  - country",
        "",
        f"Total events: {len(df):,}",
        "",
        "Events by country:",
    ]

    country_counts = (
        df["country"]
        .value_counts()
        .sort_index()
    )

    for country, count in country_counts.items():

        lines.append(
            f"  - {country}: {count:,}"
        )

    lines.extend(
        [
            "",
            "Events by year:",
        ]
    )

    year_counts = (
        pd.to_datetime(df["event_date"])
        .dt.year
        .value_counts()
        .sort_index()
    )

    for year, count in year_counts.items():

        lines.append(
            f"  - {year}: {count:,}"
        )

    lines.extend(
        [
            "",
            "Notes:",
            "  - Data retrieved through the ACLED API.",
            "  - Country and date filters were applied at API level.",
            "  - Cursor-based pagination was used.",
            "  - The raw export retains additional fields for",
            "    validation and potential future analysis.",
            "  - Duplicate event_id_cnty records are removed before",
            "    validation and saving, so validation describes the",
            "    final retained dataset.",
            "  - The completion message reports unique events retained.",
            "",
        ]
    )

    # Reproducibility hash of the reduced 5-column dataset (matches the
    # thesis's version-pinning discipline for datasets that can change upstream).
    import hashlib
    reduced_cols = ["event_date", "latitude", "longitude", "fatalities", "country"]
    reduced_bytes = (
        df[reduced_cols].sort_values(reduced_cols).to_csv(index=False).encode()
    )
    data_hash = hashlib.sha256(reduced_bytes).hexdigest()[:16]
    lines.extend([
        f"Reduced-dataset sha256[:16]: {data_hash}",
        "  (version-pin: cite this hash + download timestamp for reproducibility)",
        "",
    ])

    METADATA_FILE.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    print()
    print(f"Reduced-dataset hash (version-pin): {data_hash}")

    print()
    print("Metadata saved to:")
    print(f"  {METADATA_FILE}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ACLED RESEARCH DATA DOWNLOAD")
    print("=" * 70)

    print()
    print("Countries:")
    for country in COUNTRIES:
        print(f"  - {country}")

    print()
    print(
        f"Date range: {START_DATE} → {END_DATE}"
    )

    print()
    print("Fields:")
    for field in FIELDS:
        print(f"  - {field}")

    ensure_directories()

    username, password = get_credentials()

    access_token = get_access_token(
        username,
        password,
    )

    df = download_all_events(
        access_token
    )

    # De-duplicate on the ACLED event ID before validation so that
    # validation statistics describe the final analytical dataset.
    n_before = len(df)
    df = df.drop_duplicates(
        subset=["event_id_cnty"]
    ).reset_index(drop=True)
    n_dropped = n_before - len(df)

    if n_dropped:
        print()
        print(
            f"Removed {n_dropped:,} duplicate event(s) "
            "by event_id_cnty."
        )

    validate_data(df)

    df = save_outputs(df)

    write_metadata(df)

    print()
    print("=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Total unique ACLED events retained: {len(df):,}"
    )

    print()
    print("Output files:")

    print(
        f"  Raw:       {RAW_FILE}"
    )

    print(
        f"  Processed: {PROCESSED_FILE}"
    )

    print(
        f"  Metadata:  {METADATA_FILE}"
    )


if __name__ == "__main__":
    main()