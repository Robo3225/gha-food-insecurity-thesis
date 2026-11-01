"""
============================================================

write_manifest.py

Write dataset metadata (manifest).

Author:
Robert Odingo MSc Thesis

============================================================
"""

from pathlib import Path
from datetime import datetime
import json


def write_manifest(
    output_file,
    country,
    features,
    start_year,
    start_month,
    end_year,
    end_month,
    dataframe,
):
    """
    Save a JSON manifest describing
    a generated dataset.
    """

    manifest = {

         "dataset": Path(output_file).name,

         "country": country["name"],

         "country_code": country["code"],

         "fsc_polygons": country["fsc_polygons"],

         "labelled_units": country["labelled_units"],

         "date_range": {

         "start": f"{start_year}-{start_month:02d}",

         "end": f"{end_year}-{end_month:02d}",

             },

         "features": [

          feature["name"]

          for feature in features

          ],

         "rows": len(dataframe),

          "columns": len(dataframe.columns),

          "created": datetime.now().isoformat(),

          "earth_engine": {

          feature["name"]: feature["collection"]

           for feature in features

          }

      }

    manifest_file = (
        Path(output_file)
        .with_suffix(".manifest.json")
    )

    with open(
        manifest_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            manifest,
            f,
            indent=4,
        )

    print()

    print(f"Manifest saved:\n{manifest_file}")