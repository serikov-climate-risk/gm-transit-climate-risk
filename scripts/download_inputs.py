"""Download the TfGM GTFS feed, MSOA 2011 boundaries and Manchester daily temperatures."""
from pathlib import Path

import requests

RAW = Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)
GM = ["Bolton", "Bury", "Manchester", "Oldham", "Rochdale", "Salford", "Stockport",
      "Tameside", "Trafford", "Wigan"]
MODELS = ["CMCC_CM2_VHR4", "FGOALS_f3_H", "HiRAM_SIT_HR", "MRI_AGCM3_2_S", "EC_Earth3P_HR",
          "MPI_ESM1_2_XR", "NICAM16_8S"]
LAT, LON = 53.48, -2.24


def get(url, out, **params):
    r = requests.get(url, params=params, timeout=600)
    r.raise_for_status()
    (RAW / out).write_bytes(r.content)
    print(out, len(r.content))


get("https://odata.tfgm.com/opendata/downloads/TfGMgtfsnew.zip", "gm_gtfs.zip")
get("https://services1.arcgis.com/ESMARspQHYMw9BZ9/ArcGIS/rest/services/msoa/FeatureServer/0/query",
    "msoa2011_gm.geojson", where=" OR ".join(f"MSOA11NM LIKE '{d}%'" for d in GM),
    outFields="MSOA11CD,MSOA11NM", returnGeometry="true", outSR=27700, f="geojson")
get("https://archive-api.open-meteo.com/v1/archive", "tmax_era5.json", latitude=LAT,
    longitude=LON, start_date="1991-01-01", end_date="2025-12-31",
    daily="temperature_2m_max", timezone="Europe/London")
get("https://climate-api.open-meteo.com/v1/climate", "tmax_cmip6.json", latitude=LAT,
    longitude=LON, start_date="1991-01-01", end_date="2050-12-31", models=",".join(MODELS),
    daily="temperature_2m_max")
