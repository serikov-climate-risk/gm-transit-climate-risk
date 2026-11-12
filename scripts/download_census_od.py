"""Download the 2011 Census journey-to-work matrix (WU03EW, MSOA level) for Greater Manchester.

Source: Nomis API, dataset NM_1208_1 (Open Government Licence). Residence and workplace are
the 346 MSOAs of the ten Greater Manchester districts; flows by method of travel to work.
2011 is used rather than 2021 because the 2021 Census was taken during a lockdown and its
public origin-destination tables do not carry the travel mode.
"""
from __future__ import annotations

import io
import re
from pathlib import Path

import pandas as pd
import requests

API = "https://www.nomisweb.co.uk/api/v01/dataset/NM_1208_1"
GM = ["Bolton", "Bury", "Manchester", "Oldham", "Rochdale", "Salford", "Stockport",
      "Tameside", "Trafford", "Wigan"]
MODES = {0: "all", 1: "home", 2: "metro_tram", 3: "train", 4: "bus", 7: "car_driver",
         8: "car_passenger", 9: "bicycle", 10: "foot"}
PAGE = 25000


def gm_msoas() -> list[int]:
    d = requests.get(f"{API}/usual_residence/TYPE297.def.sdmx.json", timeout=120).json()
    codes = d["structure"]["codelists"]["codelist"][0]["code"]
    return [c["value"] for c in codes
            if re.sub(r" \d+$", "", c["description"]["value"]) in GM]


def main() -> None:
    out = Path("data/raw/census2011_wu03ew_gm.csv")
    ids = gm_msoas()
    assert len(ids) == 346, len(ids)
    geo = ",".join(map(str, ids))
    frames = []
    for mode in MODES:
        offset = 0
        while True:
            params = {
                "usual_residence": geo, "place_of_work": geo, "transport_powpew11": mode,
                "measures": 20100, "select": "usual_residence_code,place_of_work_code,obs_value",
                "RecordOffset": offset, "RecordLimit": PAGE,
            }
            r = requests.post(f"{API}.data.csv", data=params, timeout=600)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            df["mode"] = MODES[mode]
            frames.append(df)
            print(mode, offset, len(df), flush=True)
            if len(df) < PAGE:
                break
            offset += PAGE
    od = pd.concat(frames)
    od.columns = [c.lower() for c in od.columns]
    od = od[od.obs_value > 0]
    od.to_csv(out, index=False)
    print(out, len(od))


if __name__ == "__main__":
    main()
