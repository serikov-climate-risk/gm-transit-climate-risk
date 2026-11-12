"""Heat on the Metrolink: restriction days now and in the 2040s, and their cost to commuters.

TfGM reported to the Greater Manchester Transport Committee (Metrolink Service
Performance, 16 September 2022) that the extreme temperatures of 18 and 19 July 2022
"significantly impacted overhead line equipment and required temporary speed
restrictions and service cancellations", and that speed restrictions were applied in
August 2022 with temperatures "exceeding 30°C". Here:
- level 1, daily maximum above HOT_C: speed restrictions on all tram segments, run times
  multiplied by SLOW_FACTOR (assumed);
- level 2, daily maximum of EXTREME_C or more (the July 2022 days): in addition, tram
  service at half frequency (assumed).
Buses are not affected.

Temperatures: daily maximum at Manchester from ERA5 (Open-Meteo archive), 1991-2025. The
2041-2050 climate shifts each ERA5 day by the change in the matching quantile of summer
daily maxima between 1991-2014 and 2041-2050 in each of seven HighResMIP models
(Open-Meteo climate API), so the models' own biases cancel. ERA5 smooths the urban
maximum (ERA5 gives 30.1°C on the hottest day of the August 2022 spell), so the
thresholds are also checked one degree lower.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

HOT_C, EXTREME_C = 30.0, 34.0
SLOW_FACTOR = 1.4
SUMMER = (5, 6, 7, 8, 9)


def load_era5(path: str) -> pd.Series:
    a = json.load(open(path))
    return pd.Series(a["daily"]["temperature_2m_max"], index=pd.to_datetime(a["daily"]["time"]),
                     dtype=float)


def load_models(path: str) -> dict[str, pd.Series]:
    b = json.load(open(path))
    t = pd.to_datetime(b["daily"]["time"])
    return {k.replace("temperature_2m_max_", ""): pd.Series(v, index=t, dtype=float)
            for k, v in b["daily"].items() if k != "time"}


def quantile_delta(era5: pd.Series, model: pd.Series, hist=("1991", "2014"),
                   fut=("2041", "2050")) -> pd.Series:
    """ERA5 days shifted by the model's change in the matching summer quantile."""
    out = era5.copy()
    s = era5.index.month.isin(SUMMER)
    h = model[hist[0]:hist[1]]
    f = model[fut[0]:fut[1]]
    h, f = h[h.index.month.isin(SUMMER)].dropna(), f[f.index.month.isin(SUMMER)].dropna()
    q = np.linspace(0.005, 0.995, 199)
    dq = np.quantile(f, q) - np.quantile(h, q)
    p = era5[s].rank(pct=True).to_numpy()
    out[s] = era5[s].to_numpy() + np.interp(p, q, dq)
    return out


def days_per_year(t: pd.Series, hot=HOT_C, extreme=EXTREME_C) -> dict:
    years = t.index.year.nunique()
    lvl2 = (t >= extreme).sum() / years
    lvl1 = ((t > hot) & (t < extreme)).sum() / years
    return {"level1_days": lvl1, "level2_days": lvl2}


def edge_costs(g, net, level: int):
    """Edge costs on a heat day: slower trams, and at level 2 longer waits."""
    cost = g.cost.copy()
    mode = net.ride.pattern.map(net.patterns.set_index("pattern")["mode"]).to_numpy()
    ride = g.ride_row >= 0
    tram_ride = ride.copy()
    tram_ride[ride] = mode[g.ride_row[ride]] == "tram"
    cost[tram_ride] *= SLOW_FACTOR
    if level >= 2:
        from . import network as nw
        # board edges onto tram line nodes: the wait doubles with half the frequency
        tram_nodes = set(g.head[tram_ride]) | set(g.tail[tram_ride])
        board = (g.kind == 2) & np.isin(g.head, list(tram_nodes))
        wait = (cost[board] - nw.BOARD_PENALTY) / nw.WAIT_WEIGHT
        cost[board] = nw.WAIT_WEIGHT * np.minimum(2 * wait, 2 * nw.WAIT_CAP) + nw.BOARD_PENALTY
    return cost
