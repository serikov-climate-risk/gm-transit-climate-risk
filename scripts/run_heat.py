"""Heat restriction days on the Metrolink and their expected annual cost to commuters."""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from gmtransit import demand as dm
from gmtransit import heat as ht
from gmtransit import impact as im

m = pickle.load(open("data/processed/model.pkl", "rb"))
g, net, zones = m["graph"], m["net"], m["zones"]
d = pickle.load(open(sorted(Path("data/processed").glob("losses_*km.pkl"))[0], "rb"))
od, t, gc0, L = d["od"], d["t"], d["gc0"], d["logit"]
n = od[dm.MODES].sum(axis=1).to_numpy()

loss = {}
for level in (1, 2):
    gc = g.zone_costs(ht.edge_costs(g, net, level))[od.o, od.d]
    w = L.welfare_loss(gc0, gc, t)
    loss[level] = (n * w).sum() * im.VOT / 60 * im.TRIPS_PER_DAY
    lost = (n * (L.shares(gc0, t)[:, 0] - L.shares(gc, t)[:, 0])).sum()
    print(f"level {level}: loss per working day {loss[level]:,.0f} GBP, PT trips lost {lost:,.0f}")

era5 = ht.load_era5("data/raw/tmax_era5.json")
models = ht.load_models("data/raw/tmax_cmip6.json")
rows = []
for shift in (0.0, 1.0):
    hot, ext = ht.HOT_C - shift, ht.EXTREME_C - shift
    cases = {"ERA5 1991-2025": era5}
    cases.update({f"2041-2050 {k}": ht.quantile_delta(era5, s) for k, s in models.items()})
    for name, series in cases.items():
        dd = ht.days_per_year(series, hot, ext)
        eal = im.WORKDAY * (dd["level1_days"] * loss[1] + dd["level2_days"] * loss[2])
        rows.append({"thresholds": f"{hot:.0f}/{ext:.0f}", "case": name, **dd, "eal_gbp": eal})
df = pd.DataFrame(rows)
Path("results").mkdir(exist_ok=True)
df.to_csv("results/heat.csv", index=False)
pd.DataFrame({"level": [1, 2], "loss_per_day_gbp": [loss[1], loss[2]]}).to_csv(
    "results/heat_day_loss.csv", index=False)
print(df.round(2).to_string(index=False))
fut = df[df.case.str.startswith("2041")]
print(fut.groupby("thresholds").eal_gbp.agg(["min", "median", "max"]).round(0))
