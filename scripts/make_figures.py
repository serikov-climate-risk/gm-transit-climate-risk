"""Figures for the README: flooded network, losses by event, adaptation plans, heat."""
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gmtransit import impact as im
from gmtransit import optimize as op

R = Path("results")
m = pickle.load(open("data/processed/model.pkl", "rb"))
net = m["net"]
floods = pickle.load(open("data/processed/floods.pkl", "rb"))
mode = net.ride.pattern.map(net.patterns.set_index("pattern")["mode"]).to_numpy()

# 1. network and sites blocked in the 1-in-100 event, present and 2050s
fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), sharex=True, sharey=True)
for ax, clim in zip(axes, ("present", "cc01")):
    for i, g in net.geometry.items():
        ax.plot(g[:, 0] / 1000, g[:, 1] / 1000, color="0.85" if mode[i] == "bus" else "0.55",
                lw=0.3 if mode[i] == "bus" else 1.0, zorder=1)
    ef = floods[clim]
    hs = {h for v in ef.deep_sites[100].values() for h in v}
    xy = np.array([ef.hotspot_xy[h] for h in hs]) / 1000
    tram = {h for r, v in ef.deep_sites[100].items() if mode[r] == "tram" for h in v}
    txy = np.array([ef.hotspot_xy[h] for h in tram]) / 1000 if tram else np.empty((0, 2))
    ax.scatter(xy[:, 0], xy[:, 1], s=3, color="#2b6cb0", label=f"bus sites ({len(hs) - len(tram)})", zorder=2)
    if len(txy):
        ax.scatter(txy[:, 0], txy[:, 1], s=14, color="#c53030", marker="s",
                   label=f"tram sites ({len(tram)})", zorder=3)
    ax.set_title("Present climate" if clim == "present" else "2050s, UKCP18 central allowance")
    ax.set_aspect("equal")
    ax.set_xlabel("km east (British National Grid)")
    ax.legend(loc="lower left", fontsize=8)
axes[0].set_ylabel("km north")
fig.suptitle("Sites where the 1-in-100 surface water flood puts 0.3 m or more of water over 20 m of a bus or tram route")
fig.tight_layout()
fig.savefig(R / "fig_flooded_sites.png", dpi=130)

# 2. expected annual loss by return period
ev = pd.read_csv(sorted(R.glob("events_*km.csv"))[0])
t = ev.groupby(["climate", "rp"]).eal_gbp.sum().unstack(0)[["present", "cc01"]] / 1000
ax = t.plot.bar(figsize=(6, 4), color=["#718096", "#2b6cb0"], rot=0)
ax.set_xlabel("event return period (years)")
ax.set_ylabel("expected annual loss, thousand GBP")
ax.legend(["present", "2050s central"])
ax.set_title("Commuter welfare loss from surface water floods")
plt.tight_layout()
plt.savefig(R / "fig_eal_by_rp.png", dpi=130)

# 3. plans
p = pd.read_csv(R / "plans.csv")
fig, ax = plt.subplots(figsize=(7, 4.2))
for (name, st), col in zip([("plan_present", "o-"), ("plan_cc01", "s-"), ("stochastic", "^-"),
                            ("minimax_regret", "d--")], ["#2b6cb0", "#2f855a", "#6b46c1", "#c05621"]):
    q = p[p.plan == name]
    ax.plot(q.site_cost, q.total_cc01 / 1000, st, color=col, label=name.replace("_", " "))
    ax.plot(q.site_cost, q.total_present / 1000, st, color=col, alpha=0.35)
ax.set_xscale("log")
ax.set_xlabel("annualised cost per site at the 1-in-100 standard, GBP")
ax.set_ylabel("measures + expected loss, thousand GBP/yr")
ax.set_title("Plans on the re-solved network: 2050s climate (solid), present (faint)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(R / "fig_plans.png", dpi=130)

# 4. heat
h = pd.read_csv(R / "heat.csv")
h = h[h.thresholds == "30/34"]
fig, ax = plt.subplots(figsize=(7, 4))
lab = [c.replace("2041-2050 ", "") for c in h.case]
ax.barh(lab, h.eal_gbp / 1000, color=["#718096"] + ["#c05621"] * (len(h) - 1))
ax.set_xlabel("expected annual loss from Metrolink heat restrictions, thousand GBP")
ax.set_title("Heat: ERA5 1991-2025 and seven models for 2041-2050")
plt.tight_layout()
plt.savefig(R / "fig_heat.png", dpi=130)
print("figures written")
