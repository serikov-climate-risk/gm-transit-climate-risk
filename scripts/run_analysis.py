"""Flood risk to Greater Manchester bus and tram commuting, and adaptation plans.

Stages (cached in data/processed):
  A  flood maps -> flooded and blocked network edges, hotspots
  B  mode choice model, intact paths, flood events and their per-OD losses
  C  adaptation plans for a range of site costs: plans for each climate, the stochastic
     plan, the minimax regret plan, each evaluated exactly under both climates
Results go to results/*.csv.
"""
from __future__ import annotations

import argparse
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from gmtransit import demand as dm
from gmtransit import hazard as hz
from gmtransit import impact as im
from gmtransit import optimize as op

PROC, RES = Path("data/processed"), Path("results")
CLIMATES = ("present", "cc01")
SITE_COSTS = [10, 25, 50, 100, 250]  # GBP per site and year at 1-in-100 (break-even range)


def stage_a(net):
    f = PROC / "floods.pkl"
    if f.exists():
        return pickle.load(open(f, "rb"))
    tiles = Path("data/raw/flood_tiles")
    floods = {}
    for clim in CLIMATES:
        deep = hz.mosaic(tiles, clim, "d03")
        anyd = hz.mosaic(tiles, clim, "any") if any(tiles.glob(f"{clim}_any_*.npy")) else None
        t0 = time.time()
        floods[clim] = hz.edge_flooding(net, deep, anyd, clim)
        print(clim, f"edges sampled in {time.time() - t0:.0f}s")
        del deep, anyd
    pickle.dump(floods, open(f, "wb"))
    return floods


WORKERS = max(1, (os.cpu_count() or 2) - 2)
_CTX = None


def _init(path):
    global _CTX
    _CTX = pickle.load(open(path, "rb"))


def _event(task):
    s, plan, full = task
    c = _CTX
    a, lost, _ = im.event_loss(c["g"], c["net"], c["logit"], c["od"], c["t"], c["gc0"],
                               c["scs"][s], plan, c["users"])
    if full:
        return s, a, lost
    sc = c["scs"][s]
    return s, sc.prob * im.WORKDAY * a.sum(), None


def run_pool(ctx: dict, tasks: list):
    path = PROC / "eval_ctx.pkl"
    pickle.dump(ctx, open(path, "wb"))
    with ProcessPoolExecutor(WORKERS, initializer=_init, initargs=(str(path),)) as ex:
        return list(ex.map(_event, tasks, chunksize=2))


def stage_b(m, floods, cell_km, region_wide=False):
    tag = "region" if region_wide else f"{int(cell_km)}km"
    f = PROC / f"losses_{tag}.pkl"
    if f.exists():
        return pickle.load(open(f, "rb"))
    net, g, zones = m["net"], m["graph"], m["zones"]
    od = dm.load_od("data/raw/census2011_wu03ew_gm.csv", zones)
    t = dm.mode_times(od, zones)
    gc0 = g.zone_costs()[od.o, od.d]
    logit = dm.estimate(od, gc0, t)
    paths = op.base_paths(g, od)
    users = im.invert_paths(paths, len(net.ride))
    scs = im.scenarios(net, floods, cell_km, region_wide)
    all_sites = {}
    for ef in floods.values():
        for v in ef.deep_sites.values():
            for hs in v.values():
                for h in hs:
                    all_sites[h] = max(im.STANDARDS)
    none, resid, pt_lost = {}, {}, {}
    t0 = time.time()
    ctx = dict(g=g, net=net, logit=logit, od=od, t=t, gc0=gc0, scs=scs, users=users)
    for s, a, lost in run_pool(ctx, [(s, None, True) for s in range(len(scs))]):
        none[s], pt_lost[s] = a, lost
    for s, a, _ in run_pool(ctx, [(s, all_sites, True) for s, sc in enumerate(scs) if sc.sites]):
        resid[s] = a
    for s in range(len(scs)):
        resid.setdefault(s, none[s])
    print(f"{len(scs)} events in {time.time() - t0:.0f}s")
    out = dict(od=od, t=t, gc0=gc0, logit=logit, paths=paths, users=users, scs=scs,
               none=none, resid=resid, pt_lost=pt_lost)
    pickle.dump(out, open(f, "wb"))
    return out


def eal_table(d) -> pd.DataFrame:
    rows = []
    for s, sc in enumerate(d["scs"]):
        rows.append({"climate": sc.climate, "rp": sc.rp, "cell": sc.cell, "prob": sc.prob,
                     "blocked_edges": len(sc.sites), "event_loss_gbp": d["none"][s].sum(),
                     "residual_loss_gbp": d["resid"][s].sum(),
                     "pt_trips_lost": d["pt_lost"][s].sum()})
    df = pd.DataFrame(rows)
    df["eal_gbp"] = df.prob * im.WORKDAY * df.event_loss_gbp
    df["eal_residual_gbp"] = df.prob * im.WORKDAY * df.residual_loss_gbp
    return df


def exact_eal_many(m, d, plans: list[dict]) -> list[dict]:
    """Expected annual loss of each plan under each climate, on the re-solved network."""
    ctx = dict(g=m["graph"], net=m["net"], logit=d["logit"], od=d["od"], t=d["t"],
               gc0=d["gc0"], scs=d["scs"], users=d["users"])
    tasks = [(s, p, False) for p in plans for s in range(len(d["scs"]))]
    res = run_pool(ctx, tasks)
    out = [dict.fromkeys(CLIMATES, 0.0) for _ in plans]
    n = len(d["scs"])
    for k, (s, v, _) in enumerate(res):
        out[k // n][d["scs"][s].climate] += v
    return out


def stage_c(m, d):
    net = m["net"]
    mode = net.ride.pattern.map(net.patterns.set_index("pattern")["mode"]).to_numpy()
    groups = op.build_groups(d["scs"], d["paths"], d["none"], d["resid"],
                             np.flatnonzero(mode == "tram"))
    eal = eal_table(d)
    base = eal.groupby("climate").eal_gbp.sum().to_dict()
    resid = eal.groupby("climate").eal_residual_gbp.sum().to_dict()
    print("EAL without measures", base, "with every site at the highest standard", resid)
    print("groups", len(groups.weight), "hotspots", len({h for hs in groups.sites for h in hs}))
    rows, todo = [], []
    for c in SITE_COSTS:
        plans = {}
        known = {}
        for cl in CLIMATES:
            p, cost, info = op.solve(d["scs"], groups, c, {cl: 1.0})
            plans[f"plan_{cl}"] = p
            known[cl] = cost[cl]
        p, cost, info = op.solve(d["scs"], groups, c, {cl: 0.5 for cl in CLIMATES})
        plans["stochastic"] = p
        p, cost, info = op.solve(d["scs"], groups, c, {}, regret_base=known)
        plans["minimax_regret"] = p
        for name, p in plans.items():
            todo.append((c, name, p))
        print(c, {k: len(v) for k, v in plans.items()}, flush=True)
    t0 = time.time()
    evals = exact_eal_many(m, d, [p for _, _, p in todo])
    print(f"{len(todo)} plans evaluated in {time.time() - t0:.0f}s")
    plan_sites = []
    for (c, name, p), e in zip(todo, evals):
        sites_cost = c * sum(op.COST_RATIO[k] for k in p.values())
        row = {"site_cost": c, "plan": name, "sites": len(p),
               "sites_1in1000": sum(1 for k in p.values() if k == 1000),
               "measure_cost": sites_cost}
        for cl in CLIMATES:
            row[f"eal_{cl}"] = e[cl]
            row[f"total_{cl}"] = sites_cost + e[cl]
        row["total_expected"] = np.mean([row[f"total_{cl}"] for cl in CLIMATES])
        rows.append(row)
        plan_sites += [{"site_cost": c, "plan": name, "hotspot": h, "standard": k} for h, k in p.items()]
    pd.DataFrame(plan_sites).to_csv(RES / "plan_sites.csv", index=False)
    df = pd.DataFrame(rows)
    for cl in CLIMATES:
        best = df.groupby("site_cost")[f"total_{cl}"].transform("min")
        df[f"regret_{cl}"] = df[f"total_{cl}"] - best
    df["max_regret"] = df[[f"regret_{cl}" for cl in CLIMATES]].max(axis=1)
    return df, eal, base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell-km", type=float, default=im.CELL_KM)
    ap.add_argument("--stage", default="abc")
    ap.add_argument("--region-wide", action="store_true")
    args = ap.parse_args()
    RES.mkdir(exist_ok=True)
    m = pickle.load(open(PROC / "model.pkl", "rb"))
    floods = stage_a(m["net"])
    summ = pd.concat([hz.summary(m["net"], ef) for ef in floods.values()])
    summ.to_csv(RES / "flooded_network.csv", index=False)
    print(summ.to_string(index=False))
    if "b" not in args.stage:
        return
    d = stage_b(m, floods, args.cell_km, args.region_wide)
    tag = "region" if args.region_wide else f"{int(args.cell_km)}km"
    L = d["logit"]
    pd.DataFrame({"param": ["b_time", "asc_car", "asc_bicycle", "asc_walk", "ln_km_car",
                            "ln_km_bicycle", "ln_km_walk"],
                  "value": [L.b_pt, *L.asc.values(), *L.b.values()],
                  "se": L.se}).assign(rho2=L.rho2, n=L.n).to_csv(RES / "mode_choice.csv", index=False)
    eal = eal_table(d)
    eal.to_csv(RES / f"events_{tag}.csv", index=False)
    print(eal.groupby(["climate", "rp"])[["event_loss_gbp", "eal_gbp", "pt_trips_lost"]].sum())
    if "c" not in args.stage:
        return
    plans, _, _ = stage_c(m, d)
    plans.to_csv(RES / "plans.csv", index=False)


if __name__ == "__main__":
    main()
