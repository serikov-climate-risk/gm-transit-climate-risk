"""Choosing adaptation sites: a two-stage stochastic program and its robust variants.

First stage: which flood hotspots get a measure, and to which design standard
(present-climate 1-in-100 or 1-in-1000 storm), at an annualised cost per site.
Second stage: in every flood event, passengers reroute or change mode; the loss comes from
the logit model on the re-solved network.

The recourse is represented by groups. For a scenario s and an OD pair, the avoidable
loss is its full loss in s minus what remains with every protectable site dry
(shallow-water delays, sites no measure can hold). It is split over the blocking units on
the OD's intact best path (each flooded bus edge, and all flooded tram edges together);
a unit's share is recovered when all its sites are kept dry. Shares with the same
(s, unit) are summed into one group with weight W. The program is

    min  sum_h,k c_k x_hk + sum_g W_g (1 - y_g)
    s.t. y_g <= sum_{k protects in s(g)} x_hk  for every h in H_g,
         sum_k x_hk <= 1,  x binary, 0 <= y <= 1.

The split is an approximation (the logsum loss is not additive, and a better detour through
other protected sites is not credited); every plan is then
evaluated exactly on the re-solved network (`impact.expected_annual_loss`).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import highspy
import numpy as np
import pandas as pd

from . import impact as im

COST_RATIO = {100: 1.0, 1000: 2.5}  # cost of the 1-in-1000 standard relative to 1-in-100


def base_paths(g, od: pd.DataFrame) -> list[tuple[int, ...]]:
    """Ride rows on the intact best path of each OD pair."""
    eidx = g.edge_index()
    out = [()] * len(od)
    by_o = defaultdict(list)
    for i, (o, d) in enumerate(zip(od.o.to_numpy(), od.d.to_numpy())):
        by_o[o].append((i, d))
    for o, lst in by_o.items():
        _, pred = g.paths(o)
        for i, d in lst:
            rows, v = [], d
            while v != o and v >= 0:
                u = pred[v]
                if u < 0:
                    rows = []
                    break
                r = g.ride_row[eidx[(u, v)]]
                if r >= 0:
                    rows.append(int(r))
                v = u
            out[i] = tuple(reversed(rows))
    return out


@dataclass
class Groups:
    scen: np.ndarray  # scenario index per group
    sites: list  # tuple of hotspots per group
    weight: np.ndarray  # expected annual GBP at stake
    climate: np.ndarray


def build_groups(scs, paths, loss_none: dict, loss_resid: dict, tram_rows=None) -> Groups:
    """loss_none[s], loss_resid[s]: per-OD event loss in GBP for scenario s.

    The loss of an OD pair is split equally over its blocking units: each flooded bus edge
    on its intact path is one unit (a bus diverts at each site, so the delays add up), and
    all flooded tram edges together are one unit (a tram line is only back when every
    site on the way is dry).
    """
    tram_rows = set() if tram_rows is None else set(tram_rows)
    acc = {}
    for s, sc in enumerate(scs):
        if not sc.sites:
            continue
        w = (loss_none[s] - loss_resid[s]) * sc.prob * im.WORKDAY
        flooded = sc.sites
        for i in np.flatnonzero(w > 1e-9):
            on = [r for r in paths[i] if r in flooded]
            if not on:
                continue
            units = [tuple(sorted(set(flooded[r]))) for r in on if r not in tram_rows]
            tram = sorted({h for r in on if r in tram_rows for h in flooded[r]})
            if tram:
                units.append(tuple(tram))
            units = list(dict.fromkeys(units))
            for u in units:
                key = (s, u)
                acc[key] = acc.get(key, 0.0) + w[i] / len(units)
    keys = list(acc)
    return Groups(np.array([k[0] for k in keys]), [k[1] for k in keys],
                  np.array([acc[k] for k in keys]), np.array([scs[k[0]].climate for k in keys]))


def solve(scs, groups: Groups, site_cost: float, climate_weight: dict, regret_base=None,
          time_limit: float = 600.0, gap: float = 1e-4):
    """Stochastic plan for climate weights {climate: probability}.

    With `regret_base` {climate: optimal cost when that climate is known}, minimises the
    largest regret over the climates instead (minimax regret).
    Returns plan {hotspot: standard}, objective and solver info.
    """
    hot = sorted({h for hs in groups.sites for h in hs})
    hi = {h: i for i, h in enumerate(hot)}
    std = list(im.STANDARDS)
    nx = len(hot) * len(std)
    ng = len(groups.weight)
    h_ = highspy.Highs()
    h_.setOptionValue("output_flag", False)
    h_.setOptionValue("time_limit", time_limit)
    h_.setOptionValue("mip_rel_gap", gap)
    inf = highspy.kHighsInf
    robust = regret_base is not None
    # x variables, then y variables, then (robust) the regret r
    for i in range(nx):
        k = std[i % len(std)]
        c = 0.0 if robust else site_cost * COST_RATIO[k]
        h_.addVar(0.0, 1.0)
        h_.changeColCost(i, c)
    for j in range(ng):
        cl = groups.climate[j]
        c = 0.0 if robust else -climate_weight.get(cl, 0.0) * groups.weight[j]
        h_.addVar(0.0, 1.0)
        h_.changeColCost(nx + j, c)
    h_.changeColsIntegrality(nx, np.arange(nx, dtype=np.int32),
                             np.full(nx, highspy.HighsVarType.kInteger))
    for i in range(len(hot)):
        idx = np.array([i * len(std) + a for a in range(len(std))], np.int32)
        h_.addRow(-inf, 1.0, len(idx), idx, np.ones(len(idx)))
    for j in range(ng):
        sc = scs[groups.scen[j]]
        ok = [a for a, k in enumerate(std) if im.site_protected(k, sc.climate, sc.rp)]
        for h in groups.sites[j]:
            idx = np.array([nx + j] + [hi[h] * len(std) + a for a in ok], np.int32)
            val = np.array([1.0] + [-1.0] * len(ok))
            h_.addRow(-inf, 0.0, len(idx), idx, val)
    const = {cl: groups.weight[groups.climate == cl].sum() for cl in set(groups.climate)}
    if robust:
        r = nx + ng
        h_.addVar(-inf, inf)
        h_.changeColCost(r, 1.0)
        for cl, base in regret_base.items():
            # cost_cl(x, y) - base <= r
            idx = list(range(nx)) + [nx + j for j in range(ng) if groups.climate[j] == cl] + [r]
            val = [site_cost * COST_RATIO[std[i % len(std)]] for i in range(nx)] + \
                  [-groups.weight[j] for j in range(ng) if groups.climate[j] == cl] + [-1.0]
            h_.addRow(-inf, base - const.get(cl, 0.0), len(idx), np.array(idx, np.int32),
                      np.array(val))
    h_.run()
    sol = np.array(h_.getSolution().col_value)
    x = sol[:nx].round().astype(int)
    plan = {hot[i // len(std)]: std[i % len(std)] for i in np.flatnonzero(x)}
    info = h_.getInfo()
    y = sol[nx:nx + ng]
    cost_by = {}
    for cl in set(groups.climate):
        m = groups.climate == cl
        cost_by[cl] = site_cost * sum(COST_RATIO[k] for k in plan.values()) + \
            float((groups.weight[m] * (1 - y[m])).sum())
    return plan, cost_by, {"status": h_.modelStatusToString(h_.getModelStatus()),
                           "gap": info.mip_gap, "n_x": nx, "n_groups": ng,
                           "objective": info.objective_function_value}
