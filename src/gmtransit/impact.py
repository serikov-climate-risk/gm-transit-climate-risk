"""Flood events, their effect on travel and the welfare loss of commuters.

Events. Surface water floods come from convective storms a few kilometres across, so the
region is divided into square storm cells of CELL_KM; a flood event floods one cell at
the extent of one return period (30, 100 or 1000 years) and leaves the rest dry. Cells
are independent, so the expected annual loss is the sum over cells and return-period bins
of the bin probability times the event loss. The region-wide case (all cells flood
together) is the other extreme of spatial correlation: larger events, but fewer of them.

Adaptation. A measure at a hotspot keeps the site dry up to a design standard (the
present-climate 1-in-100 or 1-in-1000 storm). Under the 2050s climate change scenario
(UKCP18 central allowance, about a quarter more peak rainfall) an event of return period
T is taken to be as intense as a present event of about three times T, so a site designed
for the present 1-in-100 storm stays dry in the 2050s 1-in-30 event but floods in the
2050s 1-in-100 event: the measure reaches its limit.

Loss. The event costs commuters on one working day (both directions); the change in
logsum of the mode choice model gives the welfare loss in generalised minutes, valued at
VOT per hour. Event probability on a working day is 5/7.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import demand as dm
from . import graph as gr
from . import hazard as hz
from . import network as nw

CELL_KM = 10.0
VOT = 12.0  # GBP per generalised hour, illustrative
TRIPS_PER_DAY = 2
WORKDAY = 5 / 7
RP_BINS = {30: 1 / 30 - 1 / 100, 100: 1 / 100 - 1 / 1000, 1000: 1 / 1000}
STANDARDS = (100, 1000)
CC_FACTOR = 3  # 2050s event of return period T ~ present event of CC_FACTOR * T


def site_protected(standard: int, climate: str, rp: int) -> bool:
    eq = rp * (CC_FACTOR if climate == "cc01" else 1)
    return standard >= eq


def cell_of(x: float, y: float, cell_km: float = CELL_KM) -> int:
    s = cell_km * 1000
    return int((x - hz.X0) // s) * 1000 + int((y - hz.Y0) // s)


@dataclass
class Scenario:
    climate: str
    rp: int
    cell: int  # -1 = whole region
    prob: float
    # ride row -> hotspots of the edge that flood in this event
    sites: dict = field(default_factory=dict)
    shallow_rows: np.ndarray | None = None
    shallow_share: np.ndarray | None = None


def scenarios(net: nw.Network, floods: dict, cell_km: float = CELL_KM,
              region_wide: bool = False) -> list[Scenario]:
    mid = np.array([g[len(g) // 2] for g in (net.geometry[i] for i in range(len(net.ride)))])
    row_cell = np.array([cell_of(x, y, cell_km) for x, y in mid])
    out = []
    for clim, ef in floods.items():
        hcell = {h: cell_of(*xy, cell_km) for h, xy in ef.hotspot_xy.items()}
        for rp, p in RP_BINS.items():
            sh = ef.shallow_frac[rp]
            if region_wide:
                cells = [-1]
            else:
                cells = sorted({hcell[h] for v in ef.deep_sites[rp].values() for h in v}
                               | set(row_cell[sh > 0]))
            for c in cells:
                sites = {}
                for r, hs in ef.deep_sites[rp].items():
                    k = tuple(h for h in hs if c == -1 or hcell[h] == c)
                    if k:
                        sites[r] = k
                m = (sh > 0) & ((row_cell == c) | (c == -1))
                out.append(Scenario(clim, rp, c, p, sites, np.flatnonzero(m), sh[m]))
    return out


def edge_costs(g: gr.Graph, net: nw.Network, sc: Scenario, plan: dict | None = None):
    """Edge costs and kept-edge mask in a scenario, given a plan {hotspot: standard}."""
    plan = plan or {}
    cost = g.cost.copy()
    keep = np.ones(len(cost), bool)
    ride_edge = np.full(len(net.ride), -1)
    m = g.ride_row >= 0
    ride_edge[g.ride_row[m]] = np.flatnonzero(m)
    mode = net.ride.pattern.map(net.patterns.set_index("pattern")["mode"]).to_numpy()
    if sc.shallow_rows is not None and len(sc.shallow_rows):
        e = ride_edge[sc.shallow_rows]
        cost[e] *= 1 + sc.shallow_share * (hz.SHALLOW_SLOWDOWN - 1)
    for r, hs in sc.sites.items():
        n = sum(1 for h in hs if not (h in plan and site_protected(plan[h], sc.climate, sc.rp)))
        if n == 0:
            continue
        e = ride_edge[r]
        if mode[r] == "tram":
            keep[e] = False
        else:
            cost[e] += n * hz.BUS_DIVERSION
    return cost, keep


def invert_paths(paths, n_rows: int) -> list[np.ndarray]:
    """Ride row -> OD pairs whose intact best path uses it."""
    acc = [[] for _ in range(n_rows)]
    for i, rows in enumerate(paths):
        for r in rows:
            acc[r].append(i)
    return [np.array(a, int) for a in acc]


def event_loss(g, net, logit, od, t, gc0, sc, plan=None, users=None):
    """Per-OD welfare loss of one event (GBP), PT trips lost and new PT costs.

    Costs only rise in an event, so an OD pair whose intact best path avoids every
    affected edge keeps its cost; with `users` (from `invert_paths`) only the origins of
    the other pairs are re-solved.
    """
    cost, keep = edge_costs(g, net, sc, plan)
    gc = gc0.copy()
    if users is None:
        C = g.zone_costs(cost, keep)
        gc = C[od.o, od.d]
    else:
        changed = (cost != g.cost) | ~keep
        rows = g.ride_row[changed & (g.ride_row >= 0)]
        hit = np.unique(np.concatenate([users[r] for r in rows])) if len(rows) else np.array([], int)
        if len(hit):
            org = np.unique(od.o.to_numpy()[hit])
            C = g.zone_costs(cost, keep, origins=org)
            pos = np.searchsorted(org, od.o.to_numpy()[hit])
            gc[hit] = C[pos, od.d.to_numpy()[hit]]
    n = od[dm.MODES].sum(axis=1).to_numpy()
    w = logit.welfare_loss(gc0, gc, t)
    gbp_od = n * w * VOT / 60 * TRIPS_PER_DAY
    pt_lost = n * (logit.shares(gc0, t)[:, 0] - logit.shares(gc, t)[:, 0])
    return gbp_od, pt_lost, gc


def expected_annual_loss(g, net, logit, od, t, gc0, scs, plan=None, climate=None, users=None):
    rows = []
    for sc in scs:
        if climate and sc.climate != climate:
            continue
        gbp, lost, _ = event_loss(g, net, logit, od, t, gc0, sc, plan, users)
        rows.append({"climate": sc.climate, "rp": sc.rp, "cell": sc.cell, "prob": sc.prob,
                     "loss_gbp": gbp.sum(), "eal_gbp": sc.prob * WORKDAY * gbp.sum(),
                     "pt_trips_lost": lost.sum()})
    return pd.DataFrame(rows)
