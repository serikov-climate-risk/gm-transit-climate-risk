"""Route-node graph, zone connectors and generalised-cost shortest paths.

Node numbering: zones 0..Z-1, stop clusters Z..Z+S-1, then one line node per
(pattern, position). Edge kinds: 0 connector (zone <-> stop), 1 walk (stop <-> stop),
2 board, 3 ride, 4 alight. Ride edges keep the index of their row in `Network.ride`, so
a flooded ride edge can be removed or slowed in a scenario.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from . import network as nw

CONNECT_INSIDE = 5.0  # walk minutes assumed from a home or workplace to a stop in its zone
CONNECT_OUT_M = 300.0  # stops this close outside the zone boundary are also connected


@dataclass
class Graph:
    n_nodes: int
    n_zones: int
    tail: np.ndarray
    head: np.ndarray
    cost: np.ndarray  # generalised minutes
    kind: np.ndarray
    ride_row: np.ndarray  # row in Network.ride for ride edges, else -1

    def matrix(self, cost: np.ndarray | None = None, keep: np.ndarray | None = None) -> csr_matrix:
        c = self.cost if cost is None else cost
        m = np.ones(len(c), bool) if keep is None else keep
        # csgraph treats explicit zeros as missing edges; use a tiny positive cost instead
        return csr_matrix((np.maximum(c[m], 1e-6), (self.tail[m], self.head[m])),
                          shape=(self.n_nodes, self.n_nodes))

    def zone_costs(self, cost=None, keep=None, origins=None) -> np.ndarray:
        """Zone-to-zone generalised cost (minutes), inf when unreachable."""
        org = np.arange(self.n_zones) if origins is None else np.asarray(origins)
        d = dijkstra(self.matrix(cost, keep), directed=True, indices=org)
        return d[:, : self.n_zones]

    def paths(self, origin: int, cost=None, keep=None):
        d, pred = dijkstra(self.matrix(cost, keep), directed=True, indices=origin,
                           return_predecessors=True)
        return d, pred

    def edge_index(self) -> dict:
        return {(int(t), int(h)): i for i, (t, h) in enumerate(zip(self.tail, self.head))}


def point_in_polygon(px, py, ring: np.ndarray) -> np.ndarray:
    x, y = ring[:, 0], ring[:, 1]
    inside = np.zeros(len(px), bool)
    j = len(ring) - 1
    for i in range(len(ring)):
        cond = ((y[i] > py) != (y[j] > py)) & (
            px < (x[j] - x[i]) * (py - y[i]) / (y[j] - y[i] + 1e-12) + x[i])
        inside ^= cond
        j = i
    return inside


def zone_rings(geojson: dict) -> list[list[np.ndarray]]:
    out = []
    for f in geojson["features"]:
        g = f["geometry"]
        polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        out.append([np.asarray(p[0]) for p in polys])
    return out


def build(net: nw.Network, zones: pd.DataFrame, rings: list[list[np.ndarray]]) -> Graph:
    Z, S = len(zones), net.n_stop
    sx, sy = net.stops.x.to_numpy(), net.stops.y.to_numpy()
    tail, head, cost, kind, rrow = [], [], [], [], []

    def add(t, h, c, k, r=-1):
        tail.append(t); head.append(h); cost.append(c); kind.append(k); rrow.append(r)

    # connectors: stops inside the zone, plus stops just outside its boundary
    for z, polys in enumerate(rings):
        inside = np.zeros(S, bool)
        for ring in polys:
            inside |= point_in_polygon(sx, sy, ring)
        bnd = np.vstack([nw.densify(r, 50.0) for r in polys])
        dist, _ = cKDTree(bnd).query(np.c_[sx, sy])
        near = (~inside) & (dist < CONNECT_OUT_M)
        for s in np.flatnonzero(inside):
            c = nw.WALK_WEIGHT * CONNECT_INSIDE
            add(z, Z + s, c, 0); add(Z + s, z, c, 0)
        for s in np.flatnonzero(near):
            c = nw.WALK_WEIGHT * (CONNECT_INSIDE + nw.WALK_DETOUR * dist[s] / nw.WALK_SPEED)
            add(z, Z + s, c, 0); add(Z + s, z, c, 0)

    # walking between stop clusters
    tree = cKDTree(np.c_[sx, sy])
    for i, j in tree.query_pairs(nw.WALK_M):
        d = np.hypot(sx[i] - sx[j], sy[i] - sy[j])
        c = nw.WALK_WEIGHT * nw.WALK_DETOUR * d / nw.WALK_SPEED
        add(Z + i, Z + j, c, 1); add(Z + j, Z + i, c, 1)

    # line nodes
    ride = net.ride
    keys = pd.MultiIndex.from_frame(pd.concat([
        ride[["pattern", "pos"]], ride[["pattern"]].assign(pos=ride.pos + 1)]).drop_duplicates())
    line_id = pd.Series(np.arange(len(keys)) + Z + S, index=keys)
    for r, row in enumerate(ride.itertuples(index=False)):
        a = line_id[(row.pattern, row.pos)]
        b = line_id[(row.pattern, row.pos + 1)]
        add(Z + row.u, a, nw.WAIT_WEIGHT * row.wait + nw.BOARD_PENALTY, 2)
        add(a, b, row.minutes, 3, r)
        add(b, Z + row.v, 0.0, 4)
    n = Z + S + len(keys)
    g = Graph(n, Z, np.array(tail), np.array(head), np.array(cost, float),
              np.array(kind, np.int8), np.array(rrow))
    # drop duplicate board/alight edges created by repeated keys
    key = pd.DataFrame({"t": g.tail, "h": g.head, "k": g.kind})
    first = ~key.duplicated(["t", "h", "k"]).to_numpy()
    for f in ("tail", "head", "cost", "kind", "ride_row"):
        setattr(g, f, getattr(g, f)[first])
    return g
