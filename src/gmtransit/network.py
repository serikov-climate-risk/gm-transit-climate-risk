"""Public transport network of Greater Manchester from the TfGM GTFS feed.

The network is a frequency-based route-node graph for the weekday morning peak:
- a stop node per stop cluster (GTFS stops within `CLUSTER_M` of each other merged),
- a line node per (service pattern, stop position),
- board edges stop -> line node carrying the expected wait (half the headway, capped)
  plus a boarding penalty, ride edges between consecutive line nodes carrying the
  scheduled run time, alight edges line node -> stop at no cost, and walk edges between
  stop clusters within `WALK_M`.
Generalised cost is in in-vehicle minutes; walking and waiting are weighted.
Each ride edge keeps its GTFS shape geometry, so flood maps can be sampled along it.
"""
from __future__ import annotations

import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

BBOX = (350000, 410000, 380000, 425000)  # Greater Manchester, EPSG:27700
CLUSTER_M = 120.0
WALK_M = 500.0
WALK_SPEED = 80.0  # m per minute (4.8 km/h)
WALK_DETOUR = 1.25
WALK_WEIGHT = 2.0
WAIT_WEIGHT = 2.0
WAIT_CAP = 15.0  # minutes of expected wait at most
BOARD_PENALTY = 5.0  # minutes per boarding
PEAK = (7 * 3600, 9 * 3600)

_TO_BNG = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)


def to_bng(lon, lat):
    return _TO_BNG.transform(np.asarray(lon), np.asarray(lat))


def _secs(s: pd.Series) -> np.ndarray:
    p = s.str.split(":", expand=True).astype(int)
    return (p[0] * 3600 + p[1] * 60 + p[2]).to_numpy()


def active_services(z: zipfile.ZipFile, date: str) -> set:
    cal = pd.read_csv(z.open("calendar.txt"))
    day = pd.Timestamp(date).day_name().lower()
    d = int(date.replace("-", ""))
    s = set(cal.loc[(cal[day] == 1) & (cal.start_date <= d) & (cal.end_date >= d), "service_id"])
    cd = pd.read_csv(z.open("calendar_dates.txt"))
    cd = cd[cd.date == d]
    s |= set(cd.loc[cd.exception_type == 1, "service_id"])
    s -= set(cd.loc[cd.exception_type == 2, "service_id"])
    return s


@dataclass
class Network:
    stops: pd.DataFrame  # cluster id, x, y, name, n_gtfs
    patterns: pd.DataFrame  # pattern id, route, mode, trips in peak, headway
    ride: pd.DataFrame  # pattern, seq, from/to cluster, minutes, shape geometry key
    geometry: dict = field(default_factory=dict)  # ride index -> (n, 2) array of BNG points
    stop_of_gtfs: dict = field(default_factory=dict)

    @property
    def n_stop(self) -> int:
        return len(self.stops)


def _cluster(x: np.ndarray, y: np.ndarray, r: float) -> np.ndarray:
    tree = cKDTree(np.c_[x, y])
    parent = np.arange(len(x))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in tree.query_pairs(r):
        a, b = find(i), find(j)
        if a != b:
            parent[max(a, b)] = min(a, b)
    roots = np.array([find(i) for i in range(len(x))])
    _, lab = np.unique(roots, return_inverse=True)
    return lab


def _cut_shape(pts: np.ndarray, dist: np.ndarray, d0: float, d1: float) -> np.ndarray:
    """Part of a shape polyline between two distances along it."""
    if d1 <= d0:
        return np.empty((0, 2))
    inner = (dist > d0) & (dist < d1)
    p0 = np.c_[np.interp(d0, dist, pts[:, 0]), np.interp(d0, dist, pts[:, 1])]
    p1 = np.c_[np.interp(d1, dist, pts[:, 0]), np.interp(d1, dist, pts[:, 1])]
    return np.vstack([p0, pts[inner], p1])


def build(gtfs: Path, date: str = "2026-11-17") -> Network:
    z = zipfile.ZipFile(gtfs)
    services = active_services(z, date)
    trips = pd.read_csv(z.open("trips.txt"), dtype={"shape_id": str, "trip_id": str})
    trips = trips[trips.service_id.isin(services)]
    routes = pd.read_csv(z.open("routes.txt"), dtype={"route_short_name": str})
    trips = trips.merge(routes[["route_id", "route_short_name", "route_type"]], on="route_id")

    st = pd.read_csv(z.open("stop_times.txt"), dtype={"trip_id": str},
                     usecols=["trip_id", "arrival_time", "departure_time", "stop_id",
                              "stop_sequence", "shape_dist_traveled"])
    st = st[st.trip_id.isin(set(trips.trip_id))].sort_values(["trip_id", "stop_sequence"])
    st["arr"] = _secs(st.arrival_time)
    st["dep"] = _secs(st.departure_time)

    stops = pd.read_csv(z.open("stops.txt"))
    stops["x"], stops["y"] = to_bng(stops.stop_lon, stops.stop_lat)
    x0, x1, y0, y1 = BBOX
    stops = stops[(stops.x >= x0) & (stops.x < x1) & (stops.y >= y0) & (stops.y < y1)]
    stops = stops[stops.stop_id.isin(set(st.stop_id))].reset_index(drop=True)
    stops["cluster"] = _cluster(stops.x.to_numpy(), stops.y.to_numpy(), CLUSTER_M)
    cl = (stops.groupby("cluster")
          .agg(x=("x", "mean"), y=("y", "mean"), name=("stop_name", "first"), n_gtfs=("stop_id", "size"))
          .reset_index())
    stop_of = dict(zip(stops.stop_id, stops.cluster))

    st = st[st.stop_id.isin(stop_of)]
    st["cluster"] = st.stop_id.map(stop_of)
    # drop consecutive repeats inside one cluster
    st = st[~((st.trip_id == st.trip_id.shift()) & (st.cluster == st.cluster.shift()))]

    # service patterns: route, direction and exact cluster sequence
    seq = st.groupby("trip_id").cluster.apply(tuple).rename("seq")
    first_dep = st.groupby("trip_id").dep.first()
    tp = trips.set_index("trip_id").join(seq, how="inner").join(first_dep)
    tp = tp[tp.seq.map(len) >= 2]
    tp["pattern_key"] = list(zip(tp.route_id, tp.direction_id, tp.seq))

    # trips that pass any stop of the pattern in the peak window
    st_peak = st[(st.dep >= PEAK[0]) & (st.dep < PEAK[1])]
    peak_trips = set(st_peak.trip_id)
    tp = tp[tp.index.isin(peak_trips)]
    keys = {k: i for i, k in enumerate(dict.fromkeys(tp.pattern_key))}
    tp["pattern"] = tp.pattern_key.map(keys)
    hours = (PEAK[1] - PEAK[0]) / 3600
    pat = (tp.reset_index().groupby("pattern")
           .agg(route=("route_short_name", "first"), route_type=("route_type", "first"),
                trips=("trip_id", "size"), shape_id=("shape_id", "first"),
                trip_example=("trip_id", "first"))
           .reset_index())
    pat["mode"] = np.where(pat.route_type == 0, "tram", "bus")
    pat["headway"] = 60 * hours / pat.trips
    pat["wait"] = np.minimum(pat.headway / 2, WAIT_CAP)

    # run times: median over the pattern's peak trips
    stp = st[st.trip_id.isin(set(tp.index))].copy()
    stp["pattern"] = stp.trip_id.map(tp.pattern)
    stp["pos"] = stp.groupby("trip_id").cumcount()
    stp["next_arr"] = stp.groupby("trip_id").arr.shift(-1)
    stp["run"] = (stp.next_arr - stp.dep) / 60
    stp["next_cluster"] = stp.groupby("trip_id").cluster.shift(-1)
    ride = (stp.dropna(subset=["next_cluster"])
            .groupby(["pattern", "pos"])
            .agg(u=("cluster", "first"), v=("next_cluster", "first"), minutes=("run", "median"))
            .reset_index())
    ride["v"] = ride.v.astype(int)
    ride["minutes"] = ride.minutes.clip(lower=0.25)

    # expected wait at boarding: departures of the route and direction at that stop in the
    # peak, over all its patterns, so route variants do not split the frequency
    pat_rd = tp.groupby("pattern")[["route_id", "direction_id"]].first()
    pat = pat.join(pat_rd, on="pattern")
    sp_ = st_peak.merge(trips[["trip_id", "route_id", "direction_id"]], on="trip_id")
    dep_n = sp_.groupby(["route_id", "direction_id", "cluster"]).size().rename("n_dep")
    ride = ride.join(pat.set_index("pattern")[["route_id", "direction_id"]], on="pattern")
    ride = ride.join(dep_n, on=["route_id", "direction_id", "u"])
    ride["n_dep"] = ride.n_dep.fillna(1)
    ride["wait"] = np.minimum(60 * hours / ride.n_dep / 2, WAIT_CAP)

    # geometry from the shape of an example trip per pattern
    shapes = pd.read_csv(z.open("shapes.txt"), dtype={"shape_id": str})
    shapes = shapes[shapes.shape_id.isin(set(pat.shape_id))].sort_values(["shape_id", "shape_pt_sequence"])
    shapes["x"], shapes["y"] = to_bng(shapes.shape_pt_lon, shapes.shape_pt_lat)
    shape_pts = {k: (g[["x", "y"]].to_numpy(), g.shape_dist_traveled.to_numpy())
                 for k, g in shapes.groupby("shape_id")}
    ex = st[st.trip_id.isin(set(pat.trip_example))]
    ex_d = {t: g.shape_dist_traveled.to_numpy() for t, g in ex.groupby("trip_id")}
    geometry = {}
    pat_i = pat.set_index("pattern")
    for i, r in enumerate(ride.itertuples(index=False)):
        p = pat_i.loc[r.pattern]
        sp = shape_pts.get(p.shape_id)
        d = ex_d.get(p.trip_example)
        k = int(r.pos)
        if sp is not None and d is not None and k + 1 < len(d) and np.isfinite(d[k + 1]):
            g = _cut_shape(sp[0], sp[1], d[k], d[k + 1])
        else:
            g = np.empty((0, 2))
        if len(g) < 2:  # straight line between the clusters as a fallback
            g = cl.loc[[r.u, r.v], ["x", "y"]].to_numpy()
        geometry[i] = g
    return Network(stops=cl, patterns=pat, ride=ride, geometry=geometry, stop_of_gtfs=stop_of)


def densify(g: np.ndarray, step: float = 10.0) -> np.ndarray:
    """Points every `step` metres along a polyline."""
    seg = np.diff(g, axis=0)
    ln = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.r_[0, np.cumsum(ln)]
    if cum[-1] == 0:
        return g[:1]
    s = np.arange(0, cum[-1] + 1e-9, step)
    return np.c_[np.interp(s, cum, g[:, 0]), np.interp(s, cum, g[:, 1])]
