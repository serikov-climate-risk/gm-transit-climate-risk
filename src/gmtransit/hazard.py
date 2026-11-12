"""Surface water flooding on the network: blocked segments, delays and flood hotspots.

Flood maps (see scripts/download_flood_tiles.py) are 5 m rasters with one bit per
likelihood band: 1 High (flooded in the 1-in-30 event), 2 Medium (1-in-100), 4 Low
(1-in-1000). The bands are nested in the data, and are read cumulatively here: a cell
flooded in a more likely band is also flooded in the rarer events.

Each ride edge is sampled every 5 m along its GTFS shape.
- Deep water (0.3 m or more) over at least MIN_RUN_M of an edge blocks it. A tram cannot
  pass and the segment is out of service; a bus diverts around the site at an assumed
  delay of BUS_DIVERSION minutes per site.
- Shallow water elsewhere on the edge slows it: travel time over the flooded length is
  divided by the speed ratio of the depth-disruption function of Pregnolato et al. (2017)
  at an assumed depth of SHALLOW_DEPTH_MM.
- Each deep run is assigned to a flood hotspot, the 100 m grid square holding its middle
  point. Hotspots are the candidate sites for adaptation measures (drainage, raising or
  sealing the track or road at that site).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import network as nw

X0, X1, Y0, Y1 = 350000, 410000, 380000, 425000
RES, TILE_M = 5.0, 5000
RP = {30: 1, 100: 2, 1000: 4}  # return period -> band bit
MIN_RUN_M = 20.0
BUS_DIVERSION = 8.0  # minutes per blocked site
SHALLOW_DEPTH_MM = 100.0
HOTSPOT_M = 100.0


def pregnolato_speed(depth_mm: float) -> float:
    """Vehicle speed (km/h) in standing water, Pregnolato et al. (2017); 0 from 300 mm."""
    if depth_mm >= 300:
        return 0.0
    return max(0.0009 * depth_mm ** 2 - 0.5529 * depth_mm + 86.9448, 0.0)


SHALLOW_SLOWDOWN = pregnolato_speed(0.0) / pregnolato_speed(SHALLOW_DEPTH_MM)


def mosaic(tile_dir: Path, climate: str, depth: str) -> np.ndarray:
    ny, nx = int((Y1 - Y0) / RES), int((X1 - X0) / RES)
    out = np.zeros((ny, nx), np.uint8)
    n = int(TILE_M / RES)
    missing = 0
    for x in range(X0, X1, TILE_M):
        for y in range(Y0, Y1, TILE_M):
            f = tile_dir / f"{climate}_{depth}_{x}_{y}.npy"
            if not f.exists():
                missing += 1
                continue
            r0 = int((Y1 - y - TILE_M) / RES)
            c0 = int((x - X0) / RES)
            out[r0:r0 + n, c0:c0 + n] = np.load(f)
    if missing:
        print(f"warning: {missing} tiles missing for {climate} {depth}")
    return out


def cumulative(bits: np.ndarray, rp: int) -> np.ndarray:
    """Flooded in the 1-in-rp event: the band of rp or any more likely band."""
    mask = sum(b for r, b in RP.items() if r <= rp)
    return (bits & mask) > 0


def sample(raster: np.ndarray, pts: np.ndarray) -> np.ndarray:
    c = ((pts[:, 0] - X0) / RES).astype(int)
    r = ((Y1 - pts[:, 1]) / RES).astype(int)
    ok = (r >= 0) & (r < raster.shape[0]) & (c >= 0) & (c < raster.shape[1])
    out = np.zeros(len(pts), raster.dtype)
    out[ok] = raster[r[ok], c[ok]]
    return out


def runs(flag: np.ndarray) -> list[tuple[int, int]]:
    """Start and end (exclusive) of runs of True."""
    d = np.diff(np.r_[0, flag.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


@dataclass
class EdgeFlood:
    """Flooding of ride edges for one climate case and all return periods."""
    climate: str
    deep_sites: dict  # rp -> {ride row: tuple of hotspot ids}
    shallow_frac: dict  # rp -> array over ride rows, flooded share of length (not deep)
    hotspot_xy: dict  # hotspot id -> (x, y)


def hotspot_id(x: float, y: float) -> int:
    return int(x // HOTSPOT_M) * 100000 + int(y // HOTSPOT_M)


def edge_flooding(net: nw.Network, deep: np.ndarray, anyd: np.ndarray | None, climate: str,
                  step: float = RES) -> EdgeFlood:
    min_n = int(np.ceil(MIN_RUN_M / step))
    deep_sites = {rp: {} for rp in RP}
    shallow = {rp: np.zeros(len(net.ride)) for rp in RP}
    hxy = {}
    for i in range(len(net.ride)):
        pts = nw.densify(net.geometry[i], step)
        if len(pts) < 2:
            continue
        db = sample(deep, pts)
        ab = sample(anyd, pts) if anyd is not None else None
        for rp in RP:
            f = cumulative(db, rp)
            sites = []
            if f.any():
                for a, b in runs(f):
                    if b - a >= min_n:
                        mid = pts[(a + b) // 2]
                        h = hotspot_id(*mid)
                        hxy.setdefault(h, (float(mid[0]), float(mid[1])))
                        sites.append(h)
            if sites:
                deep_sites[rp][i] = tuple(dict.fromkeys(sites))
            if ab is not None:
                s = cumulative(ab, rp) & ~f
                shallow[rp][i] = s.mean()
    return EdgeFlood(climate, deep_sites, shallow, hxy)


def summary(net: nw.Network, ef: EdgeFlood) -> pd.DataFrame:
    rows = []
    mode = net.ride.pattern.map(net.patterns.set_index("pattern")["mode"]).to_numpy()
    for rp in RP:
        blocked = np.array(sorted(ef.deep_sites[rp]), int)
        hs = {h for v in ef.deep_sites[rp].values() for h in v}
        rows.append({"climate": ef.climate, "rp": rp, "blocked_edges": len(blocked),
                     "blocked_bus": int((mode[blocked] == "bus").sum()) if len(blocked) else 0,
                     "blocked_tram": int((mode[blocked] == "tram").sum()) if len(blocked) else 0,
                     "hotspots": len(hs),
                     "mean_shallow_share": float(ef.shallow_frac[rp].mean())})
    return pd.DataFrame(rows)
