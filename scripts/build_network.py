"""Build the network, the graph and the zone data, and save them to data/processed."""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from gmtransit import graph as gr
from gmtransit import network as nw

out = Path("data/processed")
out.mkdir(parents=True, exist_ok=True)
net = nw.build(Path("data/raw/gm_gtfs.zip"))
gj = json.load(open("data/raw/msoa2011_gm.geojson"))
rings = gr.zone_rings(gj)
rows = []
for f, polys in zip(gj["features"], rings):
    r = max(polys, key=len)
    x, y = r[:, 0], r[:, 1]
    a = 0.5 * (x[:-1] * y[1:] - x[1:] * y[:-1])
    A = a.sum()
    cx = ((x[:-1] + x[1:]) * a).sum() / (3 * A)
    cy = ((y[:-1] + y[1:]) * a).sum() / (3 * A)
    area = sum(abs(0.5 * (p[:-1, 0] * p[1:, 1] - p[1:, 0] * p[:-1, 1]).sum()) for p in polys)
    rows.append({"msoa": f["properties"]["MSOA11CD"], "name": f["properties"]["MSOA11NM"],
                 "x": cx, "y": cy, "area_km2": area / 1e6})
zones = pd.DataFrame(rows)
g = gr.build(net, zones, rings)
print("stops", net.n_stop, "patterns", len(net.patterns), "ride", len(net.ride))
print("nodes", g.n_nodes, "edges", len(g.tail), np.bincount(g.kind))
pickle.dump({"net": net, "graph": g, "zones": zones}, open(out / "model.pkl", "wb"))

# 5 km flood map tiles crossed by the network (for scripts/download_flood_tiles.py)
pts = np.vstack([nw.densify(geo, 50) for geo in net.geometry.values()])
x0, x1, y0, y1 = nw.BBOX
tiles = {(int(x // 5000 * 5000), int(y // 5000 * 5000)) for x, y in pts
         if x0 <= x < x1 and y0 <= y < y1}
Path("data/raw/needed_tiles.txt").write_text(
    "\n".join(f"{a} {b}" for a, b in sorted(tiles)))
zones.to_csv(out / "zones.csv", index=False)
