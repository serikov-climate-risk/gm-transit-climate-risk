"""Download Environment Agency surface water flood maps (NaFRA2 RoFSW) for Greater Manchester.

The EA publishes the Risk of Flooding from Surface Water through a WMS (Open Government
Licence). Its GetMap request also returns the polygons as GeoJSON, with the likelihood
band of each polygon (High: 3.3% annual chance or more, Medium: 1%, Low: 0.1%). The
polygons are fetched per 5 km tile and burnt into a 5 m raster per layer, one bit per band
(1 High, 2 Medium, 4 Low); the bands overlap, so a cell can carry several bits.

Layers: flooding of any depth and of 0.3 m or more, for the present climate and for the
EA climate change scenario 1 (UKCP18 central allowance, 2050s epoch 2040-2060, RCP8.5).
Only tiles crossed by the network are needed (`--tiles`); tiles already burnt are skipped, so the script can be rerun after a timeout.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw

SERVICES = {
    "present": "https://environment.data.gov.uk/spatialdata/nafra2-risk-of-flooding-from-surface-water/wms",
    "cc01": "https://environment.data.gov.uk/spatialdata/nafra2-risk-of-flooding-from-surface-water-climate-change/wms",
}
LAYERS = {
    "present": {"any": "rofsw", "d03": "rofsw_0_3m_depth"},
    "cc01": {"any": "rofsw_cc01", "d03": "rofsw_cc01_0_3m_depth"},
}
BITS = {"High": 1, "Medium": 2, "Low": 4}
# Greater Manchester, British National Grid (EPSG:27700), rounded outwards to 5 km
X0, X1, Y0, Y1 = 350000, 410000, 380000, 425000
TILE_M, RES = 5000, 5.0
NPX = int(TILE_M / RES)


def tiles():
    for x in range(X0, X1, TILE_M):
        for y in range(Y0, Y1, TILE_M):
            yield x, y


def burn(features: list, x0: float, y0: float) -> np.ndarray:
    """Rasterise polygons of a tile; row 0 is the northern edge."""
    out = np.zeros((NPX, NPX), np.uint8)
    for band, bit in BITS.items():
        img = Image.new("1", (NPX, NPX), 0)
        draw = ImageDraw.Draw(img)
        for f in features:
            if f["properties"].get("risk_band") != band or not f.get("geometry"):
                continue
            g = f["geometry"]
            polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
            for poly in polys:
                for k, ring in enumerate(poly):
                    r = np.asarray(ring, float)
                    px = (r[:, 0] - x0) / RES
                    py = (y0 + TILE_M - r[:, 1]) / RES
                    draw.polygon(list(zip(px, py)), fill=0 if k else 1)
        out |= np.array(img, np.uint8) * bit
    return out


def fetch(url: str, params: dict, timeout: int, retries: int):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200 and "json" in r.headers.get("content-type", ""):
                return r.json()
            print(f"  HTTP {r.status_code} {r.headers.get('content-type')}", flush=True)
        except (requests.RequestException, json.JSONDecodeError) as exc:
            print(f"  {type(exc).__name__}", flush=True)
        time.sleep(5 * (attempt + 1))
    return None


def one(clim: str, depth: str, x: int, y: int, out: Path, timeout: int, retries: int) -> bool:
    f = out / f"{clim}_{depth}_{x}_{y}.npy"
    if f.exists():
        return True
    params = {
        "service": "WMS", "version": "1.3.0", "request": "GetMap",
        "layers": LAYERS[clim][depth], "styles": "", "crs": "EPSG:27700",
        "bbox": f"{x},{y},{x + TILE_M},{y + TILE_M}", "width": 500, "height": 500,
        "format": "application/json;type=geojson",
    }
    t0 = time.time()
    d = fetch(SERVICES[clim], params, timeout, retries)
    if d is None:
        print(f"{f.name} FAILED", flush=True)
        return False
    np.save(f, burn(d["features"], x, y))
    print(f"{f.name} {len(d['features'])} features {time.time() - t0:.1f}s", flush=True)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/flood_tiles")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--depth", nargs="+", default=["d03", "any"])
    ap.add_argument("--tiles", default=None, help="file with 'x y' lower-left corners to fetch")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    todo = list(tiles())
    if args.tiles:
        keep = {tuple(map(int, ln.split())) for ln in Path(args.tiles).read_text().splitlines() if ln}
        todo = [t for t in todo if t in keep]
    jobs = [(c, d, x, y) for d in args.depth for c in LAYERS for x, y in todo]
    with ThreadPoolExecutor(args.workers) as ex:
        res = list(ex.map(lambda j: one(*j, out, args.timeout, args.retries), jobs))
    failed = res.count(False)
    print(f"done, {failed} failed")


if __name__ == "__main__":
    main()
