"""Journey-to-work demand and an aggregate multinomial logit mode choice model.

Demand: 2011 Census flows between the 346 Greater Manchester MSOAs (WU03EW). Modes in the
choice set: public transport (bus, minibus, coach, metro, tram), car (driver and
passenger), bicycle and walking. Rail commuters (not in the bus and tram feed),
home workers and other modes are left out; intrazonal flows are left out because zone
centroids say little about their distance.

Utilities: V_pt = b_t * GC_pt, V_m = ASC_m + b_t * t_m + g_m * ln(km) for car, bicycle
and walk, with GC_pt the generalised cost on the network (in-vehicle minutes with walking
and waiting weighted), t_m door-to-door times from the centroid distance and one time
coefficient for all modes; the log-distance terms let the attractiveness of a mode
change with trip length (car ownership and parking, effort of walking and cycling). The parameters are
estimated by maximum likelihood on the observed counts (an aggregate logit). Welfare
changes use the logsum, expressed in generalised public transport minutes (divided by
-b_pt).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

MODES = ["pt", "car", "bicycle", "walk"]
DETOUR = 1.3
CAR_KMH, BIKE_KMH, WALK_KMH = 25.0, 15.0, 4.8
CAR_TERMINAL = 5.0  # minutes for parking and walking at both ends


def load_od(path: str, zones: pd.DataFrame) -> pd.DataFrame:
    od = pd.read_csv(path)
    od = od.pivot_table(index=["usual_residence_code", "place_of_work_code"], columns="mode",
                        values="obs_value", aggfunc="sum", fill_value=0).reset_index()
    idx = {c: i for i, c in enumerate(zones.msoa)}
    od["o"] = od.usual_residence_code.map(idx)
    od["d"] = od.place_of_work_code.map(idx)
    od = od.dropna(subset=["o", "d"])
    od[["o", "d"]] = od[["o", "d"]].astype(int)
    out = pd.DataFrame({"o": od.o, "d": od.d})
    out["pt"] = od.get("bus", 0) + od.get("metro_tram", 0)
    out["car"] = od.get("car_driver", 0) + od.get("car_passenger", 0)
    out["bicycle"] = od.get("bicycle", 0)
    out["walk"] = od.get("foot", 0)
    out["tram"] = od.get("metro_tram", 0)
    out = out[(out.o != out.d) & (out[MODES].sum(axis=1) > 0)].reset_index(drop=True)
    return out


def mode_times(od: pd.DataFrame, zones: pd.DataFrame) -> pd.DataFrame:
    x, y = zones.x.to_numpy(), zones.y.to_numpy()
    km = DETOUR * np.hypot(x[od.o] - x[od.d], y[od.o] - y[od.d]) / 1000
    return pd.DataFrame({
        "km": km,
        "car": CAR_TERMINAL + 60 * km / CAR_KMH,
        "bicycle": 60 * km / BIKE_KMH,
        "walk": 60 * km / WALK_KMH,
    })


@dataclass
class Logit:
    b_pt: float
    asc: dict
    b: dict  # log-distance coefficients by mode
    loglik: float
    loglik0: float
    n: float

    @property
    def rho2(self) -> float:
        return 1 - self.loglik / self.loglik0

    def utilities(self, gc_pt: np.ndarray, t: pd.DataFrame) -> np.ndarray:
        V = np.empty((len(gc_pt), 4))
        with np.errstate(invalid="ignore"):
            V[:, 0] = np.where(np.isfinite(gc_pt), self.b_pt * gc_pt, -np.inf)
        lk = np.log(t["km"].to_numpy())
        for j, m in enumerate(MODES[1:], start=1):
            V[:, j] = self.asc[m] + self.b_pt * t[m].to_numpy() + self.b[m] * lk
        return V

    def shares(self, gc_pt, t) -> np.ndarray:
        V = self.utilities(gc_pt, t)
        return np.exp(V - logsumexp(V, axis=1, keepdims=True))

    def logsum(self, gc_pt, t) -> np.ndarray:
        return logsumexp(self.utilities(gc_pt, t), axis=1)

    def welfare_loss(self, gc_base, gc_new, t) -> np.ndarray:
        """Loss per traveller, in generalised public transport minutes."""
        return (self.logsum(gc_base, t) - self.logsum(gc_new, t)) / -self.b_pt


def estimate(od: pd.DataFrame, gc_pt: np.ndarray, t: pd.DataFrame) -> Logit:
    N = od[MODES].to_numpy(float)
    ok = np.isfinite(gc_pt)
    N, gc, tt = N[ok], gc_pt[ok], t[ok].reset_index(drop=True)

    def unpack(p):
        return Logit(p[0], dict(zip(MODES[1:], p[1:4])), dict(zip(MODES[1:], p[4:7])), 0, 0, 0)

    def nll(p):
        V = unpack(p).utilities(gc, tt)
        lp = V - logsumexp(V, axis=1, keepdims=True)
        return -(N * lp).sum()

    p0 = np.array([-0.03, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0])
    res = minimize(nll, p0, method="BFGS", options={"maxiter": 2000})
    m = unpack(res.x)
    m.loglik = -res.fun
    m.loglik0 = (N.sum(axis=1) * np.log(1 / 4)).sum()
    m.n = N.sum()
    m.se = np.sqrt(np.diag(res.hess_inv)) if hasattr(res, "hess_inv") else None
    return m
