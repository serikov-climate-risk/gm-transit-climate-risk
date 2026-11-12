import numpy as np
import pandas as pd

from gmtransit import demand as dm
from gmtransit import graph as gr
from gmtransit import hazard as hz
from gmtransit import impact as im
from gmtransit import network as nw
from gmtransit import optimize as op


def test_pregnolato_speed():
    assert abs(hz.pregnolato_speed(0) - 86.9448) < 1e-9
    assert hz.pregnolato_speed(300) == 0.0
    assert 1 < hz.SHALLOW_SLOWDOWN < 3


def test_cumulative_bands():
    bits = np.array([0, 1, 2, 4, 7, 6])
    assert cumulative_list(bits, 30) == [False, True, False, False, True, False]
    assert cumulative_list(bits, 1000) == [False, True, True, True, True, True]


def cumulative_list(bits, rp):
    return hz.cumulative(bits, rp).tolist()


def test_runs():
    f = np.array([0, 1, 1, 0, 1, 1, 1, 0], bool)
    assert hz.runs(f) == [(1, 3), (4, 7)]


def test_site_protected_reaches_limit_under_climate_change():
    assert im.site_protected(100, "present", 100)
    assert not im.site_protected(100, "present", 1000)
    assert im.site_protected(100, "cc01", 30)
    assert not im.site_protected(100, "cc01", 100)
    assert im.site_protected(1000, "cc01", 100)
    assert not im.site_protected(1000, "cc01", 1000)


def test_cut_shape_and_densify():
    pts = np.array([[0, 0], [10, 0], [20, 0]], float)
    dist = np.array([0, 10, 20], float)
    g = nw._cut_shape(pts, dist, 5, 15)
    assert np.allclose(g, [[5, 0], [10, 0], [15, 0]])
    d = nw.densify(np.array([[0, 0], [0, 20]], float), 5)
    assert len(d) == 5


def test_point_in_polygon():
    ring = np.array([[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]], float)
    inside = gr.point_in_polygon(np.array([5, 15]), np.array([5, 5]), ring)
    assert inside.tolist() == [True, False]


def _logit():
    return dm.Logit(-0.05, {"car": 0.0, "bicycle": -2.0, "walk": 0.0},
                    {"car": 0.0, "bicycle": 0.0, "walk": -1.0}, 0, 0, 0)


def test_welfare_loss_bounds():
    L = _logit()
    t = pd.DataFrame({"km": [5.0, 5.0], "car": [20.0, 20.0], "bicycle": [20.0, 20.0],
                      "walk": [60.0, 60.0]})
    gc0 = np.array([30.0, 30.0])
    gc1 = np.array([40.0, np.inf])
    w = L.welfare_loss(gc0, gc1, t)
    assert (w > 0).all()
    assert w[0] < 10  # less than the cost increase, other modes absorb part of it
    assert w[1] > w[0]  # losing public transport costs more than a 10 minute delay


def test_estimate_recovers_parameters():
    rng = np.random.default_rng(0)
    n = 4000
    km = rng.uniform(1, 20, n)
    t = pd.DataFrame({"km": km, "car": 5 + 60 * km / 25, "bicycle": 60 * km / 15,
                      "walk": 60 * km / 4.8})
    gc = 20 + 4 * km + rng.uniform(0, 20, n)
    true = dm.Logit(-0.04, {"car": 0.5, "bicycle": -1.5, "walk": 1.0},
                    {"car": 0.2, "bicycle": -0.3, "walk": -0.5}, 0, 0, 0)
    P = true.shares(gc, t)
    N = np.array([rng.multinomial(200, p) for p in P])
    od = pd.DataFrame(N, columns=dm.MODES)
    est = dm.estimate(od, gc, t)
    assert abs(est.b_pt - true.b_pt) < 0.01
    assert abs(est.asc["bicycle"] - true.asc["bicycle"]) < 0.3


def test_optimizer_protects_when_cheap_and_not_when_dear():
    sc = [im.Scenario("present", 100, 0, 0.01, {0: (1,)}),
          im.Scenario("present", 1000, 0, 0.001, {0: (1, 2)})]
    groups = op.Groups(np.array([0, 1]), [(1,), (1, 2)], np.array([50.0, 30.0]),
                       np.array(["present", "present"]))
    plan, cost, info = op.solve(sc, groups, site_cost=4.0, climate_weight={"present": 1.0})
    assert plan == {1: 1000, 2: 1000}  # 1-in-1000 standard needed for the rare event
    plan, cost, info = op.solve(sc, groups, site_cost=40.0, climate_weight={"present": 1.0})
    assert plan == {1: 100}
    plan, cost, info = op.solve(sc, groups, site_cost=60.0, climate_weight={"present": 1.0})
    assert plan == {}


def test_minimax_regret_balances_climates():
    sc = [im.Scenario("present", 100, 0, 0.01, {0: (1,)}),
          im.Scenario("cc01", 100, 0, 0.01, {0: (2,)})]
    groups = op.Groups(np.array([0, 1]), [(1,), (2,)], np.array([30.0, 30.0]),
                       np.array(["present", "cc01"]))
    base = {"present": 20.0, "cc01": 30.0}
    plan, cost, info = op.solve(sc, groups, 20.0, {}, regret_base=base)
    assert info["status"] == "Optimal"
