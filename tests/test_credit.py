"""Unit tests for the shared credit-policy logic.
Run with `pytest -q` or `python tests/test_credit.py`."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import credit as cr  # noqa: E402


def test_psi_is_zero_for_identical_distributions():
    x = np.random.default_rng(0).normal(size=5000)
    assert cr.psi(x, x) < 1e-9


def test_psi_detects_a_shift():
    rng = np.random.default_rng(1)
    assert cr.psi(rng.normal(size=5000), rng.normal(1, 1, 5000)) > 0.25


def test_binner_is_monotone():
    rng = np.random.default_rng(2)
    x = pd.Series(rng.uniform(0, 1, 20000))
    y = pd.Series((rng.uniform(size=20000) < 0.02 + 0.2 * x).astype(int))
    b = cr.Binner("x").fit(x, y)
    t = b.table_.drop(index="Missing", errors="ignore")
    order = sorted(t.index, key=lambda s: float(s.split(",")[0].strip("([")))
    rates = t.loc[order, "bad_rate"].values
    assert np.all(np.diff(rates) >= 0), rates


def test_decide_regions_are_ordered():
    """For a fixed loan, rising PD must move the action APPROVE -> REFER -> DECLINE, never back."""
    pdg = np.linspace(0, 0.5, 501)
    act, _, _ = cr.decide(pdg, np.full_like(pdg, 5000), np.full_like(pdg, 0.09), np.full_like(pdg, 0.68), {})
    rank = pd.Series(act).map({"APPROVE": 0, "REFER": 1, "DECLINE": 2}).values
    assert np.all(np.diff(rank) >= 0)
    assert act[0] == "APPROVE" and act[-1] == "DECLINE"


def test_breakeven_matches_formula():
    econ = pd.DataFrame({"net_good": [0.2], "py_good": [1.8], "net_bad": [-0.58], "py_bad": [0.27]}, index=["49-60m"])
    e = cr.unit_economics(econ, {"cost_of_funds": 0.06, "opex": 0.02}).iloc[0]
    assert abs(e.breakeven_pd - e.gain / (e.gain + e.loss)) < 1e-12
    # at break-even, approving is worth exactly zero
    ev = (1 - e.breakeven_pd) * e.gain - e.breakeven_pd * e.loss
    assert abs(ev) < 1e-12


def test_saved_scorecard_reproduces_saved_scores():
    card_path = ROOT / "artifacts/scorecard.json"
    if not card_path.exists():
        return  # artifacts are produced by the notebook
    card = json.load(open(card_path))
    t = pd.read_csv(ROOT / "artifacts/test_scored.csv.gz").sample(500, random_state=0)
    t["rating_rank"] = t.customer_risk_rating.map({g: i for i, g in enumerate(cr.RATING_ORDER)})
    feats = list(card["features"])
    for _, r in t.iterrows():
        app = {f: (None if pd.isna(r[f]) else r[f]) for f in feats}
        assert sum(cr.card_points(card, app).values()) == r.score


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
