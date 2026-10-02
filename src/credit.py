"""Reusable building blocks for the credit-policy project.

Everything the notebook and the Streamlit simulator share lives here so that the
policy that is analysed is exactly the policy that is simulated:

* feature engineering on application-time fields only
* a monotone WoE scorecard (binning, IV, logistic fit, points scaling)
* reason codes from scorecard points
* unit economics and the approve / refer / decline decision rule
* PSI for drift monitoring
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Feature engineering (application-time information only)
# ---------------------------------------------------------------------------
RATING_ORDER = ["AA", "A", "B", "C", "D", "E", "F", "G", "HR"]
TERM_BUCKETS = [0, 12, 24, 36, 48, 60, 240]
TERM_LABELS = ["<=12m", "13-24m", "25-36m", "37-48m", "49-60m", ">60m"]


def term_bucket(term):
    return pd.cut(term, TERM_BUCKETS, labels=TERM_LABELS).astype(str)


def add_features(d: pd.DataFrame, price_ref: pd.Series | None = None) -> pd.DataFrame:
    """Derive affordability and context features. Uses only fields known when
    the application is decided (amount, term, price, income, grade, country,
    timestamp). `price_ref` = median rate per (grade, term bucket), learned on
    the training window only; if None it is computed from `d`."""
    d = d.copy()
    r = d["initial_interest_rate"] / 12
    n = d["initial_loan_duration"]
    d["monthly_instalment"] = d["issued_amount"] * r / (1 - (1 + r) ** (-n))
    inc = d["combined_income"].where(d["combined_income"] > 0)
    d["payment_to_income"] = d["monthly_instalment"] / inc
    d["loan_to_income"] = d["issued_amount"] / inc
    d["rating_rank"] = d["customer_risk_rating"].map({g: i for i, g in enumerate(RATING_ORDER)})
    d["application_hour"] = d["issued_at"].dt.hour
    d["term_bucket"] = term_bucket(d["initial_loan_duration"])
    # price premium vs. the median price of the same grade & term (how far the
    # pricing engine deviated from "typical" for this risk grade)
    if price_ref is None:
        price_ref = d.groupby(["customer_risk_rating", "term_bucket"], observed=True)["initial_interest_rate"].median()
    key = pd.MultiIndex.from_frame(d[["customer_risk_rating", "term_bucket"]])
    med = pd.Series(price_ref.reindex(key).values, index=d.index)
    med = med.fillna(d["initial_interest_rate"].median())
    d["price_premium"] = d["initial_interest_rate"] - med
    return d


# ---------------------------------------------------------------------------
# WoE binning and scorecard
# ---------------------------------------------------------------------------
EPS = 0.5  # Laplace smoothing for WoE


def _woe_table(x_bin: pd.Series, y: pd.Series) -> pd.DataFrame:
    t = pd.crosstab(x_bin, y)
    t.columns = ["good", "bad"]
    t["n"] = t.good + t.bad
    tg, tb = t.good.sum(), t.bad.sum()
    t["bad_rate"] = t.bad / t.n
    t["share"] = t.n / t.n.sum()
    t["woe"] = np.log(((t.good + EPS) / tg) / ((t.bad + EPS) / tb))
    t["iv"] = ((t.good + EPS) / tg - (t.bad + EPS) / tb) * t.woe
    return t


class Binner:
    """Monotone fine->coarse binning for a numeric feature, or category
    grouping for a categorical one. Missing values always get their own bin."""

    def __init__(self, name, kind="numeric", max_bins=20, min_share=0.04, monotone=True):
        self.name, self.kind = name, kind
        self.max_bins, self.min_share, self.monotone = max_bins, min_share, monotone

    def fit(self, x: pd.Series, y: pd.Series):
        if self.kind == "categorical":
            vc = x.value_counts(normalize=True)
            self.keep_ = [c for c in vc.index if vc[c] >= 0.01]
            xb = self._cat(x)
        else:
            xn = x.dropna()
            qs = np.unique(np.quantile(xn, np.linspace(0, 1, self.max_bins + 1)[1:-1]))
            edges = [-np.inf, *qs, np.inf]
            edges = self._merge(xn, y.loc[xn.index], edges)
            self.edges_ = edges
            xb = self._num(x)
        self.table_ = _woe_table(xb, y)
        self.woe_ = self.table_["woe"].to_dict()
        self.iv_ = float(self.table_["iv"].sum())
        return self

    def _merge(self, x, y, edges):
        # 1) enforce a minimum bin share, 2) enforce monotone bad rate
        def stats(e):
            b = pd.cut(x, e, right=True, include_lowest=True)
            g = y.groupby(b, observed=False)
            return g.mean().values, g.size().values

        changed = True
        while changed and len(edges) > 3:
            changed = False
            rate, cnt = stats(edges)
            share = cnt / cnt.sum()
            i = int(np.argmin(share))
            if share[i] < self.min_share:
                j = i + 1 if i == 0 else (i if i == len(share) - 1 else (i + 1 if share[i + 1] < share[i - 1] else i))
                edges = edges[:j] + edges[j + 1:]
                changed = True
                continue
            if self.monotone:
                direction = np.sign(np.corrcoef(np.arange(len(rate)), rate)[0, 1])
                diffs = np.diff(rate) * direction
                if (diffs < 0).any():
                    k = int(np.argmin(diffs))
                    edges = edges[: k + 1] + edges[k + 2:]
                    changed = True
        return edges

    def _num(self, x):
        lab = pd.cut(x, self.edges_, right=True, include_lowest=True).astype(str)
        return lab.where(x.notna(), "Missing")

    def _cat(self, x):
        return x.where(x.isin(self.keep_), "Other").fillna("Missing").astype(str)

    def transform_bin(self, x):
        return self._cat(x) if self.kind == "categorical" else self._num(x)

    def transform(self, x):
        b = self.transform_bin(x)
        return b.map(self.woe_).fillna(0.0).astype(float)


class Scorecard:
    """Logistic regression on WoE with classic points scaling:
    600 points = 50:1 good:bad odds, 20 points to double the odds."""

    def __init__(self, base_score=600, base_odds=50, pdo=20):
        self.base_score, self.base_odds, self.pdo = base_score, base_odds, pdo

    def fit(self, X: pd.DataFrame, y: pd.Series, binners: dict, C=1.0):
        from sklearn.linear_model import LogisticRegression
        self.binners = binners
        self.features = list(binners)
        W = self.woe(X)
        self.lr = LogisticRegression(C=C, max_iter=1000).fit(W, y)
        self._build_points()
        return self

    def woe(self, X):
        return pd.DataFrame({f: self.binners[f].transform(X[f]) for f in self.features}, index=X.index)

    def _build_points(self):
        factor = self.pdo / np.log(2)
        offset = self.base_score - factor * np.log(self.base_odds)
        b0, betas = self.lr.intercept_[0], self.lr.coef_[0]
        n = len(self.features)
        rows = []
        for f, b in zip(self.features, betas):
            t = self.binners[f].table_
            for attr, r in t.iterrows():
                pts = -(b * r.woe) * factor + (offset - factor * b0) / n
                rows.append(dict(feature=f, attribute=attr, n=int(r.n), share=r.share, bad_rate=r.bad_rate,
                                 woe=r.woe, iv=r.iv, coef=b, points=pts))
        self.table = pd.DataFrame(rows)
        self.table["points"] = self.table["points"].round().astype(int)
        self.factor, self.offset = factor, offset

    def points(self, X):
        out = {}
        for f in self.features:
            m = self.table[self.table.feature == f].set_index("attribute")["points"]
            b = self.binners[f].transform_bin(X[f])
            out[f] = b.map(m).fillna(m.median()).astype(int)
        return pd.DataFrame(out, index=X.index)

    def score(self, X):
        return self.points(X).sum(axis=1)

    def pd_from_score(self, s):
        # ln(odds_good) = (score - offset)/factor
        return 1 / (1 + np.exp((s - self.offset) / self.factor))


def export_card(sc: Scorecard) -> dict:
    """Version-independent scorecard: ordered numeric bands and category maps,
    so a scoring service (or the Streamlit app) needs nothing but JSON."""
    card = {"factor": sc.factor, "offset": sc.offset, "base_score": sc.base_score,
            "base_odds": sc.base_odds, "pdo": sc.pdo, "features": {}}
    for f in sc.features:
        b = sc.binners[f]
        m = sc.table[sc.table.feature == f].set_index("attribute")["points"]
        fallback = int(m.median())
        entry = {"kind": b.kind, "missing": int(m.get("Missing", fallback)), "fallback": fallback}
        if b.kind == "categorical":
            entry["categories"] = {str(k): int(v) for k, v in m.items() if k != "Missing"}
            entry["other"] = int(m.get("Other", fallback))
        else:
            bands = []
            for lo, hi in zip(b.edges_[:-1], b.edges_[1:]):
                lab = str(pd.cut(pd.Series([hi if np.isfinite(hi) else lo + 1]), b.edges_, right=True, include_lowest=True).astype(str).iloc[0])
                bands.append({"lo": None if not np.isfinite(lo) else float(lo), "hi": None if not np.isfinite(hi) else float(hi),
                              "label": lab, "points": int(m.get(lab, fallback))})
            entry["bands"] = bands
        card["features"][f] = entry
    return card


def card_points(card: dict, applicant: dict) -> dict:
    """Points per characteristic for one applicant, from the exported JSON card."""
    out = {}
    for f, e in card["features"].items():
        v = applicant.get(f)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            out[f] = e["missing"]
        elif e["kind"] == "categorical":
            out[f] = e["categories"].get(str(v), e["other"])
        else:
            pts = e["fallback"]
            for bnd in e["bands"]:
                lo = -np.inf if bnd["lo"] is None else bnd["lo"]
                hi = np.inf if bnd["hi"] is None else bnd["hi"]
                if (v > lo or (lo == -np.inf)) and v <= hi:
                    pts = bnd["points"]
                    break
            out[f] = pts
    return out


def card_pd(card: dict, score: float) -> float:
    return float(1 / (1 + np.exp((score - card["offset"]) / card["factor"])))


# ---------------------------------------------------------------------------
# Reason codes
# ---------------------------------------------------------------------------
REASON_TEXT = {
    "rating_rank": "Internal risk grade is weak",
    "country": "Market has elevated observed losses",
    "combined_income": "Verified income is low",
    "payment_to_income": "Monthly payment is high relative to income",
    "loan_to_income": "Loan amount is high relative to income",
    "issued_amount": "Requested amount is in a higher-risk range",
    "initial_loan_duration": "Requested term is in a higher-risk range",
    "price_premium": "Price is above the typical rate for this grade",
    "application_hour": "Application timing pattern linked to higher risk",
    "initial_interest_rate": "Interest rate band associated with higher risk",
    "monthly_instalment": "Monthly instalment is in a higher-risk range",
}


def reason_codes(points: pd.DataFrame, max_points: dict, k=3):
    """Top-k characteristics where the applicant lost the most points versus
    the best attainable attribute (the 'points below maximum' method used for
    adverse-action notices)."""
    short = pd.DataFrame({f: max_points[f] - points[f] for f in points.columns}, index=points.index)
    order = np.argsort(-short.values, axis=1)[:, :k]
    cols = np.array(points.columns)
    out = pd.DataFrame(cols[order], index=points.index, columns=[f"reason_{i+1}" for i in range(k)])
    for i in range(k):
        lost = short.values[np.arange(len(short)), order[:, i]]
        out[f"reason_{i+1}"] = np.where(lost > 0, out[f"reason_{i+1}"], "")
    return out


# ---------------------------------------------------------------------------
# Economics and decision rule
# ---------------------------------------------------------------------------
DEFAULT_PARAMS = dict(
    cost_of_funds=0.06,   # annual funding cost on drawn principal
    opex=0.02,            # acquisition + servicing, share of principal
    lgd_multiplier=1.0,   # stress on the empirical loss of a 12m-default loan
    gain_multiplier=1.0,  # stress on the empirical return of a performing loan
    review_cost=20.0,     # EUR per manual review
    review_catch=0.35,    # share of true bads a reviewer stops
    review_false_decline=0.05,  # share of true goods a reviewer wrongly stops
)


def unit_economics(econ: pd.DataFrame, params: dict) -> pd.DataFrame:
    """econ has, per term bucket: net_good, py_good, net_bad, py_bad
    (realised cash return per EUR lent, and principal-years of exposure).
    Returns gain-if-good and loss-if-bad per EUR lent."""
    p = {**DEFAULT_PARAMS, **params}
    e = econ.copy()
    e["gain"] = (e.net_good - p["cost_of_funds"] * e.py_good - p["opex"]) * p["gain_multiplier"]
    # a defaulted loan also ties up funding through the ~1 year collection period
    e["loss"] = (-e.net_bad + p["cost_of_funds"] * (e.py_bad + 1.0) + p["opex"]) * p["lgd_multiplier"]
    e["breakeven_pd"] = e.gain / (e.gain + e.loss)
    return e


def decide(pd_, amount, gain_ratio, loss_ratio, params: dict, allow_refer=True):
    """Return (action, ev_approve, ev_review). action in {APPROVE, REFER, DECLINE}.
    A reviewer stops `review_catch` of the bads and `review_false_decline` of
    the goods, at `review_cost` per file."""
    p = {**DEFAULT_PARAMS, **params}
    G = amount * gain_ratio
    L = amount * loss_ratio
    ev_a = (1 - pd_) * G - pd_ * L
    ev_r = (1 - pd_) * (1 - p["review_false_decline"]) * G - pd_ * (1 - p["review_catch"]) * L - p["review_cost"]
    if not allow_refer:
        ev_r = np.full_like(np.asarray(ev_a, dtype=float), -np.inf)
    stack = np.vstack([np.asarray(ev_a, float), np.asarray(ev_r, float), np.zeros_like(np.asarray(ev_a, float))])
    idx = stack.argmax(axis=0)
    action = np.array(["APPROVE", "REFER", "DECLINE"])[idx]
    return action, ev_a, ev_r


# ---------------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------------
def psi(expected, actual, bins=10, categorical=False):
    """Population Stability Index. <0.10 stable, 0.10-0.25 watch, >0.25 act."""
    if categorical:
        e = pd.Series(expected).astype(str).value_counts(normalize=True)
        a = pd.Series(actual).astype(str).value_counts(normalize=True)
        idx = e.index.union(a.index)
        e, a = e.reindex(idx).fillna(0), a.reindex(idx).fillna(0)
    else:
        expected, actual = pd.Series(expected).dropna(), pd.Series(actual).dropna()
        edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        e = pd.cut(expected, edges).value_counts(normalize=True, sort=False)
        a = pd.cut(actual, edges).value_counts(normalize=True, sort=False)
    e, a = np.clip(e.values, 1e-4, None), np.clip(a.values, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def ks_stat(y, s):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.max(tpr - fpr))
