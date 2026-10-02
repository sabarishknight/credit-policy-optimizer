"""Credit Policy Simulator
Run from the project root:  streamlit run app/streamlit_app.py

Change the economics, the reviewer assumptions or a downturn stress, and see how
the approve / refer / decline policy and its profit respond on 81k real loans
from the Jan-Jun 2025 out-of-time cohort. Or score a single applicant and read
the reason codes.
"""
import json
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import credit as cr  # noqa: E402

st.set_page_config(page_title="Credit Policy Simulator", layout="wide")

COL = {"APPROVE": "#1baf7a", "REFER": "#eda100", "DECLINE": "#e34948"}
ACTIONS = ["APPROVE", "REFER", "DECLINE"]


@st.cache_data
def load():
    policy = json.load(open(ROOT / "artifacts/policy.json"))
    card = json.load(open(ROOT / "artifacts/scorecard.json"))
    test = pd.read_csv(ROOT / "artifacts/test_scored.csv.gz", parse_dates=["issued_at"])
    live = pd.read_csv(ROOT / "artifacts/live_scored.csv.gz", parse_dates=["issued_at"])
    ref = pd.read_csv(ROOT / "artifacts/train_reference.csv.gz")
    econ = pd.DataFrame(policy["unit_economics"]).set_index("term_bucket")
    return policy, card, test, live, ref, econ


policy, card, test, live, ref, econ = load()


def eur(x):
    a = abs(x)
    s = f"€{a/1e6:.2f}M" if a >= 1e6 else (f"€{a/1e3:.0f}k" if a >= 1e3 else f"€{a:,.0f}")
    return ("−" if x < 0 else "") + s


# ----------------------------------------------------------------------------- sidebar
st.sidebar.title("Assumptions")
st.sidebar.caption("Defaults = the base case in the notebook.")
p0 = policy["params"]
with st.sidebar.expander("Unit economics", expanded=True):
    cof = st.slider("Cost of funds (p.a.)", 0.02, 0.12, float(p0["cost_of_funds"]), 0.005, format="%.3f")
    opex = st.slider("Opex (% of principal)", 0.0, 0.05, float(p0["opex"]), 0.005, format="%.3f")
    lgd = st.slider("Loss on default (× measured)", 0.7, 1.4, 1.0, 0.05)
    gain = st.slider("Return on performing loans (× measured)", 0.7, 1.3, 1.0, 0.05)
with st.sidebar.expander("Manual review", expanded=True):
    use_refer = st.toggle("Use a refer band", True)
    rcost = st.slider("Cost per review (€)", 0, 80, int(p0["review_cost"]), 5)
    catch = st.slider("Share of bads the reviewer stops", 0.0, 0.8, float(p0["review_catch"]), 0.05)
    fdec = st.slider("Share of goods wrongly stopped", 0.0, 0.3, float(p0["review_false_decline"]), 0.01)
    cap = st.slider("Review capacity (% of applications)", 0.0, 0.25, float(policy["refer_cap"]), 0.01)
with st.sidebar.expander("Stress & rule", expanded=True):
    stress = st.slider("Default stress (× predicted PD)", 0.8, 2.0, 1.0, 0.05,
                       help="Inflates both the model's PDs and, for the 'expected' view, the outcomes.")
    rule = st.radio("Decision rule", ["Expected value (term-aware)", "Single score cut-off"], index=0)
    cutoff = st.slider("Score cut-off (approve if score ≥)", 480, 620,
                       int(np.ceil(card["offset"] + card["factor"] * np.log((1 - policy["pd_cutoff_single"]) / policy["pd_cutoff_single"]))),
                       disabled=rule != "Single score cut-off")

params = dict(cost_of_funds=cof, opex=opex, lgd_multiplier=lgd, gain_multiplier=gain,
              review_cost=rcost, review_catch=catch, review_false_decline=fdec)
ue = cr.unit_economics(econ, params)


def run_policy(df):
    x = df.copy()
    x["pd_s"] = np.clip(x.pd * stress, 0, 1)
    e = ue.loc[x.term_bucket]
    x["G"], x["L"] = x.issued_amount * e.gain.values, x.issued_amount * e.loss.values
    if rule == "Single score cut-off":
        act = np.where(x.score >= cutoff, "APPROVE", "DECLINE")
        ev_a = (1 - x.pd_s) * x.G - x.pd_s * x.L
        ev_r = np.full(len(x), -np.inf)
    else:
        act, ev_a, ev_r = cr.decide(x.pd_s.values, x.issued_amount.values, e.gain.values, e.loss.values, params,
                                    allow_refer=use_refer)
        act = act.copy()
        if use_refer and (act == "REFER").mean() > cap:
            gain_r = np.where(act == "REFER", ev_r - np.maximum(ev_a, 0), -np.inf)
            keep = np.argsort(-gain_r)[: int(cap * len(x))]
            mask = act == "REFER"
            mask[keep] = False
            act[mask] = np.where(np.asarray(ev_a)[mask] > 0, "APPROVE", "DECLINE")
    x["action"] = act
    x["ev_a"], x["ev_r"] = ev_a, ev_r
    if "y" in x:
        # realised value when stress = 1; under stress use expected value with stressed PD
        if stress == 1.0:
            approve_val = np.where(x.y == 0, x.G, -x.L)
            refer_val = np.where(x.y == 0, (1 - fdec) * x.G, -(1 - catch) * x.L) - rcost
        else:
            approve_val = (1 - x.pd_s) * x.G - x.pd_s * x.L
            refer_val = (1 - x.pd_s) * (1 - fdec) * x.G - x.pd_s * (1 - catch) * x.L - rcost
        x["value"] = np.select([x.action == "APPROVE", x.action == "REFER"], [approve_val, refer_val], 0.0)
        x["value_all"] = approve_val
    return x


res = run_policy(test)

# ----------------------------------------------------------------------------- header
st.title("Approve, Refer or Decline?")
st.markdown(
    "Profit-optimal credit policy simulator · scorecard model · **81,464 real loans** issued Jan–Jun 2025 "
    "(Bondora / Go & Grow public data), scored out-of-time. "
    + ("Values are **realised** (actual 12-month outcomes × measured unit economics)." if stress == 1.0
       else f"Default stress ×{stress:.2f} applied: values are **expected** under stressed PDs.")
)

t1, t2, t3, t4 = st.tabs(["Portfolio impact", "Score an applicant", "Segments", "Drift monitor"])

# ----------------------------------------------------------------------------- tab 1
with t1:
    sq, pv = res.value_all.sum(), res.value.sum()
    shares = res.action.value_counts(normalize=True).reindex(ACTIONS).fillna(0)
    approved = res[res.action == "APPROVE"]
    c = st.columns(5)
    c[0].metric("Policy profit", eur(pv), f"{eur(pv - sq)} vs approve-all")
    c[1].metric("Approve", f"{shares['APPROVE']:.1%}")
    c[2].metric("Refer", f"{shares['REFER']:.1%}")
    c[3].metric("Decline", f"{shares['DECLINE']:.1%}")
    c[4].metric("Bad rate of approved", f"{approved.y.mean():.2%}", f"{approved.y.mean() - res.y.mean():+.2%} vs all",
                delta_color="inverse")

    left, right = st.columns([1.4, 1])
    with left:
        st.subheader("Profit curve (approve safest first)")
        o = res.sort_values("pd_s")
        n = len(o)
        rates = np.linspace(.2, 1, 81)
        cum = o.value_all.cumsum().values
        curve = pd.DataFrame({"approval rate": rates, "profit (€M)": cum[(rates * n).astype(int) - 1] / 1e6})
        line = alt.Chart(curve).mark_line(color="#2a78d6", strokeWidth=2.5).encode(
            x=alt.X("approval rate:Q", axis=alt.Axis(format="%")), y=alt.Y("profit (€M):Q", scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip("approval rate:Q", format=".0%"), alt.Tooltip("profit (€M):Q", format=".2f")])
        pt = pd.DataFrame({"approval rate": [(res.action != "DECLINE").mean()], "profit (€M)": [pv / 1e6], "label": ["current policy"]})
        dot = alt.Chart(pt).mark_point(size=160, filled=True, color="#e34948").encode(x="approval rate:Q", y="profit (€M):Q",
                                                                                     tooltip=["label", alt.Tooltip("profit (€M):Q", format=".2f")])
        st.altair_chart((line + dot).properties(height=330), use_container_width=True)
        st.caption("Blue: approve/decline only, sorted by stressed PD. Red: your current policy (incl. referrals).")
    with right:
        st.subheader("Where the money comes from")
        g = res.groupby("action").agg(applications=("y", "size"), default_rate=("y", "mean"), avg_amount=("issued_amount", "mean"),
                                      profit=("value", "sum")).reindex(ACTIONS)
        st.dataframe(g.style.format({"applications": "{:,.0f}", "default_rate": "{:.1%}", "avg_amount": "€{:,.0f}",
                                     "profit": lambda v: eur(v)}), use_container_width=True)
        st.subheader("Break-even PD by term")
        st.dataframe(ue[["gain", "loss", "breakeven_pd"]].style.format({"gain": "{:+.1%}", "loss": "{:.1%}", "breakeven_pd": "{:.1%}"}),
                     use_container_width=True)

# ----------------------------------------------------------------------------- tab 2
with t2:
    st.subheader("Score a single application")
    a, b, c_ = st.columns(3)
    grade = a.selectbox("Internal risk grade", cr.RATING_ORDER[:-1], index=2)
    country = a.selectbox("Country", ["Finland", "Estonia", "Netherlands", "Latvia", "Denmark"])
    amount = b.number_input("Amount (€)", 100, 30000, 3000, 100)
    term = b.select_slider("Term (months)", [6, 12, 18, 24, 36, 48, 60, 72, 84, 96, 120], value=60)
    income = c_.number_input("Verified monthly income (€)", 0, 20000, 2200, 100)
    applicant = {"rating_rank": cr.RATING_ORDER.index(grade), "country": country, "initial_loan_duration": term,
                 "combined_income": income if income > 0 else None,
                 "loan_to_income": amount / income if income > 0 else None}
    pts = cr.card_points(card, applicant)
    score = sum(pts.values())
    pd_ = min(1.0, cr.card_pd(card, score) * stress)
    tb = str(cr.term_bucket(pd.Series([term])).iloc[0])
    act, ev_a, ev_r = cr.decide(np.array([pd_]), np.array([amount]), np.array([ue.loc[tb, "gain"]]),
                                np.array([ue.loc[tb, "loss"]]), params, allow_refer=use_refer)
    action = act[0] if rule != "Single score cut-off" else ("APPROVE" if score >= cutoff else "DECLINE")

    m = st.columns(4)
    m[0].metric("Score", f"{score}")
    m[1].metric("12-month PD", f"{pd_:.1%}")
    m[2].metric("Expected profit if approved", eur(float(ev_a[0])))
    m[3].markdown(f"<div style='padding:14px 0 0 0;font-size:28px;font-weight:700;color:{COL[action]}'>{action}</div>",
                  unsafe_allow_html=True)

    maxp = policy["max_points"]
    rows = [{"characteristic": f, "points": p, "max points": maxp[f], "points lost": maxp[f] - p} for f, p in pts.items()]
    pts_df = pd.DataFrame(rows).sort_values("points lost", ascending=False)
    l, r = st.columns([1, 1])
    l.markdown("**Points breakdown**")
    l.dataframe(pts_df, hide_index=True, use_container_width=True)
    r.markdown("**Reason codes** (largest points lost vs. best attribute)")
    if action == "APPROVE":
        r.success("Approved: no adverse-action reasons required.")
    for _, row in pts_df.head(3).iterrows():
        if row["points lost"] > 0:
            r.write(f"• {cr.REASON_TEXT.get(row.characteristic, row.characteristic)}  (−{row['points lost']} pts)")
    if country == "Denmark":
        r.warning("Denmark was not in the training window: the scorecard treats it as an unseen market. "
                  "Recommend a launch override (refer or cap volume) until 12 months of outcomes exist.")

# ----------------------------------------------------------------------------- tab 3
with t3:
    dim = st.selectbox("Segment by", ["country", "customer_risk_rating", "term_bucket"])
    sg = res.groupby(dim).agg(applications=("y", "size"), observed_default=("y", "mean"), predicted_pd=("pd_s", "mean"),
                              approve=("action", lambda s: (s == "APPROVE").mean()), refer=("action", lambda s: (s == "REFER").mean()),
                              decline=("action", lambda s: (s == "DECLINE").mean()), profit=("value", "sum"))
    st.dataframe(sg.style.format({"applications": "{:,.0f}", "observed_default": "{:.1%}", "predicted_pd": "{:.1%}",
                                  "approve": "{:.0%}", "refer": "{:.0%}", "decline": "{:.0%}", "profit": lambda v: eur(v)}),
                 use_container_width=True)
    long = sg[["approve", "refer", "decline"]].reset_index().melt(dim, var_name="action", value_name="share")
    long["action"] = long.action.str.upper()
    st.altair_chart(alt.Chart(long).mark_bar().encode(
        y=alt.Y(f"{dim}:N", title=None), x=alt.X("share:Q", stack="normalize", axis=alt.Axis(format="%")),
        color=alt.Color("action:N", scale=alt.Scale(domain=ACTIONS, range=[COL[a_] for a_ in ACTIONS])),
        order=alt.Order("action_order:Q"), tooltip=[dim, "action", alt.Tooltip("share:Q", format=".1%")]
    ).transform_calculate(action_order="indexof(['APPROVE','REFER','DECLINE'], datum.action)").properties(height=260),
        use_container_width=True)

# ----------------------------------------------------------------------------- tab 4
with t4:
    st.subheader("Population Stability Index vs. training window")
    st.caption("< 0.10 stable · 0.10–0.25 investigate · > 0.25 act")
    live["q"] = live.issued_at.dt.to_period("Q").astype(str)
    test["q"] = test.issued_at.dt.to_period("Q").astype(str)
    allq = pd.concat([test, live])
    q = pd.DataFrame({"quarter": sorted(allq.q.unique())})
    q["score PSI"] = [cr.psi(ref.score, allq[allq.q == k].score) for k in q.quarter]
    q["status"] = pd.cut(q["score PSI"], [-1, .1, .25, 99], labels=["stable", "investigate", "act"]).astype(str)
    st.altair_chart(alt.Chart(q).mark_bar(color="#4a3aa7").encode(
        x=alt.X("quarter:N"), y=alt.Y("score PSI:Q"), tooltip=["quarter", alt.Tooltip("score PSI:Q", format=".3f"), "status"]
    ).properties(height=260), use_container_width=True)
    feat_psi = []
    for f in ["country", "rating_rank", "initial_loan_duration", "loan_to_income", "combined_income"]:
        if f == "country":
            v = cr.psi(ref[f], live[f], categorical=True)
        elif f == "rating_rank":
            v = cr.psi(ref[f], live.customer_risk_rating.map({g: i for i, g in enumerate(cr.RATING_ORDER)}), categorical=True)
        elif f == "initial_loan_duration":
            v = cr.psi(cr.term_bucket(ref[f]), live.term_bucket, categorical=True)
        else:
            v = cr.psi(ref[f], live[f])
        feat_psi.append({"feature": f, "PSI live vs train": v})
    st.dataframe(pd.DataFrame(feat_psi).style.format({"PSI live vs train": "{:.3f}"}), hide_index=True)
    live_res = run_policy(live)
    st.write(f"Live population (Jul 2025 – Jun 2026, {len(live):,} loans, no outcomes yet): mean predicted PD "
             f"**{live.pd.mean():.2%}**, policy would decline **{(live_res.action == 'DECLINE').mean():.1%}** and refer "
             f"**{(live_res.action == 'REFER').mean():.1%}**.")

st.divider()
st.caption("Built on public Bondora / Go & Grow loan-level data. Profit uses unit economics measured on 2021–22 cohorts; "
           "reviewer effectiveness and costs are assumptions. See notebooks/credit_policy.ipynb for the full analysis.")
