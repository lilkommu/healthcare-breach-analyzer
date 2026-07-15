"""Healthcare Breach Impact Analyzer — Streamlit dashboard.

Data: HHS OCR Breach Portal (breaches of unsecured PHI affecting 500+ individuals),
cleaned and clustered in notebooks/ (see notebooks/eda.ipynb, notebooks/clustering.ipynb).
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Healthcare Breach Impact Analyzer", layout="wide")

BASE = Path(__file__).resolve().parent.parent
DATA_PATH = BASE / "data" / "breach_clustered.csv"

CLUSTER_NAMES = {
    0: "0 — Provider hacking (modern era)",
    1: "1 — Provider human error / insider",
    2: "2 — Health plan breaches",
    3: "3 — Business associate / vendor",
    4: "4 — Physical theft era (legacy)",
}


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["Breach Submission Date"])
    df["Cluster"] = df["cluster"].map(CLUSTER_NAMES)
    return df


@st.cache_data
def load_vectors() -> pd.DataFrame:
    """Attack-vector / remediation features mined from OCR narratives (notebooks/text_mining.ipynb)."""
    return pd.read_csv(BASE / "data" / "breach_vectors.csv")


df = load_data()

px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = px.colors.qualitative.Safe

st.title("Healthcare Breach Impact Analyzer")
st.markdown(
    f"U.S. healthcare data breaches affecting 500+ individuals, as reported to the "
    f"HHS Office for Civil Rights — **{len(df):,} breaches**, "
    f"{df['Breach Submission Date'].min():%B %Y} to {df['Breach Submission Date'].max():%B %Y}."
)
st.divider()

# ------------------------------------------------------------------ filters
with st.sidebar:
    st.header("Filters")
    breach_types = st.multiselect(
        "Breach type", sorted(df["Primary Breach Type"].dropna().unique()))
    entity_types = st.multiselect(
        "Covered entity type", sorted(df["Covered Entity Type"].dropna().unique()))
    states = st.multiselect("State", sorted(df["State"].dropna().unique()))
    yr_min, yr_max = int(df["Year"].min()), int(df["Year"].max())
    years = st.slider("Submission year", yr_min, yr_max, (yr_min, yr_max))
    clusters = st.multiselect("Cluster segment", list(CLUSTER_NAMES.values()))
    st.caption("Empty filter = all values.")

f = df[df["Year"].between(*years)]
if breach_types:
    f = f[f["Primary Breach Type"].isin(breach_types)]
if entity_types:
    f = f[f["Covered Entity Type"].isin(entity_types)]
if states:
    f = f[f["State"].isin(states)]
if clusters:
    f = f[f["Cluster"].isin(clusters)]

if f.empty:
    st.warning("No breaches match the current filters.")
    st.stop()

# ------------------------------------------------------------------ KPIs
k1, k2, k3, k4 = st.columns(4)
k1.metric("Breaches", f"{len(f):,}")
k2.metric("Individuals affected", f"{f['Individuals Affected'].sum() / 1e6:,.1f} M")
k3.metric("Median breach size", f"{f['Individuals Affected'].median():,.0f}")
k4.metric("Hacking/IT share", f"{(f['Primary Breach Type'] == 'Hacking/IT Incident').mean():.0%}")

tab_trends, tab_breakdown, tab_clusters, tab_vectors, tab_cost, tab_data = st.tabs(
    ["Trends", "Breakdown", "Clusters", "How breaches happen", "Cost estimate", "Data"])

# ------------------------------------------------------------------ trends
with tab_trends:
    c1, c2 = st.columns(2)
    by_year = f.groupby("Year").agg(
        breaches=("Year", "size"), individuals=("Individuals Affected", "sum")).reset_index()
    c1.plotly_chart(px.bar(by_year, x="Year", y="breaches",
                           title="Breaches by submission year"), width="stretch")
    c2.plotly_chart(px.bar(by_year, x="Year", y="individuals",
                           title="Individuals affected by year"), width="stretch")

    mix = (f.groupby(["Year", "Primary Breach Type"]).size()
             .rename("n").reset_index())
    st.plotly_chart(px.area(mix, x="Year", y="n", color="Primary Breach Type",
                            groupnorm="fraction",
                            title="Breach-type mix by year (share of reports)"),
                    width="stretch")
    if f["Year"].max() >= 2026:
        st.caption("2026 is a partial year (data through July 1, 2026).")

# ------------------------------------------------------------------ breakdown
with tab_breakdown:
    c1, c2 = st.columns(2)
    bt = f["Primary Breach Type"].value_counts().reset_index()
    c1.plotly_chart(px.bar(bt, x="count", y="Primary Breach Type", orientation="h",
                           title="Breaches by type"), width="stretch")
    et = f.groupby("Covered Entity Type")["Individuals Affected"].agg(["size", "sum"]).reset_index()
    et.columns = ["Covered Entity Type", "breaches", "individuals"]
    c2.plotly_chart(px.bar(et, x="Covered Entity Type", y=["breaches"],
                           title="Breaches by entity type"), width="stretch")

    c3, c4 = st.columns(2)
    top_ct = f["State"].value_counts().head(15).reset_index()
    c3.plotly_chart(px.bar(top_ct, x="count", y="State", orientation="h",
                           title="Top states by breach count"), width="stretch")
    top_af = (f.groupby("State")["Individuals Affected"].sum()
                .nlargest(15).reset_index())
    c4.plotly_chart(px.bar(top_af, x="Individuals Affected", y="State", orientation="h",
                           title="Top states by individuals affected"), width="stretch")
    st.caption("State totals reflect where the covered entity is registered, not where "
               "affected patients live — e.g. MN's total is dominated by Change Healthcare (192.7M, 2024).")

# ------------------------------------------------------------------ clusters
with tab_clusters:
    st.markdown(
        "K-Means (k=5) on log-severity, year, business-associate flag, breach type, and entity "
        "type. State was tested as a feature and excluded — it added no geographic structure "
        "(details in `notebooks/clustering.ipynb`)."
    )
    prof = (f.groupby("Cluster")
              .agg(breaches=("Cluster", "size"),
                   median_affected=("Individuals Affected", "median"),
                   total_affected=("Individuals Affected", "sum"),
                   dominant_type=("Primary Breach Type", lambda s: s.mode().iat[0]),
                   dominant_entity=("Covered Entity Type", lambda s: s.mode().iat[0]))
              .reset_index())
    st.dataframe(prof, width="stretch", hide_index=True)

    sample = f.sample(min(len(f), 4000), random_state=0)
    st.plotly_chart(
        px.strip(sample, x="Year", y="Individuals Affected", color="Cluster", log_y=True,
                 hover_data=["Name of Covered Entity", "State", "Primary Breach Type"],
                 title="Breaches by year, severity (log scale), and cluster"
                       + (" — 4,000-breach sample" if len(f) > 4000 else "")),
        width="stretch")

# ------------------------------------------------------------------ vectors
with tab_vectors:
    v = load_vectors()
    v = v[v["Year"].between(*years)]
    if breach_types:
        v = v[v["Primary Breach Type"].isin(breach_types)]
    if entity_types:
        v = v[v["Covered Entity Type"].isin(entity_types)]
    if states:
        v = v[v["State"].isin(states)]

    st.markdown(
        "Attack vectors and post-breach actions mined from OCR case narratives with a rule-based "
        "classifier (`notebooks/text_mining.ipynb`). Narratives are written when a case is "
        "**closed**, so recent years are under-covered — shares below are of *narrated* breaches "
        "only. The cluster filter does not apply to this tab."
    )
    nv = v[v["has_narrative"]]
    if nv.empty:
        st.warning("No narrated breaches match the current filters.")
    else:
        c0, c1, c2 = st.columns(3)
        c0.metric("Narrated breaches", f"{len(nv):,} / {len(v):,}")
        c1.metric("Ransomware share", f"{(nv['Primary Vector'] == 'Ransomware').mean():.0%}")
        c2.metric("Phishing/email share",
                  f"{(nv['Primary Vector'] == 'Phishing / email compromise').mean():.0%}")

        vc = (nv["Primary Vector"].value_counts(normalize=True) * 100).reset_index()
        vc.columns = ["Primary Vector", "share"]
        col_a, col_b = st.columns(2)
        col_a.plotly_chart(px.bar(vc, x="share", y="Primary Vector", orientation="h",
                                  title="Primary attack vector (% of narrated breaches)"),
                           width="stretch")
        sev = (nv.groupby("Primary Vector")["Individuals Affected"].median()
                 .sort_values().reset_index())
        col_b.plotly_chart(px.bar(sev, x="Individuals Affected", y="Primary Vector",
                                  orientation="h",
                                  title="Median individuals affected by vector"),
                           width="stretch")

        main_vecs = ["Ransomware", "Phishing / email compromise", "Other hacking / IT intrusion",
                     "Stolen device / burglary", "Insider snooping / misuse",
                     "Misdirected / inadvertent disclosure"]
        tr = (nv[nv["Primary Vector"].isin(main_vecs)]
                .groupby(["Year", "Primary Vector"]).size().rename("n").reset_index())
        totals = nv.groupby("Year").size().rename("total").reset_index()
        tr = tr.merge(totals, on="Year")
        tr["share"] = tr["n"] / tr["total"]
        st.plotly_chart(px.line(tr, x="Year", y="share", color="Primary Vector", markers=True,
                                title="Vector share of narrated breaches by year "
                                      "(post-2023 dip is partly a coverage artifact)"),
                        width="stretch")

        rem_cols = [c for c in nv.columns if c.startswith("rem: ")]
        rem = (nv[rem_cols].mean() * 100).sort_values().reset_index()
        rem.columns = ["action", "share"]
        rem["action"] = rem["action"].str[5:]
        st.plotly_chart(px.bar(rem, x="share", y="action", orientation="h",
                               title="Post-breach actions mentioned in narratives (% of narrated breaches)"),
                        width="stretch")
        st.caption("Rule-based extraction — no trained classifier; ~17% of narratives state no "
                   "clear mechanism and are labeled Other/unspecified.")

# ------------------------------------------------------------------ cost
with tab_cost:
    st.warning(
        "**Estimate, not data.** The HHS breach portal contains **no financial field**. "
        "Figures below multiply individuals affected by an adjustable external benchmark — "
        "IBM's *Cost of a Data Breach* report puts healthcare at roughly **$408 per breached "
        "record** (2025). Per-record scaling overstates costs for mega-breaches (costs don't "
        "scale linearly) and ignores fixed costs for small ones. Directional only."
    )
    per_record = st.slider("Assumed cost per breached record (USD)", 50, 600, 408, 10)
    total = f["Individuals Affected"].sum() * per_record
    c1, c2 = st.columns(2)
    c1.metric("Estimated total exposure (filtered)", f"${total / 1e9:,.1f} B")
    c2.metric("Estimated median cost per breach",
              f"${f['Individuals Affected'].median() * per_record / 1e6:,.2f} M")
    by_et = (f.groupby("Covered Entity Type")["Individuals Affected"].sum() * per_record
             ).reset_index(name="Estimated cost")
    st.plotly_chart(px.bar(by_et, x="Covered Entity Type", y="Estimated cost",
                           title=f"Estimated exposure by entity type (@ ${per_record}/record)"),
                    width="stretch")

# ------------------------------------------------------------------ data
with tab_data:
    show = f.drop(columns=["cluster"]).sort_values("Breach Submission Date", ascending=False)
    st.dataframe(show, width="stretch", hide_index=True)
    st.download_button("Download filtered data (CSV)",
                       show.to_csv(index=False).encode(),
                       "breaches_filtered.csv", "text/csv")

st.divider()
st.caption("Source: [HHS OCR Breach Portal](https://ocrportal.hhs.gov/ocr/breach/breach_report.jsf) · "
           "Cost benchmark: [IBM Cost of a Data Breach](https://www.ibm.com/reports/data-breach) · "
           "No PHI in this dataset — entity-level reports only.")
