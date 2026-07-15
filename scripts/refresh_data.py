"""Refresh pipeline: pull fresh HHS data, rebuild derived datasets.

Run monthly by .github/workflows/refresh-data.yml. Network steps degrade
gracefully: if a fetch fails, existing data is kept and the rebuild still runs.

Usage: python scripts/refresh_data.py [--skip-fetch]
"""
from __future__ import annotations

import json
import re
import sys
import warnings
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message=".*match groups.*")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"

PORTAL_URL = "https://ocrportal.hhs.gov/ocr/breach/breach_report.jsf"
ENFORCEMENT_URL = ("https://www.hhs.gov/hipaa/for-professionals/"
                   "compliance-enforcement/agreements/index.html")

# ----------------------------------------------------------------- fetching

def fetch_portal() -> bool:
    """Try to download fresh active/archived exports from the OCR portal.

    The portal is a JSF app: we need a session, the ViewState token, and the
    ids of the export buttons. Page structure changes occasionally — any
    failure keeps the committed CSVs.
    """
    try:
        import requests
        from bs4 import BeautifulSoup

        s = requests.Session()
        s.headers["User-Agent"] = "healthcare-breach-analyzer/1.0 (research)"
        page = s.get(PORTAL_URL, timeout=60)
        page.raise_for_status()
        soup = BeautifulSoup(page.text, "html.parser")
        viewstate = soup.find("input", {"name": "javax.faces.ViewState"})
        form = soup.find("form")
        if not viewstate or not form:
            print("portal: page structure not recognized, keeping existing data")
            return False

        form_id = form.get("id", "ocrForm")
        saved = 0
        # export buttons for "under investigation" and "archive" views
        for label, filename in [("Under Investigation", "breach_active.csv"),
                                ("Archive", "breach_archived.csv")]:
            export_ids = [b.get("id") for b in form.find_all(["button", "a"])
                          if b.get("id") and "export" in (b.get("id") or "").lower()]
            if not export_ids:
                print(f"portal: no export control found for {label}")
                continue
            payload = {form_id: form_id, "javax.faces.ViewState": viewstate["value"],
                       export_ids[0]: export_ids[0]}
            r = s.post(PORTAL_URL, data=payload, timeout=120)
            if r.ok and r.text.count(",") > 1000 and "javax.faces" in r.text[:400]:
                (DATA / filename).write_text(r.text, encoding="utf-8")
                saved += 1
                print(f"portal: refreshed {filename} ({len(r.text):,} bytes)")
        return saved == 2
    except Exception as exc:  # noqa: BLE001 - never break the pipeline on fetch
        print(f"portal fetch failed ({exc!r}); keeping existing data")
        return False


def fetch_enforcement() -> None:
    """Refresh the enforcement list from the official HHS agreements page.

    Only rows whose amount is stated in the listing title are added — amounts
    are never guessed. Existing curated rows are kept.
    """
    try:
        import requests
        from bs4 import BeautifulSoup

        r = requests.get(ENFORCEMENT_URL, timeout=60,
                         headers={"User-Agent": "healthcare-breach-analyzer/1.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        amount_re = re.compile(
            r"\$\s?([\d,.]+)\s?(million|m\b)?", re.IGNORECASE)
        date_re = re.compile(r"(January|February|March|April|May|June|July|August|"
                             r"September|October|November|December)\s?\d{1,2},?\s?\d{4}")
        found = []
        for a in soup.select("a[href]"):
            title = a.get_text(" ", strip=True)
            m = amount_re.search(title)
            if not m:
                continue
            amount = float(m.group(1).replace(",", ""))
            if m.group(2):
                amount *= 1_000_000
            tail = a.parent.get_text(" ", strip=True)
            dm = date_re.search(tail)
            date = pd.to_datetime(dm.group(0)) if dm else pd.NaT
            found.append({"date": date, "entity": title, "amount_usd": int(amount),
                          "action_type": "penalty" if "penalt" in title.lower() else "settlement",
                          "source_url": a["href"]})
        if not found:
            print("enforcement: nothing parsed, keeping existing file")
            return
        new = pd.DataFrame(found).dropna(subset=["date"])
        cur = pd.read_csv(DATA / "ocr_enforcement.csv", parse_dates=["date"])
        add = new[~new["source_url"].isin(cur["source_url"])]
        if len(add):
            out = pd.concat([cur, add], ignore_index=True).sort_values("date", ascending=False)
            out.to_csv(DATA / "ocr_enforcement.csv", index=False)
            print(f"enforcement: added {len(add)} new actions (titles need review)")
        else:
            print("enforcement: no new actions")
    except Exception as exc:  # noqa: BLE001
        print(f"enforcement fetch failed ({exc!r}); keeping existing data")


# ----------------------------------------------------------------- rebuild

VECTOR_RULES = {
    "Ransomware": r"ransomware|\bransom\b",
    "Phishing / email compromise":
        r"phishing|phished|email account[s]?\b.{0,60}(?:compromis|access|hack)|compromised.{0,30}email",
    "Other hacking / IT intrusion":
        r"\bhack|malware|virus\b|cyber|intrusion|breach of (?:its|the|their).{0,30}(?:network|system|server)"
        r"|network server.{0,40}(?:compromis|breach|access)"
        r"|unauthorized (?:access|party).{0,40}(?:network|server|system)|it system|security incident",
    "Insider snooping / misuse":
        r"(?:employee|workforce member|staff member|nurse|physician)s?\b"
        r".{0,90}(?:impermissibl|improperly|without authorization|unauthorized|snoop|inappropriately)",
    "Stolen device / burglary":
        r"(?:stolen|theft|burglar|broke into|break-in).{0,80}"
        r"(?:laptop|desktop|computer|thumb drive|flash drive|hard drive|phone|tablet|device|server|binder|paper|record|film|car|vehicle|office)"
        r"|(?:laptop|device|computer)s?\b.{0,40}(?:was|were) stolen",
    "Lost device / records": r"\blost\b|misplaced|could not be located|missing",
    "Improper disposal": r"disposal|disposed|dumpster|recycl|shredd",
    "Misdirected / inadvertent disclosure":
        r"mailing error|mailing vendor|misdirect|wrong (?:recipient|address|patient|fax)"
        r"|\bfax(?:ed)?\b.{0,40}(?:wrong|incorrect|error)|envelope|mailed.{0,60}(?:error|incorrect|wrong|other)"
        r"|mistakenly (?:posted|sent|emailed|mailed|disclosed|attached)"
        r"|inadvertently (?:posted|disclosed|sent|emailed|exposed|made|attached)"
        r"|(?:posted|accessible|viewable).{0,40}(?:website|internet|publicly|online)|human error",
}
REMEDIATION_RULES = {
    "New/updated safeguards": r"safeguard",
    "Staff retrained": r"retrain|additional training|workforce training|re-educat",
    "Credit monitoring offered": r"credit monitoring|identity (?:theft )?protection",
    "Corrective action plan": r"corrective action",
    "OCR technical assistance": r"technical assistance",
    "Discipline / termination": r"sanction|disciplin|terminat|dismissed",
    "Encryption implemented": r"implement.{0,50}encrypt|encrypt(?:ed|ion).{0,30}(?:devices|laptops|all)",
    "Policies revised": r"polic(?:y|ies).{0,50}(?:revis|updat|implement|develop|strengthen)",
}
EXCLUDE_BREACH_MATCH = {"os, inc.", "university health", "health care service corporation",
                        "new", "washington university school of medicine", "centra",
                        "regional medical center", "blue cross blue shield"}


def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    cols = list(df.columns)
    cols[0] = "Name of Covered Entity"
    cols[-2] = "Business Associate Present"
    df.columns = cols
    return df


def rebuild() -> None:
    active = load_raw(DATA / "breach_active.csv").assign(Status="Under Investigation")
    archived = load_raw(DATA / "breach_archived.csv").assign(Status="Archived")
    df = pd.concat([active, archived], ignore_index=True)
    df["Individuals Affected"] = pd.to_numeric(df["Individuals Affected"], errors="coerce")
    df["Breach Submission Date"] = pd.to_datetime(df["Breach Submission Date"], format="%m/%d/%Y")
    df["Name of Covered Entity"] = df["Name of Covered Entity"].str.strip()
    df["Web Description"] = df["Web Description"].str.replace(r"\s+", " ", regex=True).str.strip()
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "Status"]).reset_index(drop=True)
    df["Year"] = df["Breach Submission Date"].dt.year
    df["Primary Breach Type"] = df["Type of Breach"].str.split(",").str[0].str.strip()
    df.to_csv(DATA / "breach_clean.csv", index=False)
    print(f"rebuilt breach_clean.csv ({len(df):,} rows)")

    # --- text vectors
    raw = df["Web Description"].fillna("")
    df["has_narrative"] = (raw.str.len() > 0) & (raw.str.strip() != "\\N")
    wd = raw.str.lower().where(df["has_narrative"], "")
    for name, pattern in VECTOR_RULES.items():
        df[f"vec: {name}"] = wd.str.contains(pattern, regex=True)
    for name, pattern in REMEDIATION_RULES.items():
        df[f"rem: {name}"] = wd.str.contains(pattern, regex=True)
    primary = pd.Series("Other / unspecified", index=df.index)
    for name in reversed(list(VECTOR_RULES)):
        primary[df[f"vec: {name}"]] = name
    primary[~df["has_narrative"]] = "No narrative"
    df["Primary Vector"] = primary
    vcols = (["Name of Covered Entity", "State", "Covered Entity Type", "Individuals Affected",
              "Year", "Primary Breach Type", "Business Associate Present", "has_narrative",
              "Primary Vector"]
             + [c for c in df.columns if c.startswith(("vec: ", "rem: "))])
    df[vcols].to_csv(DATA / "breach_vectors.csv", index=False)
    print(f"rebuilt breach_vectors.csv")

    # --- cluster assignment with the persisted pipeline
    import joblib
    pipeline = joblib.load(MODELS / "kmeans_breach_clusters.joblib")
    feat_cols = json.loads((MODELS / "feature_columns.json").read_text())
    d = df.dropna(subset=["Individuals Affected", "Covered Entity Type", "Type of Breach"]).copy()
    feats = pd.DataFrame({
        "log_affected": np.log10(d["Individuals Affected"]),
        "year": d["Year"],
        "ba_present": (d["Business Associate Present"] == "Yes").astype(int)})
    feats = pd.concat([feats,
                       pd.get_dummies(d["Primary Breach Type"], prefix="bt").astype(int),
                       pd.get_dummies(d["Covered Entity Type"], prefix="et").astype(int)], axis=1)
    feats = feats.reindex(columns=feat_cols, fill_value=0)
    d["cluster"] = pipeline.predict(feats)
    d[["Name of Covered Entity", "State", "Covered Entity Type", "Individuals Affected",
       "Breach Submission Date", "Year", "Type of Breach", "Primary Breach Type",
       "Location of Breached Information", "Business Associate Present", "Status",
       "cluster"]].to_csv(DATA / "breach_clustered.csv", index=False)
    print(f"rebuilt breach_clustered.csv ({len(d):,} rows)")

    # --- enforcement join
    def norm(s: str) -> str:
        s = str(s).lower()
        s = re.sub(r"\b(inc|llc|llp|ltd|corp|corporation|company|co|pc|pa|dba|d/b/a|the)\b", "", s)
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s)).strip()

    def score(en: str, x: str) -> float:
        if min(len(en), len(x)) >= 10 and (en in x or x in en):
            return 1.0
        et, xt = set(en.split()), set(x.split())
        small = et if len(et) <= len(xt) else xt
        if len(small) >= 2 and len(" ".join(small)) >= 10 and (et <= xt or xt <= et):
            return 0.95
        return SequenceMatcher(None, en, x).ratio()

    enf = pd.read_csv(DATA / "ocr_enforcement.csv", parse_dates=["date"])
    df["_n"] = df["Name of Covered Entity"].map(norm)
    rows = []
    for _, act in enf.iterrows():
        en, ey = norm(act["entity"]), act["date"].year
        cand = df[(df["Year"] <= ey) & (df["Year"] >= ey - 9)].copy()
        cand["_s"] = cand["_n"].apply(lambda x: score(en, x))
        cand = cand[(cand["_s"] >= 0.90) &
                    (~cand["Name of Covered Entity"].str.lower().isin(EXCLUDE_BREACH_MATCH))]
        if len(cand):
            best = cand.sort_values(["_s", "Individuals Affected"], ascending=False).iloc[0]
            rows.append({"enforcement_entity": act["entity"], "enforcement_date": act["date"],
                         "amount_usd": act["amount_usd"], "action_type": act["action_type"],
                         "source_url": act["source_url"],
                         "breach_entity": best["Name of Covered Entity"],
                         "breach_year": best["Year"],
                         "individuals_affected": best["Individuals Affected"],
                         "breach_type": best["Type of Breach"],
                         "match_score": round(best["_s"], 2)})
    pd.DataFrame(rows).to_csv(DATA / "breach_penalties.csv", index=False)
    print(f"rebuilt breach_penalties.csv ({len(rows)} matches)")


if __name__ == "__main__":
    if "--skip-fetch" not in sys.argv:
        fetch_portal()
        fetch_enforcement()
    rebuild()
    print("refresh complete")
