

# app.py
# Customer Spend Prediction — Streamlit dashboard (Raw Upload -> Cleaning -> RFM -> ML -> Segmentation + Excel-friendly Download)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sklearn 
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

# Optional ML (falls back to heuristics if sklearn missing)
try:
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

# ---------- Page & theme ----------
st.set_page_config(page_title="Customer Spend Prediction", layout="wide")
st.markdown(
    """
    <style>
      html, body, [class*="stApp"] { background-color: #0e1117; color: #FAFAFA; }
      h1, h2, h3, h4, h5, h6 { color: #FAFAFA; }
      .stCaption, .st-emotion-cache-1kyxreq { color: #C9CED6 !important; }
      .stButton > button { background:#2F80ED; color:white; border:1px solid #2F80ED; border-radius:6px; }
      .stSlider { padding-top: 0.25rem; }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- Utilities ----------
REQUIRED_ORDER = [
    "customer_id","frequency","pred_prob","spend_90_total","pred_spend","spend_actual_vs_pred"
]

def make_demo_data(n=2500, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    freq = rng.integers(0, 180, size=n)
    pred_prob = np.clip(rng.normal(0.55, 0.18, size=n), 0, 1)
    pred_spend = np.clip(rng.normal(120, 60, size=n) + 0.7 * pred_prob * 200, 0, None)
    actual_spend = np.clip(pred_spend + rng.normal(0, 150, size=n) - (1 - pred_prob) * 120, 0, None)
    diff = actual_spend - pred_spend
    return pd.DataFrame(
        {
            "customer_id": np.arange(1, n + 1),
            "frequency": freq,
            "pred_prob": pred_prob.round(3),
            "spend_90_total": np.round(actual_spend, 2),
            "pred_spend": np.round(pred_spend, 2),
            "spend_actual_vs_pred": np.round(diff, 2),
        }
    )

def coerce_numeric(df: pd.DataFrame, cols) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def compute_shortfall(df: pd.DataFrame) -> pd.Series:
    if {"pred_spend","spend_90_total"}.issubset(df.columns):
        s = pd.to_numeric(df["pred_spend"], errors="coerce") - pd.to_numeric(df["spend_90_total"], errors="coerce")
    elif "spend_actual_vs_pred" in df.columns:
        s = -pd.to_numeric(df["spend_actual_vs_pred"], errors="coerce")
    else:
        s = pd.Series(0, index=df.index, dtype="float64")
    return s.clip(lower=0)

# ---------- RFM / Cleaning pipeline (works for raw Online Retail.xlsx or similar) ----------
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    # common mappings to canonical names
    col_map = {
        "customerid": "customer_id",
        "cust_id": "customer_id",
        "customer_id": "customer_id",
        "invoiceno": "invoice_no",
        "invoice": "invoice_no",
        "invoicedate": "invoice_date",
        "date": "invoice_date",
        "stockcode": "sku",
        "quantity": "quantity",
        "unitprice": "unit_price",
        "price": "unit_price",
        "amount": "amount",
        "total": "amount",
        "country": "country",
        "description": "description"
    }
    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)
    return df

def clean_transactions(raw_txn: pd.DataFrame) -> pd.DataFrame:
    df = standardize_columns(raw_txn)
    # Keep only needed columns if present
    needed = [c for c in ["customer_id","invoice_no","invoice_date","quantity","unit_price","amount","country","description"] if c in df.columns]
    if needed:
        df = df[needed + [c for c in df.columns if c not in needed]]  # keep extras too
    # Basic cleaning
    if "invoice_date" in df.columns:
        df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
    # Remove rows with no customer_id
    if "customer_id" in df.columns:
        df = df[~df["customer_id"].isna()]
    # Remove credit notes/cancellations if invoice starts with 'C'
    if "invoice_no" in df.columns:
        df = df[~df["invoice_no"].astype(str).str.startswith("C")]
    # Ensure numeric
    for c in ["quantity","unit_price","amount"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # If amount missing, compute = quantity * unit_price
    if "amount" not in df.columns and {"quantity","unit_price"}.issubset(df.columns):
        df["amount"] = df["quantity"] * df["unit_price"]
    # Remove wrong/negative values
    for c in ["quantity","unit_price","amount"]:
        if c in df.columns:
            df = df[df[c].notna()]
    if "quantity" in df.columns:
        df = df[df["quantity"] > 0]
    if "unit_price" in df.columns:
        df = df[df["unit_price"] >= 0]
    if "amount" in df.columns:
        df = df[df["amount"] >= 0]
    # Outlier trimming on amount (simple quantile cap)
    if "amount" in df.columns and len(df) > 0:
        hi = df["amount"].quantile(0.999)
        df.loc[df["amount"] > hi, "amount"] = hi
    return df

def make_customer_features(txn: pd.DataFrame) -> pd.DataFrame:
    """Compute RFM + 90/365 spend + frequency etc. Returns one row per customer."""
    if txn.empty:
        return pd.DataFrame(columns=["customer_id","recency","frequency","monetary","spend_90_total","spend_365_total"])
    t = txn.copy()
    if "invoice_date" in t.columns:
        t["invoice_date"] = pd.to_datetime(t["invoice_date"], errors="coerce")
        last_date = t["invoice_date"].max()
    else:
        # If no dates, synthesize all as last_date and we can't compute recency windows well
        last_date = pd.Timestamp.today().normalize()
        t["invoice_date"] = last_date
    # Monetary per row must exist
    if "amount" not in t.columns:
        # Fallback if amount still missing
        t["amount"] = 0.0
    # Grouping keys
    if "customer_id" not in t.columns:
        # fabricate ids if absent
        t["customer_id"] = 1
    # Frequency (unique invoices if invoice_no exists, else count of rows)
    if "invoice_no" in t.columns:
        freq_df = t.groupby("customer_id")["invoice_no"].nunique().rename("frequency")
    else:
        freq_df = t.groupby("customer_id").size().rename("frequency")
    # Monetary = total spend overall
    mon_df = t.groupby("customer_id")["amount"].sum().rename("monetary")
    # Recency = days since last purchase
    rec_df = t.groupby("customer_id")["invoice_date"].max().rename("last_date")
    rec_df = (last_date - rec_df).dt.days.rename("recency")
    # Spend windows
    cutoff_90 = last_date - pd.Timedelta(days=90)
    cutoff_365 = last_date - pd.Timedelta(days=365)
    spend_90 = t.loc[t["invoice_date"] >= cutoff_90].groupby("customer_id")["amount"].sum().rename("spend_90_total")
    spend_365 = t.loc[t["invoice_date"] >= cutoff_365].groupby("customer_id")["amount"].sum().rename("spend_365_total")
    # Combine
    features = pd.concat([freq_df, mon_df, rec_df, spend_90, spend_365], axis=1).fillna(0).reset_index()
    # Ensure types
    for c in ["frequency","monetary","recency","spend_90_total","spend_365_total"]:
        if c in features.columns:
            features[c] = pd.to_numeric(features[c], errors="coerce").fillna(0)
    return features

def _rule_pred_prob(df: pd.DataFrame) -> np.ndarray:
    # Higher prob if frequent & recent & historically higher spend
    rec = df["recency"].replace(0, 0.1)
    freq = df["frequency"]
    mny = df["monetary"]
    s365 = df["spend_365_total"]
    # Normalize-ish
    p = (
        0.45 * (1 / (1 + rec/30)) +      # recent buys → higher prob
        0.25 * np.tanh(freq / 5) +       # more orders → higher prob
        0.15 * np.tanh(mny / (df["monetary"].median() + 1e-9)) +
        0.15 * np.tanh(s365 / (df["spend_365_total"].median() + 1e-9))
    )
    return np.clip(p, 0.01, 0.99)

def _rule_pred_spend(df: pd.DataFrame) -> np.ndarray:
    # Base spend ~ fraction of spend_365_total, boosted by frequency and recency
    base = 0.2 * df["spend_365_total"] * (1 / (1 + df["recency"]/60)) * (1 + np.tanh(df["frequency"]/10))
    # Ensure numeric
    base = pd.to_numeric(base, errors="coerce").fillna(0).values
    return np.clip(base, 0, None)

def ml_or_rules_predict(cust: pd.DataFrame, retrain: bool) -> pd.DataFrame:
    """
    Produce pred_prob and pred_spend. If sklearn available and retrain=True, fit simple models.
    Otherwise, use transparent heuristics from RFM features.
    """
    df = cust.copy()
    for c in ["recency","frequency","monetary","spend_90_total","spend_365_total"]:
        if c not in df.columns:
            df[c] = 0.0
    if SKLEARN_OK and retrain and len(df) >= 50:
        # Binary target: spent in last 90 days?
        y_cls = (df["spend_90_total"] > 0).astype(int)
        X = df[["recency","frequency","monetary","spend_365_total"]].copy()
        cls = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=1000))])
        try:
            cls.fit(X, y_cls)
            pred_prob = cls.predict_proba(X)[:, 1]
        except Exception:
            # fallback to rules if training fails
            pred_prob = _rule_pred_prob(df)
        # Regression target: spend_90_total (non-negative)
        # To avoid trivial zeros, train on those with some historic spend_365_total
        mask_reg = df["spend_365_total"] > 0
        if mask_reg.sum() >= 20:
            Xr = df.loc[mask_reg, ["recency","frequency","monetary","spend_365_total"]]
            yr = df.loc[mask_reg, "spend_90_total"]
            regr = Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())])
            try:
                base_pred = regr.predict(df[["recency","frequency","monetary","spend_365_total"]])
                base_pred = np.clip(base_pred, 0, None)
            except Exception:
                base_pred = _rule_pred_spend(df)
        else:
            base_pred = _rule_pred_spend(df)
        df["pred_prob"] = np.clip(pred_prob, 0, 1)
        # Couple spend with propensity so zero-prob → zero spend
        df["pred_spend"] = (df["pred_prob"] * base_pred).astype(float)
    else:
        # Heuristic rules
        df["pred_prob"] = _rule_pred_prob(df)
        df["pred_spend"] = _rule_pred_spend(df) * df["pred_prob"]
    # Spend actual vs predicted if we carry actual window
    df["spend_actual_vs_pred"] = df.get("spend_90_total", 0) - df["pred_spend"]
    return df

# ---------- Header ----------
st.markdown("### Customer Spend Prediction")
st.markdown("<h1 style='margin-top:0.25rem;'>Welcome to the Customer Analytics Dashboard</h1>", unsafe_allow_html=True)
st.caption("Explore customers by Predicted Spend versus Actual Spend during a 90-day evaluation period.")

# ---------- Layout ----------
left, right = st.columns([1.3, 3.7], gap="large")
with left:
    st.subheader("Spend Actual vs Predicted")
    st.write("Segment customers that were predicted to spend but didn’t, then target them with campaigns.")

# ---------- Upload ----------
with left:
    uploaded = st.file_uploader(
        "Upload dataset (CSV or Excel)",
        type=["csv","xlsx","xls"],
        help="Upload raw transactions (e.g., Online Retail.xlsx). Columns like CustomerID, InvoiceDate, Quantity, UnitPrice are ideal."
    )
    retrain = st.checkbox(
        "Retrain simple model on this data",
        value=True,
        help="If off, use fast heuristics. If on (recommended), trains lightweight models on uploaded data."
    )

# ---------- Data load ----------
if uploaded is not None:
    try:
        if uploaded.name.lower().endswith((".xlsx",".xls")):
            raw_txn = pd.read_excel(uploaded)
        else:
            raw_txn = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        raw_txn = None
else:
    raw_txn = None

# If no upload, use demo customer-level data (already aggregated)
if raw_txn is None:
    df_customers = make_demo_data()
    df_customers["recency"] = np.random.randint(0, 180, len(df_customers))
    df_customers["monetary"] = np.random.gamma(2.0, 120.0, len(df_customers))
    df_customers["frequency"] = df_customers["frequency"].fillna(0)
else:
    # Full pipeline on raw transactions
    tx = clean_transactions(raw_txn)
    # Build per-customer features (RFM + windows)
    feats = make_customer_features(tx)
    # Run ML or rules to get predictions
    df_customers = ml_or_rules_predict(feats, retrain=retrain)

# Ensure canonical columns exist
for c in REQUIRED_ORDER:
    if c not in df_customers.columns:
        df_customers[c] = np.nan

# Coerce types & compute shortfall (final)
df_customers = coerce_numeric(df_customers, ["pred_spend","spend_90_total","spend_actual_vs_pred","pred_prob","frequency"])
df_customers["shortfall"] = compute_shortfall(df_customers)
df_customers["pred_prob"] = df_customers["pred_prob"].fillna(0)
df_customers["shortfall"] = df_customers["shortfall"].fillna(0)

# ---------- Data-driven slider defaults ----------
sf = df_customers["shortfall"].fillna(0)
sf_pos = sf[sf > 0]
sf_max = int(max(50, float(sf.max()))) if len(df_customers) else 50
sf_med = int(np.percentile(sf_pos, 50)) if not sf_pos.empty else 0

with left:
    prob_min = st.slider("Min propensity", 0.0, 1.0, 0.40, 0.05,
                         help="Include customers at or above this predicted propensity.")
    threshold = st.slider("Shortfall threshold ($)", 0, sf_max, sf_med, 25, format="$%d",
                          help="How much Predicted exceeds Actual to qualify.")
    delimiter_choice = st.radio("CSV delimiter", ["Excel hint (sep=,)", "Comma (,)", "Semicolon (;)"],
                                index=0, horizontal=True)

# ---------- Filter & segment ----------
keep_cols = [c for c in ["customer_id","frequency","pred_prob","spend_90_total","pred_spend",
                         "spend_actual_vs_pred","shortfall","recency","monetary","spend_365_total"]
             if c in df_customers.columns]
mask = (df_customers["pred_prob"] >= prob_min) & (df_customers["shortfall"] >= threshold)
segment = df_customers.loc[mask, keep_cols].copy()

# ---------- Download payload (Excel-friendly) ----------
if delimiter_choice == "Semicolon (;)":
    csv_bytes = segment.to_csv(index=False, sep=";").encode("utf-8")
elif delimiter_choice == "Comma (,)":
    csv_bytes = segment.to_csv(index=False).encode("utf-8")
else:
    csv_bytes = ("sep=,\n" + segment.to_csv(index=False)).encode("utf-8")

with left:
    st.caption(f"Segment size: {len(segment):,}")
    st.download_button(
        "Download Segmentation",
        data=csv_bytes,
        file_name="segmentation.csv",
        mime="text/csv",
        disabled=segment.empty,
        help="Exports customers that meet the current filters."
    )
    with st.expander("Preview segment"):
        st.dataframe(segment.head(50), use_container_width=True)

# ---------- Main scatter (right) ----------
# Choose a robust color series that truly varies
df = df_customers
if "spend_actual_vs_pred" in df.columns and df["spend_actual_vs_pred"].dropna().nunique() >= 2:
    color_series = df["spend_actual_vs_pred"]
else:
    color_series = -df["shortfall"]  # invert to match legend naming expectation
# Fallbacks for color variation
if color_series.dropna().nunique() < 2:
    color_series = df["shortfall"]
if color_series.dropna().nunique() < 2 and {"spend_90_total","pred_spend"}.issubset(df.columns):
    color_series = df["spend_90_total"] - df["pred_spend"]
if color_series.dropna().nunique() < 2 and "frequency" in df.columns:
    color_series = df["frequency"]

valid = color_series.dropna()
prc = np.nanpercentile(np.abs(valid), 99) if not valid.empty else 100
rng = float(max(10, min(1500, prc)))
range_color = [-rng, rng]
colorbar_title = "spend_actual_vs_pred" if "spend_actual_vs_pred" in df.columns else "shortfall"

# Build hover_data dynamically so only existing columns are included (avoids ValueError)
hover_data = {}
# Always-on basics if present
for col in ["customer_id", "frequency", "shortfall"]:
    if col in df.columns:
        hover_data[col] = True
# Formatted pred_prob if present
if "pred_prob" in df.columns:
    hover_data["pred_prob"] = ':.3f'
# Optional fields if present
for col in ["spend_90_total", "pred_spend", "spend_actual_vs_pred", "recency", "monetary", "spend_365_total"]:
    if col in df.columns:
        hover_data[col] = True

fig = px.scatter(
    df,
    x="frequency",
    y="pred_prob",
    color=color_series,
    color_continuous_scale="RdYlBu_r",
    range_color=range_color,
    opacity=0.75,
    template="plotly_dark",
    hover_data=hover_data,
    labels={"frequency": "frequency", "pred_prob": "pred_prob"},
)
fig.update_traces(marker=dict(size=7, line=dict(width=0)))
fig.update_coloraxes(colorbar_title_text=colorbar_title)
fig.update_layout(
    coloraxis_colorbar=dict(title=colorbar_title),
    xaxis=dict(title="frequency", zeroline=False, gridcolor="rgba(255,255,255,0.1)"),
    yaxis=dict(title="pred_prob", range=[0, 1], zeroline=False, gridcolor="rgba(255,255,255,0.1)"),
    margin=dict(l=10, r=10, t=10, b=10),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)

with right:
    st.plotly_chart(fig, use_container_width=True)

# ---------- Calibration + Brier Score ----------

if "spend_90_total" in df_customers.columns and "pred_prob" in df_customers.columns:
    try:
        # y_true = actual outcome (1 if spent in last 90 days else 0)
        y_true = (df_customers["spend_90_total"] > 0).astype(int).values
        # y_pred = model ki predicted probability
        y_pred = df_customers["pred_prob"].fillna(0).values

        # Calibration curve
        prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=10)

        fig_cal, ax = plt.subplots()
        ax.plot(prob_pred, prob_true, marker='o', label="Model")
        ax.plot([0, 1], [0, 1], linestyle="--", label="Perfectly Calibrated")
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Actual Return Rate")
        ax.legend()
        st.pyplot(fig_cal)

        # Brier Score
        brier = brier_score_loss(y_true, y_pred)
        st.write("Brier Score:", round(brier, 4))
    except Exception as e:
        st.warning(f"Calibration check failed: {e}")
