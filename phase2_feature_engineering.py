"""
Phase 2 — Feature Engineering
Customer Segmentation Tutorial

Input  : master_df.csv  (output of Phase 1)
Output : features_df.csv — scaled feature matrix ready for segmentation

Sections:
    1. Load & sanity-check master_df
    2. RFM core features
    3. Behavioural features
    4. Customer profile features
    5. RFM scoring (quintile bins)
    6. Scale features for ML
    7. Inspect & save
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings("ignore")

DATA_DIR   = r"D:\data use for learning"
AS_OF_DATE = datetime(2026, 6, 10)

# ─────────────────────────────────────────────────────────────
# STEP 1 — LOAD MASTER_DF
# ─────────────────────────────────────────────────────────────

print("=" * 60)
print("STEP 1 — Loading master_df")
print("=" * 60)

master = pd.read_csv(f"{DATA_DIR}/master_df.csv", parse_dates=["Registration_Date",
                                                                 "first_order_date",
                                                                 "last_order_date",
                                                                 "last_interaction_date"])

print(f"  Loaded: {master.shape[0]:,} rows × {master.shape[1]} cols")
assert master["Customer_ID"].nunique() == len(master), "Duplicate Customer_IDs found!"
print("  Duplicate ID check: OK")


# ─────────────────────────────────────────────────────────────
# STEP 2 — RFM CORE FEATURES
# ─────────────────────────────────────────────────────────────
# RFM is the backbone of customer segmentation.
#
#   Recency   — how recently did they buy?  Lower = better.
#   Frequency — how often do they buy?     Higher = better.
#   Monetary  — how much do they spend?    Higher = better.
#
# We compute raw values first, then score them into quintiles
# in Step 5.  Keep raw + scored versions — raw for ML,
# scored for rule-based segmentation (Phase 3a SQL / 3b Python).

print("\n" + "=" * 60)
print("STEP 2 — RFM core features")
print("=" * 60)

feat = pd.DataFrame({"Customer_ID": master["Customer_ID"]})

# ── Recency ──────────────────────────────────────────────────
# Days since last completed order.
# We use completed orders only — a refunded last order doesn't
# mean the customer is still active.

feat["R_days"] = master["recency_days"]

# Log-transform: recency is right-skewed (most customers bought
# recently; a long tail of dormant ones).  Log pulls the tail in
# so it doesn't dominate distance-based algorithms.
feat["R_log"] = np.log1p(master["recency_days"])

print(f"  Recency  — mean: {feat['R_days'].mean():.0f} days  "
      f"median: {feat['R_days'].median():.0f} days  "
      f"max: {feat['R_days'].max():.0f} days")


# ── Frequency ────────────────────────────────────────────────
# Completed orders only — refunds/cancellations don't count as
# genuine purchases.

feat["F_orders"]    = master["completed_orders"]
feat["F_orders_log"] = np.log1p(master["completed_orders"])

print(f"  Frequency — mean: {feat['F_orders'].mean():.1f} orders  "
      f"median: {feat['F_orders'].median():.0f}  "
      f"max: {feat['F_orders'].max():.0f}")


# ── Monetary ─────────────────────────────────────────────────
# Two angles: total spend (lifetime value) and average order
# value (spend per trip).  Both matter:
#   - A customer with 1 huge order looks like Premium on total
#     but New on frequency.
#   - AOV reveals the "type" of shopper regardless of history.

feat["M_total_revenue"] = master["total_revenue"]
feat["M_aov"]           = master["avg_order_value"].fillna(0)
feat["M_revenue_log"]   = np.log1p(master["total_revenue"])
feat["M_aov_log"]       = np.log1p(master["avg_order_value"].fillna(0))

print(f"  Monetary  — total rev mean: ${feat['M_total_revenue'].mean():,.0f}  "
      f"AOV mean: ${feat['M_aov'].mean():.0f}")


# ─────────────────────────────────────────────────────────────
# STEP 3 — BEHAVIOURAL FEATURES
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 3 — Behavioural features")
print("=" * 60)

# ── Purchase velocity ─────────────────────────────────────────
# Orders per active month — normalises frequency by tenure so a
# 6-month customer with 10 orders isn't penalised vs a 3-year
# customer with 30.

tenure_months = (master["Days_Since_Registration"] / 30).clip(lower=1)
feat["B_orders_per_month"] = (master["total_orders"] / tenure_months).round(4)

# Customer lifespan: days between first and last order.
# A high lifespan = loyal long-term buyer.
feat["B_lifespan_days"] = master["customer_lifespan_days"].fillna(0)

print(f"  Purchase velocity — mean: {feat['B_orders_per_month'].mean():.2f} orders/month")


# ── Basket behaviour ──────────────────────────────────────────
feat["B_avg_items_per_order"] = (
    master["total_items"] / master["total_orders"].replace(0, np.nan)
).fillna(0).round(2)

feat["B_revenue_per_item"] = master["revenue_per_item"].fillna(0)

print(f"  Avg items/order   — mean: {feat['B_avg_items_per_order'].mean():.1f}")


# ── Discount sensitivity ──────────────────────────────────────
# Customers who always buy on discount are price-sensitive and
# less profitable.  High discount usage is a churn risk signal.

feat["B_avg_discount_pct"]  = master["avg_discount_pct"].fillna(0)
feat["B_discount_revenue_ratio"] = (
    master["total_discount_amt"] / master["total_revenue"].replace(0, np.nan)
).fillna(0).clip(0, 1).round(4)

print(f"  Avg discount used — mean: {feat['B_avg_discount_pct'].mean():.1f}%")


# ── Refund / cancel behaviour ─────────────────────────────────
# High refund rate = product-fit problem or fraudulent behaviour.
# High cancel rate = checkout friction or impulse buying.

feat["B_refund_rate"]  = master["refund_rate"].fillna(0)
feat["B_cancel_rate"]  = master["cancel_rate"].fillna(0)

print(f"  Refund rate       — mean: {feat['B_refund_rate'].mean():.3f}  "
      f"max: {feat['B_refund_rate'].max():.3f}")


# ── Digital engagement ────────────────────────────────────────
# How engaged is this customer across digital channels?
# Email CTR and web visits signal intent even when not purchasing.

feat["B_email_ctr"]           = master["email_ctr"].fillna(0)
feat["B_web_visits"]          = master["web_visits"].fillna(0)
feat["B_app_opens"]           = master["app_opens"].fillna(0)
feat["B_total_interactions"]  = master["total_interactions"].fillna(0)

# Interaction rate: interactions per day of tenure.
# Normalises engagement by how long the customer has been around.
feat["B_interaction_rate"] = (
    master["total_interactions"] / master["Days_Since_Registration"].clip(lower=1)
).round(6)

print(f"  Email CTR         — mean: {feat['B_email_ctr'].mean():.3f}")
print(f"  Interaction rate  — mean: {feat['B_interaction_rate'].mean():.4f}/day")


# ── Support burden ────────────────────────────────────────────
# Tickets per order: a customer who raises a ticket on every
# other purchase is expensive to serve regardless of revenue.

feat["B_support_tickets"]          = master["support_tickets"].fillna(0)
feat["B_tickets_per_order"] = (
    master["support_tickets"] / master["total_orders"].replace(0, np.nan)
).fillna(0).round(4)

print(f"  Tickets/order     — mean: {feat['B_tickets_per_order'].mean():.3f}")


# ── Churn risk signal ─────────────────────────────────────────
# A composite binary flag: customer is "at risk" if:
#   - Last order was 90+ days ago, AND
#   - Their interaction rate is below the 25th percentile
# This catches customers going quiet before they fully churn.

interaction_rate_p25 = feat["B_interaction_rate"].quantile(0.25)
feat["B_churn_risk_flag"] = (
    (feat["R_days"] >= 90) &
    (feat["B_interaction_rate"] <= interaction_rate_p25)
).astype(int)

churn_count = feat["B_churn_risk_flag"].sum()
print(f"  Churn risk flag   — {churn_count} customers flagged "
      f"({churn_count/len(feat)*100:.1f}%)")


# ─────────────────────────────────────────────────────────────
# STEP 4 — CUSTOMER PROFILE FEATURES
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 4 — Customer profile features")
print("=" * 60)

# ── Tenure ────────────────────────────────────────────────────
feat["P_tenure_days"]   = master["Days_Since_Registration"]
feat["P_tenure_months"] = (master["Days_Since_Registration"] / 30).round(1)

# ── Loyalty ───────────────────────────────────────────────────
# Points per dollar spent: a normalised loyalty engagement score.
# High ratio = customer actively redeeming / earning points.

feat["P_loyalty_points"] = master["Loyalty_Points_Balance"].fillna(0)
feat["P_loyalty_ratio"]  = (
    master["Loyalty_Points_Balance"] /
    master["total_revenue"].replace(0, np.nan)
).fillna(0).round(4)

# ── Satisfaction ──────────────────────────────────────────────
feat["P_satisfaction"] = master["Satisfaction_Score_1_10"].fillna(
    master["Satisfaction_Score_1_10"].median()   # impute median for nulls
)

# ── Subscription & marketing flags ────────────────────────────
feat["P_is_subscriber"]    = master["Subscription_Member"].astype(int)
feat["P_marketing_opt_in"] = master["Marketing_Consent"].astype(int)

print(f"  Tenure      — mean: {feat['P_tenure_months'].mean():.0f} months")
print(f"  Loyalty ratio — mean: {feat['P_loyalty_ratio'].mean():.3f} pts/$")
print(f"  Satisfaction  — mean: {feat['P_satisfaction'].mean():.2f} / 10")
print(f"  Subscribers   — {feat['P_is_subscriber'].sum():,} "
      f"({feat['P_is_subscriber'].mean()*100:.1f}%)")


# ─────────────────────────────────────────────────────────────
# STEP 5 — RFM QUINTILE SCORING
# ─────────────────────────────────────────────────────────────
# Score each RFM dimension 1–5 (quintiles).
# Note the direction:
#   R: LOWER recency = better → score 5 to 1 (reversed)
#   F: HIGHER frequency = better → score 1 to 5
#   M: HIGHER monetary = better → score 1 to 5
#
# These integer scores are used by the rule-based segmentation
# in Phase 3a (SQL) and Phase 3b (Python).

print("\n" + "=" * 60)
print("STEP 5 — RFM quintile scoring (1–5)")
print("=" * 60)

def quintile_score(series: pd.Series, ascending: bool = True) -> pd.Series:
    """
    Bin a series into 5 equal-frequency buckets (quintiles).
    ascending=True  → higher value = higher score (F, M)
    ascending=False → lower value  = higher score (R)
    Uses rank + cut to handle ties and duplicates gracefully.
    """
    ranked = series.rank(method="first", ascending=ascending)
    return pd.qcut(ranked, q=5, labels=[1, 2, 3, 4, 5]).astype(int)


feat["R_score"] = quintile_score(feat["R_days"],          ascending=False)  # low recency = good
feat["F_score"] = quintile_score(feat["F_orders"],        ascending=True)
feat["M_score"] = quintile_score(feat["M_total_revenue"], ascending=True)

# Composite RFM score: simple sum (3–15) and weighted version
feat["RFM_sum"]  = feat["R_score"] + feat["F_score"] + feat["M_score"]

# Weighted: Monetary weighted slightly higher for revenue focus
feat["RFM_weighted"] = (
    feat["R_score"] * 0.30 +
    feat["F_score"] * 0.35 +
    feat["M_score"] * 0.35
).round(3)

# RFM segment label from the composite score
# These are the rule-based predicted segments for Phase 3
def rfm_label(row: pd.Series) -> str:
    r, f, m = row["R_score"], row["F_score"], row["M_score"]
    score   = row["RFM_sum"]
    if r >= 4 and f >= 4 and m >= 4:
        return "Premium"
    elif r >= 3 and f >= 3 and m >= 3:
        return "Standard"
    elif r <= 2 and f >= 3:
        return "At-Risk"
    elif r <= 1:
        return "Churned"
    elif f <= 1:
        return "New"
    else:
        return "Budget"

feat["RFM_segment"] = feat.apply(rfm_label, axis=1)

print("  RFM score distribution:")
print(feat[["R_score", "F_score", "M_score"]].describe().loc[["mean", "min", "50%", "max"]].round(2).to_string())
print("\n  RFM segment counts:")
for seg, n in feat["RFM_segment"].value_counts().items():
    pct = n / len(feat) * 100
    bar = "█" * int(pct / 2)
    print(f"    {seg:<12} {n:>4}  ({pct:4.1f}%)  {bar}")


# ─────────────────────────────────────────────────────────────
# STEP 6 — SCALE FEATURES FOR ML
# ─────────────────────────────────────────────────────────────
# K-Means (Phase 4) uses Euclidean distance, so features on
# different scales (revenue $0–50k vs score 1–5) will dominate
# unfairly if not scaled.
#
# We use RobustScaler (scales by median + IQR) rather than
# StandardScaler (mean + std) because our features are skewed
# and have outliers.  RobustScaler is far less sensitive to them.
#
# We only scale the ML feature columns — not the scores or flags
# that are already on a fixed scale.

print("\n" + "=" * 60)
print("STEP 6 — Scaling features for ML")
print("=" * 60)

ML_FEATURES = [
    # RFM (log-transformed raw values)
    "R_log", "F_orders_log", "M_revenue_log", "M_aov_log",
    # Behavioural
    "B_orders_per_month", "B_lifespan_days",
    "B_avg_items_per_order", "B_revenue_per_item",
    "B_avg_discount_pct", "B_discount_revenue_ratio",
    "B_refund_rate", "B_cancel_rate",
    "B_email_ctr", "B_interaction_rate",
    "B_tickets_per_order",
    # Profile
    "P_tenure_days", "P_loyalty_ratio", "P_satisfaction",
]

scaler     = RobustScaler()
scaled_arr = scaler.fit_transform(feat[ML_FEATURES])
scaled_df  = pd.DataFrame(scaled_arr, columns=[f"scaled_{c}" for c in ML_FEATURES])

print(f"  Scaled {len(ML_FEATURES)} features using RobustScaler")
print(f"  Scaled matrix shape: {scaled_df.shape}")

# Quick check: scaled values should be roughly centred around 0
print("\n  Post-scaling median (should be ~0.0):")
medians = scaled_df.median().round(3)
for col, val in medians.items():
    flag = "  <- check" if abs(val) > 0.5 else ""
    print(f"    {col:<45} {val:>7}{flag}")


# ─────────────────────────────────────────────────────────────
# STEP 7 — INSPECT FEATURE CORRELATIONS & SAVE
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 7 — Feature summary & save")
print("=" * 60)

# Combine everything: Customer_ID + raw features + RFM scores + scaled
features_df = pd.concat([
    feat.reset_index(drop=True),
    scaled_df.reset_index(drop=True),
], axis=1)

# Sanity: no nulls allowed in the scaled columns
null_scaled = scaled_df.isnull().sum().sum()
print(f"  Nulls in scaled features : {null_scaled}  {'OK' if null_scaled == 0 else 'WARNING!'}")

# Feature count summary
raw_cols    = [c for c in feat.columns if c != "Customer_ID" and not c.startswith("RFM")]
score_cols  = [c for c in feat.columns if c.startswith("RFM")]
scaled_cols = list(scaled_df.columns)

print(f"\n  Raw features     : {len(raw_cols)}")
print(f"  RFM score cols   : {len(score_cols)}")
print(f"  Scaled ML cols   : {len(scaled_cols)}")
print(f"  Total columns    : {features_df.shape[1]} (incl. Customer_ID)")

# Top correlations with RFM_weighted (useful sanity check)
print("\n  Top correlations with RFM_weighted score:")
corr = features_df[raw_cols + ["RFM_weighted"]].corr()["RFM_weighted"].drop("RFM_weighted")
top_corr = corr.abs().sort_values(ascending=False).head(8)
for col, val in top_corr.items():
    direction = "+" if corr[col] >= 0 else "-"
    bar = "█" * int(abs(val) * 20)
    print(f"    {direction}{col:<40} r={corr[col]:+.3f}  {bar}")

# Save
features_df.to_csv(f"{DATA_DIR}/features_df.csv", index=False)
print(f"\n  Saved → features_df.csv  "
      f"({features_df.shape[0]:,} rows × {features_df.shape[1]} cols)")
print("\nPhase 2 complete. Run phase3a_sql_segmentation.py next.")
