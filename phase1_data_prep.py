"""
Phase 1 — Data Preparation & Cleaning
Customer Segmentation Tutorial

Run this file section by section (or all at once).
Output: master_df.csv — one clean row per customer, ready for Phase 2.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

AS_OF_DATE = datetime(2026, 6, 10)   # must match generation date

# ─────────────────────────────────────────────────────────────
# STEP 1 — LOAD RAW CSVs
# ─────────────────────────────────────────────────────────────
# We read with dtype=str first so nothing gets silently coerced.
# We'll fix types explicitly in Step 3.

print("=" * 60)
print("STEP 1 — Loading raw files")
print("=" * 60)

customers_raw    = pd.read_csv("01_customers_raw.csv",    dtype=str)
orders_raw       = pd.read_csv("02_orders_raw.csv",       dtype=str)
interactions_raw = pd.read_csv("03_interactions_raw.csv", dtype=str)
ground_truth     = pd.read_csv("00_ground_truth_segments.csv")

print(f"customers    : {customers_raw.shape[0]:>6,} rows × {customers_raw.shape[1]} cols")
print(f"orders       : {orders_raw.shape[0]:>6,} rows × {orders_raw.shape[1]} cols")
print(f"interactions : {interactions_raw.shape[0]:>6,} rows × {interactions_raw.shape[1]} cols")
print(f"ground_truth : {ground_truth.shape[0]:>6,} rows × {ground_truth.shape[1]} cols")


# ─────────────────────────────────────────────────────────────
# STEP 2 — INSPECT & PROFILE EACH TABLE
# ─────────────────────────────────────────────────────────────

def profile_table(df: pd.DataFrame, name: str) -> None:
    """Print a quick data-quality card for any DataFrame."""
    print(f"\n{'─'*50}")
    print(f"  {name}  ({df.shape[0]:,} rows × {df.shape[1]} cols)")
    print(f"{'─'*50}")

    null_counts  = df.isnull().sum()
    empty_counts = (df == "").sum()          # empty strings from CSV
    total_bad    = null_counts + empty_counts

    profile = pd.DataFrame({
        "dtype"    : df.dtypes,
        "nulls"    : null_counts,
        "empty"    : empty_counts,
        "bad_total": total_bad,
        "bad_%"    : (total_bad / len(df) * 100).round(1),
        "example"  : df.apply(lambda c: c.dropna().iloc[0] if c.dropna().shape[0] else "—"),
    })

    # Only show columns that have problems or are worth knowing about
    print(profile[profile["bad_total"] > 0].to_string() or "  No nulls or empty strings found.")

    # Numeric-looking columns: show range
    num_cols = [c for c in df.columns
                if df[c].str.replace(".", "", 1).str.replace("-", "", 1).str.isnumeric().mean() > 0.9]
    if num_cols:
        print("\n  Numeric ranges:")
        for col in num_cols[:8]:   # cap at 8 so output stays readable
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            print(f"    {col:<35} min={vals.min():.2f}  max={vals.max():.2f}  mean={vals.mean():.2f}")

print("\n" + "=" * 60)
print("STEP 2 — Profiling")
print("=" * 60)

profile_table(customers_raw,    "01_customers_raw")
profile_table(orders_raw,       "02_orders_raw")
profile_table(interactions_raw, "03_interactions_raw")


# ─────────────────────────────────────────────────────────────
# STEP 3 — CLEAN & FIX DATA TYPES
# ─────────────────────────────────────────────────────────────
# Rule: fix one table at a time, validate after each, never
# mutate the raw frames — always work on a copy.

print("\n" + "=" * 60)
print("STEP 3 — Cleaning")
print("=" * 60)

# ── 3a. CUSTOMERS ────────────────────────────────────────────

cust = customers_raw.copy()

# Dates
cust["Registration_Date"] = pd.to_datetime(cust["Registration_Date"], errors="coerce")

# Numeric columns
int_cols   = ["Age", "Loyalty_Points_Balance"]
float_cols = ["Annual_Income"]
bool_map   = {"Yes": True, "No": False}

for col in int_cols:
    cust[col] = pd.to_numeric(cust[col], errors="coerce").astype("Int64")

for col in float_cols:
    cust[col] = pd.to_numeric(cust[col], errors="coerce")

for col in ["Subscription_Member", "Marketing_Consent"]:
    cust[col] = cust[col].map(bool_map)

# Satisfaction score: must be 1–10
cust["Satisfaction_Score_1_10"] = pd.to_numeric(
    cust["Satisfaction_Score_1_10"], errors="coerce"
)
invalid_sat = cust["Satisfaction_Score_1_10"].lt(1) | cust["Satisfaction_Score_1_10"].gt(10)
print(f"  Customers: {invalid_sat.sum()} satisfaction scores outside 1–10 → set to NaN")
cust.loc[invalid_sat, "Satisfaction_Score_1_10"] = np.nan

# Categorical columns: strip whitespace, enforce known values
cat_col_values = {
    "Account_Status":    ["Active", "Inactive"],
    "Gender":            ["Male", "Female", "Non-binary", "Prefer not to say"],
    "Acquisition_Channel": [
        "Organic Search", "Paid Social", "Email Campaign",
        "Referral", "Direct", "Affiliate", "TV/Print", "In-Store"
    ],
}
for col, valid in cat_col_values.items():
    cust[col] = cust[col].str.strip()
    bad = ~cust[col].isin(valid) & cust[col].notna()
    if bad.sum():
        print(f"  Customers: {bad.sum()} unknown values in '{col}' → set to NaN")
        cust.loc[bad, col] = np.nan
    cust[col] = pd.Categorical(cust[col], categories=valid)

# Age sanity check (18–100)
age_bad = cust["Age"].lt(18) | cust["Age"].gt(100)
print(f"  Customers: {age_bad.sum()} ages outside 18–100 → set to NaN")
cust.loc[age_bad, "Age"] = pd.NA

# Days since registration
cust["Days_Since_Registration"] = (
    AS_OF_DATE - cust["Registration_Date"]
).dt.days.clip(lower=0)

print(f"  Customers cleaned: {cust.shape[0]:,} rows, "
      f"{cust.isnull().sum().sum()} total nulls remaining")


# ── 3b. ORDERS ───────────────────────────────────────────────

ord_ = orders_raw.copy()

# Dates & timestamps
ord_["Order_Date"]      = pd.to_datetime(ord_["Order_Date"],      errors="coerce")
ord_["Order_Timestamp"] = pd.to_datetime(ord_["Order_Timestamp"], errors="coerce")

# Numeric money/count columns
money_cols = ["Order_Subtotal", "Discount_Amount", "Shipping_Cost", "Tax_Amount", "Order_Total"]
for col in money_cols:
    ord_[col] = pd.to_numeric(ord_[col], errors="coerce").round(2)

ord_["Items_Count"]   = pd.to_numeric(ord_["Items_Count"],   errors="coerce").astype("Int64")
ord_["Discount_Pct"]  = pd.to_numeric(ord_["Discount_Pct"],  errors="coerce")

# Remove negative totals (data corruption)
neg_total = ord_["Order_Total"].lt(0)
print(f"  Orders: {neg_total.sum()} negative Order_Total rows → dropped")
ord_ = ord_[~neg_total].copy()

# Future-dated orders (shouldn't exist)
future_orders = ord_["Order_Date"].gt(AS_OF_DATE)
print(f"  Orders: {future_orders.sum()} future-dated orders → dropped")
ord_ = ord_[~future_orders].copy()

# Orphan orders (no matching customer)
known_customers = set(cust["Customer_ID"])
orphans = ~ord_["Customer_ID"].isin(known_customers)
print(f"  Orders: {orphans.sum()} orphan orders (no matching customer) → dropped")
ord_ = ord_[~orphans].copy()

# Categoricals
for col in ["Order_Status", "Channel", "Payment_Method", "Device_Type", "Currency"]:
    ord_[col] = ord_[col].str.strip().astype("category")

# Hour of day (useful feature later)
ord_["Order_Hour"] = ord_["Order_Timestamp"].dt.hour

print(f"  Orders cleaned: {ord_.shape[0]:,} rows")


# ── 3c. INTERACTIONS ─────────────────────────────────────────

inter = interactions_raw.copy()

inter["Interaction_Date"]      = pd.to_datetime(inter["Interaction_Date"],      errors="coerce")
inter["Interaction_Timestamp"] = pd.to_datetime(inter["Interaction_Timestamp"], errors="coerce")
inter["Duration_Minutes"]      = pd.to_numeric(inter["Duration_Minutes"],       errors="coerce")

# Satisfaction_Rating is mixed (int or empty string) — handle carefully
inter["Satisfaction_Rating"] = pd.to_numeric(
    inter["Satisfaction_Rating"].replace("", np.nan), errors="coerce"
)

# Orphan interactions
orphan_int = ~inter["Customer_ID"].isin(known_customers)
print(f"  Interactions: {orphan_int.sum()} orphan rows → dropped")
inter = inter[~orphan_int].copy()

# Future-dated
future_int = inter["Interaction_Date"].gt(AS_OF_DATE)
print(f"  Interactions: {future_int.sum()} future-dated rows → dropped")
inter = inter[~future_int].copy()

for col in ["Interaction_Type", "Channel", "Outcome"]:
    inter[col] = inter[col].str.strip().astype("category")

print(f"  Interactions cleaned: {inter.shape[0]:,} rows")


# ─────────────────────────────────────────────────────────────
# STEP 4 — JOIN INTO MASTER TABLE (1 row per customer)
# ─────────────────────────────────────────────────────────────
# We compute aggregated order and interaction features here so
# Phase 2 (feature engineering) can start from a single wide
# DataFrame rather than having to re-join every time.

print("\n" + "=" * 60)
print("STEP 4 — Joining to master table")
print("=" * 60)

# ── 4a. Order aggregates per customer ────────────────────────

completed = ord_[ord_["Order_Status"] == "Completed"]

order_agg = (
    ord_
    .groupby("Customer_ID")
    .agg(
        total_orders        = ("Order_ID",    "count"),
        completed_orders    = ("Order_ID",    lambda x: (ord_.loc[x.index, "Order_Status"] == "Completed").sum()),
        total_revenue       = ("Order_Total", "sum"),
        avg_order_value     = ("Order_Total", "mean"),
        max_order_value     = ("Order_Total", "max"),
        min_order_value     = ("Order_Total", "min"),
        first_order_date    = ("Order_Date",  "min"),
        last_order_date     = ("Order_Date",  "max"),
        total_items         = ("Items_Count", "sum"),
        total_discount_amt  = ("Discount_Amount", "sum"),
        avg_discount_pct    = ("Discount_Pct",    "mean"),
        refunded_orders     = ("Order_Status", lambda x: (x == "Refunded").sum()),
        cancelled_orders    = ("Order_Status", lambda x: (x == "Cancelled").sum()),
        preferred_channel   = ("Channel",      lambda x: x.value_counts().index[0]),
        preferred_category  = ("Category",     lambda x: x.value_counts().index[0]),
    )
    .reset_index()
)

# Days since last order (recency)
order_agg["recency_days"] = (
    AS_OF_DATE - order_agg["last_order_date"]
).dt.days.clip(lower=0)

# Customer lifetime in days (first order to last order)
order_agg["customer_lifespan_days"] = (
    order_agg["last_order_date"] - order_agg["first_order_date"]
).dt.days.clip(lower=0)

# Refund/cancel rates
order_agg["refund_rate"]   = order_agg["refunded_orders"]  / order_agg["total_orders"]
order_agg["cancel_rate"]   = order_agg["cancelled_orders"] / order_agg["total_orders"]

# Revenue per item
order_agg["revenue_per_item"] = (
    order_agg["total_revenue"] / order_agg["total_items"].replace(0, np.nan)
)

print(f"  Order aggregates: {order_agg.shape[0]:,} customers with orders")


# ── 4b. Interaction aggregates per customer ───────────────────

interaction_agg = (
    inter
    .groupby("Customer_ID")
    .agg(
        total_interactions     = ("Interaction_ID",   "count"),
        support_tickets        = ("Interaction_Type", lambda x: (x == "support_ticket").sum()),
        email_opens            = ("Interaction_Type", lambda x: (x == "email_open").sum()),
        email_clicks           = ("Interaction_Type", lambda x: (x == "email_click").sum()),
        web_visits             = ("Interaction_Type", lambda x: (x == "web_visit").sum()),
        app_opens              = ("Interaction_Type", lambda x: (x == "app_open").sum()),
        avg_duration_mins      = ("Duration_Minutes", "mean"),
        avg_int_satisfaction   = ("Satisfaction_Rating", "mean"),
        last_interaction_date  = ("Interaction_Date", "max"),
        preferred_int_channel  = ("Channel",           lambda x: x.value_counts().index[0]),
    )
    .reset_index()
)

# Days since last interaction
interaction_agg["days_since_last_interaction"] = (
    AS_OF_DATE - interaction_agg["last_interaction_date"]
).dt.days.clip(lower=0)

# Email engagement rate: clicks / opens (avoid division by zero)
interaction_agg["email_ctr"] = (
    interaction_agg["email_clicks"]
    / interaction_agg["email_opens"].replace(0, np.nan)
).fillna(0).round(4)

print(f"  Interaction aggregates: {interaction_agg.shape[0]:,} customers with interactions")


# ── 4c. Merge everything onto the customer base ───────────────

master_df = (
    cust
    .merge(order_agg,       on="Customer_ID", how="left")
    .merge(interaction_agg, on="Customer_ID", how="left")
)

# Customers with zero orders get 0 / NaT defaults
zero_order_cols = [
    "total_orders", "completed_orders", "refunded_orders", "cancelled_orders",
    "total_items", "total_interactions", "support_tickets",
    "email_opens", "email_clicks", "web_visits", "app_opens",
]
for col in zero_order_cols:
    master_df[col] = master_df[col].fillna(0).astype(int)

master_df["total_revenue"]    = master_df["total_revenue"].fillna(0)
master_df["recency_days"]     = master_df["recency_days"].fillna(master_df["Days_Since_Registration"])
master_df["refund_rate"]      = master_df["refund_rate"].fillna(0)
master_df["cancel_rate"]      = master_df["cancel_rate"].fillna(0)
master_df["email_ctr"]        = master_df["email_ctr"].fillna(0)


# ─────────────────────────────────────────────────────────────
# STEP 5 — FINAL VALIDATION
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 5 — Final validation")
print("=" * 60)

# Shape
print(f"\n  master_df shape  : {master_df.shape[0]:,} rows × {master_df.shape[1]} cols")

# Duplicate Customer IDs
dup_ids = master_df["Customer_ID"].duplicated().sum()
print(f"  Duplicate IDs    : {dup_ids}  {'OK' if dup_ids == 0 else 'WARNING!'}")

# Null rates in key columns
key_cols = [
    "Customer_ID", "Registration_Date", "Account_Status",
    "total_orders", "total_revenue", "recency_days",
    "total_interactions", "Satisfaction_Score_1_10",
]
print("\n  Null rates in key columns:")
for col in key_cols:
    null_pct = master_df[col].isnull().mean() * 100
    flag = "  <- check" if null_pct > 10 else ""
    print(f"    {col:<35} {null_pct:5.1f}%{flag}")

# Sanity: revenue must be >= 0
neg_rev = (master_df["total_revenue"] < 0).sum()
print(f"\n  Negative revenue rows : {neg_rev}  {'OK' if neg_rev == 0 else 'WARNING!'}")

# Segment distribution (ground truth join check)
master_with_gt = master_df.merge(ground_truth, on="Customer_ID", how="left")
seg_dist = master_with_gt["Segment"].value_counts()
print("\n  Ground-truth segment distribution:")
for seg, n in seg_dist.items():
    pct = n / len(master_df) * 100
    bar = "█" * int(pct / 2)
    print(f"    {seg:<12} {n:>4}  ({pct:4.1f}%)  {bar}")

# Quick summary statistics on key numeric columns
print("\n  Key column statistics:")
stat_cols = ["total_orders", "total_revenue", "avg_order_value",
             "recency_days", "total_interactions"]
print(
    master_df[stat_cols]
    .describe()
    .loc[["mean", "std", "min", "50%", "max"]]
    .round(1)
    .to_string()
)


# ─────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────

master_df.to_csv("master_df.csv", index=False)
print(f"\n  Saved → master_df.csv  ({master_df.shape[0]:,} rows × {master_df.shape[1]} cols)")
print("\nPhase 1 complete. Run phase2_feature_engineering.py next.")
