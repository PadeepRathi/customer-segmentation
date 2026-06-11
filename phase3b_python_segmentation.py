"""
Phase 3b — Python-based Segmentation with pandas
Customer Segmentation Tutorial

Input  : features_df.csv  (output of Phase 2)
Output : segments_python.csv — every customer with a predicted segment + business metrics

What this phase does:
    - Applies rule-based segmentation using RFM scores + behavioural flags
    - Profiles each segment with business metrics
    - Produces actionable recommendations per segment
    - Compares two segmentation strategies (pure RFM vs enriched RFM)
"""

import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = r"D:\data use for learning"

# ─────────────────────────────────────────────────────────────
# STEP 1 — LOAD FEATURES
# ─────────────────────────────────────────────────────────────

print("=" * 60)
print("STEP 1 — Loading features_df")
print("=" * 60)

feat = pd.read_csv(os.path.join(DATA_DIR, "features_df.csv"))

print(f"  Loaded : {feat.shape[0]:,} rows × {feat.shape[1]} cols")

# We only need a focused set of columns for segmentation rules
working = feat[[
    "Customer_ID",
    # RFM raw
    "R_days", "F_orders", "M_total_revenue", "M_aov",
    # RFM scores (1–5 quintiles from Phase 2)
    "R_score", "F_score", "M_score", "RFM_sum", "RFM_weighted",
    # Behavioural
    "B_orders_per_month", "B_churn_risk_flag",
    "B_refund_rate", "B_cancel_rate",
    "B_avg_discount_pct", "B_tickets_per_order",
    "B_email_ctr", "B_interaction_rate",
    # Profile
    "P_tenure_days", "P_satisfaction", "P_loyalty_points",
    "P_is_subscriber",
]].copy()

print(f"  Working columns selected: {working.shape[1]}")


# ─────────────────────────────────────────────────────────────
# STEP 2 — STRATEGY A: PURE RFM SEGMENTATION
# ─────────────────────────────────────────────────────────────
# The simplest professional approach:
# Assign segments using only R, F, M quintile scores.
#
# Score logic (each score is 1–5):
#   Premium   → R≥4  AND F≥4  AND M≥4   (top buyers, recent)
#   Loyal     → F≥4  AND M≥3            (frequent, decent spend)
#   At-Risk   → R≤2  AND (F≥3 OR M≥3)  (used to buy, gone quiet)
#   Churned   → R=1                     (haven't bought in ages)
#   New       → F≤1  AND R≥4            (just arrived)
#   Budget    → M≤2                     (low spenders)
#   Standard  → everyone else
#
# ORDER MATTERS: rules are evaluated top to bottom.
# Once a customer matches a rule they get that label and stop.

print("\n" + "=" * 60)
print("STEP 2 — Strategy A: Pure RFM segmentation")
print("=" * 60)

def rfm_segment_pure(row: pd.Series) -> str:
    """
    Assign a segment label using only RFM quintile scores.
    Scores are 1 (lowest) to 5 (highest).
    """
    r = row["R_score"]
    f = row["F_score"]
    m = row["M_score"]

    if r >= 4 and f >= 4 and m >= 4:
        return "Premium"
    elif r <= 1:
        return "Churned"
    elif r <= 2 and (f >= 3 or m >= 3):
        return "At-Risk"
    elif f <= 1 and r >= 4:
        return "New"
    elif f >= 4 and m >= 3:
        return "Loyal"
    elif m <= 2:
        return "Budget"
    else:
        return "Standard"

working["segment_rfm"] = working.apply(rfm_segment_pure, axis=1)

# Distribution
print("\n  Segment A (pure RFM) distribution:")
dist_a = working["segment_rfm"].value_counts()
for seg, n in dist_a.items():
    pct = n / len(working) * 100
    bar = "█" * int(pct / 2)
    print(f"    {seg:<12} {n:>4}  ({pct:4.1f}%)  {bar}")


# ─────────────────────────────────────────────────────────────
# STEP 3 — STRATEGY B: ENRICHED RFM SEGMENTATION
# ─────────────────────────────────────────────────────────────
# We add behavioural signals to the RFM rules to make them
# more precise.  Real businesses use this approach because
# pure RFM misses important nuances:
#
#   - A "Premium" RFM score with 100% refund rate is not
#     a true Premium customer.
#   - A "Standard" customer with high churn risk needs
#     different treatment than a stable Standard customer.
#   - A subscriber who opens every email is more valuable
#     than the same RFM score without that engagement.

print("\n" + "=" * 60)
print("STEP 3 — Strategy B: Enriched RFM segmentation")
print("=" * 60)

def rfm_segment_enriched(row: pd.Series) -> str:
    """
    Assign a segment using RFM scores + behavioural flags.
    Additional signals used:
        churn_risk  : recency ≥90 days AND low interaction rate
        high_refund : refund rate > 30%
        discount_dep: avg discount > 10% (price-sensitive)
        engaged     : email_ctr > 0.5 OR interaction_rate > median
    """
    r = row["R_score"]
    f = row["F_score"]
    m = row["M_score"]

    churn_risk   = row["B_churn_risk_flag"] == 1
    high_refund  = row["B_refund_rate"]      > 0.30
    discount_dep = row["B_avg_discount_pct"] > 10
    engaged      = row["B_email_ctr"]        > 0.5 or row["B_interaction_rate"] > 0.20
    subscriber   = row["P_is_subscriber"]    == 1

    # ── Tier 1: Champions (best of the best) ──────────────────
    # Recent, frequent, high spend, engaged, no refund issues
    if r >= 4 and f >= 4 and m >= 4 and not high_refund and engaged:
        return "Champion"

    # ── Tier 2: Premium (high value, slightly less engaged) ───
    if r >= 4 and f >= 4 and m >= 4:
        return "Premium"

    # ── Tier 3: Loyal subscribers ─────────────────────────────
    # Frequent buyers who are subscribed (sticky relationship)
    if f >= 4 and m >= 3 and subscriber and not churn_risk:
        return "Loyal"

    # ── Tier 4: Churned (gone silent) ─────────────────────────
    # Must check before At-Risk — churned is the worse state
    if r <= 1:
        return "Churned"

    # ── Tier 5: At-Risk (slipping away) ───────────────────────
    if churn_risk or (r <= 2 and (f >= 3 or m >= 3)):
        return "At-Risk"

    # ── Tier 6: New customers ─────────────────────────────────
    if f <= 1 and r >= 4:
        return "New"

    # ── Tier 7: Budget / price-sensitive ──────────────────────
    # Low spend OR heavily discount-dependent
    if m <= 2 or discount_dep:
        return "Budget"

    # ── Tier 8: Standard (solid middle) ───────────────────────
    return "Standard"

working["segment_enriched"] = working.apply(rfm_segment_enriched, axis=1)

# Distribution
print("\n  Segment B (enriched RFM) distribution:")
dist_b = working["segment_enriched"].value_counts()
for seg, n in dist_b.items():
    pct = n / len(working) * 100
    bar = "█" * int(pct / 2)
    print(f"    {seg:<12} {n:>4}  ({pct:4.1f}%)  {bar}")

# How many customers moved between strategies?
changed = (working["segment_rfm"] != working["segment_enriched"]).sum()
print(f"\n  Customers re-segmented by enriched rules: {changed} "
      f"({changed/len(working)*100:.1f}%)")


# ─────────────────────────────────────────────────────────────
# STEP 4 — SEGMENT PROFILING
# ─────────────────────────────────────────────────────────────
# For each segment, compute business metrics.
# This is what you'd present to a marketing or CRM team —
# not scores, but revenue, behaviour, and engagement numbers.

print("\n" + "=" * 60)
print("STEP 4 — Segment profiles (enriched strategy)")
print("=" * 60)

profile_metrics = {
    "Customers"       : ("Customer_ID",        "count"),
    "Avg Recency(d)"  : ("R_days",             "mean"),
    "Avg Orders"      : ("F_orders",           "mean"),
    "Avg Revenue($)"  : ("M_total_revenue",    "mean"),
    "Avg AOV($)"      : ("M_aov",              "mean"),
    "Avg Satisfaction": ("P_satisfaction",     "mean"),
    "Churn Flag%"     : ("B_churn_risk_flag",  "mean"),
    "Avg Discount%"   : ("B_avg_discount_pct", "mean"),
    "Refund Rate"     : ("B_refund_rate",       "mean"),
    "Subscriber%"     : ("P_is_subscriber",    "mean"),
}

agg_dict = {display: pd.NamedAgg(column=col, aggfunc=fn)
            for display, (col, fn) in profile_metrics.items()}

segment_profile = (
    working
    .groupby("segment_enriched")
    .agg(**agg_dict)
    .round(2)
)

# Add % of total customers
segment_profile.insert(
    1, "% of Total",
    (segment_profile["Customers"] / len(working) * 100).round(1)
)

# Add revenue contribution %
segment_profile["Revenue %"] = (
    working.groupby("segment_enriched")["M_total_revenue"].sum()
    / working["M_total_revenue"].sum() * 100
).round(1)

# Sort by avg revenue descending
segment_profile = segment_profile.sort_values("Avg Revenue($)", ascending=False)

# Convert rate columns to % for readability
segment_profile["Churn Flag%"]  = (segment_profile["Churn Flag%"]  * 100).round(1)
segment_profile["Refund Rate"]  = (segment_profile["Refund Rate"]  * 100).round(1)
segment_profile["Subscriber%"]  = (segment_profile["Subscriber%"]  * 100).round(1)

print("\n", segment_profile.to_string())


# ─────────────────────────────────────────────────────────────
# STEP 5 — REVENUE CONCENTRATION (80/20 RULE CHECK)
# ─────────────────────────────────────────────────────────────
# In most businesses, ~20% of customers generate ~80% of revenue.
# This check tells you how concentrated your revenue is.

print("\n" + "=" * 60)
print("STEP 5 — Revenue concentration analysis")
print("=" * 60)

revenue_by_seg = (
    working.groupby("segment_enriched")["M_total_revenue"]
    .agg(["sum", "count"])
    .assign(revenue_pct = lambda x: x["sum"] / x["sum"].sum() * 100,
            customer_pct= lambda x: x["count"] / x["count"].sum() * 100)
    .sort_values("sum", ascending=False)
)

print("\n  Revenue vs Customer share per segment:")
print(f"  {'Segment':<14} {'Rev %':>7}  {'Cust %':>7}  {'Rev/Cust ratio':>15}")
print(f"  {'-'*50}")
for seg, row in revenue_by_seg.iterrows():
    ratio = row["revenue_pct"] / row["customer_pct"]
    bar   = "▲" if ratio > 1 else "▼"
    print(f"  {seg:<14} {row['revenue_pct']:>6.1f}%  "
          f"{row['customer_pct']:>6.1f}%  "
          f"{ratio:>10.2f}x  {bar}")

# Cumulative: which segments make up 80% of revenue?
cumrev = revenue_by_seg["revenue_pct"].cumsum()
segs_for_80 = (cumrev < 80).sum() + 1
print(f"\n  Top {segs_for_80} segment(s) account for 80% of revenue")


# ─────────────────────────────────────────────────────────────
# STEP 6 — SEGMENT TRANSITION COMPARISON
# ─────────────────────────────────────────────────────────────
# Show which customers moved between Strategy A and B.
# This reveals where the behavioural signals changed the call.

print("\n" + "=" * 60)
print("STEP 6 — Strategy A vs B: where did labels change?")
print("=" * 60)

moves = (
    working[working["segment_rfm"] != working["segment_enriched"]]
    .groupby(["segment_rfm", "segment_enriched"])
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)

if len(moves):
    print("\n  Pure RFM → Enriched RFM  (top movements):")
    print(f"  {'From':<14} {'To':<14} {'Count':>6}")
    print(f"  {'-'*36}")
    for _, row in moves.head(10).iterrows():
        print(f"  {row['segment_rfm']:<14} → {row['segment_enriched']:<14} {row['count']:>6}")
else:
    print("  No differences between strategies.")


# ─────────────────────────────────────────────────────────────
# STEP 7 — BUSINESS ACTION RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────
# Translate segments into concrete CRM actions.
# This is the deliverable a marketing team actually uses.

print("\n" + "=" * 60)
print("STEP 7 — Business action recommendations")
print("=" * 60)

ACTIONS = {
    "Champion": {
        "priority" : "HIGH",
        "goal"     : "Retain & leverage advocacy",
        "actions"  : [
            "Exclusive early access to new products",
            "VIP loyalty tier upgrade",
            "Referral programme invitation",
            "Personal account manager (if B2B)",
        ],
        "avoid"    : "Do not over-discount — they buy at full price",
    },
    "Premium": {
        "priority" : "HIGH",
        "goal"     : "Retain & upsell",
        "actions"  : [
            "Personalised product recommendations",
            "Loyalty reward acceleration (double points)",
            "Invite to Premium membership/subscription",
        ],
        "avoid"    : "Avoid generic mass emails — they notice",
    },
    "Loyal": {
        "priority" : "MEDIUM-HIGH",
        "goal"     : "Deepen relationship",
        "actions"  : [
            "Subscription renewal reminders with incentive",
            "Cross-sell adjacent categories",
            "Birthday / anniversary offers",
        ],
        "avoid"    : "Don't take them for granted — they can churn quietly",
    },
    "Standard": {
        "priority" : "MEDIUM",
        "goal"     : "Increase order frequency",
        "actions"  : [
            "Triggered email after 30 days of inactivity",
            "Bundle deals to increase AOV",
            "Subscription trial offer",
        ],
        "avoid"    : "Avoid heavy discounting — trains price sensitivity",
    },
    "New": {
        "priority" : "MEDIUM-HIGH",
        "goal"     : "Convert to repeat buyer fast",
        "actions"  : [
            "Welcome series email (3–5 emails over 2 weeks)",
            "Second-purchase incentive within 14 days",
            "Onboarding guide / best-sellers showcase",
        ],
        "avoid"    : "Don't overwhelm — one clear CTA per email",
    },
    "Budget": {
        "priority" : "LOW-MEDIUM",
        "goal"     : "Increase AOV or accept low margin",
        "actions"  : [
            "Free shipping threshold nudge ('Add $X for free shipping')",
            "Low-cost upsells at checkout",
            "Value-focused messaging (not luxury)",
        ],
        "avoid"    : "Don't push premium products — wrong fit",
    },
    "At-Risk": {
        "priority" : "HIGH",
        "goal"     : "Win back before full churn",
        "actions"  : [
            "Win-back email: 'We miss you' + time-limited offer",
            "Survey: why did you stop buying?",
            "Personalised reminder of past purchases",
        ],
        "avoid"    : "Don't offer huge discounts first — try engagement first",
    },
    "Churned": {
        "priority" : "LOW",
        "goal"     : "Low-cost re-activation attempt",
        "actions"  : [
            "One re-activation email with strong offer",
            "Suppress from regular campaigns (reduces spam rate)",
            "If no response in 90 days: archive",
        ],
        "avoid"    : "Don't invest heavily — most won't return",
    },
}

for seg, info in ACTIONS.items():
    count = (working["segment_enriched"] == seg).sum()
    if count == 0:
        continue
    pct   = count / len(working) * 100
    print(f"\n  ┌─ {seg.upper()} ({count} customers, {pct:.1f}%)")
    print(f"  │  Priority : {info['priority']}")
    print(f"  │  Goal     : {info['goal']}")
    for i, action in enumerate(info["actions"], 1):
        print(f"  │  Action {i} : {action}")
    print(f"  └─ Avoid   : {info['avoid']}")


# ─────────────────────────────────────────────────────────────
# STEP 8 — SAVE
# ─────────────────────────────────────────────────────────────

# Final output: Customer_ID + both segment labels + key metrics
output_cols = [
    "Customer_ID",
    "segment_rfm", "segment_enriched",
    "R_score", "F_score", "M_score", "RFM_sum",
    "R_days", "F_orders", "M_total_revenue", "M_aov",
    "B_churn_risk_flag", "B_refund_rate",
    "P_satisfaction", "P_is_subscriber",
]
result = working[output_cols].copy()
result.to_csv(os.path.join(DATA_DIR, "segments_python.csv"), index=False)

print("\n" + "=" * 60)
print(f"  Saved → segments_python.csv  ({result.shape[0]:,} rows)")
print("  Phase 3b complete. Run phase4_clustering.py next.")
print("=" * 60)
