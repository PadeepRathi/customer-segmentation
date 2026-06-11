"""
Phase 4 — K-Means Clustering
Customer Segmentation Tutorial

Input  : features_df.csv   (output of Phase 2)
         segments_python.csv (output of Phase 3b — for comparison)
Output : clusters_kmeans.csv — every customer with ML cluster label

What this phase does:
    1. Prepare the scaled feature matrix
    2. Elbow method — find the optimal number of clusters (K)
    3. Silhouette analysis — confirm the best K
    4. Fit K-Means with the chosen K
    5. PCA — reduce to 2D and visualise clusters
    6. Profile each cluster with business metrics
    7. Name clusters based on their characteristics
    8. Compare ML clusters vs rule-based segments (Phase 3b)
    9. Save results

Key concept — WHY K-Means after rule-based segmentation?
    Rule-based (Phase 3b) = you decide the rules upfront.
      Pro: explainable, easy to present to business.
      Con: your rules might miss natural groupings in the data.
    K-Means = the algorithm finds groupings with no rules.
      Pro: discovers patterns you didn't know existed.
      Con: clusters need interpretation — they have no names.
    Best practice: do both and compare. They should broadly agree.
    Where they disagree is where the most interesting insights are.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend for Colab/script
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
import warnings
warnings.filterwarnings("ignore")

DATA_DIR   = r"D:\data use for learning"
RANDOM_STATE = 42

# ─────────────────────────────────────────────────────────────
# STEP 1 — LOAD & PREPARE FEATURE MATRIX
# ─────────────────────────────────────────────────────────────

print("=" * 60)
print("STEP 1 — Loading feature matrix")
print("=" * 60)

feat = pd.read_csv(os.path.join(DATA_DIR, "features_df.csv"))
seg3 = pd.read_csv(os.path.join(DATA_DIR, "segments_python.csv"))

# The 18 scaled columns — these are what K-Means will cluster on.
# We use scaled (RobustScaler) log-transformed values from Phase 2
# so no single feature dominates due to scale differences.
SCALED_COLS = [c for c in feat.columns if c.startswith("scaled_")]

X = feat[SCALED_COLS].values   # numpy array, shape (1000, 18)

print(f"  Feature matrix : {X.shape[0]:,} customers × {X.shape[1]} features")
print(f"  Features used  :")
for i, col in enumerate(SCALED_COLS, 1):
    print(f"    {i:>2}. {col.replace('scaled_', '')}")


# ─────────────────────────────────────────────────────────────
# STEP 2 — ELBOW METHOD  (find optimal K)
# ─────────────────────────────────────────────────────────────
# K-Means requires you to specify K (number of clusters) upfront.
# The elbow method runs K-Means for K=2..12 and plots the
# inertia (sum of squared distances from each point to its
# cluster centre) for each K.
#
# As K increases, inertia always falls — more clusters = tighter
# fit. The "elbow" is where adding one more cluster stops giving
# a meaningful improvement. That's your optimal K.

print("\n" + "=" * 60)
print("STEP 2 — Elbow method")
print("=" * 60)

K_RANGE   = range(2, 13)
inertias  = []
sil_scores = []

print("  Fitting K-Means for K = 2 to 12 ...")
for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    km.fit(X)
    inertias.append(km.inertia_)
    sil = silhouette_score(X, km.labels_, sample_size=500,
                           random_state=RANDOM_STATE)
    sil_scores.append(sil)
    print(f"    K={k:>2}  inertia={km.inertia_:>10,.1f}  silhouette={sil:.4f}")

# Find elbow: largest drop in inertia reduction rate
inertia_drops = [inertias[i-1] - inertias[i] for i in range(1, len(inertias))]
drop_ratios   = [inertia_drops[i] / inertia_drops[i-1]
                 for i in range(1, len(inertia_drops))]
elbow_k = list(K_RANGE)[1 + drop_ratios.index(min(drop_ratios))]

# Best silhouette K
best_sil_k = list(K_RANGE)[sil_scores.index(max(sil_scores))]

print(f"\n  Elbow method suggests   K = {elbow_k}")
print(f"  Best silhouette score   K = {best_sil_k}  "
      f"(score={max(sil_scores):.4f})")

# Plot elbow curve
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(list(K_RANGE), inertias, "bo-", linewidth=2, markersize=6)
ax1.axvline(x=elbow_k, color="red", linestyle="--", alpha=0.7,
            label=f"Elbow K={elbow_k}")
ax1.set_xlabel("Number of Clusters (K)")
ax1.set_ylabel("Inertia (Within-cluster Sum of Squares)")
ax1.set_title("Elbow Method — Optimal K")
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(list(K_RANGE), sil_scores, "go-", linewidth=2, markersize=6)
ax2.axvline(x=best_sil_k, color="red", linestyle="--", alpha=0.7,
            label=f"Best K={best_sil_k}")
ax2.set_xlabel("Number of Clusters (K)")
ax2.set_ylabel("Silhouette Score")
ax2.set_title("Silhouette Score — Cluster Separation Quality")
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, "elbow_silhouette.png"), dpi=150,
            bbox_inches="tight")
plt.close()
print(f"\n  Saved → elbow_silhouette.png")


# ─────────────────────────────────────────────────────────────
# STEP 3 — FIT FINAL K-MEANS MODEL
# ─────────────────────────────────────────────────────────────
# We use the elbow K. If elbow and silhouette disagree, prefer
# the one closer to the number of business segments you expect.
# Here we have 6–8 natural business segments, so we pick
# whichever K is more meaningful in that range.

print("\n" + "=" * 60)
print("STEP 3 — Fitting final K-Means model")
print("=" * 60)

# Choose K — use elbow, but cap between 5 and 9 for business sense
CHOSEN_K = max(5, min(9, elbow_k))
print(f"  Chosen K = {CHOSEN_K}  (elbow={elbow_k}, sil_best={best_sil_k})")

# n_init=20: run 20 times with different starting points, keep best
# max_iter=500: allow enough iterations to converge
kmeans = KMeans(
    n_clusters   = CHOSEN_K,
    random_state = RANDOM_STATE,
    n_init       = 20,
    max_iter     = 500,
)
kmeans.fit(X)

feat["cluster"] = kmeans.labels_
labels          = kmeans.labels_

final_sil = silhouette_score(X, labels)
print(f"  Final silhouette score : {final_sil:.4f}")
print(f"  Inertia                : {kmeans.inertia_:,.1f}")

# Silhouette score interpretation:
#   > 0.5  = strong structure, well-separated clusters
#   0.2–0.5 = reasonable structure (typical for real customer data)
#   < 0.2  = weak structure, clusters overlap significantly
if final_sil > 0.5:
    interpretation = "Strong — clusters are well separated"
elif final_sil > 0.3:
    interpretation = "Reasonable — typical for real customer data"
elif final_sil > 0.2:
    interpretation = "Weak but acceptable — some cluster overlap"
else:
    interpretation = "Poor — consider fewer clusters or different features"
print(f"  Interpretation         : {interpretation}")

# Cluster sizes
print("\n  Cluster size distribution:")
for c, n in sorted(feat["cluster"].value_counts().items()):
    pct = n / len(feat) * 100
    bar = "█" * int(pct / 2)
    print(f"    Cluster {c}  {n:>4} customers  ({pct:4.1f}%)  {bar}")


# ─────────────────────────────────────────────────────────────
# STEP 4 — PCA VISUALISATION
# ─────────────────────────────────────────────────────────────
# 18 features can't be visualised directly.
# PCA (Principal Component Analysis) compresses 18 dimensions
# into 2 while preserving as much variance as possible.
# We plot those 2 dimensions so we can SEE the clusters.
#
# Important: PCA is ONLY for visualisation here.
# The actual clustering was done on all 18 features.

print("\n" + "=" * 60)
print("STEP 4 — PCA visualisation")
print("=" * 60)

pca       = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca     = pca.fit_transform(X)
var_ratio = pca.explained_variance_ratio_

print(f"  PC1 explains : {var_ratio[0]*100:.1f}% of variance")
print(f"  PC2 explains : {var_ratio[1]*100:.1f}% of variance")
print(f"  Total        : {sum(var_ratio)*100:.1f}% of variance captured in 2D")

# What do PC1 and PC2 represent?
# Show the top features driving each component
components_df = pd.DataFrame(
    pca.components_.T,
    index=[c.replace("scaled_", "") for c in SCALED_COLS],
    columns=["PC1", "PC2"]
)
print("\n  Top features driving PC1 (the main axis of variation):")
pc1_top = components_df["PC1"].abs().sort_values(ascending=False).head(5)
for feat_name, loading in pc1_top.items():
    direction = "+" if components_df.loc[feat_name, "PC1"] > 0 else "-"
    print(f"    {direction}{feat_name:<35} loading={abs(loading):.3f}")

print("\n  Top features driving PC2:")
pc2_top = components_df["PC2"].abs().sort_values(ascending=False).head(5)
for feat_name, loading in pc2_top.items():
    direction = "+" if components_df.loc[feat_name, "PC2"] > 0 else "-"
    print(f"    {direction}{feat_name:<35} loading={abs(loading):.3f}")

# Plot clusters in 2D PCA space
colors = cm.tab10(np.linspace(0, 0.9, CHOSEN_K))
fig, ax = plt.subplots(figsize=(10, 7))

for cluster_id in range(CHOSEN_K):
    mask = labels == cluster_id
    ax.scatter(
        X_pca[mask, 0], X_pca[mask, 1],
        c=[colors[cluster_id]], label=f"Cluster {cluster_id}",
        alpha=0.6, s=40, edgecolors="none"
    )

# Plot cluster centres in PCA space
centres_pca = pca.transform(kmeans.cluster_centers_)
ax.scatter(
    centres_pca[:, 0], centres_pca[:, 1],
    c="black", marker="X", s=200, zorder=5, label="Centroids"
)
for i, (cx, cy) in enumerate(centres_pca):
    ax.annotate(f"  C{i}", (cx, cy), fontsize=9, fontweight="bold")

ax.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}% variance)")
ax.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}% variance)")
ax.set_title(f"K-Means Clusters in PCA Space  (K={CHOSEN_K})")
ax.legend(loc="upper right", fontsize=8)
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, "pca_clusters.png"), dpi=150,
            bbox_inches="tight")
plt.close()
print(f"\n  Saved → pca_clusters.png")


# ─────────────────────────────────────────────────────────────
# STEP 5 — CLUSTER PROFILING
# ─────────────────────────────────────────────────────────────
# Clusters have no names — just numbers. We profile each one
# using raw business metrics to understand what it represents.

print("\n" + "=" * 60)
print("STEP 5 — Cluster profiling")
print("=" * 60)

# Merge cluster labels onto feature data
profile_data = feat[[
    "Customer_ID", "cluster",
    "R_days", "F_orders", "M_total_revenue", "M_aov",
    "B_orders_per_month", "B_churn_risk_flag",
    "B_refund_rate", "B_avg_discount_pct",
    "B_email_ctr", "B_interaction_rate",
    "P_tenure_days", "P_satisfaction",
    "P_loyalty_points", "P_is_subscriber",
]].copy()

cluster_profile = (
    profile_data.groupby("cluster")
    .agg(
        customers        = ("Customer_ID",       "count"),
        avg_recency_days = ("R_days",            "mean"),
        avg_orders       = ("F_orders",          "mean"),
        avg_revenue      = ("M_total_revenue",   "mean"),
        avg_aov          = ("M_aov",             "mean"),
        avg_orders_month = ("B_orders_per_month","mean"),
        churn_flag_pct   = ("B_churn_risk_flag", "mean"),
        avg_refund_rate  = ("B_refund_rate",     "mean"),
        avg_discount_pct = ("B_avg_discount_pct","mean"),
        avg_email_ctr    = ("B_email_ctr",       "mean"),
        avg_satisfaction = ("P_satisfaction",    "mean"),
        subscriber_pct   = ("P_is_subscriber",   "mean"),
    )
    .round(2)
)

# Revenue share
total_rev = profile_data["M_total_revenue"].sum()
cluster_profile["rev_pct"] = (
    profile_data.groupby("cluster")["M_total_revenue"].sum()
    / total_rev * 100
).round(1)

# Revenue per customer ratio (vs average)
cluster_profile["rev_ratio"] = (
    cluster_profile["rev_pct"] /
    (cluster_profile["customers"] / len(profile_data) * 100)
).round(2)

# Format % columns
cluster_profile["churn_flag_pct"] = (cluster_profile["churn_flag_pct"] * 100).round(1)
cluster_profile["subscriber_pct"] = (cluster_profile["subscriber_pct"] * 100).round(1)
cluster_profile["avg_refund_rate"] = (cluster_profile["avg_refund_rate"] * 100).round(1)

cluster_profile = cluster_profile.sort_values("avg_revenue", ascending=False)

print("\n", cluster_profile.to_string())


# ─────────────────────────────────────────────────────────────
# STEP 6 — NAME THE CLUSTERS
# ─────────────────────────────────────────────────────────────
# Based on the profile, assign a business-friendly name to each
# cluster. This is a human judgement call — you look at the
# metrics and decide what each cluster represents.
#
# Naming logic (applied in priority order):
#   High revenue + low recency + high frequency = VIP/Champion
#   High revenue + moderate recency             = Loyal/Premium
#   High recency (inactive) + some history      = At-Risk/Churned
#   Low frequency + low recency                 = New
#   Low revenue across the board                = Budget

print("\n" + "=" * 60)
print("STEP 6 — Naming clusters")
print("=" * 60)

def name_cluster(row: pd.Series) -> str:
    """
    Assign a business name to a cluster based on its profile.
    Uses the cluster_profile row (indexed by cluster id).
    """
    rev     = row["avg_revenue"]
    recency = row["avg_recency_days"]
    orders  = row["avg_orders"]
    aov     = row["avg_aov"]
    churn   = row["churn_flag_pct"]
    sub     = row["subscriber_pct"]

    # Rank thresholds based on overall data
    rev_p75    = cluster_profile["avg_revenue"].quantile(0.75)
    rev_p25    = cluster_profile["avg_revenue"].quantile(0.25)
    rec_p75    = cluster_profile["avg_recency_days"].quantile(0.75)

    if rev >= rev_p75 and recency <= 30 and orders >= 30:
        return "VIP / Champion"
    elif rev >= rev_p75 and recency <= 60:
        return "Loyal / Premium"
    elif rev >= rev_p75 and recency > 60:
        return "High-Value At-Risk"
    elif churn >= 30 or recency > 150:
        return "Churned / Dormant"
    elif orders <= 3:
        return "New / One-Time"
    elif rev <= rev_p25 and aov <= 80:
        return "Budget / Price-Sensitive"
    else:
        return "Standard / Mid-Tier"

cluster_names = cluster_profile.apply(name_cluster, axis=1)
cluster_profile["cluster_name"] = cluster_names

# Build cluster_id → name mapping
cluster_map = cluster_names.to_dict()

print("\n  Cluster name assignments:")
for cluster_id, name in sorted(cluster_map.items()):
    n    = cluster_profile.loc[cluster_id, "customers"]
    rev  = cluster_profile.loc[cluster_id, "avg_revenue"]
    pct  = cluster_profile.loc[cluster_id, "rev_pct"]
    print(f"    Cluster {cluster_id} → {name:<25} "
          f"({n} customers, avg ${rev:,.0f}, {pct}% of revenue)")

feat["cluster_name"] = feat["cluster"].map(cluster_map)


# ─────────────────────────────────────────────────────────────
# STEP 7 — SILHOUETTE PLOT  (cluster quality per sample)
# ─────────────────────────────────────────────────────────────
# The overall silhouette score tells you the average quality.
# This plot shows the score for EVERY customer — which clusters
# are tight and which are fuzzy.
#
# A good cluster has most samples above the average score line.
# A bad cluster has many samples below 0 (misclassified).

print("\n" + "=" * 60)
print("STEP 7 — Per-sample silhouette plot")
print("=" * 60)

sil_samples = silhouette_samples(X, labels)

fig, ax = plt.subplots(figsize=(10, 6))
y_lower = 10

for cluster_id in range(CHOSEN_K):
    cluster_sil = np.sort(sil_samples[labels == cluster_id])
    size        = cluster_sil.shape[0]
    y_upper     = y_lower + size

    color = cm.tab10(cluster_id / CHOSEN_K)
    ax.fill_betweenx(
        np.arange(y_lower, y_upper),
        0, cluster_sil,
        facecolor=color, edgecolor=color, alpha=0.7
    )
    ax.text(-0.05, y_lower + 0.5 * size,
            f"C{cluster_id}\n({cluster_map.get(cluster_id,'?')[:10]})",
            fontsize=7)
    y_lower = y_upper + 10

ax.axvline(x=final_sil, color="red", linestyle="--",
           label=f"Avg silhouette = {final_sil:.3f}")
ax.set_xlabel("Silhouette Coefficient")
ax.set_ylabel("Cluster")
ax.set_title(f"Silhouette Plot — K={CHOSEN_K}")
ax.legend(loc="upper right")
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, "silhouette_plot.png"), dpi=150,
            bbox_inches="tight")
plt.close()
print(f"  Saved → silhouette_plot.png")
print(f"  Average silhouette : {final_sil:.4f}")

# Per-cluster silhouette
print("\n  Per-cluster silhouette scores:")
for cluster_id in range(CHOSEN_K):
    cluster_sil_mean = sil_samples[labels == cluster_id].mean()
    name = cluster_map.get(cluster_id, "?")
    flag = "  ← weak" if cluster_sil_mean < 0.2 else ""
    print(f"    Cluster {cluster_id} ({name:<25}) "
          f"score={cluster_sil_mean:.4f}{flag}")


# ─────────────────────────────────────────────────────────────
# STEP 8 — COMPARE ML CLUSTERS VS RULE-BASED SEGMENTS
# ─────────────────────────────────────────────────────────────
# The cross-tabulation shows how K-Means clusters map onto the
# Phase 3b enriched segments. Where they agree = strong signal.
# Where they disagree = interesting — the ML found a nuance the
# rules missed, or the rules captured domain knowledge ML can't.

print("\n" + "=" * 60)
print("STEP 8 — ML clusters vs rule-based segments")
print("=" * 60)

comparison = feat[["Customer_ID", "cluster", "cluster_name"]].merge(
    seg3[["Customer_ID", "segment_enriched"]], on="Customer_ID"
)

# Cross-tabulation: rows = ML cluster name, cols = rule-based segment
crosstab = pd.crosstab(
    comparison["cluster_name"],
    comparison["segment_enriched"],
    margins=True
)
print("\n  Cross-tabulation (rows=ML cluster, cols=rule-based segment):")
print(crosstab.to_string())

# Agreement rate: what % of customers have matching labels?
# We map cluster names to rule-based names for comparison
CLUSTER_TO_RULE = {
    "VIP / Champion"       : ["Champion", "Premium"],
    "Loyal / Premium"      : ["Champion", "Premium", "Loyal"],
    "High-Value At-Risk"   : ["At-Risk"],
    "Churned / Dormant"    : ["Churned"],
    "New / One-Time"       : ["New"],
    "Budget / Price-Sensitive": ["Budget"],
    "Standard / Mid-Tier"  : ["Standard", "Loyal"],
}

matches = comparison.apply(
    lambda r: r["segment_enriched"] in
              CLUSTER_TO_RULE.get(r["cluster_name"], []),
    axis=1
)
agreement = matches.mean() * 100
print(f"\n  Label agreement rate : {agreement:.1f}%")
print(f"  ({matches.sum()} of {len(matches)} customers match between "
      f"ML and rule-based)")


# ─────────────────────────────────────────────────────────────
# STEP 9 — SAVE
# ─────────────────────────────────────────────────────────────

output = feat[["Customer_ID", "cluster", "cluster_name"]].merge(
    seg3[["Customer_ID", "segment_enriched",
          "R_score", "F_score", "M_score",
          "R_days", "F_orders", "M_total_revenue"]],
    on="Customer_ID"
)

output.to_csv(os.path.join(DATA_DIR, "clusters_kmeans.csv"), index=False)

print("\n" + "=" * 60)
print(f"  Saved → clusters_kmeans.csv  ({output.shape[0]:,} rows)")
print("\n  Output files:")
print("    clusters_kmeans.csv    — cluster labels per customer")
print("    elbow_silhouette.png   — K selection charts")
print("    pca_clusters.png       — 2D cluster visualisation")
print("    silhouette_plot.png    — per-sample cluster quality")
print("\nPhase 4 complete. Run phase5_validation.py next.")
print("=" * 60)
