"""
Synthetic Customer Data Generator (Faker-optimized)
Generates 4 interconnected CSVs for customer segmentation practice.
"""

import os
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from faker import Faker
from tqdm import tqdm

fake = Faker("en_US")


# ─────────────────────────────────────────────
# SEGMENT CONFIG
# ─────────────────────────────────────────────

@dataclass
class SegmentConfig:
    proportion:   float
    order_range:  tuple[int, int]
    aov_range:    tuple[float, float]
    recency_days: tuple[int, int]
    interaction_range: tuple[int, int]
    sat_range:    tuple[int, int]
    loyalty_rate: tuple[float, float]   # (low, high) fraction of spend
    referral_range: tuple[int, int]


SEGMENTS: dict[str, SegmentConfig] = {
    "Premium":  SegmentConfig(0.15, (25,150), (150,500),  (1,90),   (50,300), (7,10), (0.10,0.20), (2,15)),
    "Standard": SegmentConfig(0.30, (10,50),  (50,150),   (1,90),   (20,120), (5,9),  (0.06,0.12), (0,8)),
    "Budget":   SegmentConfig(0.20, (3,20),   (15,50),    (1,90),   (10,60),  (5,8),  (0.04,0.08), (0,5)),
    "New":      SegmentConfig(0.15, (1,5),    (30,100),   (1,30),   (5,25),   (5,9),  (0.05,0.10), (0,3)),
    "At-Risk":  SegmentConfig(0.12, (5,30),   (40,120),   (90,180), (8,40),   (3,6),  (0.03,0.06), (0,3)),
    "Churned":  SegmentConfig(0.08, (1,15),   (20,80),    (180,730),(2,20),   (1,5),  (0.02,0.04), (0,2)),
}

# ─────────────────────────────────────────────
# STATIC LOOKUP TABLES  (weighted choices)
# ─────────────────────────────────────────────

ACQ_CHANNELS, ACQ_W = zip(*[
    ("Organic Search",0.25),("Paid Social",0.20),("Email Campaign",0.15),
    ("Referral",0.12),("Direct",0.10),("Affiliate",0.08),
    ("TV/Print",0.05),("In-Store",0.05),
])

ORDER_CHANNELS, ORDER_CHANNEL_W = zip(*[
    ("Online",0.40),("Mobile App",0.30),("In-Store",0.15),("Phone",0.10),("Social Media",0.05),
])

CATEGORIES, CAT_W = zip(*[
    ("Electronics",0.20),("Clothing",0.18),("Home & Garden",0.15),("Sports",0.12),
    ("Books",0.10),("Health & Beauty",0.12),("Food & Beverage",0.08),("Automotive",0.05),
])

DEVICES, DEVICE_W     = zip(*[("Desktop",0.45),("Mobile",0.40),("Tablet",0.15)])
PAYMENTS, PAYMENT_W   = zip(*[("Credit Card",0.45),("PayPal",0.20),("Debit Card",0.15),
                                ("Apple Pay",0.10),("Google Pay",0.07),("BNPL",0.03)])
GENDERS, GENDER_W     = zip(*[("Male",0.48),("Female",0.48),("Non-binary",0.02),("Prefer not to say",0.02)])

INT_TYPES, INT_W = zip(*[
    ("email_open",0.25),("email_click",0.15),("email_sent",0.20),("support_ticket",0.12),
    ("web_visit",0.15),("app_open",0.08),("ad_click",0.03),("survey_response",0.02),
])

ORDER_STATUSES  = ["Completed"]*9 + ["Refunded","Cancelled","Pending"]
COUPON_POOL     = ["SAVE10","WELCOME","SUMMER25","FALL15","FLASH20","LOYALTY5","NEWBIE",""]
SUPPORT_CATS    = ["Billing","Shipping","Returns","Product Issue","Account","General Inquiry"]
EMAIL_CAMPAIGNS = ["Welcome Series","Weekly Deals","New Arrivals","Abandoned Cart",
                   "Birthday Offer","Loyalty Rewards","Re-engagement","Seasonal Sale"]
WEB_PAGES       = ["Homepage","Product Page","Cart","Checkout","Search","Category Page","Account","Blog"]
AD_PLATFORMS    = ["Facebook","Google","Instagram","TikTok","Twitter"]
INT_CHANNELS    = ["Email","Web","Mobile App","Phone","Social Media","In-Store"]


def _wchoice(population, weights):
    """Single weighted random choice (faster than np.random.choice for scalars)."""
    return random.choices(population, weights=weights, k=1)[0]


# ─────────────────────────────────────────────
# TABLE BUILDERS
# ─────────────────────────────────────────────

def _build_customers(
    n: int,
    segment_labels: list[str],
    as_of: datetime,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Build the CRM customers table."""

    rows = []
    for i in tqdm(range(n), desc="Customers", unit="cust"):
        seg  = segment_labels[i]
        cfg  = SEGMENTS[seg]

        # --- identity (Faker) ---
        fname  = fake.first_name()
        lname  = fake.last_name()
        email  = fake.email()          # realistic, unique-ish
        phone  = fake.phone_number()
        addr   = fake.street_address()
        city   = fake.city()
        state  = fake.state_abbr()
        zipcode= fake.zipcode()

        # --- demographics ---
        age    = int(np.clip(rng.normal(42, 14), 18, 75))
        gender = _wchoice(GENDERS, GENDER_W)
        income = int(np.clip(rng.lognormal(11.2, 0.6), 15_000, 350_000))

        # --- dates ---
        reg_date = as_of - timedelta(days=int(rng.integers(1, 1826)))

        # --- account ---
        if seg == "Churned":
            status = "Inactive"
        elif seg == "At-Risk":
            status = "Inactive" if rng.random() < 0.30 else "Active"
        else:
            status = "Active"

        rows.append({
            "Customer_ID":             f"CUST-{1000 + i:06d}",
            "First_Name":              fname,
            "Last_Name":               lname,
            "Email":                   email,
            "Phone":                   phone,
            "Age":                     age,
            "Gender":                  gender,
            "Annual_Income":           income,
            "Address_Line_1":          addr,
            "City":                    city,
            "State":                   state,
            "ZIP_Code":                zipcode,
            "Country":                 "USA",
            "Registration_Date":       reg_date.strftime("%Y-%m-%d"),
            "Account_Status":          status,
            "Subscription_Member":     "Yes" if rng.random() < 0.35 else "No",
            "Marketing_Consent":       "Yes" if rng.random() < 0.75 else "No",
            "Acquisition_Channel":     _wchoice(ACQ_CHANNELS, ACQ_W),
            "Satisfaction_Score_1_10": int(rng.integers(*cfg.sat_range)),
            "Loyalty_Points_Balance":  0,          # back-filled after orders
            "Segment":                 seg,
        })

    return pd.DataFrame(rows)


def _build_orders(
    df_customers: pd.DataFrame,
    as_of: datetime,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Build the transactional orders table."""

    rows = []
    order_counter = 100_000

    for _, cust in tqdm(df_customers.iterrows(), total=len(df_customers),
                        desc="Orders", unit="cust"):
        cid  = cust["Customer_ID"]
        seg  = cust["Segment"]
        cfg  = SEGMENTS[seg]
        reg  = datetime.strptime(cust["Registration_Date"], "%Y-%m-%d")
        span = max((as_of - reg).days, 1)

        n_orders    = int(rng.integers(*cfg.order_range))
        aov_lo, aov_hi = cfg.aov_range

        # last-order recency
        rec_lo = min(cfg.recency_days[0], span)
        rec_hi = min(cfg.recency_days[1], span)
        if rec_lo >= rec_hi:
            rec_lo, rec_hi = 1, max(2, span)
        last_order = as_of - timedelta(days=int(rng.integers(rec_lo, rec_hi + 1)))

        # all order dates
        if n_orders == 1:
            dates = [last_order]
        else:
            offsets = sorted(rng.integers(0, span, size=n_orders - 1).tolist())
            dates   = [reg + timedelta(days=int(d)) for d in offsets] + [last_order]

        pref_cat = _wchoice(CATEGORIES, CAT_W)

        for od in dates:
            subtotal     = round(max(5.0, float(rng.uniform(aov_lo, aov_hi)) + float(rng.uniform(-10, 20))), 2)
            n_items      = max(1, int(subtotal / float(rng.uniform(15, 50))))
            disc_pct     = random.choice([0,0,0,0,0,5,10,15,20]) if rng.random() < 0.30 else 0
            disc_amt     = round(subtotal * disc_pct / 100, 2)
            shipping     = 0.0 if subtotal > 50 else round(float(rng.uniform(4.99, 9.99)), 2)
            tax          = round((subtotal - disc_amt) * 0.08, 2)
            total        = round(subtotal - disc_amt + shipping + tax, 2)
            timestamp    = od + timedelta(hours=int(rng.integers(0, 24)), minutes=int(rng.integers(0, 60)))
            category     = pref_cat if rng.random() < 0.70 else _wchoice(CATEGORIES, CAT_W)

            rows.append({
                "Order_ID":         f"ORD-{order_counter}",
                "Customer_ID":      cid,
                "Order_Date":       od.strftime("%Y-%m-%d"),
                "Order_Timestamp":  timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "Order_Status":     random.choice(ORDER_STATUSES),
                "Channel":          _wchoice(ORDER_CHANNELS, ORDER_CHANNEL_W),
                "Category":         category,
                "Subcategory":      random.choice(["Sub_A","Sub_B","Sub_C","Sub_D","Sub_E"]),
                "Items_Count":      n_items,
                "Order_Subtotal":   subtotal,
                "Discount_Amount":  disc_amt,
                "Discount_Pct":     disc_pct,
                "Shipping_Cost":    shipping,
                "Tax_Amount":       tax,
                "Order_Total":      total,
                "Payment_Method":   _wchoice(PAYMENTS, PAYMENT_W),
                "Device_Type":      _wchoice(DEVICES, DEVICE_W),
                "Coupon_Code":      random.choice(COUPON_POOL) if disc_pct > 0 else "",
                "Currency":         "USD",
            })
            order_counter += 1

    return pd.DataFrame(rows)


def _interaction_detail(int_type: str, rng: np.random.Generator) -> tuple:
    """Return (detail, outcome, duration, agent, rating) for a given interaction type."""
    if int_type == "support_ticket":
        outcome_pool = ["Resolved","Resolved","Resolved","Resolved","Escalated","Pending","Closed - No Action"]
        rating_pool  = [1,2,3,4,5,"","","",""]
        return (random.choice(SUPPORT_CATS),
                random.choice(outcome_pool),
                int(rng.integers(2, 46)),
                f"AGT-{rng.integers(100, 1000)}",
                random.choice(rating_pool))
    if int_type in ("email_open","email_click","email_sent"):
        outcome = {"email_sent":"Delivered","email_open":"Opened","email_click":"Clicked"}[int_type]
        return (random.choice(EMAIL_CAMPAIGNS), outcome, int(rng.integers(0, 6)), "", "")
    if int_type == "web_visit":
        return (random.choice(WEB_PAGES),
                random.choice(["Bounce","Browse","Add to Cart","Purchase","Sign Up"]),
                int(rng.integers(1, 31)), "", "")
    if int_type == "app_open":
        return ("Mobile App",
                random.choice(["Browse","Search","Purchase","Check Order"]),
                int(rng.integers(2, 21)), "", "")
    if int_type == "ad_click":
        return (random.choice(AD_PLATFORMS), "Clicked", int(rng.integers(0, 3)), "", "")
    # survey_response
    return ("NPS Survey", f"Score: {rng.integers(0, 11)}", int(rng.integers(3, 11)), "", "")


def _build_interactions(
    df_customers: pd.DataFrame,
    as_of: datetime,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows = []
    int_counter = 1_000_000

    for _, cust in tqdm(df_customers.iterrows(), total=len(df_customers),
                        desc="Interactions", unit="cust"):
        cid  = cust["Customer_ID"]
        seg  = cust["Segment"]
        cfg  = SEGMENTS[seg]
        reg  = datetime.strptime(cust["Registration_Date"], "%Y-%m-%d")
        span = max((as_of - reg).days, 1)

        n_int = int(rng.integers(*cfg.interaction_range))

        # vectorised date offsets
        offsets  = rng.integers(0, span, size=n_int)
        int_type_arr = rng.choice(list(INT_TYPES), size=n_int,
                                  p=[w/sum(INT_W) for w in INT_W])

        for offset, int_type in zip(offsets, int_type_arr):
            int_date  = reg + timedelta(days=int(offset))
            timestamp = int_date + timedelta(
                hours=int(rng.integers(0, 24)), minutes=int(rng.integers(0, 60))
            )
            detail, outcome, duration, agent, rating = _interaction_detail(int_type, rng)

            rows.append({
                "Interaction_ID":        f"INT-{int_counter}",
                "Customer_ID":           cid,
                "Interaction_Date":      int_date.strftime("%Y-%m-%d"),
                "Interaction_Timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "Interaction_Type":      int_type,
                "Channel":               random.choice(INT_CHANNELS),
                "Detail":                detail,
                "Outcome":               outcome,
                "Duration_Minutes":      duration,
                "Agent_ID":              agent,
                "Satisfaction_Rating":   rating,
            })
            int_counter += 1

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────

def generate_synthetic_customer_data(
    n_customers: int = 1000,
    seed: int = 42,
    as_of_date: Optional[datetime] = None,
    output_dir: str = ".",
) -> dict[str, pd.DataFrame]:
    """
    Generate 3 interconnected raw tables + ground truth for customer
    segmentation practice and save them as CSV files.

    Parameters
    ----------
    n_customers : int
        Number of customers to generate (default 1 000).
    seed : int
        Random seed for reproducibility.
    as_of_date : datetime, optional
        Reference "today" for recency calculations. Defaults to now.
    output_dir : str
        Directory in which to write the CSV files.

    Returns
    -------
    dict with keys 'customers', 'orders', 'interactions', 'ground_truth'
    """
    # seed everything
    Faker.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    rng = np.random.default_rng(seed)

    as_of = as_of_date or datetime.now()

    # --- segment assignment ---
    seg_names = list(SEGMENTS.keys())
    seg_probs  = [SEGMENTS[s].proportion for s in seg_names]
    segment_labels = rng.choice(seg_names, size=n_customers, p=seg_probs).tolist()

    # --- build tables ---
    df_customers   = _build_customers(n_customers, segment_labels, as_of, rng)
    df_orders      = _build_orders(df_customers, as_of, rng)
    df_interactions = _build_interactions(df_customers, as_of, rng)

    # --- back-fill loyalty points from real order totals ---
    spend_map = df_orders.groupby("Customer_ID")["Order_Total"].sum()
    for idx, row in df_customers.iterrows():
        cid  = row["Customer_ID"]
        cfg  = SEGMENTS[row["Segment"]]
        spend = spend_map.get(cid, 0)
        rate  = rng.uniform(*cfg.loyalty_rate)
        df_customers.at[idx, "Loyalty_Points_Balance"] = int(spend * rate)

    # --- split ground truth ---
    df_ground_truth   = df_customers[["Customer_ID", "Segment"]].copy()
    df_customers_out  = df_customers.drop(columns=["Segment"]).copy()

    # --- save ---
    os.makedirs(output_dir, exist_ok=True)
    df_customers_out.to_csv( f"{output_dir}/01_customers_raw.csv",         index=False)
    df_orders.to_csv(         f"{output_dir}/02_orders_raw.csv",            index=False)
    df_interactions.to_csv(   f"{output_dir}/03_interactions_raw.csv",      index=False)
    df_ground_truth.to_csv(   f"{output_dir}/00_ground_truth_segments.csv", index=False)

    print("\n" + "=" * 60)
    print("SYNTHETIC DATA GENERATED SUCCESSFULLY")
    print("=" * 60)
    for label, df in [
        ("01_customers_raw.csv",         df_customers_out),
        ("02_orders_raw.csv",            df_orders),
        ("03_interactions_raw.csv",      df_interactions),
        ("00_ground_truth_segments.csv", df_ground_truth),
    ]:
        print(f"  {label:<40} {df.shape[0]:>8,} rows × {df.shape[1]} cols")
    print(f"\n  Saved to: {os.path.abspath(output_dir)}/")

    return {
        "customers":    df_customers_out,
        "orders":       df_orders,
        "interactions": df_interactions,
        "ground_truth": df_ground_truth,
    }


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    data = generate_synthetic_customer_data(
        n_customers = 1000,
        seed        = 42,
        as_of_date  = datetime(2026, 6, 10),
        output_dir  = ".",
    )
