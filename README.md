# Customer Segmentation Pipeline

Python pipeline that segments 1,000 customers using RFM analysis,
behavioural feature engineering, and K-Means clustering.

## Pipeline

| Phase | Script | Output |
|-------|--------|--------|
| 0 | generate_customer_data.py | 3 raw CSV files |
| 1 | phase1_data_prep.py | master_df.csv |
| 2 | phase2_feature_engineering.py | features_df.csv |
| 3b | phase3b_python_segmentation.py | segments_python.csv |
| 4 | phase4_clustering.py | clusters_kmeans.csv |

## How to run

pip install -r requirements.txt
python generate_customer_data.py
python phase1_data_prep.py
...

## Key results

- 8 customer segments identified
- Champions (10.8% of customers) generate 29.1% of revenue
- K-Means silhouette score: 0.371

## Tools
Python · pandas · scikit-learn · Faker · matplotlib
