import duckdb

con = duckdb.connect()
con.execute("""
    CREATE VIEW gold AS 
    SELECT * FROM read_parquet('/tmp/gold_distress/**/*.parquet', hive_partitioning=true)
""")

# 1. Zone distribution by company type
print("\n=== Zone distribution by company type ===")
print(con.execute("""
    SELECT 
        distress_label,
        distress_zone,
        COUNT(*) as n_periods,
        ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY distress_label), 1) as pct
    FROM gold
    WHERE distress_zone IS NOT NULL
    GROUP BY distress_label, distress_zone
    ORDER BY distress_label, distress_zone
""").fetchdf())

# 2. Mean Z-Score by company type
print("\n=== Mean Z-Score by company type ===")
print(con.execute("""
    SELECT 
        distress_label,
        ROUND(AVG(altman_z_score), 2) as mean_z,
        ROUND(MEDIAN(altman_z_score), 2) as median_z,
        COUNT(altman_z_score) as n_scores
    FROM gold
    GROUP BY distress_label
""").fetchdf())

# 3. % of LoPucki companies that were in distress zone at least once
print("\n=== Recall: LoPucki companies flagged at least once ===")
print(con.execute("""
    SELECT 
        COUNT(DISTINCT CASE WHEN distress_zone = 'distress' THEN cik END) as flagged,
        COUNT(DISTINCT cik) as total,
        ROUND(100.0 * COUNT(DISTINCT CASE WHEN distress_zone = 'distress' THEN cik END) / COUNT(DISTINCT cik), 1) as pct
    FROM gold
    WHERE distress_label = 1
""").fetchdf())