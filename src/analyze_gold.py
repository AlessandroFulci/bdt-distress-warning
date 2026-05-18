"""Extract presentation result numbers from the Gold layer (run once)."""
import duckdb
from pathlib import Path

GOLD = Path(__file__).resolve().parent.parent / "data" / "cache" / "gold_distress"
g = f"read_parquet('{GOLD}/**/*.parquet', hive_partitioning=true)"
con = duckdb.connect()

def show(title, sql):
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)
    print(con.sql(sql).df().to_string(index=False))

show("[1] Z'-Score by group  ->  Slide 13 table", f"""
  SELECT CASE distress_label WHEN 1 THEN 'Distressed (LoPucki)'
                             ELSE 'Healthy (S&P 500)' END         AS company_group,
         COUNT(DISTINCT cik)             AS companies,
         COUNT(*)                        AS periods_total,
         COUNT(altman_z_score)           AS periods_scored,
         ROUND(AVG(altman_z_score), 2)   AS mean_z,
         ROUND(MEDIAN(altman_z_score),2) AS median_z
  FROM {g}
  GROUP BY distress_label ORDER BY distress_label
""")

show("[2] Distress-zone distribution by group  ->  Slide 13 + Slide 15 precision", f"""
  SELECT CASE distress_label WHEN 1 THEN 'Distressed (LoPucki)'
                             ELSE 'Healthy (S&P 500)' END AS company_group,
         ROUND(100.0*COUNT(*) FILTER (WHERE distress_zone='distress')/COUNT(*),1) AS pct_distress,
         ROUND(100.0*COUNT(*) FILTER (WHERE distress_zone='grey')    /COUNT(*),1) AS pct_grey,
         ROUND(100.0*COUNT(*) FILTER (WHERE distress_zone='safe')    /COUNT(*),1) AS pct_safe
  FROM {g}
  WHERE altman_z_score IS NOT NULL
  GROUP BY distress_label ORDER BY distress_label
""")

show("[3] Recall on LoPucki bankruptcies  ->  Slide 13 + Slide 15 recall", f"""
  WITH lop AS (SELECT * FROM {g} WHERE distress_label = 1)
  SELECT COUNT(DISTINCT cik)                                          AS lopucki_with_data,
         COUNT(DISTINCT cik) FILTER (WHERE altman_z_score IS NOT NULL) AS lopucki_with_zscore,
         COUNT(DISTINCT cik) FILTER (WHERE distress_zone='distress')   AS reached_distress,
         ROUND(100.0*COUNT(DISTINCT cik) FILTER (WHERE distress_zone='distress')
               /NULLIF(COUNT(DISTINCT cik) FILTER (WHERE altman_z_score IS NOT NULL),0),1)
                                                                       AS recall_pct
  FROM lop
""")

show("[4] Coverage / data sparsity  ->  Slide 5 + Slide 15", f"""
  SELECT COUNT(*)                                      AS gold_rows_total,
         COUNT(altman_z_score)                         AS gold_rows_scored,
         ROUND(100.0*COUNT(altman_z_score)/COUNT(*),1) AS pct_scored
  FROM {g}
""")