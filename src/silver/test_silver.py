from dotenv import load_dotenv
load_dotenv()
import os
import duckdb

con = duckdb.connect()
con.execute(f"""
    CREATE SECRET minio (
        TYPE s3,
        KEY_ID     '{os.getenv("MINIO_ACCESS_KEY")}',
        SECRET     '{os.getenv("MINIO_SECRET_KEY")}',
        ENDPOINT   '{os.getenv("MINIO_ENDPOINT")}',
        URL_STYLE  'path',
        USE_SSL    false
    )
""")

# Check 1: row count and columns
df = con.execute("""
    SELECT *
    FROM read_parquet('s3://silver/financial_facts/**/*.parquet')
    LIMIT 5
""").df()
print("=== Columns ===")
print(df.columns.tolist())
print()
print("=== Sample rows ===")
print(df[['cik','company_name','period_end','assets_total','ebit','working_capital']].to_string())

# Check 2: no duplicate (cik, period_end)
dupes = con.execute("""
    SELECT cik, period_end, COUNT(*) as n
    FROM read_parquet('s3://silver/financial_facts/**/*.parquet')
    GROUP BY cik, period_end
    HAVING n > 1
""").df()
print(f"\n=== Duplicate periods (should be 0) ===")
print(len(dupes))

# Check 3: null rates for key fields
nulls = con.execute("""
    SELECT
        COUNT(*)                                                    as total_rows,
        SUM(CASE WHEN ebit           IS NULL THEN 1 ELSE 0 END)    as missing_ebit,
        SUM(CASE WHEN working_capital IS NULL THEN 1 ELSE 0 END)   as missing_wc,
        SUM(CASE WHEN assets_total   IS NULL THEN 1 ELSE 0 END)    as missing_assets
    FROM read_parquet('s3://silver/financial_facts/**/*.parquet')
""").df()
print(f"\n=== Null check ===")
print(nulls.to_string())

# Check 4: fiscal year coverage
years = con.execute("""
    SELECT fiscal_year, COUNT(*) as rows, COUNT(DISTINCT cik) as companies
    FROM read_parquet('s3://silver/financial_facts/**/*.parquet')
    GROUP BY fiscal_year
    ORDER BY fiscal_year
""").df()
print(f"\n=== Fiscal year coverage ===")
print(years.to_string())