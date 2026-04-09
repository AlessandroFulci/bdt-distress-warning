import os
from dotenv import load_dotenv
from minio import Minio
import requests

load_dotenv()

# Test 1: SEC EDGAR is reachable
print("Testing SEC EDGAR connection...")
headers = {"User-Agent": os.getenv("SEC_USER_AGENT")}
r = requests.get(
    "https://data.sec.gov/submissions/CIK0000320193.json",
    headers=headers,
    timeout=10,
)
print(f"  EDGAR status: {r.status_code}")
print(f"  Apple's name from EDGAR: {r.json().get('name')}")

# Test 2: MinIO is reachable
print("\nTesting MinIO connection...")
client = Minio(
    os.getenv("MINIO_ENDPOINT"),
    access_key=os.getenv("MINIO_ACCESS_KEY"),
    secret_key=os.getenv("MINIO_SECRET_KEY"),
    secure=os.getenv("MINIO_SECURE") == "true",
)

bronze_bucket = os.getenv("MINIO_BRONZE_BUCKET")
if not client.bucket_exists(bronze_bucket):
    client.make_bucket(bronze_bucket)
    print(f"  Created bucket: {bronze_bucket}")
else:
    print(f"  Bucket already exists: {bronze_bucket}")

print("\nAll good. Setup is working.")
