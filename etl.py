import argparse
import os
import re
import pandas as pd
import duckdb

parser = argparse.ArgumentParser(description="ETL: Excel (Sheet1) -> DuckDB (sales) for Pharma-GPT")
parser.add_argument("excel_path", help="Full path to your Excel file")
parser.add_argument("--db", default="pharma.duckdb", help="Output DuckDB file (default: pharma.duckdb)")
args = parser.parse_args()

excel_path = args.excel_path
out_db = args.db

# ----- 1) Load Sheet1 explicitly -----
# dtype=str keeps Persian text & dates stable; we will convert numbers ourselves.
df = pd.read_excel(excel_path, sheet_name="Sheet1", engine="openpyxl")

# ----- 2) Normalize Farsi/English headers -----
def norm_header(s: str) -> str:
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = s.strip()
    s = s.replace("ي", "ی").replace("ك", "ک")  # Arabic -> Persian forms
    s = re.sub(r"\s+", " ", s)                 # collapse spaces
    return s

df.columns = [norm_header(c) for c in df.columns]

# Known headers (normalized text)
H = {
    "generic_code": "کد ژنریک",
    "generic_name_fa": "نام ژنریک",
    "molecule_fa": "مولکول دارویی",
    "trade_name_fa": "نام تجاری فرآورده",
    "company_fa": "شرکت تامین کننده",
    "jalali_date": "تاریخ شناسه گذاری",
    "year_fa": "سال",
    "origin_type": "تولیدی/وارداتی",
    "temp_permit": "پروانه موقت",
    "price_raw": "قیمت",
    "qty_raw": "تعداد تامین شده",
    "sales_rial_raw": "ارزش ریالی",
    "drug_name_en": "drug name",
    "route": "route",
    "dosage_form": "dosage form",
    "atc_code": "atc code",
    "anatomical": "Anatomical"
}

# Map present headers to internal names (tolerant to partial matches)
present = {}
for k, fa in H.items():
    hit = None
    for col in df.columns:
        if col == fa or fa in col:
            hit = col
            break
    if hit is not None:
        present[k] = hit

df = df.rename(columns={present[k]: k for k in present})

# ----- 3) Numeric cleanup -----
def to_num(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return 0.0
    if isinstance(x, (int, float)) and not pd.isna(x):
        return float(x)
    s = str(x).strip().replace(",", "")
    try:
        return float(s)
    except Exception:
        return 0.0

for c in ["price_raw", "qty_raw", "sales_rial_raw"]:
    if c in df.columns:
        df[c] = df[c].map(to_num)

df["unit_price_rial"] = df["price_raw"].astype(float) if "price_raw" in df.columns else 0.0
df["qty"] = df["qty_raw"].astype(float) if "qty_raw" in df.columns else 0.0
df["sales_rial"] = df["sales_rial_raw"].astype(float) if "sales_rial_raw" in df.columns else df["unit_price_rial"] * df["qty"]

# ----- 4) Year/Month (Jalali) -----
if "year_fa" in df.columns:
    df["jyear"] = df["year_fa"].astype(str).str.slice(0,4)
elif "jalali_date" in df.columns:
    df["jyear"] = df["jalali_date"].astype(str).str.slice(0,4)
else:
    df["jyear"] = ""

if "jalali_date" in df.columns:
    df["jmonth"] = df["jalali_date"].astype(str).str.slice(5,7)
else:
    df["jmonth"] = ""

df["jmonth"] = df["jmonth"].apply(lambda m: ("0"+str(int(m)))[-2:] if str(m).strip().isdigit() else str(m))
df["jym"] = df["jyear"].astype(str) + "-" + df["jmonth"].astype(str)

# ----- 5) ATC levels (robust) -----
def clean_atc(x):
    if pd.isna(x): return ""
    s = str(x).strip().upper().replace(" ", "")
    return s

if "atc_code" in df.columns:
    df["atc_code"] = df["atc_code"].map(clean_atc)
    df["atc1"] = df["atc_code"].str.slice(0,1)
    df["atc2"] = df["atc_code"].str.slice(0,3)
    df["atc3"] = df["atc_code"].str.slice(0,4)
    df["atc4"] = df["atc_code"].str.slice(0,5)
    df["atc5"] = df["atc_code"].str.slice(0,7)
else:
    for c in ["atc1","atc2","atc3","atc4","atc5"]:
        df[c] = ""

# ----- 6) Company normalization -----
def norm_company(s):
    if pd.isna(s): return ""
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("ي","ی").replace("ك","ک")
    return s

df["company_norm"] = df["company_fa"].map(norm_company) if "company_fa" in df.columns else ""

# ----- 7) Select final columns -----
cols = [
    "jyear", "jmonth", "jym",
    "atc1", "atc2", "atc3", "atc4", "atc5", "atc_code",
    "route", "dosage_form",
    "company_fa", "company_norm",
    "generic_code", "generic_name_fa", "drug_name_en", "trade_name_fa",
    "unit_price_rial", "qty", "sales_rial",
    "jalali_date", "origin_type", "temp_permit", "anatomical"
]
existing = [c for c in cols if c in df.columns]
clean = df[existing].copy()

# ----- 8) Save to DuckDB -----
con = duckdb.connect(out_db)
con.execute("DROP TABLE IF EXISTS sales;")
con.execute("CREATE TABLE sales AS SELECT * FROM clean;")
# Optional: small index to speed up reads (DuckDB will auto-optimize, but safe to create projections)
# DuckDB doesn't use traditional indexes; leaving as-is.
con.close()

print(f"✅ DB created: {os.path.abspath(out_db)} (table: sales)")
print("Columns in table:", ", ".join(existing))
print("Row count:", len(clean))
