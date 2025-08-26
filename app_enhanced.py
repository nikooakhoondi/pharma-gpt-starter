import os
import re
import json
import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st

DB_PATH = "pharma.duckdb"

st.set_page_config(page_title="Pharma-GPT — Advanced Filters", layout="wide")
st.title("💊 Pharma-GPT — Advanced Filters & Table")

if not os.path.exists(DB_PATH):
    st.error("Database not found. Run:  python etl.py YOUR_FILE.xlsx --db pharma.duckdb")
    st.stop()

con = duckdb.connect(DB_PATH)

# ---------- Helpers ----------
def qdf(sql, params=None):
    try:
        if params:
            return con.execute(sql, params).df()
        return con.execute(sql).df()
    except Exception as e:
        st.error(f"SQL error: {e}")
        return pd.DataFrame()

def draw_chart(df, x_col="jym", y_col="sales_rial", color=None):
    if df.empty: 
        st.warning("No data to chart.")
        return
    if x_col in df.columns and y_col in df.columns and df[x_col].nunique() > 1:
        fig = px.line(df, x=x_col, y=y_col, color=color)
        st.plotly_chart(fig, use_container_width=True)
    else:
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = [c for c in df.columns if c not in num_cols]
        if not num_cols:
            st.dataframe(df)
            return
        fig = px.bar(df, x=cat_cols[0] if cat_cols else df.columns[0], y=num_cols[0])
        st.plotly_chart(fig, use_container_width=True)

# ---------- Fetch uniques for selectors ----------
distinct_cols = {
    "company_fa": "شرکت تامین کننده",
    "molecule_fa": "مولکول دارویی",
    "generic_name_fa": "نام ژنریک",
    "trade_name_fa": "نام تجاری فرآورده",
    "route": "route",
    "dosage_form": "dosage form",
    "origin_type": "تولیدی/وارداتی",
    "atc_code": "ATC code"
}

uniques = {}
for col in distinct_cols.keys():
    if col in qdf("PRAGMA table_info('sales');")["name"].tolist():
        uniques[col] = qdf(f"SELECT DISTINCT {col} FROM sales WHERE {col} IS NOT NULL AND {col} <> '' ORDER BY {col} LIMIT 20000")[col].tolist()
    else:
        uniques[col] = []

years = qdf("SELECT DISTINCT jyear FROM sales WHERE jyear <> '' ORDER BY jyear")["jyear"].tolist()
minmax_price = qdf("SELECT MIN(unit_price_rial) AS minp, MAX(unit_price_rial) AS maxp FROM sales")
min_price = float(minmax_price["minp"].iloc[0] or 0.0)
max_price = float(minmax_price["maxp"].iloc[0] or 0.0)

# ---------- Sidebar filters ----------
st.sidebar.header("Filters")

# ATC level & code
atc_level = st.sidebar.radio("ATC level", ["atc5","atc4","atc3","atc2","atc1"], horizontal=True)
atc_code = st.sidebar.text_input("ATC code (exact, optional)", value="")

# Company (type & choose)
company_sel = st.sidebar.multiselect("شرکت تامین کننده (type to search)", options=uniques.get("company_fa", []))

# Molecule, Generic, Trade
molecule_sel = st.sidebar.multiselect("مولکول دارویی", options=uniques.get("molecule_fa", []))
generic_sel  = st.sidebar.multiselect("نام ژنریک", options=uniques.get("generic_name_fa", []))
trade_sel    = st.sidebar.multiselect("نام تجاری فرآورده", options=uniques.get("trade_name_fa", []))

# Route / Dosage form / Origin
route_sel  = st.sidebar.multiselect("route", options=uniques.get("route", []))
dosage_sel = st.sidebar.multiselect("dosage form", options=uniques.get("dosage_form", []))
origin_sel = st.sidebar.multiselect("تولیدی/وارداتی", options=uniques.get("origin_type", []))

# Years
year_sel = st.sidebar.multiselect("سال (Jalali)", options=years, default=[])

# Price range
if max_price > 0:
    price_range = st.sidebar.slider("قیمت (unit_price_rial)", min_value=float(min_price), max_value=float(max_price), value=(float(min_price), float(max_price)))
else:
    price_range = (0.0, 0.0)

# Free text contains (applied to trade_name_fa and generic_name_fa)
free_text = st.sidebar.text_input("Contains (in نام ژنریک/نام تجاری)", value="")

# Aggregation mode
agg_mode = st.sidebar.radio("Aggregation", ["Monthly (by jym, company, atc5)", "Raw rows"], index=0)

# ---------- Build WHERE clause safely ----------
where = []
params = {}

def add_list_filter(col, values, pname):
    if values:
        placeholders = ",".join([f":{pname}{i}" for i in range(len(values))])
        where.append(f"{col} IN ({placeholders})")
        for i, v in enumerate(values):
            params[f"{pname}{i}"] = v

# ATC level/code
if atc_code.strip():
    where.append(f"{atc_level} = :atc_code")
    params["atc_code"] = atc_code.strip().upper()

# Multi-select filters
add_list_filter("company_fa", company_sel, "comp")
add_list_filter("molecule_fa", molecule_sel, "mol")
add_list_filter("generic_name_fa", generic_sel, "gen")
add_list_filter("trade_name_fa", trade_sel, "trd")
add_list_filter("route", route_sel, "rte")
add_list_filter("dosage_form", dosage_sel, "dsg")
add_list_filter("origin_type", origin_sel, "org")
add_list_filter("jyear", year_sel, "yr")

# Price range
if max_price > 0:
    where.append("unit_price_rial BETWEEN :pmin AND :pmax")
    params["pmin"], params["pmax"] = price_range

# Free text
if free_text.strip():
    where.append("(generic_name_fa LIKE :ft OR trade_name_fa LIKE :ft)")
    params["ft"] = f"%{free_text.strip()}%"

where_sql = " WHERE " + " AND ".join(where) if where else ""

# ---------- Columns to display ----------
all_cols = [
    "jyear","jmonth","jym",
    "company_fa",
    "molecule_fa","generic_name_fa","trade_name_fa",
    "route","dosage_form","origin_type",
    "atc_code","atc1","atc2","atc3","atc4","atc5",
    "unit_price_rial","qty","sales_rial"
]
present_cols = [c for c in all_cols if c in qdf("PRAGMA table_info('sales');")["name"].tolist()]

st.sidebar.header("Columns")
default_cols = [c for c in ["jym","company_fa","atc5","generic_name_fa","trade_name_fa","dosage_form","route","qty","sales_rial"] if c in present_cols]
show_cols = st.sidebar.multiselect("Select columns to show (table):", options=present_cols, default=default_cols)

# ---------- Run query ----------
st.subheader("Results")

if agg_mode.startswith("Monthly"):
    # Aggregate by month + selected key dims
    group_dims = [c for c in ["jym","company_fa","atc5"] if c in present_cols]
    select_dims = ", ".join(group_dims) if group_dims else ""
    select_expr = (select_dims + ", ") if select_dims else ""
    sql = f"""
        SELECT {select_expr} SUM(qty) AS qty, SUM(sales_rial) AS sales_rial
        FROM sales
        {where_sql}
        GROUP BY {select_dims} {"," if select_dims else ""} 
        ORDER BY {", ".join(group_dims) if group_dims else "sales_rial DESC"}
    """
    df = qdf(sql, params)
    # For the table, reorder to user-selected columns if possible
    for c in ["qty","sales_rial"]:
        if c not in df.columns:
            df[c] = None
    ordered = [c for c in show_cols if c in df.columns] + [c for c in ["qty","sales_rial"] if c in df.columns and c not in show_cols]
    st.dataframe(df[ordered] if ordered else df, use_container_width=True)
    # Chart
    if "jym" in df.columns:
        draw_chart(df, x_col="jym", y_col="sales_rial", color="company_fa" if "company_fa" in df.columns else None)
else:
    # Raw rows, limited for safety
    select_cols = ", ".join(show_cols) if show_cols else "*"
    sql = f"""
        SELECT {select_cols}
        FROM sales
        {where_sql}
        LIMIT 10000
    """
    df = qdf(sql, params)
    st.dataframe(df, use_container_width=True)

st.caption("Tip: Use the sidebar to filter by company, molecule, generic, trade, route, dosage form, origin, year, ATC level/code, and price range. Choose which columns appear in the table.")
