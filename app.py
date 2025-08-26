import os
import re
import json
import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

DB_PATH = "pharma.duckdb"
st.set_page_config(page_title="Pharma-GPT", layout="wide")
st.title("💊 Pharma-GPT — Chat over your Farsi sales data")
st.caption("Ask in English or Persian; data fields/values remain in Farsi.")

if not os.path.exists(DB_PATH):
    st.error("Run ETL first:  python etl.py your_file.xlsx --db pharma.duckdb")
    st.stop()

con = duckdb.connect(DB_PATH)

FORBIDDEN = re.compile(r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|EXPORT|COPY|PRAGMA)\b", re.I)
def run_sql(sql: str) -> pd.DataFrame:
    if FORBIDDEN.search(sql) or ";" in sql:
        st.error("Forbidden SQL detected.")
        return pd.DataFrame()
    try:
        return con.execute(sql).df()
    except Exception as e:
        st.error(f"SQL error: {e}")
        return pd.DataFrame()

def draw_chart(df):
    if df.empty:
        st.warning("No data.")
        return
    cols = df.columns.tolist()
    if "jym" in cols and df["jym"].nunique() > 1:
        y_cands = [c for c in ["sales_rial", "qty"] if c in cols]
        y = y_cands[0] if y_cands else cols[-1]
        fig = px.line(df, x="jym", y=y, color=cols[1] if len(cols) > 2 else None)
    else:
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = [c for c in cols if c not in num_cols]
        if not num_cols:
            st.dataframe(df)
            return
        y = num_cols[0]
        x = cat_cols[0] if cat_cols else cols[0]
        fig = px.bar(df, x=x, y=y)
    st.plotly_chart(fig, use_container_width=True)

SYSTEM_PROMPT = """
You translate business questions (English or Persian) about Iran pharma sales into SAFE DuckDB SQL over table `sales`.
Return ONLY a compact JSON object: {"sql": "...", "intent": "...", "chart": "line|bar|table"}.

Schema (sales):
  jyear(str), jmonth(str), jym(str YYYY-MM),
  atc1, atc2, atc3, atc4, atc5,
  route, dosage_form,
  company_fa, company_norm,
  generic_code, generic_name_fa, drug_name_en, trade_name_fa,
  unit_price_rial(float), qty(float), sales_rial(float),
  jalali_date(str), origin_type, temp_permit, anatomical

Rules:
- SELECT-only (GROUP BY/WHERE/ORDER), never modify schema.
- "last 5 years" (Jalali) ~ jyear IN ['1399','1400','1401','1402','1403'] unless user specifies otherwise.
- "market share" = company sales / total sales under SAME filters (ATC, dosage_form, route, time).
- Use sales_rial for revenue by default; use qty if user says 'units' or 'تعدادی'.
- Trends: SELECT jym, SUM(metric) GROUP BY jym ORDER BY jym.
- ATC levels: atc5 (full, 7 chars), atc4(5), atc3(4), atc2(3), atc1(1).
- Company Abidi may appear as company_fa LIKE '%عبیدی%' or company_norm LIKE '%عبیدی%'
"""

USE_GPT = bool(OPENAI_API_KEY)
if USE_GPT:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        st.warning(f"OpenAI client not available: {e}")
        USE_GPT = False

samples = st.expander("Quick examples")
with samples:
    c1, c2, c3 = st.columns(3)
    if c1.button("Trend: ATC L02BX03 (1402)"):
        st.session_state["q"] = "Trend of revenue for ATC L02BX03 in Jalali year 1402"
    if c2.button("Abidi market share (last 5 years, N02BE01)"):
        st.session_state["q"] = "Market share of شرکت داروسازی دعبیدی for ATC N02BE01 in last 5 years"
    if c3.button("Top-5 companies in J05AF06 (1402)"):
        st.session_state["q"] = "Top 5 companies by revenue in ATC J05AF06 during 1402"

q = st.chat_input("Ask e.g.: Market share of دعبیدی in last 5 years (ATC N02BE01)") or st.session_state.get("q")
if q:
    st.chat_message("user").write(q)
    if USE_GPT:
        try:
            prompt = SYSTEM_PROMPT + "\nUser: " + q + "\nAssistant:"
            resp = client.responses.create(model="gpt-5", input=prompt, temperature=0.1)
            content = resp.output[0].content[0].text if hasattr(resp, "output") else resp.output_text
            try:
                plan = json.loads(content)
            except Exception:
                m = re.search(r"\{[\s\S]*\}", content)
                plan = json.loads(m.group(0)) if m else {"sql": "", "chart": "table", "intent": ""}
            sql = plan.get("sql", "").strip()
            chart = plan.get("chart", "table")
            st.code(sql or "-- no sql", language="sql")
            if sql:
                df = run_sql(sql)
                st.dataframe(df, use_container_width=True)
                if chart in ("line", "bar"):
                    draw_chart(df)
        except Exception as e:
            st.error(f"GPT error: {e}")
    else:
        st.info("No OpenAI API key set; use the manual Query Builder below.")

st.divider()
st.subheader("Manual Query Builder (no GPT)")
colA, colB, colC, colD = st.columns(4)
level = colA.selectbox("ATC level", ["atc5", "atc4", "atc3", "atc2", "atc1"])
code = colB.text_input("ATC code (optional)")
company = colC.text_input("Company (Farsi substring)")
year = colD.text_input("Jalali year (e.g., ۱۴۰۲)")
metric = st.radio("Metric", ["sales_rial", "qty"], horizontal=True)

if st.button("Run"):
    where = []
    if code: where.append(f"{level} = '{code.strip()}'")
    if company: where.append(f"(company_fa LIKE '%{company.strip()}%' OR company_norm LIKE '%{company.strip()}%')")
    if year: where.append(f"jyear = '{year.strip()}'")
    where_sql = (" WHERE " + " AND ".join(where)) if where else ""
    sql = f"""
    SELECT jym, SUM({metric}) AS {metric}
    FROM sales
    {where_sql}
    GROUP BY jym
    ORDER BY jym
    """
    st.code(sql, language="sql")
    df = run_sql(sql)
    st.dataframe(df, use_container_width=True)
    draw_chart(df)

st.caption("© Pharma-GPT — Streamlit + DuckDB")
