# app_enhanced_v2.py
import os
import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import streamlit_authenticator as stauth

# -----------------------------------------------------------
# Page config
# -----------------------------------------------------------
st.set_page_config(page_title="Pharma-GPT — Advanced Filters (v2)", layout="wide")

# -----------------------------------------------------------
# --- LOGIN (username/password) ---
# -----------------------------------------------------------
# --- robust login call that works across authenticator versions ---
def do_login(authenticator):
    # Newer versions: login(location="main")  (no form name)
    try:
        return authenticator.login(location="main")
    except TypeError:
        pass
    # Older versions: login("Login", "main")
    try:
        return authenticator.login("Login", "main")
    except TypeError:
        # Slightly newer old-version that needs keyword for location
        return authenticator.login("Login", location="main")
# ---- Build authenticator from Streamlit secrets ----
def _build_authenticator_from_secrets():
    AUTH = st.secrets.get("auth", {})

    # cookie settings with sensible defaults
    cookie_name = str(AUTH.get("cookie_name", "pharma_gpt_auth"))
    cookie_key  = str(AUTH.get("cookie_key",  "change-me"))

    # copy credentials from secrets → plain dict (st.secrets is read-only)
    users = {}
    creds_in = AUTH.get("credentials", {}).get("usernames", {})
    for uname, info in creds_in.items():
        users[str(uname)] = {
            "name":     str(info.get("name", "")),
            "email":    str(info.get("email", "")),
            "password": str(info.get("password", "")),
        }

    if not users:
        st.error("No users found in secrets. Add them to `.streamlit/secrets.toml` under [auth].")
        st.stop()

    # instantiate the authenticator (positional args for widest compatibility)
    return stauth.Authenticate(
        {"usernames": users},  # credentials
        cookie_name,           # cookie name
        cookie_key,            # cookie key/secret
        7,                     # cookie expiry days
    )

authenticator = _build_authenticator_from_secrets()

name, auth_status, username = do_login(authenticator)

if not auth_status:
    if auth_status is False:
        st.error("Incorrect username or password")
    else:
        st.info("Please log in to continue.")
    st.stop()
else:
    # Show logout in sidebar (support both signatures)
    try:
        authenticator.logout("Logout", location="sidebar")
    except TypeError:
        authenticator.logout("Logout", "sidebar")

# -----------------------------------------------------------
# Header (shown only after login)
# -----------------------------------------------------------
st.title("💊 Pharma-GPT — Advanced Filters & Table (v2)")
st.caption("⚡ Filters auto‑update results (no Run button).")

# -----------------------------------------------------------
# DB setup (only after login)
# -----------------------------------------------------------
DB_PATH = "pharma.duckdb"
if not os.path.exists(DB_PATH):
    st.error("Database not found. Run:  python etl.py YOUR_FILE.xlsx --db pharma.duckdb")
    st.stop()

con = duckdb.connect(DB_PATH)

def qdf(sql: str, params=None) -> pd.DataFrame:
    """Run a DuckDB query and return a DataFrame."""
    try:
        if params:
            return con.execute(sql, params).df()
        return con.execute(sql).df()
    except Exception as e:
        st.error(f"SQL error: {e}")
        return pd.DataFrame()

def draw_chart(df: pd.DataFrame, x_col="jym", y_col="sales_rial", color=None):
    """Draw a simple line/bar chart depending on the data."""
    if df.empty:
        return
    if x_col in df.columns and y_col in df.columns and df[x_col].nunique() > 1:
        fig = px.line(df, x=x_col, y=y_col, color=color)
        st.plotly_chart(fig, use_container_width=True)
    else:
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = [c for c in df.columns if c not in num_cols]
        if num_cols:
            x = cat_cols[0] if cat_cols else df.columns[0]
            fig = px.bar(df, x=x, y=num_cols[0])
            st.plotly_chart(fig, use_container_width=True)

# ---------- Schema & basics ----------
schema = qdf("PRAGMA table_info('sales');")
present_cols_set = set(schema["name"].tolist())

def get_uniques(col: str):
    if col in present_cols_set:
        return qdf(
            f"SELECT DISTINCT {col} FROM sales "
            f"WHERE {col} IS NOT NULL AND {col} <> '' "
            f"ORDER BY {col} LIMIT 20000"
        )[col].tolist()
    return []

years = qdf("SELECT DISTINCT jyear FROM sales WHERE jyear <> '' ORDER BY jyear")["jyear"].tolist()
mm = qdf("SELECT MIN(unit_price_rial) AS minp, MAX(unit_price_rial) AS maxp FROM sales")
min_val = mm["minp"].iloc[0] if not mm.empty else None
max_val = mm["maxp"].iloc[0] if not mm.empty else None
pmin = float(min_val) if pd.notna(min_val) else 0.0
pmax = float(max_val) if pd.notna(max_val) else 0.0

# -----------------------------------------------------------
# Sidebar filters
# -----------------------------------------------------------
st.sidebar.header("Filters")

atc_level = st.sidebar.radio("ATC level", ["atc5", "atc4", "atc3", "atc2", "atc1"], horizontal=True)
atc_code = st.sidebar.text_input("ATC code (exact, optional)", value="")

company_sel  = st.sidebar.multiselect("شرکت تامین کننده", options=get_uniques("company_fa"))
molecule_sel = st.sidebar.multiselect("مولکول دارویی", options=get_uniques("molecule_fa"))
generic_sel  = st.sidebar.multiselect("نام ژنریک", options=get_uniques("generic_name_fa"))
trade_sel    = st.sidebar.multiselect("نام تجاری فرآورده", options=get_uniques("trade_name_fa"))
route_sel    = st.sidebar.multiselect("route", options=get_uniques("route"))
dosage_sel   = st.sidebar.multiselect("dosage form", options=get_uniques("dosage_form"))
origin_sel   = st.sidebar.multiselect("تولیدی/وارداتی", options=get_uniques("origin_type"))
year_sel     = st.sidebar.multiselect("سال (Jalali)", options=years, default=[])

if pmax > 0:
    price_range = st.sidebar.slider(
        "قیمت (unit_price_rial)",
        min_value=float(pmin),
        max_value=float(pmax),
        value=(float(pmin), float(pmax)),
    )
else:
    price_range = (0.0, 0.0)

free_text = st.sidebar.text_input("Contains (نام ژنریک/نام تجاری)", value="")
agg_mode = st.sidebar.radio("Aggregation", ["Monthly (by jym, company, atc5)", "Raw rows"], index=0)

# ---------- WHERE builder ----------
where_clauses = []
params = []

def add_in_list(col: str, values):
    if values:
        placeholders = ",".join(["?"] * len(values))
        where_clauses.append(f"{col} IN ({placeholders})")
        params.extend(values)

if atc_code.strip():
    where_clauses.append(f"{atc_level} = ?")
    params.append(atc_code.strip().upper())

add_in_list("company_fa", company_sel)
add_in_list("molecule_fa", molecule_sel)
add_in_list("generic_name_fa", generic_sel)
add_in_list("trade_name_fa", trade_sel)
add_in_list("route", route_sel)
add_in_list("dosage_form", dosage_sel)
add_in_list("origin_type", origin_sel)
add_in_list("jyear", year_sel)

if pmax > 0:
    where_clauses.append("unit_price_rial BETWEEN ? AND ?")
    params.extend([price_range[0], price_range[1]])

if free_text.strip():
    where_clauses.append("(generic_name_fa LIKE ? OR trade_name_fa LIKE ?)")
    like_val = f"%{free_text.strip()}%"
    params.extend([like_val, like_val])

where_sql = (" WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

# ---------- Column chooser ----------
all_cols = [
    "jyear","jmonth","jym",
    "company_fa",
    "molecule_fa","generic_name_fa","trade_name_fa",
    "route","dosage_form","origin_type",
    "atc_code","atc1","atc2","atc3","atc4","atc5",
    "unit_price_rial","qty","sales_rial"
]
present_cols = [c for c in all_cols if c in present_cols_set]

st.sidebar.header("Columns")
default_cols = [c for c in ["jym","company_fa","atc5","generic_name_fa","trade_name_fa","dosage_form","route","qty","sales_rial"] if c in present_cols]
show_cols = st.sidebar.multiselect("Select columns to show (table):", options=present_cols, default=default_cols)

# -----------------------------------------------------------
# Results
# -----------------------------------------------------------
st.subheader("Results")

if agg_mode.startswith("Monthly"):
    group_dims = [c for c in ["jym","company_fa","atc5"] if c in present_cols_set]
    select_dims = ", ".join(group_dims)
    select_expr = (select_dims + ", ") if select_dims else ""
    group_by = f"GROUP BY {select_dims}" if select_dims else ""
    order_by = f"ORDER BY {', '.join(group_dims)}" if group_dims else "ORDER BY sales_rial DESC"

    sql = f"""
        SELECT {select_expr} SUM(qty) AS qty, SUM(sales_rial) AS sales_rial
        FROM sales
        {where_sql}
        {group_by}
        {order_by}
    """
    df = qdf(sql, params)

    for c in ["qty","sales_rial"]:
        if c not in df.columns:
            df[c] = None
    ordered = [c for c in show_cols if c in df.columns] + [c for c in ["qty","sales_rial"] if c in df.columns and c not in show_cols]
    st.dataframe(df[ordered] if ordered else df, use_container_width=True)

    if "jym" in df.columns:
        draw_chart(df, x_col="jym", y_col="sales_rial", color=("company_fa" if "company_fa" in df.columns else None))
else:
    select_cols = ", ".join(show_cols) if show_cols else "*"
    sql = f"SELECT {select_cols} FROM sales {where_sql} LIMIT 10000"
    df = qdf(sql, params)
    st.dataframe(df, use_container_width=True)

st.caption("Tip: Use the sidebar to filter by company, molecule, generic, trade name, route, dosage form, origin type, year, ATC level/code, and price range. Choose which columns appear in the table.")

# -----------------------------------------------------------
# GPT Chat (Persian/English)
# -----------------------------------------------------------
load_dotenv()
API_KEY = (st.secrets.get("OPENAI_API_KEY") if "OPENAI_API_KEY" in st.secrets else None) or os.getenv("OPENAI_API_KEY")
if not API_KEY:
    st.markdown("---")
    st.subheader("💬 Chat with GPT (Persian/English)")
    st.warning("OpenAI API key not found in Secrets or .env — chat is disabled.")
else:
    client = OpenAI(api_key=API_KEY)
    st.markdown("---")
    st.subheader("💬 Chat with GPT (Persian/English)")
    st.caption(f"API key loaded: {bool(API_KEY)}")

    persian_to_col = {
        "شرکت تامین کننده": "company_fa",
        "مولکول دارویی": "molecule_fa",
        "نام ژنریک": "generic_name_fa",
        "نام تجاری فرآورده": "trade_name_fa",
        "سال": "jyear",
        "تولیدی/وارداتی": "origin_type",
        "مسیر مصرف": "route",
        "route": "route",
        "شکل دارویی": "dosage_form",
        "dosage form": "dosage_form",
        "قیمت": "unit_price_rial",
        "ارزش ریالی": "sales_rial",
        "تعداد": "qty",
        "کد ATC": "atc_code",
        "atc1": "atc1", "atc2": "atc2", "atc3": "atc3", "atc4": "atc4", "atc5": "atc5",
    }

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        st.chat_message(msg["role"]).write(msg["content"])

    user_q = st.chat_input("سوالت را اینجا بنویس (ask about your data)…")
    if user_q:
        st.chat_message("user").write(user_q)
        st.session_state.chat_history.append({"role": "user", "content": user_q})

        schema_cols = ", ".join(sorted(list(present_cols_set)))
        system_prompt = (
            "You are a SQL generator for DuckDB. "
            "Given a Persian or English question about the `sales` table, "
            "return ONLY a valid SQL query that DuckDB can execute. No prose. No markdown fences. "
            "Use only existing columns."
        )

        mapping_lines = "\n".join([f"- {k} -> {v}" for k, v in persian_to_col.items()])
        user_prompt = f"""
Question: {user_q}

Important mappings from Persian terms to columns (use these when needed):
{mapping_lines}

Table: sales
Columns: {schema_cols}

Rules:
- Use exact column names above.
- Prefer WHERE filters that match the user's question.
- For market share, compute share = SUM(sales_rial for group)/SUM(sales_rial total), grouped by year/company as needed.
- Jalali year is `jyear` (e.g., 1402).
- If ATC is referenced, you can filter on atc1..atc5 or atc_code.
- Always LIMIT 5000 for raw rows.
Return ONLY the SQL query.
""".strip()

        sql_query = None
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
            )
            sql_query = resp.choices[0].message.content.strip()
        except Exception as e:
            st.error(f"OpenAI error: {e}")

        if sql_query:
            st.chat_message("assistant").write(f"SQL پیشنهادی:\n```sql\n{sql_query}\n```")
            try:
                df_chat = qdf(sql_query)
                if df_chat.empty:
                    st.info("نتیجه‌ای یافت نشد.")
                else:
                    st.dataframe(df_chat, use_container_width=True)
                    draw_chart(df_chat)
            except Exception as e:
                st.error(f"Query failed: {e}")

            st.session_state.chat_history.append({"role": "assistant", "content": sql_query})
