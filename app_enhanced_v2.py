# app_enhanced_v2.py  — Supabase-only version
from typing import Optional
import pandas as pd
import streamlit as st
import streamlit_authenticator as stauth
from supabase import create_client, Client

# -----------------------------------------------------------
# Page config
# -----------------------------------------------------------
st.set_page_config(page_title="Pharma-GPT — Advanced Filters (v2)", layout="wide")

# -----------------------------------------------------------
# --- LOGIN (username/password) ---
# -----------------------------------------------------------
def do_login(authenticator):
    # Try multiple signatures for compatibility with different versions
    try:
        return authenticator.login(location="main")
    except TypeError:
        pass
    try:
        return authenticator.login("Login", "main")
    except TypeError:
        return authenticator.login("Login", location="main")

def _build_authenticator_from_secrets():
    AUTH = st.secrets.get("auth", {})
    cookie_name = str(AUTH.get("cookie_name", "pharma_gpt_auth"))
    cookie_key  = str(AUTH.get("cookie_key",  "change-me"))
    cookie_days = int(AUTH.get("cookie_expiry_days", 7))

    # copy credentials from secrets → plain dict
    users = {}
    creds_in = AUTH.get("credentials", {}).get("usernames", {})
    for uname, info in creds_in.items():
        users[str(uname)] = {
            "name":     str(info.get("name", "")),
            "email":    str(info.get("email", "")),
            "password": str(info.get("password", "")),
        }

    if not users:
        st.error("No users found in secrets. Add them under [auth] → [auth.credentials.usernames.*] in `.streamlit/secrets.toml`.")
        st.stop()

    return stauth.Authenticate({"usernames": users}, cookie_name, cookie_key, cookie_days)

authenticator = _build_authenticator_from_secrets()
name, auth_status, username = do_login(authenticator)

if not auth_status:
    if auth_status is False:
        st.error("Incorrect username or password")
    else:
        st.info("Please log in to continue.")
    st.stop()
else:
    try:
        authenticator.logout("Logout", location="sidebar")
    except TypeError:
        authenticator.logout("Logout", "sidebar")

# -----------------------------------------------------------
# Header
# -----------------------------------------------------------
st.title("💊 Pharma-GPT — Advanced Filters & Table (v2)")
st.caption("⚡ Server-side pivots on Supabase (no local DB).")

# -----------------------------------------------------------
# Supabase connection
# -----------------------------------------------------------
@st.cache_resource
def get_supabase() -> Optional[Client]:
    url = st.secrets.get("SUPABASE_URL")
    key = st.secrets.get("SUPABASE_KEY")
    if not url or not key:
        st.error("Missing SUPABASE_URL / SUPABASE_KEY in Streamlit Secrets.")
        return None
    return create_client(url, key)

sb = get_supabase()
if sb is None:
    st.stop()

# -----------------------------------------------------------
# Pivot UI (uses Supabase RPC: pivot_2d_numeric)
# -----------------------------------------------------------
allowed_dims = [
    "سال",
    "کد ژنریک",
    "نام ژنریک",
    "مولکول دارویی",
    "نام تجاری فرآورده",
    "شرکت تامین کننده",
    "تولیدی/وارداتی",
    "route",
    "dosage form",
    "atc code",
    "Anatomical",
]
metrics = ["ارزش ریالی", "قیمت", "تعداد تامین شده"]

st.header("Pivot (server-side over ALL data)")
c1, c2, c3 = st.columns(3)
dim1   = c1.selectbox("Dimension 1", allowed_dims, index=0)
dim2   = c2.selectbox("Dimension 2", allowed_dims, index=5)
metric = c3.selectbox("Metric (sum of)", metrics, index=0)
y1, y2 = st.slider("Year range (سال)", min_value=1390, max_value=1500, value=(1400, 1404))

@st.cache_data(ttl=300)
def run_pivot(dim1: str, dim2: str, metric: str, y1: int, y2: int) -> pd.DataFrame:
    try:
        res = sb.rpc(
            "pivot_2d_numeric",
            {"dim1": dim1, "dim2": dim2, "metric": metric,
             "year_from": int(y1), "year_to": int(y2)}
        ).execute()
        return pd.DataFrame(res.data or [])
    except Exception as e:
        st.error(f"Supabase RPC failed: {e}")
        return pd.DataFrame()

df = run_pivot(dim1, dim2, metric, y1, y2)
st.caption(f"Returned {len(df)} aggregated rows.")
st.dataframe(df)  # columns expected: d1, d2, total_value, rows

# --- SAFE CHART (handles missing d1/d2) ---
if not df.empty and "total_value" in df.columns:
    idx_cols = [c for c in ("d1", "d2") if c in df.columns]
    chart_df = df.copy().sort_values("total_value", ascending=False)

    if idx_cols:
        chart_df = chart_df.set_index(idx_cols)[["total_value"]]
    else:
        chart_df = chart_df[["total_value"]]

    st.bar_chart(chart_df)


st.info("Need the old detailed filter/table view? We can add a Supabase-backed table page next.")
