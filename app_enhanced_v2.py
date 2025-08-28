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

# ---- SUPER-SAFE CHART ----
if not df.empty:
    # st.write("Columns:", list(df.columns))  # debug if needed

    if "total_value" in df.columns:
        # Build a single label column from whatever dims exist, then plot total_value
        label_parts = []
        if "d1" in df.columns:
            label_parts.append(df["d1"].astype(str))
        if "d2" in df.columns:
            label_parts.append(df["d2"].astype(str))

        if label_parts:
            # element-wise concat: s1, then s1 + " — " + s2, ...
            label = label_parts[0].fillna("")
            for s in label_parts[1:]:
                label = label.str.cat(s.fillna(""), sep=" — ")
        else:
            # no d1/d2 returned; make a synthetic label index
            label = pd.Series([f"row {i+1}" for i in range(len(df))])

        chart_df = (
            pd.DataFrame({"label": label, "total_value": df["total_value"]})
            .sort_values("total_value", ascending=False)
        )
        st.bar_chart(chart_df.set_index("label")[["total_value"]])
    else:
        st.info("No 'total_value' column returned from the RPC, so chart is skipped.")

st.info("Need the old detailed filter/table view? We can add a Supabase-backed table page next.")

# -----------------------------------------------------------
# Data-Aware Chat (uses Supabase data, no raw SQL)
# -----------------------------------------------------------
import os, json
from openai import OpenAI

API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
TABLE = 'Amarnameh_sheet1'  # Supabase table name

st.markdown("---")
st.subheader("💬 Data-Aware Chat (Persian/English)")

if not API_KEY:
    st.warning("OpenAI API key not found — data chat is disabled.")
else:
    client = OpenAI(api_key=API_KEY)

    # 1) Whitelists so the model can only ask for safe things
    allowed_dims = [
        "سال","کد ژنریک","نام ژنریک","مولکول دارویی","نام تجاری فرآورده",
        "شرکت تامین کننده","تولیدی/وارداتی","route","dosage form","atc code","Anatomical",
    ]
    allowed_metrics = ["ارزش ریالی","قیمت","تعداد تامین شده"]
    # columns you permit for filtering in raw rows:
    allowed_filter_cols = [
        "سال","کد ژنریک","نام ژنریک","مولکول دارویی","نام تجاری فرآورده",
        "شرکت تامین کننده","تولیدی/وارداتی","route","dosage form","atc code","Anatomical",
    ]
    # map some common Persian synonyms -> canonical column names (optional)
    synonyms = {
        "شرکت": "شرکت تامین کننده",
        "تامین کننده": "شرکت تامین کننده",
        "ژنریک": "نام ژنریک",
        "نام تجاری": "نام تجاری فرآورده",
        "کد": "کد ژنریک",
        "سال شمسی": "سال",
        "مسیر": "route",
        "شکل": "dosage form",
        "ارزش": "ارزش ریالی",
        "قیمت واحد": "قیمت",
        "تعداد": "تعداد تامین شده",
        "ATC": "atc code",
    }

    guide = {
        "allowed_dims": allowed_dims,
        "allowed_metrics": allowed_metrics,
        "allowed_filters": allowed_filter_cols,
    }

    if "data_chat" not in st.session_state:
        st.session_state.data_chat = []

    for msg in st.session_state.data_chat:
        st.chat_message(msg["role"]).write(msg["content"])

    user_q = st.chat_input("مثلاً: سهم ارزش ریالی هر شرکت در سال‌های ۱۴۰۰ تا ۱۴۰۲؟")
    if user_q:
        st.chat_message("user").write(user_q)
        st.session_state.data_chat.append({"role": "user", "content": user_q})

        # 2) Ask the model to produce a JSON plan (pivot OR rows)
        system_prompt = f"""
You are a planner that outputs ONLY compact JSON (no prose). You control a data tool with:
- pivot(dim1, dim2, metric, year_from, year_to)  # both dims must be from allowed_dims; metric from allowed_metrics
- rows(filters, limit)  # filters is an object of column -> value OR list of values; only columns in allowed_filters

Rules:
- Use Persian or English inputs.
- If the user asks for shares or totals by categories, use "pivot".
- If the user wants raw examples/records, use "rows".
- Respect year ranges if mentioned; otherwise leave them null.
- If user gives synonyms, normalize using this map: {synonyms}
- Keep JSON small; do not include analysis, only fields below.

Allowed:
{json.dumps(guide, ensure_ascii=False)}

Output schema (one of these):
{{"intent":"pivot","dim1":"...", "dim2":"...", "metric":"...", "year_from":1400, "year_to":1404, "top_n":10}}
OR
{{"intent":"rows","filters":{{"ستون":"مقدار" }}, "limit": 200}}
""".strip()

        try:
            plan_resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_q},
                ],
                temperature=0,
            )
            plan_text = plan_resp.choices[0].message.content.strip()
            # Extract JSON (in case model adds stray text)
            start = plan_text.find("{")
            end = plan_text.rfind("}")
            if start == -1 or end == -1:
                raise ValueError("No JSON plan returned.")
            plan = json.loads(plan_text[start:end+1])
        except Exception as e:
            st.chat_message("assistant").write(f"⚠️ برنامه‌ریز نتوانست برنامه بدهد: {e}")
            st.session_state.data_chat.append({"role": "assistant", "content": f"⚠️ برنامه‌ریز نتوانست برنامه بدهد: {e}"})
            plan = None

        answer = None
        if plan:
            try:
                if plan.get("intent") == "pivot":
                    dim1 = plan.get("dim1")
                    dim2 = plan.get("dim2")
                    metric = plan.get("metric", "ارزش ریالی")
                    y1 = plan.get("year_from")
                    y2 = plan.get("year_to")
                    top_n = int(plan.get("top_n") or 20)

                    # validate
                    if dim1 not in allowed_dims or dim2 not in allowed_dims:
                        raise ValueError("Invalid dimension(s).")
                    if metric not in allowed_metrics:
                        raise ValueError("Invalid metric.")

                    # call your server-side RPC
                    res = sb.rpc("pivot_2d_numeric", {
                        "dim1": dim1, "dim2": dim2, "metric": metric,
                        "year_from": int(y1) if y1 else None,
                        "year_to": int(y2) if y2 else None
                    }).execute()
                    df_ans = pd.DataFrame(res.data or [])
                    if not df_ans.empty:
                        df_ans = df_ans.sort_values("total_value", ascending=False).head(top_n)
                        st.dataframe(df_ans, use_container_width=True)
                        answer = f"نتیجه‌ی Pivot برای «{dim1} × {dim2}» روی «{metric}»" + (f" در بازه‌ی {y1}-{y2}" if y1 and y2 else "") + f" (Top {top_n})."
                    else:
                        answer = "هیچ نتیجه‌ای برای این Pivot پیدا نشد."

                elif plan.get("intent") == "rows":
                    filters = plan.get("filters") or {}
                    limit = int(plan.get("limit") or 200)
                    # validate and build a PostgREST filter
                    q = sb.table(TABLE).select("*")
                    for col, val in filters.items():
                        # normalize using synonyms
                        col = synonyms.get(col, col)
                        if col not in allowed_filter_cols:
                            continue
                        if isinstance(val, list):
                            # use in_ filter
                            q = q.in_(col, val)
                        else:
                            q = q.eq(col, val)
                    q = q.limit(limit)
                    res = q.execute()
                    df_ans = pd.DataFrame(res.data or [])
                    if not df_ans.empty:
                        st.dataframe(df_ans, use_container_width=True)
                        answer = f"{len(df_ans)} ردیف مطابق فیلترها نمایش داده شد (حداکثر {limit})."
                    else:
                        answer = "ردیفی مطابق شرایط پیدا نشد."

                else:
                    answer = "جهت پاسخ نیاز است مشخص کنید Pivot می‌خواهید یا ردیف‌های خام."

            except Exception as e:
                answer = f"⚠️ اجرای برنامه شکست خورد: {e}"

        if answer is None:
            answer = "سوال را واضح‌تر بپرس یا مثال بده تا Pivot یا فیلتر مناسب بسازم."

        st.chat_message("assistant").write(answer)
        st.session_state.data_chat.append({"role": "assistant", "content": answer})

