# app_enhanced_v2.py — Supabase-only, with friendly preface + advanced filters
from typing import Optional
import os, json
import pandas as pd
import streamlit as st
import streamlit_authenticator as stauth
from supabase import create_client, Client
from openai import OpenAI

# -----------------------------------------------------------
# Page config
# -----------------------------------------------------------
st.set_page_config(page_title="Pharma-GPT (v2)", layout="wide")

# -----------------------------------------------------------
# --- LOGIN (username/password) ---
# -----------------------------------------------------------
def do_login(authenticator):
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

TABLE = "Amarname_sheet1"  # ✅ fixed the typo (was Amarname_sheet1)

# -----------------------------------------------------------
# Friendly Preface (concise)
# -----------------------------------------------------------
st.title("💊 Pharma-GPT")
st.caption("Pivot like Excel. Or just ask in natural language — I’ll answer with data, tables, and charts from your Supabase.")

with st.expander("راهنمای سریع / Quick Start", expanded=True):
    st.markdown(
        """
**دو راه استفاده دارید:**
1) **Pivot**: دو بُعد + یک متریک را انتخاب کنید تا جمع‌ها را روی کل دیتابیس ببینید.  
2) **فیلتر/جدول**: با جعبه‌های جستجو، داده را بر اساس **مولکول، برند، شکل دارویی، مسیر مصرف، تامین‌کننده، سال، ATC یا تولیدی/وارداتی** فیلتر کنید، مرتب‌سازی و خروجی CSV بگیرید.  

**Chat**: به فارسی/انگلیسی بپرسید: «سهم ارزش ریالی هر شرکت طی ۱۴۰۰ تا ۱۴۰۲؟»  
**نکته ATC**: هم انتخاب دقیق (Exact) دارید هم جستجوی پیشوند (مثلاً `N06A%`).  
"""
    )

# -----------------------------------------------------------
# Tabs
# -----------------------------------------------------------
tab_pivot, tab_table, tab_chat = st.tabs(["📊 Pivot", "📋 Filter/Table", "💬 Chat"])

# -----------------------------------------------------------
# PIVOT (server-side RPC)
# -----------------------------------------------------------
with tab_pivot:
    allowed_dims = [
        "سال","کد ژنریک","نام ژنریک","مولکول دارویی","نام تجاری فرآورده",
        "شرکت تامین کننده","تولیدی/وارداتی","route","dosage form","atc code","Anatomical",
    ]
    metrics = ["ارزش ریالی", "قیمت", "تعداد تامین شده"]

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

    df_pivot = run_pivot(dim1, dim2, metric, y1, y2)
    st.caption(f"Returned {len(df_pivot)} aggregated rows.")
    st.dataframe(df_pivot, use_container_width=True)  # expected: d1, d2, total_value, rows

    if not df_pivot.empty and "total_value" in df_pivot.columns:
        # Build a label column (d1 — d2) and plot
        parts = []
        if "d1" in df_pivot.columns: parts.append(df_pivot["d1"].astype(str))
        if "d2" in df_pivot.columns: parts.append(df_pivot["d2"].astype(str))
        if parts:
            label = parts[0].fillna("")
            for s in parts[1:]:
                label = label.str.cat(s.fillna(""), sep=" — ")
        else:
            label = pd.Series([f"row {i+1}" for i in range(len(df_pivot))])
        chart_df = pd.DataFrame({"label": label, "total_value": df_pivot["total_value"]}).sort_values("total_value", ascending=False)
        st.bar_chart(chart_df.set_index("label")[["total_value"]])
    else:
        st.info("No 'total_value' column returned from the RPC, so chart is skipped.")

# -----------------------------------------------------------
# FILTER/TABLE VIEW (searchable dropdowns)
# -----------------------------------------------------------
with tab_table:
    COLS = {
        "مولکول دارویی": "مولکول دارویی",
        "نام برند": "نام تجاری فرآورده",
        "شکل دارویی": "dosage form",
        "طریقه مصرف": "route",
        "نام تامین کننده": "شرکت تامین کننده",
        "سال": "سال",
        "ATC code": "atc code",
        "وارداتی/تولید داخل": "تولیدی/وارداتی",
        # for sorting/visible extras
        "ارزش ریالی": "ارزش ریالی",
        "تعداد تامین شده": "تعداد تامین شده",
        "قیمت": "قیمت",
    }

    @st.cache_data(ttl=600)
    def get_unique(col: str, limit: int = 50000):
        r = sb.table(TABLE).select(col).neq(col, "").not_.is_(col, None).limit(limit).execute()
        vals = [row.get(col) for row in (r.data or []) if row.get(col) not in (None, "")]
        return sorted(set(vals))

    st.subheader("فیلترها")
    c1, c2 = st.columns(2)
    with c1:
        mols   = st.multiselect("مولکول دارویی", options=get_unique(COLS["مولکول دارویی"]))
        brands = st.multiselect("نام برند", options=get_unique(COLS["نام برند"]))
        forms  = st.multiselect("شکل دارویی", options=get_unique(COLS["شکل دارویی"]))
        routes = st.multiselect("طریقه مصرف", options=get_unique(COLS["طریقه مصرف"]))
    with c2:
        provs  = st.multiselect("نام تامین کننده", options=get_unique(COLS["نام تامین کننده"]))
        years  = st.multiselect("سال", options=get_unique(COLS["سال"]))
        atc_exact = st.multiselect("ATC code (Exact)", options=get_unique(COLS["ATC code"]))
        atc_prefix = st.text_input("فیلتر ATC بر اساس پیشوند (مثل N06A)", value="")

    prod_type = st.multiselect("وارداتی/تولید داخل", options=get_unique(COLS["وارداتی/تولید داخل"]))

    st.markdown("---")
    colA, colB, colC = st.columns(3)
    with colA:
        sort_by = st.selectbox(
            "مرتب‌سازی بر اساس",
            options=[COLS["ارزش ریالی"], COLS["تعداد تامین شده"], COLS["قیمت"], COLS["سال"]],
            format_func=lambda c: [k for k, v in COLS.items() if v == c][0]
        )
    with colB:
        descending = st.toggle("نزولی", value=True)
    with colC:
        limit_rows = st.number_input("حداکثر ردیف", value=20000, min_value=1000, step=1000)

    reset = st.button("بازنشانی فیلترها")
    if reset:
        st.experimental_rerun()

    @st.cache_data(ttl=300)
    def query_with_filters(
        mols, brands, forms, routes, provs, years, atc_exact, atc_prefix, prod_type, sort_by, descending, limit_rows
    ):
        q = sb.table(TABLE).select("*")

        if mols:      q = q.in_(COLS["مولکول دارویی"], mols)
        if brands:    q = q.in_(COLS["نام برند"], brands)
        if forms:     q = q.in_(COLS["شکل دارویی"], forms)
        if routes:    q = q.in_(COLS["طریقه مصرف"], routes)
        if provs:     q = q.in_(COLS["نام تامین کننده"], provs)
        if years:     q = q.in_(COLS["سال"], years)
        if prod_type: q = q.in_(COLS["وارداتی/تولید داخل"], prod_type)

        # ATC: exact has priority; otherwise prefix
        if atc_exact:
            q = q.in_(COLS["ATC code"], atc_exact)
        elif atc_prefix.strip():
            # if PostgREST supports ilike in your setup:
            q = q.ilike(COLS["ATC code"], atc_prefix.strip() + "%")

        # Server-side order where available
        q = q.order(sort_by, desc=descending).limit(int(limit_rows))
        res = q.execute()
        df = pd.DataFrame(res.data or [])

        # Client-side fallback sort
        if not df.empty and sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=not descending)

        return df

    df = query_with_filters(mols, brands, forms, routes, provs, years, atc_exact, atc_prefix, prod_type, sort_by, descending, limit_rows)
    st.markdown("### خروجی فیلتر شده")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button("دانلود CSV", df.to_csv(index=False).encode("utf-8-sig"), "filtered.csv", "text/csv")

# -----------------------------------------------------------
# DATA-AWARE CHAT (Persian/English)
# -----------------------------------------------------------
with tab_chat:
    API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    st.subheader("گفتگو با دیتابیس")

    if not API_KEY:
        st.warning("OpenAI API key not found — data chat is disabled.")
    else:
        client = OpenAI(api_key=API_KEY)

        allowed_dims = [
            "سال","کد ژنریک","نام ژنریک","مولکول دارویی","نام تجاری فرآورده",
            "شرکت تامین کننده","تولیدی/وارداتی","route","dosage form","atc code","Anatomical",
        ]
        allowed_metrics = ["ارزش ریالی","قیمت","تعداد تامین شده"]
        allowed_filter_cols = allowed_dims[:]  # same list

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

        guide = {"allowed_dims": allowed_dims, "allowed_metrics": allowed_metrics, "allowed_filters": allowed_filter_cols}

        if "data_chat" not in st.session_state:
            st.session_state.data_chat = []

        for msg in st.session_state.data_chat:
            st.chat_message(msg["role"]).write(msg["content"])

        user_q = st.chat_input("مثلاً: سهم ارزش ریالی هر شرکت در سال‌های ۱۴۰۰ تا ۱۴۰۲؟")
        if user_q:
            st.chat_message("user").write(user_q)
            st.session_state.data_chat.append({"role": "user", "content": user_q})

            system_prompt = f"""
You are a planner that outputs ONLY compact JSON (no prose). You control a data tool with:
- pivot(dim1, dim2, metric, year_from, year_to)
- rows(filters, limit)

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
{{"intent":"rows","filters":{{"ستون":"مقدار"}}, "limit": 200}}
""".strip()

            try:
                plan_resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "system", "content": system_prompt},
                              {"role": "user", "content": user_q}],
                    temperature=0,
                )
                plan_text = plan_resp.choices[0].message.content.strip()
                start = plan_text.find("{"); end = plan_text.rfind("}")
                if start == -1 or end == -1:
                    raise ValueError("No JSON plan returned.")
                plan = json.loads(plan_text[start:end+1])
            except Exception as e:
                msg = f"⚠️ برنامه‌ریز نتوانست برنامه بدهد: {e}"
                st.chat_message("assistant").write(msg)
                st.session_state.data_chat.append({"role": "assistant", "content": msg})
                plan = None

            answer = None
            if plan:
                try:
                    if plan.get("intent") == "pivot":
                        dim1 = plan.get("dim1")
                        dim2 = plan.get("dim2")
                        metric = plan.get("metric", "ارزش ریالی")
                        y1 = plan.get("year_from"); y2 = plan.get("year_to")
                        top_n = int(plan.get("top_n") or 20)

                        if dim1 not in allowed_dims or dim2 not in allowed_dims:
                            raise ValueError("Invalid dimension(s).")
                        if metric not in allowed_metrics:
                            raise ValueError("Invalid metric.")

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
                        q = sb.table(TABLE).select("*")
                        for col, val in filters.items():
                            col = synonyms.get(col, col)
                            if col not in allowed_filter_cols:
                                continue
                            if isinstance(val, list):
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
