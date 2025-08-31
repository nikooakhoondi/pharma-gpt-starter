# app_enhanced_v2.py — Supabase-only: Pivot + Filter/Table + GPT Chat (fixed indentation)
from typing import Optional
import os, json
import pandas as pd
import streamlit as st
import streamlit_authenticator as stauth
from supabase import create_client, Client
from openai import OpenAI

# ---------------------------- Page config ----------------------------
st.set_page_config(page_title="Pharma-GPT (Supabase)", layout="wide")

# ---------------------------- Auth helpers ----------------------------
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
    cookie_key  = str(AUTH.get("cookie_key", "change-me"))
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

# ---------------------------- Supabase ----------------------------
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

# ---------------------------- Cached OpenAI client ----------------------------
@st.cache_resource
def get_openai_client():
    key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    return OpenAI(api_key=key)

TABLE = "Amarname_sheet1"  # keep exactly as user requested

# ---------------------------- Shared constants ----------------------------
ALLOWED_DIMS = [
    "سال","کد ژنریک","نام ژنریک","مولکول دارویی","نام تجاری فرآورده",
    "شرکت تامین کننده","تولیدی/وارداتی","route","dosage form","atc code","Anatomical",
]
ALLOWED_METRICS = ["ارزش ریالی", "قیمت", "تعداد تامین شده"]

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

SYNONYMS = {
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

# ---------------------------- Cached helpers (TOP-LEVEL) ----------------------------
@st.cache_data(ttl=300)
def run_pivot_rpc(dim1: str, dim2: str, metric: str, y1: int, y2: int) -> pd.DataFrame:
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

@st.cache_data(ttl=1800)  # cache 30 minutes
def get_unique(col: str, limit: int = 20000):
    """
    Ultra-robust distinct fetch for problematic column names (spaces, slashes, non-ASCII).
    Strategy: pull a page with select('*') then dedupe client-side.
    """
    try:
        r = sb.table(TABLE).select("*").limit(limit).execute()
    except Exception as e:
        st.error(f"Supabase select failed for uniques on '{col}': {e}")
        return []

    data = r.data or []
    vals = []
    for row in data:
        v = row.get(col)
        if v is None:
            continue
        if isinstance(v, str) and v.strip() == "":
            continue
        vals.append(v)

    try:
        return sorted(set(vals))
    except TypeError:
        return sorted({str(v) for v in vals})

@st.cache_data(ttl=600)
def query_with_filters(
    mols, brands, forms, routes, provs, years, atc_exact, atc_prefix, prod_type,
    sort_by, descending, limit_rows
):
    q = sb.table(TABLE).select("*")

    if mols:      q = q.in_(COLS["مولکول دارویی"], mols)
    if brands:    q = q.in_(COLS["نام برند"], brands)
    if forms:     q = q.in_(COLS["شکل دارویی"], forms)
    if routes:    q = q.in_(COLS["طریقه مصرف"], routes)
    if provs:     q = q.in_(COLS["نام تامین کننده"], provs)
    if years:     q = q.in_(COLS["سال"], years)
    if prod_type: q = q.in_(COLS["وارداتی/تولید داخل"], prod_type)

    # ATC: exact first; else prefix
    if atc_exact:
        q = q.in_(COLS["ATC code"], atc_exact)
    elif atc_prefix and atc_prefix.strip():
        try:
            q = q.ilike(COLS["ATC code"], atc_prefix.strip() + "%")
        except Exception:
            q = q.like(COLS["ATC code"], atc_prefix.strip() + "%")

    # Avoid server-side order() on non-ASCII names; sort client-side instead
    try:
        res = q.limit(int(limit_rows)).execute()
    except Exception as e:
        st.error(f"Supabase query failed: {e}")
        return pd.DataFrame()

    df = pd.DataFrame(res.data or [])
    if not df.empty and sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=not descending, kind="mergesort")
    return df

# ---------------------------- Preface ----------------------------
st.title("💊 Pharma-GPT")
st.caption("Pivot like Excel — or ask in natural language. Results come from your Supabase table: Amarname_sheet1.")

with st.expander("راهنمای سریع / Quick Start", expanded=True):
    st.markdown(
        """
**دو مسیر دارید:**
1) **Pivot**: دو بُعد + یک متریک را انتخاب کنید تا جمع‌ها را روی کل دیتابیس ببینید.  
2) **فیلتر/جدول**: با جعبه‌های جستجو، داده را بر اساس **مولکول، برند، شکل دارویی، مسیر مصرف، تامین‌کننده، سال، ATC یا تولیدی/وارداتی** فیلتر و مرتب کنید و CSV بگیرید.  

**Chat**: به فارسی/انگلیسی بپرسید (مثلاً: «سهم ارزش ریالی هر شرکت در ۱۴۰۰–۱۴۰۲؟») تا خروجی جدول/چارت بگیرید.  
**نکته ATC**: هم انتخاب دقیق دارید، هم پیشوند (مثل `N06A%`).  
"""
    )

# ---------------------------- Auth gate ----------------------------
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

# ---------------------------- Tabs ----------------------------
tab_table, tab_chat = st.tabs(["📋 Filter/Table", "💬 Chat"])

# ============================ FILTER / TABLE ============================
with tab_table:
    st.subheader("فیلترها")

    # ---- Debounced filter form ----
    with st.form("filters_form", clear_on_submit=False):
        c1, c2 = st.columns(2)
        with c1:
            mols = st.multiselect("مولکول دارویی", options=get_unique(COLS["مولکول دارویی"]))
            brands = st.multiselect("نام برند", options=get_unique(COLS["نام برند"]))
            forms = st.multiselect("شکل دارویی", options=get_unique(COLS["شکل دارویی"]))
            routes = st.multiselect("طریقه مصرف", options=get_unique(COLS["طریقه مصرف"]))
        with c2:
            provs = st.multiselect("نام تامین کننده", options=get_unique(COLS["نام تامین کننده"]))
            years = st.multiselect("سال", options=get_unique(COLS["سال"]))
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

        applied = st.form_submit_button("اعمال فیلترها")

    # initialise once
    if "filters_last" not in st.session_state:
        st.session_state.filters_last = None

    # Build a signature of current filters to know if anything changed
    signature = (
        tuple(sorted(mols)), tuple(sorted(brands)), tuple(sorted(forms)), tuple(sorted(routes)),
        tuple(sorted(provs)), tuple(sorted(years)), tuple(sorted(atc_exact)),
        (atc_prefix or "").strip(), tuple(sorted(prod_type)), sort_by, bool(descending), int(limit_rows)
    )

    # Only run the DB query if user pressed "اعمال فیلترها" OR first load OR filters actually changed
    should_query = applied or (st.session_state.filters_last is None) or (st.session_state.filters_last != signature)
    if should_query:
        df = query_with_filters(
            mols, brands, forms, routes, provs, years, atc_exact, atc_prefix, prod_type,
            sort_by, descending, limit_rows
        )
        st.session_state.filters_last = signature
        st.session_state.filtered_df = df
    else:
        df = st.session_state.get("filtered_df", pd.DataFrame())

    st.markdown("### خروجی فیلتر شده")
    st.dataframe(df, use_container_width=True, hide_index=True)
    if not df.empty:
        st.download_button(
            label="دانلود CSV",
            data=df.to_csv(index=False).encode("utf-8-sig"),
            file_name="filtered.csv",
            mime="text/csv",
            key="download_csv_filtered"  # 👈 unique key avoids DuplicateWidgetID
        )

    # ---- Pivot-like chart from filtered rows (unchanged behaviour) ----
    st.markdown("---")
    st.subheader("نمودار از داده‌های فیلتر شده")

    agg_dims_all = [
        "سال","کد ژنریک","نام ژنریک","مولکول دارویی","نام تجاری فرآورده",
        "شرکت تامین کننده","تولیدی/وارداتی","route","dosage form","atc code","Anatomical",
    ]
    agg_metric_all = ["ارزش ریالی", "قیمت", "تعداد تامین شده"]

    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        agg_dim1 = st.selectbox("بعد اول (Dimension 1)", agg_dims_all, index=0, key="agg_dim1")
    with cc2:
        agg_dim2_sel = st.selectbox("بعد دوم (اختیاری)", ["— هیچ —"] + agg_dims_all, index=0, key="agg_dim2")
        agg_dim2 = None if agg_dim2_sel == "— هیچ —" else agg_dim2_sel
    with cc3:
        agg_metric = st.selectbox("متریک (مجموع)", agg_metric_all, index=0, key="agg_metric")

    if df.empty:
        st.info("پس از اعمال فیلترها، داده‌ای برای تجمیع وجود ندارد.")
    else:
        missing = [c for c in [agg_dim1, agg_dim2, agg_metric] if c and c not in df.columns]
        if missing:
            st.warning(f"ستون‌های مورد نیاز در خروجی یافت نشد: {missing}")
        else:
            group_cols = [c for c in [agg_dim1, agg_dim2] if c]
            try:
                g = df.groupby(group_cols, dropna=False)[agg_metric].sum().reset_index()
            except Exception:
                tmp = df.copy()
                for c in group_cols:
                    tmp[c] = tmp[c].astype(str)
                g = tmp.groupby(group_cols, dropna=False)[agg_metric].sum().reset_index()

            label = g[agg_dim1].astype(str).fillna("")
            if agg_dim2:
                label = label + " — " + g[agg_dim2].astype(str).fillna("")
            chart_df = pd.DataFrame({"label": label, "total_value": g[agg_metric]}).sort_values("total_value", ascending=False)
            st.bar_chart(chart_df.set_index("label")[["total_value"]])
            st.caption(f"ردیف‌های تجمیع‌شده: {len(g)}  |  ستون تجمیع: {agg_metric}")


    # ---- Run filtered query ----
    df = query_with_filters(
        mols, brands, forms, routes, provs, years, atc_exact, atc_prefix, prod_type,
        sort_by, descending, limit_rows
    )

    st.markdown("### خروجی فیلتر شده")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button("دانلود CSV", df.to_csv(index=False).encode("utf-8-sig"), "filtered.csv", "text/csv")

    # ---- Aggregation & Chart (Pivot-like, computed client-side from filtered rows) ----
    st.markdown("---")
    st.subheader("نمودار از داده‌های فیلتر شده")

    # Choose dimensions & metric for the chart (same idea as the old Pivot tab)
    agg_dims_all = [
        "سال","کد ژنریک","نام ژنریک","مولکول دارویی","نام تجاری فرآورده",
        "شرکت تامین کننده","تولیدی/وارداتی","route","dosage form","atc code","Anatomical",
    ]
    agg_metric_all = ["ارزش ریالی", "قیمت", "تعداد تامین شده"]

    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        agg_dim1 = st.selectbox("بعد اول (Dimension 1)", agg_dims_all, index=0)
    with cc2:
        agg_dim2 = st.selectbox("بعد دوم (اختیاری)", ["— هیچ —"] + agg_dims_all, index=0)
        agg_dim2 = None if agg_dim2 == "— هیچ —" else agg_dim2
    with cc3:
        agg_metric = st.selectbox("متریک (مجموع)", agg_metric_all, index=0)

    if df.empty:
        st.info("پس از اعمال فیلترها، داده‌ای برای تجمیع وجود ندارد.")
    else:
        missing_cols = [c for c in [agg_dim1, agg_dim2, agg_metric] if c and c not in df.columns]
        if missing_cols:
            st.warning(f"ستون‌های مورد نیاز در خروجی یافت نشد: {missing_cols}")
        else:
            # Group & sum
            group_cols = [c for c in [agg_dim1, agg_dim2] if c]
            try:
                g = df.groupby(group_cols, dropna=False)[agg_metric].sum().reset_index()
            except Exception:
                # In case types are mixed, coerce to string for grouping dims
                tmp = df.copy()
                for c in group_cols:
                    tmp[c] = tmp[c].astype(str)
                g = tmp.groupby(group_cols, dropna=False)[agg_metric].sum().reset_index()

            # Build a label column similar to the previous pivot chart
            if agg_dim2:
                label = g[agg_dim1].astype(str).fillna("") + " — " + g[agg_dim2].astype(str).fillna("")
            else:
                label = g[agg_dim1].astype(str).fillna("")

            chart_df = (
                pd.DataFrame({"label": label, "total_value": g[agg_metric]})
                .sort_values("total_value", ascending=False)
            )

            st.bar_chart(chart_df.set_index("label")[["total_value"]])
            st.caption(f"ردیف‌های تجمیع‌شده: {len(g)}  |  ستون تجمیع: {agg_metric}")


# ============================ GPT DATA CHAT ============================
with tab_chat:
    st.subheader("گفتگو با دیتابیس")

    # 1) Get cached OpenAI client (from Patch 3.1)
    client = get_openai_client()
    if not client:
        st.warning("OpenAI API key not found — data chat is disabled.")
    else:
        # 2) Safe session init
        if "data_chat" not in st.session_state:
            st.session_state.data_chat = []

        # 3) Render history safely
        for msg in st.session_state.data_chat:
            role = msg.get("role", "assistant")
            content = msg.get("content", "")
            try:
                st.chat_message(role).write(content)
            except Exception:
                # Never let a broken history item crash the app
                st.chat_message("assistant").write("⚠️ پیام قبلی قابل نمایش نیست (خطای قالب).")

        # 4) Chat input
        user_q = st.chat_input("مثلاً: سهم ارزش ریالی هر شرکت در سال‌های ۱۴۰۰ تا ۱۴۰۲؟", key="data_chat_input")
        if user_q:
            # echo user
            st.chat_message("user").write(user_q)
            st.session_state.data_chat.append({"role": "user", "content": user_q})

            # -------- Planner prompt (unchanged idea, made safer) --------
            guide = {"allowed_dims": ALLOWED_DIMS, "allowed_metrics": ALLOWED_METRICS, "allowed_filters": ALLOWED_DIMS}
            system_prompt = f"""
You are a planner that outputs ONLY compact JSON (no prose). You control a data tool with:
- pivot(dim1, dim2, metric, year_from, year_to)
- rows(filters, limit)

Rules:
- Use Persian or English inputs.
- If the user asks for shares or totals by categories, use "pivot".
- If the user wants raw examples/records, use "rows".
- Respect year ranges if mentioned; otherwise leave them null.
- If user gives synonyms, normalize using this map: {SYNONYMS}
- Keep JSON small; do not include analysis, only fields below.

Allowed:
{json.dumps(guide, ensure_ascii=False)}

Output schema (one of these):
{{"intent":"pivot","dim1":"...", "dim2":"...", "metric":"...", "year_from":1400, "year_to":1404, "top_n":10}}
OR
{{"intent":"rows","filters":{{"ستون":"مقدار"}}, "limit": 200}}
""".strip()

            # 6) Execute the plan safely and show results
            answer = None
            try:
                intent = (plan.get("intent") or "").lower()

                if intent == "pivot":
                    d1 = plan.get("dim1")
                    d2 = plan.get("dim2")
                    metric = plan.get("metric", "ارزش ریالی")
                    y1 = plan.get("year_from")
                    y2 = plan.get("year_to")
                    top_n = int(plan.get("top_n") or 20)

                    # validate
                    if d1 not in ALLOWED_DIMS or d2 not in ALLOWED_DIMS:
                        raise ValueError("Invalid dimension(s).")
                    if metric not in ALLOWED_METRICS:
                        raise ValueError("Invalid metric.")

                    # default year range if missing
                    if not y1 or not y2:
                        try:
                            yr_min = sb.table(TABLE).select("سال").order("سال").limit(1).execute()
                            min_year = int(yr_min.data[0]["سال"]) if yr_min.data else 1300
                            yr_max = sb.table(TABLE).select("سال").order("سال", desc=True).limit(1).execute()
                            max_year = int(yr_max.data[0]["سال"]) if yr_max.data else 1500
                        except Exception:
                            min_year, max_year = 1300, 1500
                        y1 = int(y1) if y1 else min_year
                        y2 = int(y2) if y2 else max_year

                    # RPC
                    res = sb.rpc(
                        "pivot_2d_numeric",
                        {"dim1": d1, "dim2": d2, "metric": metric, "year_from": int(y1), "year_to": int(y2)}
                    ).execute()

                    df_ans = pd.DataFrame(res.data or [])
                    if not df_ans.empty:
                        # show table
                        df_ans = df_ans.sort_values("total_value", ascending=False).head(top_n)
                        st.dataframe(df_ans, use_container_width=True)

                        # quick bar chart
                        parts = []
                        if "d1" in df_ans.columns:
                            parts.append(df_ans["d1"].astype(str))
                        if "d2" in df_ans.columns:
                            parts.append(df_ans["d2"].astype(str))
                        if parts:
                            label = parts[0].fillna("")
                            for s in parts[1:]:
                                label = label.str.cat(s.fillna(""), sep=" — ")
                        else:
                            label = pd.Series([f"row {i+1}" for i in range(len(df_ans))])
                        chart_df = pd.DataFrame({"label": label, "total_value": df_ans["total_value"]}).sort_values("total_value", ascending=False)
                        st.bar_chart(chart_df.set_index("label")[["total_value"]])

                        answer = f"Pivot «{d1} × {d2}» روی «{metric}» در بازهٔ {y1}–{y2} (Top {top_n})."
                    else:
                        answer = f"هیچ نتیجه‌ای برای Pivot «{d1} × {d2}» روی «{metric}» در بازهٔ {y1}–{y2} پیدا نشد."

                elif intent == "rows":
                    filters = plan.get("filters") or {}
                    limit = int(plan.get("limit") or 200)

                    q = sb.table(TABLE).select("*")
                    for col, val in filters.items():
                        col = SYNONYMS.get(col, col)
                        if col not in ALLOWED_DIMS:
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
                # <-- this 'except' closes the outer try and fixes the SyntaxError
                answer = f"⚠️ اجرای برنامه شکست خورد: {e}"

                         # 6) Execute the plan safely and show results
            answer = None
            try:
                intent = (plan.get("intent") or "").lower()

                if intent == "pivot":
                    d1 = plan.get("dim1")
                    d2 = plan.get("dim2")
                    metric = plan.get("metric", "ارزش ریالی")
                    y1 = plan.get("year_from")
                    y2 = plan.get("year_to")
                    top_n = int(plan.get("top_n") or 20)

                    # validate
                    if d1 not in ALLOWED_DIMS or d2 not in ALLOWED_DIMS:
                        raise ValueError("Invalid dimension(s).")
                    if metric not in ALLOWED_METRICS:
                        raise ValueError("Invalid metric.")

                    # default year range if missing
                    if not y1 or not y2:
                        try:
                            yr_min = sb.table(TABLE).select("سال").order("سال").limit(1).execute()
                            min_year = int(yr_min.data[0]["سال"]) if yr_min.data else 1300
                            yr_max = sb.table(TABLE).select("سال").order("سال", desc=True).limit(1).execute()
                            max_year = int(yr_max.data[0]["سال"]) if yr_max.data else 1500
                        except Exception:
                            min_year, max_year = 1300, 1500
                        y1 = int(y1) if y1 else min_year
                        y2 = int(y2) if y2 else max_year

                    # RPC
                    res = sb.rpc(
                        "pivot_2d_numeric",
                        {"dim1": d1, "dim2": d2, "metric": metric, "year_from": int(y1), "year_to": int(y2)}
                    ).execute()

                    df_ans = pd.DataFrame(res.data or [])
                    if not df_ans.empty:
                        # show table
                        df_ans = df_ans.sort_values("total_value", ascending=False).head(top_n)
                        st.dataframe(df_ans, use_container_width=True)

                        # quick bar chart
                        parts = []
                        if "d1" in df_ans.columns:
                            parts.append(df_ans["d1"].astype(str))
                        if "d2" in df_ans.columns:
                            parts.append(df_ans["d2"].astype(str))
                        if parts:
                            label = parts[0].fillna("")
                            for s in parts[1:]:
                                label = label.str.cat(s.fillna(""), sep=" — ")
                        else:
                            label = pd.Series([f"row {i+1}" for i in range(len(df_ans))])
                        chart_df = pd.DataFrame({"label": label, "total_value": df_ans["total_value"]}).sort_values("total_value", ascending=False)
                        st.bar_chart(chart_df.set_index("label")[["total_value"]])

                        answer = f"Pivot «{d1} × {d2}» روی «{metric}» در بازهٔ {y1}–{y2} (Top {top_n})."
                    else:
                        answer = f"هیچ نتیجه‌ای برای Pivot «{d1} × {d2}» روی «{metric}» در بازهٔ {y1}–{y2} پیدا نشد."

                elif intent == "rows":
                    filters = plan.get("filters") or {}
                    limit = int(plan.get("limit") or 200)

                    q = sb.table(TABLE).select("*")
                    for col, val in filters.items():
                        col = SYNONYMS.get(col, col)
                        if col not in ALLOWED_DIMS:
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
                # <-- this 'except' closes the outer try and fixes the SyntaxError
                answer = f"⚠️ اجرای برنامه شکست خورد: {e}"


                # 7) Always render an assistant message (never None)
                if not answer:
                    answer = "سوال را واضح‌تر بپرس یا مثال بده تا Pivot یا فیلتر مناسب بسازم."

                st.chat_message("assistant").write(answer)
                st.session_state.data_chat.append({"role": "assistant", "content": answer})
