import streamlit as st
import pickle
import pandas as pd
 
# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Spend Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# ---------- CSS ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500;600&display=swap');
 
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #07080f; color: #dde1f0; }
section[data-testid="stSidebar"] {
    background: #0c0d18 !important;
    border-right: 1px solid #1c1d2e;
}
 
h1 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    font-size: 2.4rem !important;
    background: linear-gradient(120deg, #f9a8d4, #c084fc, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -1.5px;
    line-height: 1.1 !important;
}
h2, h3 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    color: #b0b4cc !important;
}
 
.section-tag {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #c084fc;
    margin-bottom: 0.6rem;
    margin-top: 1.4rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-tag::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(to right, #c084fc33, transparent);
}
 
.result-wrap {
    background: linear-gradient(135deg, #0f0a1a, #0a0f1f);
    border: 1px solid #c084fc;
    border-radius: 20px;
    padding: 2.2rem 2rem;
    text-align: center;
    margin: 1.5rem 0;
    box-shadow: 0 0 60px rgba(192, 132, 252, 0.12);
    position: relative;
    overflow: hidden;
}
.result-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #7c7f99;
    margin-bottom: 0.5rem;
}
.result-amount {
    font-family: 'Syne', sans-serif;
    font-size: 4rem;
    font-weight: 800;
    letter-spacing: -3px;
    background: linear-gradient(120deg, #f9a8d4, #c084fc, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
}
 
.mini-card {
    background: #0f1020;
    border: 1px solid #1e2035;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
    transition: border-color 0.2s;
}
.mini-card:hover { border-color: #818cf855; }
.mini-label {
    font-size: 0.68rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #5c607a;
    margin-bottom: 4px;
}
.mini-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
}
 
.badge {
    display: inline-block;
    padding: 5px 16px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-top: 0.8rem;
}
.badge-vip    { background: rgba(249,168,212,0.12); color: #f9a8d4; border: 1px solid #f9a8d455; }
.badge-high   { background: rgba(192,132,252,0.12); color: #c084fc; border: 1px solid #c084fc55; }
.badge-medium { background: rgba(129,140,248,0.12); color: #818cf8; border: 1px solid #818cf855; }
.badge-low    { background: rgba(100,116,139,0.12); color: #94a3b8; border: 1px solid #94a3b855; }
 
.spend-bar-wrap {
    background: #1a1b2e;
    border-radius: 999px;
    height: 6px;
    margin: 0.8rem auto 0.3rem;
    max-width: 300px;
    overflow: hidden;
}
.spend-bar-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(to right, #818cf8, #c084fc, #f9a8d4);
}
 
div[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(135deg, #818cf8, #c084fc) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    width: 100% !important;
    padding: 0.8rem !important;
    letter-spacing: 0.5px !important;
    margin-top: 0.5rem !important;
}
div[data-testid="stFormSubmitButton"] button:hover { opacity: 0.82 !important; }
 
hr { border-color: #1c1d2e !important; }
 
button[data-baseweb="tab"] {
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    color: #5c607a !important;
}
button[data-baseweb="tab"][aria-selected="true"] { color: #c084fc !important; }
 
.pill {
    background: #0f1020;
    border: 1px solid #1e2035;
    border-radius: 8px;
    padding: 0.55rem 0.9rem;
    font-size: 0.82rem;
    color: #7c7f99;
    margin-bottom: 0.4rem;
}
 
div[data-baseweb="select"] > div,
div[data-testid="stNumberInput"] input {
    background: #0f1020 !important;
    border-color: #1e2035 !important;
    color: #dde1f0 !important;
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)
 
 
# ---------- LOAD MODEL ----------
@st.cache_resource
def load_artifacts():
    try:
        with open("customer_spend_model.pkl", "rb") as f:
            mdl = pickle.load(f)
        with open("model_features.pkl", "rb") as f:
            feat = pickle.load(f)
        return mdl, feat
    except FileNotFoundError as e:
        return None, str(e)
 
 
model, features = load_artifacts()
 
 
def get_tier(value):
    if value >= 500:
        return ("VIP", "badge-vip", "👑")
    elif value >= 200:
        return ("High Value", "badge-high", "🔥")
    elif value >= 80:
        return ("Mid Value", "badge-medium", "📈")
    else:
        return ("Low Value", "badge-low", "💤")
 
 
# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("""
    <div style='padding: 0.8rem 0 1rem;'>
        <div style='font-family: Syne, sans-serif; font-size: 1.25rem; font-weight: 800;
                    background: linear-gradient(120deg, #f9a8d4, #c084fc, #818cf8);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;'>
            SpendAI
        </div>
        <div style='color: #5c607a; font-size: 0.78rem; margin-top: 3px;'>30-Day Spend Intelligence</div>
    </div>
    """, unsafe_allow_html=True)
 
    st.markdown("---")
    st.markdown("<div style='color:#7c7f99; font-size:0.8rem; font-weight:600; letter-spacing:1px; text-transform:uppercase; margin-bottom:0.6rem;'>How it works</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='pill'>📥 Enter customer RFM signals</div>
    <div class='pill'>🤖 ML model predicts 30-day spend</div>
    <div class='pill'>🎯 Prioritize high-value customers</div>
    """, unsafe_allow_html=True)
 
    st.markdown("---")
    st.markdown("<div style='color:#7c7f99; font-size:0.8rem; font-weight:600; letter-spacing:1px; text-transform:uppercase; margin-bottom:0.6rem;'>Model Status</div>", unsafe_allow_html=True)
    if model is not None:
        st.success("✓ Model ready")
        countries_all = [f.replace("Country_", "") for f in features if "Country_" in f]
        st.markdown(f"""
        <div class='pill'>🌍 {len(countries_all)} countries supported</div>
        <div class='pill'>📐 {len(features)} input features</div>
        """, unsafe_allow_html=True)
    else:
        st.error("Model files not found")
        st.code("customer_spend_model.pkl\nmodel_features.pkl", language="text")
 
    st.markdown("---")
    st.caption("Hackathon 2025 · Built with ❤️")
 
 
# ---------- MAIN ----------
st.markdown("# 💰 Customer Spend Predictor")
st.markdown("""
<div style='color:#5c607a; font-size:0.95rem; margin-bottom:2rem; margin-top:-0.4rem;'>
    Predict how much a customer will spend in the next <strong style='color:#c084fc'>30 days</strong>
    based on their purchase behavior
</div>
""", unsafe_allow_html=True)
 
if model is None:
    st.markdown(f"""
    <div style='background:#1a0a14; border:1px solid #f43f5e55; border-radius:14px; padding:1.4rem; color:#fda4af;'>
        ⚠️ <strong>Model files not found.</strong> Make sure
        <code style='background:#2a0a14; padding:2px 6px; border-radius:4px;'>customer_spend_model.pkl</code> and
        <code style='background:#2a0a14; padding:2px 6px; border-radius:4px;'>model_features.pkl</code>
        are in the same folder as this script.
    </div>
    """, unsafe_allow_html=True)
    st.stop()
 
countries = [f.replace("Country_", "") for f in features if "Country_" in f]
 
tab1, tab2 = st.tabs(["  🎯  Single Prediction  ", "  📂  Batch Prediction  "])
 
# ===== TAB 1 =====
with tab1:
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("<div class='section-tag'>RFM Signals</div>", unsafe_allow_html=True)
        recency         = st.number_input("Recency (days since last purchase)", min_value=0, value=30)
        frequency       = st.number_input("Frequency (total orders)", min_value=0, value=5)
        monetary        = st.number_input("Monetary Value ($)", min_value=0.0, value=250.0, step=10.0)

    with col2:
        st.markdown("<div class='section-tag'>Order Metrics</div>", unsafe_allow_html=True)
        avg_order_value   = st.number_input("Average Order Value ($)", min_value=0.0, value=50.0, step=5.0)
        total_quantity    = st.number_input("Total Quantity Ordered", min_value=0, value=20)
        customer_lifetime = st.number_input("Customer Lifetime (days)", min_value=0, value=365)

    st.markdown("<div class='section-tag'>Location</div>", unsafe_allow_html=True)
    country = st.selectbox("Country", countries if countries else ["Unknown"])

    st.markdown("<br>", unsafe_allow_html=True)
    auto_update = st.toggle("Auto-update prediction", value=True)
    run_prediction = auto_update or st.button("⚡ Predict 30-Day Spend")

    if run_prediction:
        input_data = {feat: 0 for feat in features}
        input_data["Recency"]          = recency
        input_data["Frequency"]        = frequency
        input_data["Monetary"]         = monetary
        input_data["AvgOrderValue"]    = avg_order_value
        input_data["TotalQuantity"]    = total_quantity
        input_data["CustomerLifetime"] = customer_lifetime

        country_key = "Country_" + country
        if country_key in input_data:
            input_data[country_key] = 1

        try:
            input_df   = pd.DataFrame([input_data])
            prediction = model.predict(input_df)[0]
            tier_name, tier_class, tier_icon = get_tier(prediction)
            bar_pct = min(prediction / 1000 * 100, 100)

            st.markdown(f"""
            <div class='result-wrap'>
                <div class='result-label'>Predicted 30-Day Spend</div>
                <div class='result-amount'>${prediction:,.2f}</div>
                <div class='spend-bar-wrap'>
                    <div class='spend-bar-fill' style='width:{bar_pct:.1f}%'></div>
                </div>
                <span class='badge {tier_class}'>{tier_icon} {tier_name}</span>
            </div>
            """, unsafe_allow_html=True)

            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"""
                <div class='mini-card'>
                    <div class='mini-label'>Projected Annual</div>
                    <div class='mini-value' style='color:#f9a8d4'>${prediction * 12:,.0f}</div>
                </div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class='mini-card'>
                    <div class='mini-label'>Daily Average</div>
                    <div class='mini-value' style='color:#c084fc'>${prediction / 30:,.2f}</div>
                </div>""", unsafe_allow_html=True)
            with m3:
                est_orders = max(frequency / 3, 1)
                st.markdown(f"""
                <div class='mini-card'>
                    <div class='mini-label'>Est. Per Order</div>
                    <div class='mini-value' style='color:#818cf8'>${prediction / est_orders:,.2f}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.caption("💡 Use this estimate to rank and prioritize customers for retention or upsell campaigns.")

        except Exception as e:
            st.error(f"Prediction failed — feature mismatch likely.\n\n`{e}`")
 
 
# ===== TAB 2 =====
with tab2:
    st.markdown("""
    <div style='color:#5c607a; font-size:0.9rem; margin-bottom:1.5rem;'>
        Upload a CSV with one row per customer. Column names must match training features.
        A <code>predicted_spend_30d</code> column will be appended to the output.
    </div>
    """, unsafe_allow_html=True)
 
    uploaded = st.file_uploader("Drop CSV here", type=["csv"])
 
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.markdown(f"""
            <div class='pill' style='margin-bottom:1rem;'>
                📋 <strong>{len(df):,}</strong> customers · <strong>{df.shape[1]}</strong> columns detected
            </div>
            """, unsafe_allow_html=True)
            st.dataframe(df.head(), use_container_width=True)
 
            if st.button("⚡ Run Batch Prediction"):
                with st.spinner("Running predictions..."):
                    preds = model.predict(df)
                    df["predicted_spend_30d"] = preds
 
                st.success(f"✅ Done — {len(df):,} predictions complete")
 
                c1, c2, c3 = st.columns(3)
                c1.metric("Avg Spend", f"${df['predicted_spend_30d'].mean():,.2f}")
                c2.metric("Max Spend", f"${df['predicted_spend_30d'].max():,.2f}")
                c3.metric("Total Est. Revenue", f"${df['predicted_spend_30d'].sum():,.2f}")
 
                st.dataframe(df.head(10), use_container_width=True)
 
                st.download_button(
                    "📥 Download Results CSV",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name="predicted_30d_spend.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Batch prediction failed — check column names.\n\n`{e}`")
 