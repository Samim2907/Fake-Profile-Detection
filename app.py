import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import gdown 
# ── Load model artifacts ──────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model    = joblib.load("model.pkl")
    scaler   = joblib.load("scaler.pkl")
    features = joblib.load("features.pkl")
    return model, scaler, features

model, scaler, features = load_model()

# ── Helper: build feature vector from raw inputs ──────────────────────────────
def engineer(followers, following, posts, bio_len, profile_pic, ff_ratio):
    profile_pic = 1 if profile_pic else 0
    row = {
        "followers_count":         np.log1p(followers),
        "following_count":         np.log1p(following),
        "post_count":              np.log1p(posts),
        "bio_length":              bio_len,
        "profile_pic":             profile_pic,
        "follower_following_ratio": ff_ratio,
        "posts_per_follower":      posts / (followers + 1),
        "posts_per_following":     posts / (following + 1),
        "engagement_proxy":        followers / (posts + 1),
        "low_followers_flag":      int(followers < 50),
        "high_following_flag":     int(following > 1000),
        "suspicious_ratio_flag":   int(ff_ratio < 0.1),
        "has_bio":                 int(bio_len > 0),
        "no_posts_flag":           int(posts == 0),
        "ff_ratio_clipped":        min(ff_ratio, 100),
    }
    df = pd.DataFrame([row])[features]
    return scaler.transform(df)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Fake Profile Detector", page_icon="🕵️", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0d0f14; color: #e8eaf0; }

div[data-testid="stNumberInput"] input,
div[data-testid="stTextArea"] textarea,
div[data-testid="stSelectbox"] div {
    background: #131720 !important;
    border: 1px solid #1e2535 !important;
    color: #e8eaf0 !important;
    border-radius: 8px !important;
}
div[data-testid="stButton"] button {
    background: #63b3ed !important;
    color: #0d0f14 !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
    border: none !important;
}
div[data-testid="stSelectbox"] > div > div {
    background: #131720 !important;
    border: 1px solid #1e2535 !important;
    color: #e8eaf0 !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)
st.title("🔍 Fake Profile Detector")
st.divider()

# Sidebar
with st.sidebar:
    st.markdown("## 🔍 Model Info")
    st.markdown("""
    **Model:** Stacking Ensemble  
    **Base models:** Random Forest, Extra Trees, Gradient Boosting  
    **Meta learner:** Logistic Regression  
    **Accuracy:** ~96.98%  
    **Dataset:** 32,138 profiles  
    **Features used:** 15  
    """)
    st.divider()
    st.markdown("**Features fed to model:**")
    for f in features:
        st.caption(f"• {f}")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🧑 Manual Input", "📂 Batch CSV Upload"])

# ════════════════════════════════════════════════════════════
# TAB 1 — Manual Input
# ════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Enter Profile Details")

    col1, col2, col3 = st.columns(3)
    with col1:
        followers = st.number_input("Followers Count", min_value=0, value=200)
        following = st.number_input("Following Count", min_value=0, value=300)
    with col2:
        posts     = st.number_input("Post Count",      min_value=0, value=15)
        profile_pic = st.radio("Has Profile Picture?", ["Yes", "No"], horizontal=True) == "Yes"
    with col3:
        bio_text = st.text_area("Paste Bio (or leave blank)", height=80)
        bio_len  = len(bio_text.strip())
        st.caption(f"Bio length: {bio_len} characters")
        
    ff_ratio    = followers / (following + 1)


    if st.button("Predict", type="primary", use_container_width=True):
        X_input = engineer(followers, following, posts, bio_len, profile_pic, ff_ratio)

        with st.spinner("Analyzing profile..."):
            import time; time.sleep(2.0)
            pred  = model.predict(X_input)[0]
            proba = model.predict_proba(X_input)[0]

        st.divider()

        # ── Styled result card ──
        if pred == 1:
            st.markdown(f"""
            <div style="background:#1a0a0a;border:2px solid #c53030;border-radius:12px;padding:1.5rem;text-align:center;">
                <div style="font-size:2.5rem;">🚨</div>
                <div style="font-size:1.6rem;font-weight:700;color:#fc8181;">FAKE Profile</div>
                <div style="color:#8892a4;margin-top:0.3rem;">Confidence: {proba[1]*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background:#0a1a0f;border:2px solid #276749;border-radius:12px;padding:1.5rem;text-align:center;">
                <div style="font-size:2.5rem;">✅</div>
                <div style="font-size:1.6rem;font-weight:700;color:#68d391;">REAL Profile</div>
                <div style="color:#8892a4;margin-top:0.3rem;">Confidence: {proba[0]*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Confidence gauge ──
        st.markdown("<br>", unsafe_allow_html=True)
        import plotly.graph_objects as go

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba[1] * 100,
            title={"text": "Fake Probability %", "font": {"color": "#8892a4"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#8892a4"},
                "bar":  {"color": "#c53030" if pred == 1 else "#276749"},
                "bgcolor": "#131720",
                "steps": [
                    {"range": [0,  40], "color": "#0d1f12"},
                    {"range": [40, 70], "color": "#1a1a0a"},
                    {"range": [70, 100],"color": "#1a0a0a"},
                ],
                "threshold": {
                    "line": {"color": "#e8eaf0", "width": 2},
                    "thickness": 0.75,
                    "value": 50
                },
            },
            number={"suffix": "%", "font": {"color": "#e8eaf0"}},
        ))
        fig.update_layout(
            paper_bgcolor="#0d0f14",
            font_color="#e8eaf0",
            height=280,
            margin=dict(t=40, b=10, l=20, r=20)
        )
        st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════
# TAB 2 — Batch CSV Upload
# ════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Upload a CSV for Batch Prediction")
    st.info("CSV must have columns: `followers_count`, `following_count`, `post_count`, `bio_length`, `profile_pic`, `follower_following_ratio`")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded:
        df = pd.read_csv(uploaded)
        st.write(f"**{len(df)} profiles loaded.** Preview:")
        st.dataframe(df.head(), use_container_width=True)

        if st.button("Run Batch Prediction", type="primary", use_container_width=True):
            # Fix profile_pic
            df["profile_pic"] = df["profile_pic"].map(
                {"1": 1, "0": 0, "True": 1, "False": 0, 1: 1, 0: 0, True: 1, False: 0}
            ).fillna(0).astype(int)

            # Engineer features
            df["posts_per_follower"]    = df["post_count"] / (df["followers_count"] + 1)
            df["posts_per_following"]   = df["post_count"] / (df["following_count"] + 1)
            df["engagement_proxy"]      = df["followers_count"] / (df["post_count"] + 1)
            df["low_followers_flag"]    = (df["followers_count"] < 50).astype(int)
            df["high_following_flag"]   = (df["following_count"] > 1000).astype(int)
            df["suspicious_ratio_flag"] = (df["follower_following_ratio"] < 0.1).astype(int)
            df["has_bio"]               = (df["bio_length"] > 0).astype(int)
            df["no_posts_flag"]         = (df["post_count"] == 0).astype(int)
            df["ff_ratio_clipped"]      = df["follower_following_ratio"].clip(upper=100)
            df["followers_count"]       = np.log1p(df["followers_count"])
            df["following_count"]       = np.log1p(df["following_count"])
            df["post_count"]            = np.log1p(df["post_count"])

            X_batch = scaler.transform(df[features].fillna(0))
            preds   = model.predict(X_batch)
            probas  = model.predict_proba(X_batch)

            df["Prediction"]   = ["🚨 Fake" if p == 1 else "✅ Real" for p in preds]
            df["P(Real)"]      = (probas[:, 0] * 100).round(1).astype(str) + "%"
            df["P(Fake)"]      = (probas[:, 1] * 100).round(1).astype(str) + "%"

            st.divider()
            fake_count = sum(preds)
            real_count = len(preds) - fake_count
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Profiles", len(preds))
            c2.metric("✅ Real",  real_count)
            c3.metric("🚨 Fake",  fake_count)

            st.dataframe(df[["Prediction", "P(Real)", "P(Fake)"]].join(
                pd.read_csv(uploaded).reset_index(drop=True)), use_container_width=True)

  
            csv_out = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Results CSV", csv_out, "predictions.csv", "text/csv")