# --------------------------------------------------------------
# FINAL ULTRA PREMIUM AUTOAI APP — Luxury Gold Theme + Dark/Neon
# --------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import os
import time

# ==============================================================
# PAGE CONFIG
# ==============================================================
st.set_page_config(
    page_title="Car Price Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================
# THEME STATE (SESSION)
# ==============================================================
if "theme" not in st.session_state:
    st.session_state.theme = "gold"     # default theme


# ==============================================================
# THEME CSS DEFINITIONS
# ==============================================================
def apply_theme():
    theme = st.session_state.theme

    if theme == "gold":
        bg = "#0b0f12"
        card_glass = "rgba(255,255,255,0.03)"
        glow = "rgba(255,215,130,0.35)"
        text = "#e8ecf3"
        accent = "#ffd27d"
    elif theme == "dark":
        bg = "#111"
        card_glass = "rgba(255,255,255,0.05)"
        glow = "rgba(255,255,255,0.15)"
        text = "#ddd"
        accent = "#4fa8ff"
    else:  # neon
        bg = "#0c0f26"
        card_glass = "rgba(0,255,255,0.04)"
        glow = "rgba(0,255,255,0.35)"
        text = "#e4f2ff"
        accent = "#00f2ff"

    st.markdown(f"""
    <style>

    body {{
        background-color:{bg};
        color:{text};
        font-family:'Segoe UI',sans-serif;
    }}

    .hero {{
        background-image: linear-gradient(90deg, rgba(10,10,10,0.7), rgba(20,17,15,0.7)),
        url('https://images.pexels.com/photos/358070/pexels-photo-358070.jpeg');
        background-size:cover;
        padding:42px;
        border-radius:16px;
        box-shadow:0 12px 40px rgba(0,0,0,0.7);
    }}

    .hero-title {{
        font-size:38px; 
        font-weight:800; 
        color:{accent};
        animation:fadeIn 1s ease forwards;
    }}

    .hero-sub {{
        font-size:18px; 
        margin-top:6px; 
        animation:fadeIn 1.3s ease forwards;
    }}

    .title-underline {{
        width:0;
        height:3px;
        background:{accent};
        border-radius:3px;
        animation:underline 1s ease forwards;
    }}

    @keyframes underline {{
        from {{ width:0; }}
        to   {{ width:260px; }}
    }}

    @keyframes fadeIn {{
        from {{ opacity:0; transform:translateY(20px); }}
        to   {{ opacity:1; transform:translateY(0); }}
    }}

    @keyframes floaty {{
        0%{{transform:translateY(0)}}
        50%{{transform:translateY(-6px)}}
        100%{{transform:translateY(0)}}
    }}

    .glass-card {{
        background:{card_glass};
        border:1px solid rgba(255,255,255,0.05);
        border-radius:12px;
        padding:18px;
        box-shadow:0 10px 28px rgba(0,0,0,0.45);
    }}

    .glass-card:hover {{
        transform:translateY(-5px);
        box-shadow:0 0 20px {glow};
        transition:0.28s ease-in-out;
    }}

    .kpi {{
        background:{card_glass};
        padding:14px;
        border-radius:10px;
        text-align:center;
        margin-top:12px;
        animation:pulse 2.7s infinite;
    }}

    @keyframes pulse {{
        0%{{transform:scale(1)}}
        50%{{transform:scale(1.05)}}
        100%{{transform:scale(1)}}
    }}

    .kpi-value {{
        font-size:22px;
        font-weight:800;
        color:{accent};
    }}

    .kpi-label {{
        font-size:12px;
        color:{text};
    }}

    </style>
    """, unsafe_allow_html=True)


apply_theme()

# ==============================================================
# LOAD MODEL
# ==============================================================
MODEL_FILE = "priceprediction.pkl"
DATA_FILE = "cardekho_imputated.csv"

if not os.path.exists(MODEL_FILE):
    st.error("❌ Model file missing.")
    st.stop()

MODEL_FILE = "priceprediction.pkl"

model = joblib.load(MODEL_FILE)
FEATURES = list(model.feature_names_in_)

try:
    COEFS = model.coef_.ravel()
except:
    COEFS = None


# ==============================================================
# LOAD / SYNTHETIC DATA
# ==============================================================
def synth_data(n=400):
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "engine": rng.randint(800,3000,n),
        "mileage": rng.uniform(10,25,n),
        "predicted_price": rng.randint(200000,1500000,n),
        "fuel": rng.choice(["Petrol","Diesel","CNG","LPG"],n)
    })
    return df

if os.path.exists(DATA_FILE):
    try:
        vis_df = pd.read_csv(DATA_FILE)
        if "predicted_price" not in vis_df.columns:
            vis_df["predicted_price"] = np.random.randint(200000,1500000,len(vis_df))
    except:
        vis_df = synth_data()
else:
    vis_df = synth_data()


# ==============================================================
# SIDEBAR NAVIGATION (simple, clean)
# ==============================================================
with st.sidebar:
    st.markdown("## Navigation")
    page = st.radio("", ["Home", "Predict", "Visuals", "Settings"])
    st.markdown("---")
    user_img = st.file_uploader("Upload Car Image", type=["jpg","png","jpeg"])
    st.markdown("---")



# ==============================================================
# HOME PAGE (NO GAPS, Fully Filled)
# ==============================================================
if page == "Home":

    # HERO
    st.markdown("<div class='hero'>", unsafe_allow_html=True)

    h1, h2 = st.columns([1.6, 1])

    with h1:
        st.markdown("<div class='hero-title'>Car Price Analysis Dashboard</div>", unsafe_allow_html=True)
        st.markdown("<div class='title-underline'></div>", unsafe_allow_html=True)
        st.markdown("<div class='hero-sub'>A premium dashboard for car price prediction, insights, and interactive analytics.</div>", unsafe_allow_html=True)
        st.subheader("How the Model Works")
    st.write("""
    - The model was trained on historical automobile data with both numeric and categorical variables.  
    - Categorical features like **fuel type**, **seller type**, and **transmission mode** were converted using one-hot encoding.  
    - Numerical variables including **vehicle age**, **engine CC**, **mileage**, **km driven**, **max power**, and **seats** were used directly.  
    - The regression learns optimal **feature coefficients** and an **intercept** that minimize prediction error.
    """)


    with h2:
        imgs = [
            "https://images.pexels.com/photos/170811/pexels-photo-170811.jpeg",
            "https://images.pexels.com/photos/210019/pexels-photo-210019.jpeg"
        ]
        for i, img in enumerate(imgs):
            st.markdown(
                f"<img src='{img}' style='width:250px;border-radius:10px;margin-bottom:10px;animation:fadeIn 1s ease {i/3}s forwards, floaty 4s infinite;'>",
                unsafe_allow_html=True
            )

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 3 CARDS — fills the middle space
    c1, c2, c3 = st.columns(3)

    # Card 1
    with c1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.image("https://images.pexels.com/photos/358070/pexels-photo-358070.jpeg", use_container_width=True)
        st.subheader("Vehicle Insights")
        st.write("Explore key vehicle metrics like mileage, engine capacity, power and more.")
        avg_m = round(vis_df["mileage"].mean(),1)
        st.markdown(f"<div class='kpi'><div class='kpi-value'>{avg_m} kmpl</div><div class='kpi-label'>Avg Mileage</div></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Card 2
    with c2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.image("https://images.pexels.com/photos/170811/pexels-photo-170811.jpeg", use_container_width=True)
        st.subheader("Pricing Trends")
        st.write("View market pricing patterns across different attributes.")
        avg_p = int(vis_df["predicted_price"].mean())
        st.markdown(f"<div class='kpi'><div class='kpi-value'>₹ {avg_p:,}</div><div class='kpi-label'>Avg Price</div></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Card 3
    with c3:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.image("https://images.pexels.com/photos/305070/pexels-photo-305070.jpeg", use_container_width=True)
        st.subheader("Feature Behaviors")
        st.write("See how attributes influence car value mathematically.")
        st.markdown(f"<div class='kpi'><div class='kpi-value'>{len(FEATURES)}</div><div class='kpi-label'>Features</div></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)



# ==============================================================
# PREDICT PAGE
# ==============================================================
elif page == "Predict":
    st.title("Predict Car Price")

    left, right = st.columns([1.4,1])

    with left:
        st.subheader("Enter Car Details")
        vehicle_age = st.number_input("Vehicle Age",0.0,30.0,5.0)
        km = st.number_input("Kilometers Driven",0.0,400000.0,60000.0)
        mileage = st.number_input("Mileage",5.0,40.0,18.0)
        engine = st.number_input("Engine CC",600,6000,1200)
        power = st.number_input("Max Power",20.0,400.0,70.0)
        seats = st.number_input("Seats",2,10,5)

        fuel = st.selectbox("Fuel Type",["Petrol","Diesel","CNG","LPG","Electric"])
        seller = st.selectbox("Seller Type",["Individual","Dealer","Trustmark Dealer"])
        trans = st.selectbox("Transmission",["Manual","Automatic"])

        btn = st.button("Predict Price")

    with right:
        st.subheader("Preview")
        if user_img:
            st.image(user_img, use_container_width=True)
        else:
            st.image("https://images.pexels.com/photos/170811/pexels-photo-170811.jpeg", use_container_width=True)

        out = st.empty()
        g = st.empty()

        if btn:
            X = {f:0 for f in FEATURES}

            vals = {
                "vehicle_age":vehicle_age,
                "km_driven":km,
                "mileage":mileage,
                "engine":engine,
                "max_power":power,
                "seats":seats
            }
            for k,v in vals.items():
                if k in X:
                    X[k]=v

            for f in ["CNG","Diesel","Electric","LPG","Petrol"]:
                col=f"fuel_type_{f}"
                if col in X:
                    X[col]=(fuel==f)

            for s in ["Individual","Dealer","Trustmark Dealer"]:
                col=f"seller_type_{s}"
                if col in X:
                    X[col]=(seller==s)

            for t in ["Manual","Automatic"]:
                col=f"transmission_type_{t}"
                if col in X:
                    X[col]=(trans==t)

            row = pd.DataFrame([X],columns=FEATURES)
            pred = float(model.predict(row)[0])
            pred=max(pred,0)

            for i in range(1,21):
                out.markdown(f"<h3 style='color:#ffd27d;'>Estimated Price: ₹ {pred*(i/20):,.0f}</h3>",unsafe_allow_html=True)
                time.sleep(0.01)

            out.markdown(f"<h1 style='color:#ffb347;'>₹ {pred:,.0f}</h1>",unsafe_allow_html=True)

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred,
                number={"prefix":"₹ "},
                gauge={"axis":{"range":[0,pred*1.6]},
                       "bar":{"color":"#ffd27d"}}
            ))
            g.plotly_chart(fig, use_container_width=True)




# ==============================================================
# VISUALS PAGE
# ==============================================================
elif page == "Visuals":
    st.title("Data Visualizations")

    a,b = st.columns(2)
    with a:
        st.subheader("Engine vs Price")
        fig=px.scatter(vis_df,x="engine",y="predicted_price",color="fuel",trendline="ols")
        st.plotly_chart(fig,use_container_width=True)

    with b:
        st.subheader("Mileage vs Price")
        fig2=px.scatter(vis_df,x="mileage",y="predicted_price",color="fuel",trendline="ols")
        st.plotly_chart(fig2,use_container_width=True)

    st.subheader("Correlation Heatmap")
    numeric = vis_df.select_dtypes(include=["int64","float64"])
    if numeric.shape[1] > 1:
        fig3, ax = plt.subplots(figsize=(8,4))
        sns.heatmap(numeric.corr(), cmap="RdBu_r", annot=False, ax=ax)
        st.pyplot(fig3)
    else:
        st.info("Not enough numeric columns.")



# ==============================================================
# SETTINGS PAGE (THEME SWITCH + ABOUT)
# ==============================================================
elif page == "Settings":

    st.title("Settings & About")

    st.subheader("🎨 Theme")
    theme_choice = st.selectbox("Choose Theme", ["Luxury Gold","Dark Mode","Neon Blue"])
    if theme_choice == "Luxury Gold":
        st.session_state.theme = "gold"
    elif theme_choice == "Dark Mode":
        st.session_state.theme = "dark"
    else:
        st.session_state.theme = "neon"

    st.info("Theme will refresh when you navigate to another page.")

    st.markdown("---")

    st.subheader("ℹ️ About This App")
    st.write("""
    This dashboard predicts car prices using a trained Multiple Linear Regression model.
    It includes interactive analytics, visuals, animations, and customizable themes.
    """)

    st.subheader("👩🏻‍💻 Developer")
    st.write("""
    **Developed by:** Chetana Parakh  
    **Focus Areas:** Machine Learning, UI/UX, Software Engineering  
    **Tech Used:** Streamlit, Python, Linear Regression, Plotly  
    """)

    st.subheader("🔖 Version")
    st.write("v2.0 — Ultra Premium Edition")

