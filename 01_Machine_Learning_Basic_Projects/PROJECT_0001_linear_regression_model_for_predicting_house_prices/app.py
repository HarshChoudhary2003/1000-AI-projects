import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(
    page_title="House Price Estimator",
    page_icon="🏡",
    layout="wide"
)

# Baseline market data: square feet vs sale price (in $1,000s)
DEFAULT_DATA = pd.DataFrame({
    "area_sqft": [650, 800, 950, 1100, 1250, 1400, 1550, 1700, 1850, 2000, 2250, 2500],
    "price_k":   [330, 395, 480, 545,  620,  710,  780,  845,  930,  995, 1120, 1260]
})

@st.cache_resource
def get_trained_model(df: pd.DataFrame):
    X = df[["area_sqft"]].values
    y = df["price_k"].values
    
    reg = LinearRegression()
    reg.fit(X, y)
    
    preds = reg.predict(X)
    metrics = {
        "r2": r2_score(y, preds),
        "rmse": np.sqrt(mean_squared_error(y, preds)),
        "slope": reg.coef_[0],
        "intercept": reg.intercept_
    }
    return reg, metrics

model, stats = get_trained_model(DEFAULT_DATA)

# --- Sidebar Controls ---
st.sidebar.header("Property Parameters")

user_area = st.sidebar.slider(
    "Living Area (sq. ft.)",
    min_value=500,
    max_value=3500,
    value=1650,
    step=25
)

with st.sidebar.expander("Quick Presets"):
    col_p1, col_p2 = st.columns(2)
    if col_p1.button("Starter Condo"):
        user_area = 750
    if col_p2.button("Family Home"):
        user_area = 2100

# Predict for user input
raw_pred_k = model.predict([[user_area]])[0]
est_price = raw_pred_k * 1000
price_per_sqft = est_price / user_area
lower_bound = (raw_pred_k - stats["rmse"]) * 1000
upper_bound = (raw_pred_k + stats["rmse"]) * 1000

# --- Main Dashboard ---
st.title("🏡 House Price Prediction Model")
st.caption("Interactive valuation dashboard powered by simple linear regression trained on historical sales comps.")

# Key metrics row
m1, m2, m3, m4 = st.columns(4)
m1.metric("Estimated Value", f"${est_price:,.0f}")
m2.metric("Price / Sq. Ft.", f"${price_per_sqft:.2f}")
m3.metric("Model Fit (R²)", f"{stats['r2'] * 100:.1f}%")
m4.metric("Typical Error (± RMSE)", f"±${stats['rmse'] * 1000:,.0f}")

st.divider()

# Visualization
col_chart, col_info = st.columns([2, 1])

with col_chart:
    st.subheader("Market Trend & Regression Line")
    
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=120)
    
    # Historical data points
    ax.scatter(
        DEFAULT_DATA["area_sqft"], 
        DEFAULT_DATA["price_k"] * 1000, 
        color="#2563eb", 
        s=60, 
        alpha=0.85, 
        label="Recent Sales Comps",
        zorder=3
    )
    
    # Fitted trend line
    line_x = np.linspace(500, 3500, 100).reshape(-1, 1)
    line_y = model.predict(line_x) * 1000
    ax.plot(line_x, line_y, color="#dc2626", linewidth=2, label="Fitted Regression Line", zorder=2)
    
    # Error band (RMSE confidence envelope)
    ax.fill_between(
        line_x.flatten(),
        line_y - stats["rmse"] * 1000,
        line_y + stats["rmse"] * 1000,
        color="#fee2e2",
        alpha=0.5,
        label="± 1 RMSE Confidence Range"
    )
    
    # User's selection point with dashed guide lines
    ax.scatter([user_area], [est_price], color="#16a34a", s=130, zorder=5, label=f"Selected Property ({user_area} sq.ft)")
    ax.axvline(x=user_area, color="#16a34a", linestyle="--", alpha=0.4, linewidth=1.2)
    ax.axhline(y=est_price, color="#16a34a", linestyle="--", alpha=0.4, linewidth=1.2)

    ax.set_xlabel("Area (Square Feet)", fontsize=11, fontweight="medium")
    ax.set_ylabel("Price (USD)", fontsize=11, fontweight="medium")
    ax.yaxis.set_major_formatter("${x:,.0f}")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, loc="upper left")
    
    plt.tight_layout()
    st.pyplot(fig)

with col_info:
    st.subheader("Valuation Summary")
    st.write(f"For a **{user_area:,} sq. ft.** property, the linear model projects:")
    
    st.info(
        f"**Expected Price:** ${est_price:,.0f}\n\n"
        f"**Expected Range (68% CI):**\n"
        f"${lower_bound:,.0f} — ${upper_bound:,.0f}"
    )
    
    st.subheader("Model Formula")
    sign = "+" if stats["intercept"] >= 0 else "-"
    st.latex(rf"\text{{Price}} = {stats['slope']:.3f} \times \text{{SqFt}} {sign} {abs(stats['intercept']):.2f}")
    
    st.caption(
        f"Each additional square foot adds roughly **${stats['slope'] * 1000:.2f}** to estimated property value."
    )

with st.expander("View Underlying Sales Comps"):
    st.dataframe(
        DEFAULT_DATA.rename(columns={"area_sqft": "Living Area (sq.ft)", "price_k": "Sale Price ($1000s)"}),
        width="stretch"
    )
