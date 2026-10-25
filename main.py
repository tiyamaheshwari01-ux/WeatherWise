from pathlib import Path
import json
import sqlite3

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="WeatherWise | Climate Intelligence",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded"
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "weather.csv"
DB_PATH = ROOT / "data" / "weather_warehouse.db"
MODEL_DIR = ROOT / "models"

FEATURES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "rain_sum",
    "precipitation_hours",
    "wind_speed_10m_max",
    "relative_humidity_2m_mean",
    "pressure_msl_mean",
    "weather_code",
    "day_of_year"
]

# ---------------- CUSTOM DESIGN ----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background:
          radial-gradient(ellipse at 10% 0%, #172b4d 0%, transparent 42%),
          linear-gradient(135deg, #080e1b 0%, #101827 55%, #0b1424 100%);
        color: #eef4ff;
    }

    [data-testid="stSidebar"] {
        background: #0b1322;
        border-right: 1px solid #24334a;
    }

    .hero {
        padding: 30px 32px;
        border: 1px solid #34435e;
        border-radius: 24px;
        background: linear-gradient(120deg, #172b4b, #1c2740 55%, #273b57);
        margin-bottom: 24px;
        box-shadow: 0 16px 45px rgba(0,0,0,.20);
    }

    .eyebrow {
        color: #8ddcff;
        letter-spacing: 3px;
        font-size: 11px;
        font-weight: 700;
    }

    .hero h1 {
        font-size: 38px;
        color: #ffffff;
        margin: 8px 0;
    }

    .hero p {
        color: #c6d5eb;
        margin-bottom: 0;
    }

    .metric-card {
        background: rgba(25, 40, 63, .82);
        border: 1px solid #34445e;
        border-radius: 18px;
        padding: 20px;
        min-height: 125px;
    }

    .metric-label {
        color: #aabbd4;
        font-size: 13px;
    }

    .metric-value {
        color: #ffffff;
        font-size: 29px;
        font-weight: 700;
        margin: 8px 0;
    }

    .metric-note {
        color: #8ddcff;
        font-size: 12px;
    }

    h2, h3 {
        color: #e9f1ff;
    }

    div[data-testid="stMetric"] {
        background: #152238;
        border: 1px solid #34445e;
        padding: 16px;
        border-radius: 15px;
    }

    div[data-testid="stMetricValue"] {
        color: #eaf6ff;
    }

    .footer {
        color: #91a4bf;
        text-align: center;
        font-size: 12px;
        padding: 25px 0 10px 0;
    }
</style>
""", unsafe_allow_html=True)


def stop_with_message(message):
    st.warning(message)
    st.stop()


# ---------------- CHECK PROJECT FILES ----------------
if not DATA_PATH.exists():
    stop_with_message(
        "Weather dataset not found. Run "
        "'python src/data_pipeline.py' first."
    )

if not (MODEL_DIR / "temperature_model.joblib").exists():
    stop_with_message(
        "Models not found. Run "
        "'python src/train_models.py' first."
    )

if not (MODEL_DIR / "rain_model.joblib").exists():
    stop_with_message(
        "Rain model missing. Run 'python src/train_models.py'."
    )

if not (MODEL_DIR / "metrics.json").exists():
    stop_with_message(
        "Metrics missing. Run 'python src/train_models.py'."
    )

df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df["day_of_year"] = df["date"].dt.dayofyear

# Exclude incomplete rows from the model demo.
df = (
    df.dropna(subset=FEATURES)
    .sort_values("date")
    .reset_index(drop=True)
)

if df.empty:
    stop_with_message("No complete weather records were found.")

temperature_model = joblib.load(
    MODEL_DIR / "temperature_model.joblib"
)
rain_model = joblib.load(MODEL_DIR / "rain_model.joblib")

with open(MODEL_DIR / "metrics.json", "r") as file:
    metrics = json.load(file)


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 🌦️ WeatherWise")
    st.caption("CLIMATE INTELLIGENCE PLATFORM")
    st.divider()

    st.markdown("### 📍 Location")
    st.info("Delhi, India")

    st.markdown("### 🧠 ML Engine")
    st.write("Linear Regression")
    st.write("Random Forest Classifier")

    st.divider()
    st.markdown("### 🗄️ Data Warehouse")
    st.write("Star-schema design")
    st.write("SQLite analytical database")

    st.divider()
    st.caption("Data: Open-Meteo historical archive")
    st.caption("Project: Data Mining & Warehousing")


# ---------------- HERO ----------------
st.markdown("""
<div class="hero">
    <div class="eyebrow">WEATHER ANALYTICS • MACHINE LEARNING</div>
    <h1>WeatherWise ☁️</h1>
    <p>
        Discover weather patterns. Explore climate trends.
        Turn historical data into intelligent predictions.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------- DATE SELECTION ----------------
st.subheader("🗓️ Explore a day")

min_date = df["date"].min().date()
max_date = df["date"].max().date()

selected_date = st.date_input(
    "Choose the historical day to analyse",
    value=max_date,
    min_value=min_date,
    max_value=max_date
)

available = df[df["date"].dt.date <= selected_date]

if available.empty:
    stop_with_message("No weather record is available for this date.")

row = available.iloc[-1]
X = row[FEATURES].to_frame().T

predicted_temp = float(temperature_model.predict(X)[0])
rain_prediction = int(rain_model.predict(X)[0])

class_positions = list(rain_model.classes_)
rain_probability = (
    float(rain_model.predict_proba(X)[0][class_positions.index(1)])
    if 1 in class_positions else 0.0
)

prediction_date = row["date"] + pd.Timedelta(days=1)

st.caption(
    f"Using weather observations from {row['date'].date()} "
    f"to estimate conditions for {prediction_date.date()}."
)

st.info(
    "This is an educational next-day model estimate based on the "
    "selected historical record, not a live official weather forecast."
)

# ---------------- PREDICTION CARDS ----------------
st.subheader("✨ Prediction snapshot")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Next-day max temperature",
        f"{predicted_temp:.1f} °C"
    )

with c2:
    st.metric(
        "Rain prediction",
        "🌧️ Rain" if rain_prediction else "☀️ No rain"
    )

with c3:
    st.metric(
        "Rain-class probability",
        f"{rain_probability * 100:.1f}%"
    )

with c4:
    st.metric(
        "Observed max temperature",
        f"{row['temperature_2m_max']:.1f} °C"
    )

# ---------------- HISTORICAL INSIGHTS ----------------
st.divider()
st.subheader("📈 Weather intelligence")

chart_data = available.tail(30).copy()

tab1, tab2, tab3 = st.tabs([
    "🌡️ Temperature",
    "🌧️ Rainfall",
    "💨 Atmospheric conditions"
])

with tab1:
    fig_temp = px.line(
        chart_data,
        x="date",
        y=[
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean"
        ],
        markers=True,
        title="Temperature trends — last 30 available days",
        labels={
            "date": "Date",
            "value": "Temperature (°C)",
            "variable": "Measurement"
        },
        template="plotly_dark"
    )
    fig_temp.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="",
        margin=dict(l=10, r=10, t=60, b=10)
    )
    st.plotly_chart(fig_temp, use_container_width=True)

with tab2:
    fig_rain = px.bar(
        chart_data,
        x="date",
        y="precipitation_sum",
        title="Daily precipitation",
        labels={
            "date": "Date",
            "precipitation_sum": "Precipitation (mm)"
        },
        template="plotly_dark"
    )
    fig_rain.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=60, b=10)
    )
    st.plotly_chart(fig_rain, use_container_width=True)

with tab3:
    fig_conditions = px.line(
        chart_data,
        x="date",
        y=[
            "relative_humidity_2m_mean",
            "wind_speed_10m_max"
        ],
        markers=True,
        title="Humidity and wind measurements",
        labels={
            "date": "Date",
            "value": "Measurement",
            "variable": "Weather variable"
        },
        template="plotly_dark"
    )
    fig_conditions.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="",
        margin=dict(l=10, r=10, t=60, b=10)
    )
    st.plotly_chart(fig_conditions, use_container_width=True)

# ---------------- MODEL PERFORMANCE ----------------
st.divider()
st.subheader("🧠 Machine-learning performance")

m1, m2, m3, m4 = st.columns(4)

m1.metric("Temperature MAE", f"{metrics['temperature_mae']:.2f} °C")
m2.metric("Temperature RMSE", f"{metrics['temperature_rmse']:.2f} °C")
m3.metric("Rain accuracy", f"{metrics['rain_accuracy'] * 100:.1f}%")
m4.metric("Rain recall", f"{metrics['rain_recall'] * 100:.1f}%")

st.caption(
    f"Chronological evaluation: "
    f"{metrics['training_records']:,} training records and "
    f"{metrics['testing_records']:,} testing records. "
    "These are historical test-set metrics, not guaranteed future accuracy."
)

# ---------------- WAREHOUSE STATUS ----------------
st.divider()
st.subheader("🗄️ Data warehouse status")

if DB_PATH.exists():
    with sqlite3.connect(DB_PATH) as conn:
        fact_count = pd.read_sql_query(
            "SELECT COUNT(*) AS n FROM Fact_Weather", conn
        )["n"].iloc[0]

        date_count = pd.read_sql_query(
            "SELECT COUNT(*) AS n FROM Dim_Date", conn
        )["n"].iloc[0]

        location_count = pd.read_sql_query(
            "SELECT COUNT(*) AS n FROM Dim_Location", conn
        )["n"].iloc[0]

    w1, w2, w3 = st.columns(3)
    w1.metric("Weather fact records", f"{fact_count:,}")
    w2.metric("Date dimension records", f"{date_count:,}")
    w3.metric("Location dimension records", f"{location_count:,}")
else:
    st.warning("Warehouse database not found. Run the ETL script.")

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
    WEATHERWISE • DATA MINING + DATA WAREHOUSING + ML<br>
    Built for academic learning and demonstration.
</div>
""", unsafe_allow_html=True)
