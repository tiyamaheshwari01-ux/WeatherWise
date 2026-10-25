from pathlib import Path
import sqlite3
import requests
import pandas as pd

# Project paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

CSV_PATH = DATA_DIR / "weather.csv"
DB_PATH = DATA_DIR / "weather_warehouse.db"

# Delhi weather data
CITY = "Delhi"
LATITUDE = 28.6139
LONGITUDE = 77.2090

API_URL = "https://archive-api.open-meteo.com/v1/archive"

PARAMS = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "start_date": "2015-01-01",
    "end_date": "2025-12-31",
    "daily": (
        "temperature_2m_max,temperature_2m_min,"
        "temperature_2m_mean,precipitation_sum,rain_sum,"
        "precipitation_hours,wind_speed_10m_max,"
        "relative_humidity_2m_mean,pressure_msl_mean,"
        "weather_code"
    ),
    "timezone": "Asia/Kolkata"
}


def download_weather():
    print("Downloading Delhi weather data...")

    response = requests.get(API_URL, params=PARAMS, timeout=90)
    response.raise_for_status()
    result = response.json()

    if "daily" not in result:
        raise ValueError(f"No daily weather data returned: {result}")

    df = pd.DataFrame(result["daily"])
    df = df.rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df["city"] = CITY

    numeric_cols = [
        col for col in df.columns
        if col not in ("date", "city")
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.drop_duplicates(subset=["date", "city"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    # Fill only short internal gaps
    df[numeric_cols] = df[numeric_cols].interpolate(
        limit=2, limit_area="inside"
    )

    if df.empty:
        raise ValueError("The downloaded dataset is empty.")

    df.to_csv(CSV_PATH, index=False, date_format="%Y-%m-%d")

    print(f"CSV saved to: {CSV_PATH}")
    print(f"Total weather records: {len(df)}")

    return df


def load_warehouse(df):
    print("Creating SQLite data warehouse...")

    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS Dim_Date (
                date_key INTEGER PRIMARY KEY,
                full_date TEXT UNIQUE,
                year INTEGER,
                month INTEGER,
                day INTEGER,
                season TEXT
            );

            CREATE TABLE IF NOT EXISTS Dim_Location (
                location_key INTEGER PRIMARY KEY,
                city TEXT UNIQUE,
                latitude REAL,
                longitude REAL
            );

            CREATE TABLE IF NOT EXISTS Fact_Weather (
                date_key INTEGER,
                location_key INTEGER,
                temperature_max REAL,
                temperature_min REAL,
                temperature_mean REAL,
                rainfall REAL,
                rain REAL,
                humidity REAL,
                wind_speed REAL,
                pressure REAL,
                weather_code REAL,
                PRIMARY KEY (date_key, location_key),
                FOREIGN KEY (date_key) REFERENCES Dim_Date(date_key),
                FOREIGN KEY (location_key)
                    REFERENCES Dim_Location(location_key)
            );
        """)

        conn.execute("""
            INSERT OR IGNORE INTO Dim_Location
            (city, latitude, longitude)
            VALUES (?, ?, ?)
        """, (CITY, LATITUDE, LONGITUDE))

        location_key = conn.execute(
            "SELECT location_key FROM Dim_Location WHERE city = ?",
            (CITY,)
        ).fetchone()[0]

        for _, row in df.iterrows():
            date = row["date"]
            date_key = int(date.strftime("%Y%m%d"))

            if date.month in (12, 1, 2):
                season = "Winter"
            elif date.month in (3, 4, 5):
                season = "Summer"
            elif date.month in (6, 7, 8, 9):
                season = "Monsoon"
            else:
                season = "Autumn"

            conn.execute("""
                INSERT OR IGNORE INTO Dim_Date
                (date_key, full_date, year, month, day, season)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                date_key, date.strftime("%Y-%m-%d"),
                date.year, date.month, date.day, season
            ))

            def get_value(column):
                value = row.get(column)
                return None if pd.isna(value) else float(value)

            conn.execute("""
                INSERT OR REPLACE INTO Fact_Weather
                (date_key, location_key, temperature_max,
                 temperature_min, temperature_mean, rainfall,
                 rain, humidity, wind_speed, pressure, weather_code)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                date_key, location_key,
                get_value("temperature_2m_max"),
                get_value("temperature_2m_min"),
                get_value("temperature_2m_mean"),
                get_value("precipitation_sum"),
                get_value("rain_sum"),
                get_value("relative_humidity_2m_mean"),
                get_value("wind_speed_10m_max"),
                get_value("pressure_msl_mean"),
                get_value("weather_code")
            ))

    print(f"Warehouse created at: {DB_PATH}")
    print("ETL completed successfully!")


if __name__ == "__main__":
    weather_df = download_weather()
    load_warehouse(weather_df)
