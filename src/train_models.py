from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score
)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "weather.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

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


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Run python src/data_pipeline.py first."
        )

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["day_of_year"] = df["date"].dt.dayofyear

    # Targets refer to the FOLLOWING day.
    df["target_temperature"] = (
        df["temperature_2m_max"].shift(-1)
    )
    df["target_rain"] = (
        df["precipitation_sum"].shift(-1) > 1.0
    ).astype(int)

    df = df.dropna(
        subset=FEATURES + ["target_temperature"]
    ).reset_index(drop=True)

    if len(df) < 100:
        raise ValueError(
            "Not enough complete records to train the models."
        )

    # Keep chronological order to avoid future-data leakage.
    split = int(len(df) * 0.80)
    train = df.iloc[:split]
    test = df.iloc[split:]

    if train["target_rain"].nunique() < 2:
        raise ValueError(
            "Training data needs both rainy and non-rainy days."
        )

    X_train = train[FEATURES]
    X_test = test[FEATURES]

    # Model 1: temperature regression
    temp_model = LinearRegression()
    temp_model.fit(X_train, train["target_temperature"])

    temp_predictions = temp_model.predict(X_test)

    mae = mean_absolute_error(
        test["target_temperature"], temp_predictions
    )
    rmse = mean_squared_error(
        test["target_temperature"], temp_predictions
    ) ** 0.5

    # Model 2: next-day rainfall classification
    rain_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )
    rain_model.fit(X_train, train["target_rain"])

    rain_predictions = rain_model.predict(X_test)

    accuracy = accuracy_score(
        test["target_rain"], rain_predictions
    )
    precision = precision_score(
        test["target_rain"], rain_predictions,
        zero_division=0
    )
    recall = recall_score(
        test["target_rain"], rain_predictions,
        zero_division=0
    )

    joblib.dump(
        temp_model, MODEL_DIR / "temperature_model.joblib"
    )
    joblib.dump(
        rain_model, MODEL_DIR / "rain_model.joblib"
    )

    metrics = {
        "temperature_mae": round(float(mae), 3),
        "temperature_rmse": round(float(rmse), 3),
        "rain_accuracy": round(float(accuracy), 4),
        "rain_precision": round(float(precision), 4),
        "rain_recall": round(float(recall), 4),
        "training_records": int(len(train)),
        "testing_records": int(len(test)),
        "features": FEATURES
    }

    with open(MODEL_DIR / "metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    print("\nMODEL TRAINING COMPLETED")
    print(json.dumps(metrics, indent=4))


if __name__ == "__main__":
    main()
