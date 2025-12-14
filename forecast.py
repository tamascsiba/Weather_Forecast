import pandas as pd
import numpy as np
from pathlib import Path

# =========================
# Scikit-learn imports
# =========================
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# =========================
# Visualization (Plotly)
# =========================
import plotly.graph_objs as go
import plotly.io as pio


def prepare_features(df: pd.DataFrame):
    """
    Prepare input features and target variable from raw weather data.

    Steps:
    1. Combine Date and Time columns into a single datetime column.
    2. Extract useful time-based features from datetime.
    3. Remove non-numeric columns (Date, Time).
    4. Convert all remaining feature columns to numeric values.
    5. Separate Temperature as the target variable.

    Returns:
        X  : DataFrame containing model input features
        y  : Series containing target temperatures (or None if not present)
        dt : Datetime Series (used later for printing and plotting)
    """
    df = df.copy()

    # -------------------------------------------------
    # Step 1: Create a datetime column from Date + Time
    # -------------------------------------------------
    if "Date" in df.columns and "Time" in df.columns:
        # Example handled correctly: "2024-11-9" + "16:01:00"
        dt = pd.to_datetime(
            df["Date"].astype(str) + " " + df["Time"].astype(str),
            errors="coerce"
        )
    elif "Date" in df.columns:
        dt = pd.to_datetime(df["Date"], errors="coerce")
    else:
        # Fallback if no date information exists
        dt = pd.Series([pd.NaT] * len(df))

    # -------------------------------------------------
    # Step 2: Extract time-based features
    # These help the model learn seasonal and daily patterns
    # -------------------------------------------------
    df["month"] = dt.dt.month
    df["dayofyear"] = dt.dt.dayofyear
    df["hour"] = dt.dt.hour
    df["minute"] = dt.dt.minute
    df["weekday"] = dt.dt.weekday  # Monday = 0, Sunday = 6

    # -------------------------------------------------
    # Step 3: Drop original Date and Time columns
    # (They are non-numeric and cannot be used directly)
    # -------------------------------------------------
    for col in ["Date", "Time"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # -------------------------------------------------
    # Step 4: Separate target variable (Temperature)
    # -------------------------------------------------
    y = None
    if "Temperature" in df.columns:
        # Convert Temperature to numeric and coerce errors to NaN
        y = pd.to_numeric(df["Temperature"], errors="coerce")
        df = df.drop(columns=["Temperature"])

    # -------------------------------------------------
    # Step 5: Ensure all remaining features are numeric
    # This converts values like "59" or "0" into numbers
    # -------------------------------------------------
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df, y, dt


# =========================
# File paths (relative to this script)
# =========================
BASE_DIR = Path(__file__).resolve().parent
train_path = BASE_DIR / "WeatherDatas.csv"
test_path = BASE_DIR / "WeatherDatas_31.12.csv"

# =========================
# Load CSV files
# =========================
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# =========================
# Feature engineering
# =========================
X, y, _ = prepare_features(train_df)
X_new, y_new, dt_new = prepare_features(test_df)

# =========================
# Build the scikit-learn pipeline
# =========================
# Pipeline ensures that preprocessing and model training
# are always applied in the same order
model = Pipeline(steps=[
    # Replace missing values with the column mean
    ("imputer", SimpleImputer(strategy="mean")),

    # Standardize features (mean=0, std=1)
    ("scaler", StandardScaler()),

    # Random Forest regression model
    ("regressor", RandomForestRegressor(
        n_estimators=500,     # Number of trees
        random_state=42,      # Reproducibility
        n_jobs=-1             # Use all CPU cores
    ))
])

# =========================
# Train / validation split
# =========================
# Random split for basic model evaluation
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# Train the model
# =========================
model.fit(X_train, y_train)

# =========================
# Validate the model
# =========================
val_pred = model.predict(X_val)

# Mean Absolute Error (MAE)
mae = mean_absolute_error(y_val, val_pred)

# Root Mean Squared Error (RMSE)
# Calculated manually for compatibility with older sklearn versions
rmse = mean_squared_error(y_val, val_pred) ** 0.5

print(f"Validation MAE:  {mae:.2f} °C")
print(f"Validation RMSE: {rmse:.2f} °C")

# =========================
# Predict on new (future) data
# =========================
predictions = model.predict(X_new)

# =========================
# Print predictions and errors
# =========================
absolute_errors = np.abs(y_new.to_numpy() - predictions)

for i in range(len(predictions)):
    print(
        "DateTime: {}, Predicted: {:.2f} °C, Actual: {:.2f} °C, Error: {:.2f} °C"
        .format(
            dt_new.iloc[i],
            predictions[i],
            y_new.iloc[i],
            absolute_errors[i]
        )
    )

print(f"Average prediction error: {absolute_errors.mean():.2f} °C")

# =========================
# Visualization using Plotly
# =========================
fig = go.Figure()

# Predicted temperature curve
fig.add_trace(go.Scatter(
    x=dt_new,
    y=predictions,
    mode="lines+markers",
    name="Predicted Temperature"
))

# Actual temperature curve
fig.add_trace(go.Scatter(
    x=dt_new,
    y=y_new,
    mode="lines+markers",
    name="Actual Temperature"
))

fig.update_layout(
    title="Actual vs Predicted Temperature",
    xaxis_title="Date and Time",
    yaxis_title="Temperature (°C)",
    xaxis=dict(tickangle=45),
    legend=dict(x=0, y=1)
)

pio.show(fig)
