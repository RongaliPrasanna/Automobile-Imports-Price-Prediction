import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score




st.set_page_config(
    page_title="Automobile Imports Price Prediction",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Automobile Imports Price Prediction")
st.write("Predict the price of an automobile using Machine Learning.")



@st.cache_data
def load_data():

    columns = [
        "symboling",
        "normalized-losses",
        "make",
        "fuel-type",
        "aspiration",
        "num-of-doors",
        "body-style",
        "drive-wheels",
        "engine-location",
        "wheel-base",
        "length",
        "width",
        "height",
        "curb-weight",
        "engine-type",
        "num-of-cylinders",
        "engine-size",
        "fuel-system",
        "bore",
        "stroke",
        "compression-ratio",
        "horsepower",
        "peak-rpm",
        "city-mpg",
        "highway-mpg",
        "price"
    ]

    df = pd.read_csv(
        "auto_imports.csv",
        names=columns,
        header=None
    )

    # Replace ? with NaN
    df = df.replace("?", np.nan)

    # Convert numerical columns
    numeric_columns = [
        "symboling",
        "normalized-losses",
        "wheel-base",
        "length",
        "width",
        "height",
        "curb-weight",
        "engine-size",
        "bore",
        "stroke",
        "compression-ratio",
        "horsepower",
        "peak-rpm",
        "city-mpg",
        "highway-mpg",
        "price"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove rows where price is missing
    df = df.dropna(subset=["price"])

    # Fill numerical missing values
    numeric_features = [
        "normalized-losses",
        "wheel-base",
        "length",
        "width",
        "height",
        "curb-weight",
        "engine-size",
        "bore",
        "stroke",
        "compression-ratio",
        "horsepower",
        "peak-rpm",
        "city-mpg",
        "highway-mpg"
    ]

    for col in numeric_features:
        df[col] = df[col].fillna(df[col].median())

    # Fill categorical missing values
    categorical_features = [
        "make",
        "fuel-type",
        "aspiration",
        "num-of-doors",
        "body-style",
        "drive-wheels",
        "engine-location",
        "engine-type",
        "num-of-cylinders",
        "fuel-system"
    ]

    for col in categorical_features:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df


df = load_data()




X = df.drop("price", axis=1)
y = df["price"]

# One-hot encoding
X_encoded = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.20,
    random_state=42
)

# Scaling for Linear Regression
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)




lr_model = LinearRegression()

lr_model.fit(
    X_train_scaled,
    y_train
)

y_pred_lr = lr_model.predict(X_test_scaled)


dt_model = DecisionTreeRegressor(
    random_state=42
)

dt_model.fit(
    X_train,
    y_train
)

y_pred_dt = dt_model.predict(X_test)


rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(
    X_train,
    y_train
)

y_pred_rf = rf_model.predict(X_test)



def calculate_metrics(y_test, prediction):

    mae = mean_absolute_error(
        y_test,
        prediction
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            prediction
        )
    )

    r2 = r2_score(
        y_test,
        prediction
    )

    return mae, rmse, r2


mae_lr, rmse_lr, r2_lr = calculate_metrics(
    y_test,
    y_pred_lr
)

mae_dt, rmse_dt, r2_dt = calculate_metrics(
    y_test,
    y_pred_dt
)

mae_rf, rmse_rf, r2_rf = calculate_metrics(
    y_test,
    y_pred_rf
)




model_comparison = pd.DataFrame({

    "Model": [
        "Linear Regression",
        "Decision Tree",
        "Random Forest"
    ],

    "MAE": [
        mae_lr,
        mae_dt,
        mae_rf
    ],

    "RMSE": [
        rmse_lr,
        rmse_dt,
        rmse_rf
    ],

    "R2 Score": [
        r2_lr,
        r2_dt,
        r2_rf
    ]
})


# Select model with highest R2
best_model_row = model_comparison.loc[
    model_comparison["R2 Score"].idxmax()
]

selected_model_name = best_model_row["Model"]


if selected_model_name == "Linear Regression":

    selected_model = lr_model

elif selected_model_name == "Decision Tree":

    selected_model = dt_model

else:

    selected_model = rf_model



st.sidebar.header("Enter Automobile Details")


def number_input(label, column):

    return st.sidebar.number_input(
        label,
        value=float(df[column].median())
    )


def select_input(label, column):

    values = sorted(
        df[column].dropna().unique().tolist()
    )

    return st.sidebar.selectbox(
        label,
        values
    )




symboling = number_input(
    "Symboling",
    "symboling"
)

normalized_losses = number_input(
    "Normalized Losses",
    "normalized-losses"
)

make = select_input(
    "Make",
    "make"
)

fuel_type = select_input(
    "Fuel Type",
    "fuel-type"
)

aspiration = select_input(
    "Aspiration",
    "aspiration"
)

num_doors = select_input(
    "Number of Doors",
    "num-of-doors"
)

body_style = select_input(
    "Body Style",
    "body-style"
)

drive_wheels = select_input(
    "Drive Wheels",
    "drive-wheels"
)

engine_location = select_input(
    "Engine Location",
    "engine-location"
)

wheel_base = number_input(
    "Wheel Base",
    "wheel-base"
)

length = number_input(
    "Length",
    "length"
)

width = number_input(
    "Width",
    "width"
)

height = number_input(
    "Height",
    "height"
)

curb_weight = number_input(
    "Curb Weight",
    "curb-weight"
)

engine_type = select_input(
    "Engine Type",
    "engine-type"
)

num_cylinders = select_input(
    "Number of Cylinders",
    "num-of-cylinders"
)

engine_size = number_input(
    "Engine Size",
    "engine-size"
)

fuel_system = select_input(
    "Fuel System",
    "fuel-system"
)

bore = number_input(
    "Bore",
    "bore"
)

stroke = number_input(
    "Stroke",
    "stroke"
)

compression_ratio = number_input(
    "Compression Ratio",
    "compression-ratio"
)

horsepower = number_input(
    "Horsepower",
    "horsepower"
)

peak_rpm = number_input(
    "Peak RPM",
    "peak-rpm"
)

city_mpg = number_input(
    "City MPG",
    "city-mpg"
)

highway_mpg = number_input(
    "Highway MPG",
    "highway-mpg"
)




input_data = pd.DataFrame({

    "symboling": [symboling],

    "normalized-losses": [
        normalized_losses
    ],

    "make": [make],

    "fuel-type": [fuel_type],

    "aspiration": [aspiration],

    "num-of-doors": [num_doors],

    "body-style": [body_style],

    "drive-wheels": [drive_wheels],

    "engine-location": [
        engine_location
    ],

    "wheel-base": [wheel_base],

    "length": [length],

    "width": [width],

    "height": [height],

    "curb-weight": [
        curb_weight
    ],

    "engine-type": [engine_type],

    "num-of-cylinders": [
        num_cylinders
    ],

    "engine-size": [
        engine_size
    ],

    "fuel-system": [
        fuel_system
    ],

    "bore": [bore],

    "stroke": [stroke],

    "compression-ratio": [
        compression_ratio
    ],

    "horsepower": [
        horsepower
    ],

    "peak-rpm": [peak_rpm],

    "city-mpg": [city_mpg],

    "highway-mpg": [
        highway_mpg
    ]
})


# --------------------------------------------------
# ENCODE USER INPUT
# --------------------------------------------------

input_encoded = pd.get_dummies(
    input_data,
    drop_first=True
)

# Make columns identical to training data
input_encoded = input_encoded.reindex(
    columns=X_encoded.columns,
    fill_value=0
)




st.subheader("Prediction")

if st.button("Predict Automobile Price"):

    if selected_model_name == "Linear Regression":

        input_scaled = scaler.transform(
            input_encoded
        )

        prediction = selected_model.predict(
            input_scaled
        )

    else:

        prediction = selected_model.predict(
            input_encoded
        )

    predicted_price = prediction[0]

    st.success(
        f"Estimated Automobile Price: ${predicted_price:,.2f}"
    )

    st.info(
        f"Model Used: {selected_model_name}"
    )




st.subheader("Model Comparison")

st.dataframe(
    model_comparison,
    use_container_width=True
)

st.info(
    f"Selected Model based on highest R² Score: "
    f"{selected_model_name}"
)
