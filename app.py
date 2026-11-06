import streamlit as st
import pandas as pd
import requests
from io import BytesIO

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(page_title="Hotel Booking Cancellation Predictor", page_icon="🏨")

st.title("🏨 Hotel Booking Cancellation Predictor")
st.write("Predict whether a hotel booking is likely to be cancelled.")

DATA_URL = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv"

FEATURES = [
    "hotel", "lead_time", "arrival_date_year", "arrival_date_month",
    "stays_in_weekend_nights", "stays_in_week_nights", "adults",
    "children", "babies", "meal", "market_segment",
    "distribution_channel", "is_repeated_guest",
    "previous_cancellations", "previous_bookings_not_canceled",
    "reserved_room_type", "deposit_type", "customer_type", "adr",
    "required_car_parking_spaces", "total_of_special_requests"
]

NUMERIC = [
    "lead_time", "arrival_date_year", "stays_in_weekend_nights",
    "stays_in_week_nights", "adults", "children", "babies",
    "is_repeated_guest", "previous_cancellations",
    "previous_bookings_not_canceled", "adr",
    "required_car_parking_spaces", "total_of_special_requests"
]

CATEGORICAL = [c for c in FEATURES if c not in NUMERIC]


@st.cache_data
def load_data():
    response = requests.get(DATA_URL, timeout=90)
    response.raise_for_status()
    df = pd.read_csv(BytesIO(response.content), low_memory=False)

    required = FEATURES + ["is_canceled"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing dataset columns: {missing}")

    df = df[required].copy()
    df["is_canceled"] = pd.to_numeric(df["is_canceled"], errors="coerce")
    df = df.dropna(subset=["is_canceled"])
    df["is_canceled"] = df["is_canceled"].astype(int)

    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in CATEGORICAL:
        df[col] = df[col].fillna("Unknown").astype(str)

    return df


@st.cache_resource
def train_model():
    df = load_data()

    if len(df) > 30000:
        df, _ = train_test_split(
            df,
            train_size=30000,
            random_state=42,
            stratify=df["is_canceled"]
        )

    X = df[FEATURES]
    y = df["is_canceled"]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC),
        ("categorical", categorical_pipeline, CATEGORICAL)
    ])

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            max_depth=18,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20,
        random_state=42, stratify=y
    )

    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))

    return model, accuracy, len(df)


with st.spinner("Loading dataset and training model..."):
    try:
        model, accuracy, total_records = train_model()
    except Exception as e:
        st.error("Unable to load data or train the model.")
        st.exception(e)
        st.stop()


st.sidebar.header("Enter Booking Details")

hotel = st.sidebar.selectbox("Hotel Type", ["City Hotel", "Resort Hotel"])
lead_time = st.sidebar.number_input("Days before arrival", 0, 1000, 50)
arrival_year = st.sidebar.selectbox("Arrival Year", [2015, 2016, 2017])

arrival_month = st.sidebar.selectbox(
    "Arrival Month",
    ["January", "February", "March", "April", "May", "June",
     "July", "August", "September", "October", "November", "December"]
)

weekend_nights = st.sidebar.number_input("Weekend Nights", 0, 20, 1)
week_nights = st.sidebar.number_input("Week Nights", 0, 50, 2)
adults = st.sidebar.number_input("Adults", 0, 20, 2)
children = st.sidebar.number_input("Children", 0, 10, 0)
babies = st.sidebar.number_input("Babies", 0, 10, 0)

meal = st.sidebar.selectbox("Meal Plan", ["BB", "HB", "FB", "SC", "Undefined"])

market_segment = st.sidebar.selectbox(
    "Market Segment",
    ["Online TA", "Offline TA/TO", "Direct", "Corporate",
     "Groups", "Complementary", "Aviation", "Undefined"]
)

distribution_channel = st.sidebar.selectbox(
    "Distribution Channel", ["TA/TO", "Direct", "Corporate", "GDS", "Undefined"]
)

repeated_guest = st.sidebar.selectbox("Returning Guest", ["No", "Yes"])

previous_cancellations = st.sidebar.number_input("Previous Cancellations", 0, 50, 0)

previous_not_cancelled = st.sidebar.number_input(
    "Previous Bookings Not Cancelled", 0, 100, 0
)

room_type = st.sidebar.selectbox(
    "Reserved Room Type", ["A", "B", "C", "D", "E", "F", "G", "H", "L", "P"]
)

deposit_type = st.sidebar.selectbox(
    "Deposit Type", ["No Deposit", "Non Refund", "Refundable"]
)

customer_type = st.sidebar.selectbox(
    "Customer Type", ["Transient", "Transient-Party", "Contract", "Group"]
)

adr = st.sidebar.number_input("Average Daily Rate", 0.0, 2000.0, 100.0)
parking = st.sidebar.number_input("Parking Spaces", 0, 10, 0)
special_requests = st.sidebar.number_input("Special Requests", 0, 10, 0)


booking = pd.DataFrame([{
    "hotel": hotel,
    "lead_time": lead_time,
    "arrival_date_year": arrival_year,
    "arrival_date_month": arrival_month,
    "stays_in_weekend_nights": weekend_nights,
    "stays_in_week_nights": week_nights,
    "adults": adults,
    "children": children,
    "babies": babies,
    "meal": meal,
    "market_segment": market_segment,
    "distribution_channel": distribution_channel,
    "is_repeated_guest": 1 if repeated_guest == "Yes" else 0,
    "previous_cancellations": previous_cancellations,
    "previous_bookings_not_canceled": previous_not_cancelled,
    "reserved_room_type": room_type,
    "deposit_type": deposit_type,
    "customer_type": customer_type,
    "adr": adr,
    "required_car_parking_spaces": parking,
    "total_of_special_requests": special_requests
}])[FEATURES]


st.subheader("Booking Cancellation Prediction")

if st.button("Predict Booking Status", type="primary"):
    probability = model.predict_proba(booking)[0][1]
    prediction = model.predict(booking)[0]

    st.metric("Cancellation Probability", f"{probability:.2%}")

    if prediction == 1:
        st.error("This booking is likely to be cancelled.")
    else:
        st.success("This booking is likely to be confirmed.")

st.divider()
st.subheader("Model Performance")

col1, col2 = st.columns(2)
col1.metric("Model Accuracy", f"{accuracy:.2%}")
col2.metric("Dataset Records", f"{total_records:,}")

st.info(
    "This project uses Random Forest Classification "
    "to predict hotel booking cancellations."
)
