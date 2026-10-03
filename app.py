import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


st.title("Customer Churn Prediction")
st.write("Predict whether a telecom customer is likely to churn.")


# -----------------------------
# Load dataset
# -----------------------------

df = pd.read_csv("data/Telco_customer_churn(Telco_Churn).csv")


features = [
    "Gender", "Senior Citizen", "Partner", "Dependents",
    "Tenure Months", "Phone Service", "Multiple Lines",
    "Internet Service", "Online Security", "Online Backup",
    "Device Protection", "Tech Support", "Streaming TV",
    "Streaming Movies", "Contract", "Paperless Billing",
    "Payment Method", "Monthly Charges", "Total Charges"
]


X = df[features].copy()
y = df["Churn Label"].map({"No": 0, "Yes": 1})


# Fix Total Charges
X["Total Charges"] = pd.to_numeric(
    X["Total Charges"], errors="coerce"
)

X["Total Charges"] = X["Total Charges"].fillna(
    X["Total Charges"].median()
)


# -----------------------------
# Train model
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


categorical_cols = X.select_dtypes(
    include=["object", "string"]
).columns

numerical_cols = X.select_dtypes(
    include=["int64", "float64"]
).columns


preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
])


model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])


model.fit(X_train, y_train)


# -----------------------------
# User inputs
# -----------------------------

st.header("Customer Information")

input_data = {}

for col in categorical_cols:
    input_data[col] = st.selectbox(
        col,
        sorted(X[col].unique())
    )


for col in numerical_cols:
    input_data[col] = st.number_input(
        col,
        value=float(X[col].median())
    )


# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict Churn"):

    customer = pd.DataFrame([input_data])

    prediction = model.predict(customer)[0]

    probability = model.predict_proba(customer)[0][1]

    if prediction == 1:
        st.error(
            f"Customer is likely to churn "
            f"(probability: {probability:.2%})"
        )
    else:
        st.success(
            f"Customer is likely to stay "
            f"(churn probability: {probability:.2%})"
        )