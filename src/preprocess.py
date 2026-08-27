import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


categorical_features = [
    "Origin_City",
    "Destination_City",
    "Route_Type",
    "Transportation_Mode",
    "Product_Category"
]

features = [
    "Origin_City",
    "Destination_City",
    "Route_Type",
    "Transportation_Mode",
    "Product_Category",
    "Base_Lead_Time_Days",
    "Scheduled_Lead_Time_Days",
    "Geopolitical_Risk_Index",
    "Weather_Severity_Index",
    "Inflation_Rate_Pct",
    "Shipping_Cost_USD",
    "Order_Weight_Kg",
    "Order_Month",
    "Order_DayOfWeek"
]


def prepare_data(df):

    df = df.copy()

    df["Order_Date"] = pd.to_datetime(df["Order_Date"])

    df["Order_Month"] = df["Order_Date"].dt.month
    df["Order_DayOfWeek"] = df["Order_Date"].dt.dayofweek

    X = df[features]
    y = df["Delivery_Status"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            )
        ],
        remainder="passthrough"
    )

    return X, y, preprocessor