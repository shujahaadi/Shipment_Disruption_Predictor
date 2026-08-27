import pandas as pd
from joblib import load


model = load("models/decision_tree.pkl")
preprocessor = load("models/preprocessor.pkl")


# Example new shipment
new_shipment = pd.DataFrame([{
    "Order_Date": "2024-06-15",
    "Origin_City": "Mumbai, IN",
    "Destination_City": "Dubai, AE",
    "Route_Type": "Direct",
    "Transportation_Mode": "Air",
    "Product_Category": "Electronics",
    "Base_Lead_Time_Days": 10,
    "Scheduled_Lead_Time_Days": 12,
    "Geopolitical_Risk_Index": 0.3,
    "Weather_Severity_Index": 2,
    "Inflation_Rate_Pct": 4.5,
    "Shipping_Cost_USD": 5000,
    "Order_Weight_Kg": 200
}])


new_shipment["Order_Date"] = pd.to_datetime(
    new_shipment["Order_Date"]
)

new_shipment["Order_Month"] = (
    new_shipment["Order_Date"].dt.month
)

new_shipment["Order_DayOfWeek"] = (
    new_shipment["Order_Date"].dt.dayofweek
)


X_new = new_shipment[[
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
]]



X_new_processed = preprocessor.transform(X_new)


prediction = model.predict(X_new_processed)

print("Predicted Delivery Status:", prediction[0])