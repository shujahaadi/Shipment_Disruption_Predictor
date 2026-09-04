import pandas as pd
from joblib import load


model = load("models/decision_tree.pkl")
preprocessor = load("models/preprocessor.pkl")


def predict_shipment(shipment):
    new_shipment = pd.DataFrame([shipment])


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

    return prediction[0]