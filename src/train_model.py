import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from joblib import dump
from sklearn.metrics import classification_report, confusion_matrix

from preprocess import prepare_data


df = pd.read_csv("data/global_supply_chain_disruption_v1.csv")


X, y, preprocessor = prepare_data(df)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)


model = DecisionTreeClassifier(
    random_state=42
)


model.fit(X_train_processed, y_train)

y_pred = model.predict(X_test_processed)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

dump(model, "models/decision_tree.pkl")
dump(preprocessor, "models/preprocessor.pkl")

print("Model and preprocessor saved successfully!")