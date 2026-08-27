import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from joblib import dump

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
    max_depth=10,
    min_samples_leaf=10,
    random_state=42
)


model.fit(X_train_processed, y_train)


dump(model, "models/decision_tree.pkl")
dump(preprocessor, "models/preprocessor.pkl")

print("Model and preprocessor saved successfully!")