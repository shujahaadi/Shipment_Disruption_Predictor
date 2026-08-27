# Shipment Disruption Predictor

A machine learning project that predicts whether a shipment will be **Late** or **On Time** based on shipment, route, logistics, and external risk factors.

## Overview

This project explores the use of machine learning to predict shipment delivery status.

The dataset contains information such as:

- Origin and destination cities
- Transportation mode
- Route type
- Product category
- Scheduled lead time
- Shipping cost
- Order weight
- Weather severity
- Geopolitical risk
- Inflation
- Order date

The project covers the complete machine learning workflow:

**Data preprocessing → Feature engineering → Model training → Evaluation → Prediction**

## Tech Stack

- **Python**
- **Pandas** — data manipulation and analysis
- **NumPy** — numerical operations
- **Scikit-learn** — preprocessing, model training, and evaluation
- **Matplotlib & Seaborn** — data visualization
- **Jupyter Notebook** — experimentation and analysis
- **Joblib** — saving and loading trained models

## Project Structure

```text
Shipment_Disruption_Predictor/
│
├── data/
│   └── global_supply_chain_disruption_v1.csv
│
├── models/
│   ├── decision_tree.pkl
│   └── preprocessor.pkl
│
├── notebooks/
│   └── exploration.ipynb
│
├── outputs/
│
├── src/
│   ├── preprocess.py
│   ├── train_model.py
│   └── predict.py
│
├── .gitignore
├── README.md
└── requirements.txt

```

## Model Performance

The project evaluates a Decision Tree classifier and explores Random Forest as a comparison model.

### Final Tuned Decision Tree

- **Accuracy:** 93.1%
- **Late Precision:** 81%
- **Late Recall:** 59%
- **Late F1-Score:** 68%

The Decision Tree was tuned using `max_depth` and `min_samples_leaf` to improve generalization and balance model performance.

### Model Selection

Decision Tree hyperparameters were tuned using `max_depth` and `min_samples_leaf`.

Random Forest will be explored as the next model to compare against the tuned Decision Tree.

## How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd Shipment_Disruption_Predictor
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

**Windows PowerShell:**

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Train the model

```bash
python src/train_model.py
```

This will train the Decision Tree and save the trained model and preprocessor inside the `models/` directory.

### 6. Make a prediction

```bash
python src/predict.py
```

The script will output the predicted delivery status:

```text
Predicted Delivery Status: On Time
```