import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

from data_preprocessing import load_data, preprocess_data

print("Loading dataset...")

# Load CSV file
df = load_data("../data/churn.csv")

print("Preprocessing data...")
X, y, scaler, feature_names = preprocess_data(df)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training model...")

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,
    scoring="accuracy"
)

grid.fit(X_train, y_train)

model = grid.best_estimator_

# Evaluation
y_pred = model.predict(X_test)

print("\nBest Parameters:", grid.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model
os.makedirs("../models", exist_ok=True)

joblib.dump(model, "../models/churn_model.pkl")
joblib.dump(scaler, "../models/scaler.pkl")
joblib.dump(feature_names, "../models/feature_names.pkl")

print("\nModel saved successfully!")

import matplotlib.pyplot as plt
import pandas as pd

# Feature Importance
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10,6))
plt.barh(feature_importance_df["Feature"][:10],
         feature_importance_df["Importance"][:10])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 10 Feature Importances")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("../models/feature_importance.png")