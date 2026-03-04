import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(path):
    df = pd.read_csv(path)

    # Convert TotalCharges to numeric (some values are blank)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Remove missing values
    df.dropna(inplace=True)

    # Drop customerID column
    df.drop("customerID", axis=1, inplace=True)

    return df


def preprocess_data(df):
    # Convert target column (Yes/No → 1/0)
    df["Churn"] = LabelEncoder().fit_transform(df["Churn"])

    # Convert categorical columns to numbers
    df = pd.get_dummies(df, drop_first=True)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, X.columns