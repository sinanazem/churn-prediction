import numpy as np
import pandas as pd
import os

from sklearn import metrics
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score, classification_report, recall_score, confusion_matrix,
    roc_auc_score, precision_score, f1_score, roc_curve, auc
)
from sklearn.preprocessing import OrdinalEncoder

from catboost import CatBoostClassifier, Pool

def load_data(data_path):
    """
    Load the data from the specified path.
    """
    df = pd.read_csv(data_path)
    return df

def preprocess_data(df):
    """
    Preprocess the data, including handling missing values, encoding categorical variables, and converting the target variable.
    """
    # Convert TotalCharges to numeric, filling NaN values
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['tenure'] * df['MonthlyCharges'], inplace=True)

    df['SeniorCitizen'] = df['SeniorCitizen'].astype(object)

    df['MultipleLines'] = df['MultipleLines'].replace('No phone service', 'No')
    columns_to_replace = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    for column in columns_to_replace:
        df[column] = df[column].replace('No internet service', 'No')

    df['Churn'] = df['Churn'].replace({'No': 0, 'Yes': 1})

    return df

def split_data(df):
    """
    Split the data into training and testing sets using stratified shuffle split.
    """
    strat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=64)
    train_index, test_index = next(strat_split.split(df, df["Churn"]))

    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index]

    X_train = strat_train_set.drop("Churn", axis=1)
    y_train = strat_train_set["Churn"].copy()

    X_test = strat_test_set.drop("Churn", axis=1)
    y_test = strat_test_set["Churn"].copy()

    return X_train, y_train, X_test, y_test

def train_model(X_train, y_train, X_test, y_test):
    """
    Train the CatBoost model and evaluate its performance.
    """
    categorical_columns = X_train.select_dtypes(include=['object']).columns.tolist()

    cat_model = CatBoostClassifier(verbose=False, random_state=0, scale_pos_weight=3)
    cat_model.fit(X_train, y_train, cat_features=categorical_columns, eval_set=(X_test, y_test))

    y_pred = cat_model.predict(X_test)

    accuracy, recall, roc_auc, precision = [round(metric(y_test, y_pred), 4) for metric in [accuracy_score, recall_score, roc_auc_score, precision_score]]

    model_names = ['CatBoost_Model']
    result = pd.DataFrame({'Accuracy': accuracy, 'Recall': recall, 'Roc_Auc': roc_auc, 'Precision': precision}, index=model_names)

    return cat_model, result

def save_model(cat_model, model_dir):
    """
    Save the trained CatBoost model to the specified directory.
    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_path = os.path.join(model_dir, "catboost_model.cbm")
    cat_model.save_model(model_path)

def main():
    data_path = "/mnt/c/Users/user/OneDrive/Desktop/churn-prediction/data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    model_dir = "/mnt/c/Users/user/OneDrive/Desktop/churn-prediction/model"

    # Load data
    df = load_data(data_path)

    # Preprocess data
    df = preprocess_data(df)

    # Split data
    X_train, y_train, X_test, y_test = split_data(df)

    # Train model and evaluate
    cat_model, result = train_model(X_train, y_train, X_test, y_test)
    print(result)

    # Save model
    save_model(cat_model, model_dir)

if __name__ == "__main__":
    main()